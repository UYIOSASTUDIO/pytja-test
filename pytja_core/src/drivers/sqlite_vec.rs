// =============================================================================
// PYTJA SQLITE VECTOR DRIVER
// =============================================================================
//
// Zero-dependency embedded vector store using SQLite for persistence and
// pure Rust for similarity search. No external extensions required.
//
// Architecture:
//   - Vectors stored as BLOBs (little-endian f32 arrays) in SQLite
//   - Metadata stored as JSON TEXT columns
//   - Similarity search computed in Rust (not SQL) for portability
//   - Optimized batch loading with configurable scan size
//
// Trade-offs vs PgVector:
//   + Zero infrastructure dependencies (just a file)
//   + Works in Zero-Touch Bootstrap (no PostgreSQL needed)
//   + Portable across all platforms
//   - Linear scan instead of HNSW (suitable for <500k vectors per collection)
//   - No server-side filtering pushdown (filter applied post-scan in Rust)
//
// This driver is ideal for:
//   - Local development and testing
//   - Small to medium datasets
//   - Edge/IoT deployments
//   - Self-hosted single-user instances
//

use crate::repo::VectorStore;
use crate::error::PytjaError;
use crate::models::{
    VectorPoint, VectorSearchResult, VectorCollectionConfig,
    VectorFilter, DistanceMetric,
};
use async_trait::async_trait;
use sqlx::sqlite::{SqlitePool, SqliteConnectOptions};
use sqlx::Row;
use std::str::FromStr;
use tracing::info;

// =============================================================================
// PURE RUST VECTOR MATH
// =============================================================================

/// Compute cosine similarity between two vectors.
/// Returns a value between -1.0 and 1.0 (1.0 = identical direction).
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    // Process in chunks of 4 for better CPU pipeline utilization
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let a0 = a[base] as f64;
        let a1 = a[base + 1] as f64;
        let a2 = a[base + 2] as f64;
        let a3 = a[base + 3] as f64;
        let b0 = b[base] as f64;
        let b1 = b[base + 1] as f64;
        let b2 = b[base + 2] as f64;
        let b3 = b[base + 3] as f64;

        dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    }

    // Handle remaining elements
    let base = chunks * 4;
    for i in 0..remainder {
        let av = a[base + i] as f64;
        let bv = b[base + i] as f64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    (dot / denom) as f32
}

/// Compute L2 (Euclidean) distance between two vectors.
/// Returns a non-negative value (0.0 = identical).
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = (a[base] - b[base]) as f64;
        let d1 = (a[base + 1] - b[base + 1]) as f64;
        let d2 = (a[base + 2] - b[base + 2]) as f64;
        let d3 = (a[base + 3] - b[base + 3]) as f64;
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        let d = (a[base + i] - b[base + i]) as f64;
        sum += d * d;
    }

    sum.sqrt() as f32
}

/// Compute inner product (dot product) between two vectors.
#[inline]
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        sum += (a[base] as f64) * (b[base] as f64)
            + (a[base + 1] as f64) * (b[base + 1] as f64)
            + (a[base + 2] as f64) * (b[base + 2] as f64)
            + (a[base + 3] as f64) * (b[base + 3] as f64);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        sum += (a[base + i] as f64) * (b[base + i] as f64);
    }

    sum as f32
}

/// Compute a similarity score between two vectors given a distance metric.
/// Higher score = more similar (regardless of metric).
fn compute_score(a: &[f32], b: &[f32], metric: &DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(a, b),
        DistanceMetric::L2 => {
            let dist = l2_distance(a, b);
            1.0 / (1.0 + dist) // Convert distance to similarity
        },
        DistanceMetric::InnerProduct => inner_product(a, b),
    }
}

// =============================================================================
// BLOB SERIALIZATION
// =============================================================================

/// Serialize a Vec<f32> to a little-endian byte array (BLOB).
fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

/// Deserialize a little-endian byte array (BLOB) back to Vec<f32>.
fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    let count = blob.len() / 4;
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * 4;
        let bytes = [blob[offset], blob[offset + 1], blob[offset + 2], blob[offset + 3]];
        result.push(f32::from_le_bytes(bytes));
    }
    result
}

// =============================================================================
// SQLITE VEC DRIVER
// =============================================================================

/// Embedded vector store driver using SQLite for persistence and
/// pure Rust for similarity computation.
#[derive(Clone)]
pub struct SqliteVecDriver {
    pool: SqlitePool,
}

impl SqliteVecDriver {
    /// Create a new SqliteVecDriver. The path should be a filesystem path
    /// to the SQLite database file (will be created if it doesn't exist).
    ///
    /// Example: `data/vectors.db`
    pub async fn new(path: &str) -> Result<Self, PytjaError> {
        // 1. Enterprise Fix: Stripping the protocol prefix if it exists
        let clean_path = path.strip_prefix("sqlite://").unwrap_or(path);

        // 2. Ensure parent directory exists using the clean path
        if let Some(parent) = std::path::Path::new(clean_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| PytjaError::System(format!("Failed to create directory: {}", e)))?;
            }
        }

        // 3. Re-apply exactly one protocol prefix for sqlx
        let conn_str = format!("sqlite://{}", clean_path);
        let options = SqliteConnectOptions::from_str(&conn_str)
            .map_err(|e| PytjaError::VectorStoreUnavailable(
                format!("Invalid SQLite path '{}': {}", clean_path, e)
            ))?
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            // Performance tuning for vector workloads
            .pragma("cache_size", "-64000")      // 64MB page cache
            .pragma("mmap_size", "268435456")     // 256MB memory-mapped I/O
            .pragma("temp_store", "memory");       // Temp tables in memory

        let pool = SqlitePool::connect_with(options).await
            .map_err(|e| PytjaError::VectorStoreUnavailable(
                format!("Failed to open SQLite vector store: {}", e)
            ))?;

        Ok(Self { pool })
    }

    /// Sanitize collection name — alphanumeric + underscore only.
    fn sanitize_name(name: &str) -> Result<String, PytjaError> {
        let clean: String = name.chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect();

        if clean.is_empty() || clean.len() != name.len() {
            return Err(PytjaError::VectorOperationError(
                format!("Invalid collection name '{}'. Use only alphanumeric characters and underscores.", name)
            ));
        }

        if clean.len() > 63 {
            return Err(PytjaError::VectorOperationError(
                "Collection name too long (max 63 characters).".into()
            ));
        }

        Ok(format!("pytja_vec_{}", clean))
    }

    /// Check if a vector point passes a metadata filter.
    fn matches_filter(metadata: &Option<serde_json::Value>, filter: &VectorFilter) -> bool {
        // Check owner filter
        // (owner is checked separately in the SQL query, so skip here)

        // Check JSON metadata filter (containment check)
        if let Some(filter_json) = &filter.filter_json {
            if let Some(filter_obj) = filter_json.as_object() {
                match metadata {
                    Some(meta) => {
                        if let Some(meta_obj) = meta.as_object() {
                            for (key, expected_val) in filter_obj {
                                match meta_obj.get(key) {
                                    Some(actual_val) if actual_val == expected_val => continue,
                                    _ => return false,
                                }
                            }
                        } else {
                            return false;
                        }
                    },
                    None => return false,
                }
            }
        }

        true
    }
}

#[async_trait]
impl VectorStore for SqliteVecDriver {

    async fn vector_init(&self) -> Result<(), PytjaError> {
        // Create the collections metadata table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS pytja_vector_collections (
                name TEXT PRIMARY KEY,
                table_name TEXT NOT NULL UNIQUE,
                dimension INTEGER NOT NULL,
                distance_metric TEXT NOT NULL,
                owner TEXT NOT NULL,
                created_at REAL NOT NULL
            )"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        info!("SQLite vector store initialized successfully.");
        Ok(())
    }

    // =========================================================================
    // COLLECTION MANAGEMENT
    // =========================================================================

    async fn create_collection(
        &self,
        name: &str,
        dimension: u32,
        metric: DistanceMetric,
        owner: &str,
    ) -> Result<(), PytjaError> {
        let table_name = Self::sanitize_name(name)?;

        // Check if collection already exists
        let existing: Option<(String,)> = sqlx::query_as(
            "SELECT name FROM pytja_vector_collections WHERE name = ?1"
        )
            .bind(name)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if existing.is_some() {
            return Err(PytjaError::CollectionAlreadyExists(name.to_string()));
        }

        // Create vector data table
        // - embedding stored as BLOB (little-endian f32 array)
        // - metadata stored as JSON TEXT
        let create_sql = format!(
            "CREATE TABLE {} (
                id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                metadata TEXT DEFAULT '{{}}',
                owner TEXT NOT NULL,
                created_at REAL NOT NULL
            )",
            table_name
        );

        sqlx::query(&create_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(
                format!("Failed to create collection table: {}", e)
            ))?;

        // Create index on owner for RBAC-filtered queries
        let owner_idx = format!(
            "CREATE INDEX IF NOT EXISTS {}_owner_idx ON {} (owner)",
            table_name, table_name
        );

        sqlx::query(&owner_idx)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(
                format!("Failed to create owner index: {}", e)
            ))?;

        // Register in metadata table
        let now = chrono::Utc::now().timestamp() as f64;
        sqlx::query(
            "INSERT INTO pytja_vector_collections (name, table_name, dimension, distance_metric, owner, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
            .bind(name)
            .bind(&table_name)
            .bind(dimension as i32)
            .bind(metric.to_string())
            .bind(owner)
            .bind(now)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        info!("Created SQLite vector collection '{}' (dim={}, metric={})", name, dimension, metric);
        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> Result<(), PytjaError> {
        let _config = self.get_collection(name).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(name.to_string()))?;

        let table_name = Self::sanitize_name(name)?;

        let drop_sql = format!("DROP TABLE IF EXISTS {}", table_name);
        sqlx::query(&drop_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        sqlx::query("DELETE FROM pytja_vector_collections WHERE name = ?1")
            .bind(name)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        info!("Deleted SQLite vector collection '{}'", name);
        Ok(())
    }

    async fn list_collections(&self) -> Result<Vec<VectorCollectionConfig>, PytjaError> {
        let rows = sqlx::query(
            "SELECT name, dimension, distance_metric, owner, created_at FROM pytja_vector_collections ORDER BY created_at DESC"
        )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut collections = Vec::new();
        for row in rows {
            let metric_str: String = row.try_get("distance_metric").unwrap_or_else(|_| "cosine".into());
            collections.push(VectorCollectionConfig {
                name: row.try_get("name").unwrap_or_default(),
                dimension: row.try_get::<i32, _>("dimension").unwrap_or(0) as u32,
                distance_metric: metric_str.parse().unwrap_or(DistanceMetric::Cosine),
                owner: row.try_get("owner").unwrap_or_default(),
                created_at: row.try_get("created_at").unwrap_or(0.0),
            });
        }

        Ok(collections)
    }

    async fn get_collection(&self, name: &str) -> Result<Option<VectorCollectionConfig>, PytjaError> {
        let row = sqlx::query(
            "SELECT name, dimension, distance_metric, owner, created_at FROM pytja_vector_collections WHERE name = ?1"
        )
            .bind(name)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if let Some(r) = row {
            let metric_str: String = r.try_get("distance_metric").unwrap_or_else(|_| "cosine".into());
            Ok(Some(VectorCollectionConfig {
                name: r.try_get("name").unwrap_or_default(),
                dimension: r.try_get::<i32, _>("dimension").unwrap_or(0) as u32,
                distance_metric: metric_str.parse().unwrap_or(DistanceMetric::Cosine),
                owner: r.try_get("owner").unwrap_or_default(),
                created_at: r.try_get("created_at").unwrap_or(0.0),
            }))
        } else {
            Ok(None)
        }
    }

    // =========================================================================
    // VECTOR CRUD
    // =========================================================================

    async fn vector_upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> Result<u64, PytjaError> {
        if points.is_empty() {
            return Ok(0);
        }

        let config = self.get_collection(collection).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(collection.to_string()))?;

        let table_name = Self::sanitize_name(collection)?;

        // Validate dimensions
        for point in &points {
            if point.embedding.len() as u32 != config.dimension {
                return Err(PytjaError::DimensionMismatch {
                    expected: config.dimension,
                    actual: point.embedding.len() as u32,
                });
            }
        }

        // Batch upsert with transaction
        let mut tx = self.pool.begin().await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let upsert_sql = format!(
            "INSERT INTO {} (id, embedding, metadata, owner, created_at) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding, metadata = excluded.metadata, created_at = excluded.created_at",
            table_name
        );

        let mut count = 0u64;
        for point in &points {
            let blob = embedding_to_blob(&point.embedding);
            let metadata_str = match &point.metadata {
                Some(m) => m.to_string(),
                None => "{}".to_string(),
            };

            sqlx::query(&upsert_sql)
                .bind(&point.id)
                .bind(&blob)
                .bind(&metadata_str)
                .bind(&point.owner)
                .bind(point.created_at)
                .execute(&mut *tx)
                .await
                .map_err(|e| PytjaError::DatabaseError(
                    format!("Failed to upsert vector '{}': {}", point.id, e)
                ))?;

            count += 1;
        }

        tx.commit().await
            .map_err(|e| PytjaError::DatabaseError(format!("Transaction commit failed: {}", e)))?;

        Ok(count)
    }

    async fn vector_get(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> Result<Vec<VectorPoint>, PytjaError> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let _config = self.get_collection(collection).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(collection.to_string()))?;

        let table_name = Self::sanitize_name(collection)?;

        // SQLite uses ? for params, build the IN clause
        let placeholders: Vec<String> = ids.iter().enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();

        let query_sql = format!(
            "SELECT id, embedding, metadata, owner, created_at FROM {} WHERE id IN ({})",
            table_name,
            placeholders.join(", ")
        );

        let mut query = sqlx::query(&query_sql);
        for id in &ids {
            query = query.bind(id);
        }

        let rows = query.fetch_all(&self.pool).await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let blob: Vec<u8> = row.try_get("embedding").unwrap_or_default();
            let embedding = blob_to_embedding(&blob);
            let metadata_str: String = row.try_get("metadata").unwrap_or_else(|_| "{}".into());
            let metadata: Option<serde_json::Value> = serde_json::from_str(&metadata_str).ok();

            results.push(VectorPoint {
                id: row.try_get("id").unwrap_or_default(),
                embedding,
                metadata,
                owner: row.try_get("owner").unwrap_or_default(),
                created_at: row.try_get("created_at").unwrap_or(0.0),
            });
        }

        Ok(results)
    }

    async fn vector_delete(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> Result<u64, PytjaError> {
        if ids.is_empty() {
            return Ok(0);
        }

        let _config = self.get_collection(collection).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(collection.to_string()))?;

        let table_name = Self::sanitize_name(collection)?;

        let placeholders: Vec<String> = ids.iter().enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();

        let delete_sql = format!(
            "DELETE FROM {} WHERE id IN ({})",
            table_name,
            placeholders.join(", ")
        );

        let mut query = sqlx::query(&delete_sql);
        for id in &ids {
            query = query.bind(id);
        }

        let result = query.execute(&self.pool).await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        Ok(result.rows_affected())
    }

    async fn vector_delete_by_filter(
        &self,
        collection: &str,
        filter: VectorFilter,
    ) -> Result<u64, PytjaError> {
        let _config = self.get_collection(collection).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(collection.to_string()))?;

        let table_name = Self::sanitize_name(collection)?;

        // For SQLite, we load matching IDs first then delete in bulk.
        // This is because SQLite doesn't have JSONB operators.
        let mut conditions = Vec::new();
        let mut params: Vec<String> = Vec::new();

        if let Some(owner) = &filter.owner_filter {
            conditions.push(format!("owner = ?{}", params.len() + 1));
            params.push(owner.clone());
        }

        if conditions.is_empty() && filter.filter_json.is_some() {
            // For JSON metadata filtering in SQLite, we need to load and check in Rust
            // Load all IDs and metadata, filter in memory, then delete matching IDs
            let select_sql = format!("SELECT id, metadata FROM {}", table_name);
            let rows = sqlx::query(&select_sql)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

            let mut ids_to_delete = Vec::new();
            for row in rows {
                let id: String = row.try_get("id").unwrap_or_default();
                let metadata_str: String = row.try_get("metadata").unwrap_or_else(|_| "{}".into());
                let metadata: Option<serde_json::Value> = serde_json::from_str(&metadata_str).ok();

                if Self::matches_filter(&metadata, &filter) {
                    ids_to_delete.push(id);
                }
            }

            if ids_to_delete.is_empty() {
                return Ok(0);
            }

            return self.vector_delete(collection, ids_to_delete).await;
        }

        if conditions.is_empty() {
            return Err(PytjaError::VectorOperationError(
                "Cannot delete by filter with empty filter.".into()
            ));
        }

        let delete_sql = format!("DELETE FROM {} WHERE {}", table_name, conditions.join(" AND "));
        let mut query = sqlx::query(&delete_sql);
        for param in &params {
            query = query.bind(param);
        }

        let result = query.execute(&self.pool).await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        Ok(result.rows_affected())
    }

    // =========================================================================
    // SIMILARITY SEARCH
    // =========================================================================

    async fn vector_search(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        top_k: u32,
        filter: Option<VectorFilter>,
    ) -> Result<Vec<VectorSearchResult>, PytjaError> {
        let config = self.get_collection(collection).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(collection.to_string()))?;

        if query_vector.len() as u32 != config.dimension {
            return Err(PytjaError::DimensionMismatch {
                expected: config.dimension,
                actual: query_vector.len() as u32,
            });
        }

        let table_name = Self::sanitize_name(collection)?;

        // Build base query — optionally filter by owner at the SQL level
        let (select_sql, owner_param) = if let Some(ref f) = filter {
            if let Some(ref owner) = f.owner_filter {
                (
                    format!("SELECT id, embedding, metadata, owner FROM {} WHERE owner = ?1", table_name),
                    Some(owner.clone()),
                )
            } else {
                (format!("SELECT id, embedding, metadata, owner FROM {}", table_name), None)
            }
        } else {
            (format!("SELECT id, embedding, metadata, owner FROM {}", table_name), None)
        };

        let mut query = sqlx::query(&select_sql);
        if let Some(ref owner) = owner_param {
            query = query.bind(owner);
        }

        let rows = query.fetch_all(&self.pool).await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        // Compute similarities in Rust
        // Use a min-heap (BinaryHeap) to efficiently track top-k results
        let mut scored: Vec<(f32, String, Option<serde_json::Value>, String)> = Vec::with_capacity(rows.len());

        for row in &rows {
            let id: String = row.try_get("id").unwrap_or_default();
            let blob: Vec<u8> = row.try_get("embedding").unwrap_or_default();
            let metadata_str: String = row.try_get("metadata").unwrap_or_else(|_| "{}".into());
            let metadata: Option<serde_json::Value> = serde_json::from_str(&metadata_str).ok();
            let owner: String = row.try_get("owner").unwrap_or_default();

            // Apply metadata filter if present
            if let Some(ref f) = filter {
                if f.filter_json.is_some() && !Self::matches_filter(&metadata, f) {
                    continue;
                }
            }

            let embedding = blob_to_embedding(&blob);
            if embedding.len() as u32 != config.dimension {
                continue; // Skip corrupted entries
            }

            let score = compute_score(&query_vector, &embedding, &config.distance_metric);
            scored.push((score, id, metadata, owner));
        }

        // Sort by score descending (higher = more similar)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        let results: Vec<VectorSearchResult> = scored.into_iter()
            .take(top_k as usize)
            .map(|(score, id, metadata, owner)| VectorSearchResult {
                id,
                score,
                metadata,
                owner,
            })
            .collect();

        Ok(results)
    }

    // =========================================================================
    // COLLECTION STATS
    // =========================================================================

    async fn vector_count(&self, collection: &str) -> Result<u64, PytjaError> {
        let _config = self.get_collection(collection).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(collection.to_string()))?;

        let table_name = Self::sanitize_name(collection)?;
        let count_sql = format!("SELECT COUNT(*) FROM {}", table_name);

        let row: (i64,) = sqlx::query_as(&count_sql)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        Ok(row.0 as u64)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_roundtrip() {
        let original = vec![0.1f32, 0.2, 0.3, -0.5, 1.0, 0.0];
        let blob = embedding_to_blob(&original);
        let recovered = blob_to_embedding(&blob);
        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_zero() {
        let v = vec![1.0f32, 2.0, 3.0];
        let dist = l2_distance(&v, &v);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let ip = inner_product(&a, &b);
        assert!((ip - 32.0).abs() < 1e-5); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_large_vector_performance() {
        // Simulate 1536-dim OpenAI embeddings
        let a: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..1536).map(|i| ((1536 - i) as f32) * 0.001).collect();
        let _ = cosine_similarity(&a, &b);
        let _ = l2_distance(&a, &b);
        let _ = inner_product(&a, &b);
        // No assertion needed — just verifying it doesn't panic
    }
}
