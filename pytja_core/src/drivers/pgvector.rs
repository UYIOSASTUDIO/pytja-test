// =============================================================================
// PYTJA PGVECTOR DRIVER
// =============================================================================
//
// Production-grade vector store driver using PostgreSQL + pgvector extension.
// Supports HNSW indexing, cosine/L2/inner product distance metrics,
// JSON metadata filtering, and RBAC-aware queries.
//
// Prerequisites:
//   - PostgreSQL 15+ with pgvector extension installed
//   - CREATE EXTENSION vector; (executed automatically on init)
//

use crate::repo::VectorStore;
use crate::error::PytjaError;
use crate::models::{
    VectorPoint, VectorSearchResult, VectorCollectionConfig,
    VectorFilter, DistanceMetric,
};
use async_trait::async_trait;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use tracing::{info, warn};

/// PgVector driver — connects to a PostgreSQL instance with the pgvector extension.
#[derive(Clone)]
pub struct PgVectorDriver {
    pool: PgPool,
}

impl PgVectorDriver {
    /// Create a new PgVectorDriver connected to the given PostgreSQL URL.
    ///
    /// The URL should be a standard PostgreSQL connection string, e.g.:
    /// `postgres://user:pass@host:5432/dbname`
    pub async fn new(url: &str) -> Result<Self, PytjaError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .min_connections(2)
            .acquire_timeout(std::time::Duration::from_secs(10))
            .connect(url)
            .await
            .map_err(|e| PytjaError::VectorStoreUnavailable(
                format!("Failed to connect to PostgreSQL for pgvector: {}", e)
            ))?;

        Ok(Self { pool })
    }

    /// Sanitize a collection name to prevent SQL injection.
    /// Only allows alphanumeric characters and underscores.
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

    /// Convert a Vec<f32> embedding to pgvector text format: '[0.1,0.2,0.3]'
    fn embedding_to_pgvector_text(embedding: &[f32]) -> String {
        let inner: Vec<String> = embedding.iter().map(|v| format!("{}", v)).collect();
        format!("[{}]", inner.join(","))
    }

    /// Build a WHERE clause from a VectorFilter for metadata/owner filtering.
    fn build_filter_clause(filter: &VectorFilter, param_offset: usize) -> (String, Vec<String>) {
        let mut conditions = Vec::new();
        let mut params = Vec::new();

        if let Some(owner) = &filter.owner_filter {
            conditions.push(format!("owner = ${}", param_offset + params.len() + 1));
            params.push(owner.clone());
        }

        if let Some(filter_json) = &filter.filter_json {
            // Support flat key-value filter: {"key": "value", "key2": "value2"}
            if let Some(obj) = filter_json.as_object() {
                for (key, value) in obj {
                    // Use JSONB containment operator for efficient indexed lookup
                    let json_fragment = serde_json::json!({ key: value });
                    conditions.push(format!(
                        "metadata @> ${}::jsonb",
                        param_offset + params.len() + 1
                    ));
                    params.push(json_fragment.to_string());
                }
            }
        }

        if conditions.is_empty() {
            ("".to_string(), params)
        } else {
            (format!("WHERE {}", conditions.join(" AND ")), params)
        }
    }
}

#[async_trait]
impl VectorStore for PgVectorDriver {

    async fn vector_init(&self) -> Result<(), PytjaError> {
        // Enable the pgvector extension
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::VectorStoreUnavailable(
                format!("Failed to create pgvector extension. Is pgvector installed? Error: {}", e)
            ))?;

        // Create the collections metadata table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS pytja_vector_collections (
                name TEXT PRIMARY KEY,
                table_name TEXT NOT NULL UNIQUE,
                dimension INTEGER NOT NULL,
                distance_metric TEXT NOT NULL,
                owner TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL
            )"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        info!("PgVector store initialized successfully.");
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
            "SELECT name FROM pytja_vector_collections WHERE name = $1"
        )
            .bind(name)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if existing.is_some() {
            return Err(PytjaError::CollectionAlreadyExists(name.to_string()));
        }

        // Create the vector data table with HNSW index
        let create_table_sql = format!(
            "CREATE TABLE {} (
                id TEXT PRIMARY KEY,
                embedding vector({}) NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                owner TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL
            )",
            table_name, dimension
        );

        sqlx::query(&create_table_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(
                format!("Failed to create collection table: {}", e)
            ))?;

        // Create HNSW index for fast approximate nearest neighbor search
        // HNSW is preferred over IVFFlat for production: no training step needed,
        // better recall at comparable speed.
        let index_sql = format!(
            "CREATE INDEX IF NOT EXISTS {}_embedding_idx ON {} USING hnsw (embedding {}) WITH (m = 16, ef_construction = 200)",
            table_name, table_name, metric.to_pgvector_ops()
        );

        sqlx::query(&index_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(
                format!("Failed to create HNSW index: {}", e)
            ))?;

        // Create GIN index on metadata for fast JSON filtering
        let metadata_index_sql = format!(
            "CREATE INDEX IF NOT EXISTS {}_metadata_idx ON {} USING gin (metadata jsonb_path_ops)",
            table_name, table_name
        );

        sqlx::query(&metadata_index_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(
                format!("Failed to create metadata GIN index: {}", e)
            ))?;

        // Create index on owner for RBAC queries
        let owner_index_sql = format!(
            "CREATE INDEX IF NOT EXISTS {}_owner_idx ON {} (owner)",
            table_name, table_name
        );

        sqlx::query(&owner_index_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(
                format!("Failed to create owner index: {}", e)
            ))?;

        // Register collection in metadata table
        let now = chrono::Utc::now().timestamp() as f64;
        sqlx::query(
            "INSERT INTO pytja_vector_collections (name, table_name, dimension, distance_metric, owner, created_at) VALUES ($1, $2, $3, $4, $5, $6)"
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

        info!("Created vector collection '{}' (dim={}, metric={}, table={})", name, dimension, metric, table_name);
        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> Result<(), PytjaError> {
        let config = self.get_collection(name).await?
            .ok_or_else(|| PytjaError::CollectionNotFound(name.to_string()))?;

        let table_name = Self::sanitize_name(name)?;

        // Drop the vector data table
        let drop_sql = format!("DROP TABLE IF EXISTS {} CASCADE", table_name);
        sqlx::query(&drop_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        // Remove from metadata registry
        sqlx::query("DELETE FROM pytja_vector_collections WHERE name = $1")
            .bind(name)
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        info!("Deleted vector collection '{}'", name);
        Ok(())
    }

    async fn list_collections(&self) -> Result<Vec<VectorCollectionConfig>, PytjaError> {
        let rows = sqlx::query("SELECT name, dimension, distance_metric, owner, created_at FROM pytja_vector_collections ORDER BY created_at DESC")
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
        let row = sqlx::query("SELECT name, dimension, distance_metric, owner, created_at FROM pytja_vector_collections WHERE name = $1")
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

        // Batch upsert using a transaction for atomicity and performance
        let mut tx = self.pool.begin().await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let upsert_sql = format!(
            "INSERT INTO {} (id, embedding, metadata, owner, created_at) VALUES ($1, $2::vector, $3::jsonb, $4, $5)
             ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata, created_at = EXCLUDED.created_at",
            table_name
        );

        let mut count = 0u64;
        for point in &points {
            let vec_text = Self::embedding_to_pgvector_text(&point.embedding);
            let metadata_str = match &point.metadata {
                Some(m) => m.to_string(),
                None => "{}".to_string(),
            };

            sqlx::query(&upsert_sql)
                .bind(&point.id)
                .bind(&vec_text)
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

        // Build parameterized IN clause
        let placeholders: Vec<String> = ids.iter().enumerate()
            .map(|(i, _)| format!("${}", i + 1))
            .collect();

        let query_sql = format!(
            "SELECT id, embedding::text, metadata, owner, created_at FROM {} WHERE id IN ({})",
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
            let embedding_text: String = row.try_get("embedding").unwrap_or_default();
            let embedding = Self::parse_pgvector_text(&embedding_text);

            let metadata_value: Option<serde_json::Value> = row.try_get("metadata").ok();

            results.push(VectorPoint {
                id: row.try_get("id").unwrap_or_default(),
                embedding,
                metadata: metadata_value,
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
            .map(|(i, _)| format!("${}", i + 1))
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
        let (where_clause, params) = Self::build_filter_clause(&filter, 0);

        if where_clause.is_empty() {
            return Err(PytjaError::VectorOperationError(
                "Cannot delete by filter with empty filter. Use vector_delete with IDs instead.".into()
            ));
        }

        let delete_sql = format!("DELETE FROM {} {}", table_name, where_clause);
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

        // Validate query vector dimension
        if query_vector.len() as u32 != config.dimension {
            return Err(PytjaError::DimensionMismatch {
                expected: config.dimension,
                actual: query_vector.len() as u32,
            });
        }

        let table_name = Self::sanitize_name(collection)?;
        let operator = config.distance_metric.to_pgvector_operator();
        let vec_text = Self::embedding_to_pgvector_text(&query_vector);

        // Build query with optional filter
        let (where_clause, filter_params) = if let Some(ref f) = filter {
            // param_offset = 2 because $1 = query_vector, $2 = top_k
            Self::build_filter_clause(f, 2)
        } else {
            ("".to_string(), vec![])
        };

        // Set ef_search for HNSW — higher values give better recall at cost of latency.
        // Using 2x top_k or minimum 100 for good recall.
        let ef_search = std::cmp::max(top_k * 2, 100);
        let set_ef = format!("SET LOCAL hnsw.ef_search = {}", ef_search);

        // Use a transaction to scope the SET LOCAL
        let mut tx = self.pool.begin().await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        sqlx::query(&set_ef)
            .execute(&mut *tx)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let search_sql = format!(
            "SELECT id, metadata, owner, (embedding {} $1::vector) AS distance
             FROM {} {}
             ORDER BY embedding {} $1::vector
             LIMIT $2",
            operator, table_name, where_clause, operator
        );

        let mut query = sqlx::query(&search_sql)
            .bind(&vec_text)
            .bind(top_k as i32);

        for param in &filter_params {
            query = query.bind(param);
        }

        let rows = query.fetch_all(&mut *tx).await
            .map_err(|e| PytjaError::DatabaseError(
                format!("Vector search failed: {}", e)
            ))?;

        tx.commit().await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let distance: f64 = row.try_get("distance").unwrap_or(f64::MAX);
            let metadata_value: Option<serde_json::Value> = row.try_get("metadata").ok();

            // Convert distance to a similarity score:
            // - Cosine distance: score = 1.0 - distance (pgvector cosine returns distance, not similarity)
            // - L2 distance: score = 1.0 / (1.0 + distance)
            // - Inner product: score = -distance (pgvector negates for ORDER BY compatibility)
            let score = match config.distance_metric {
                DistanceMetric::Cosine => 1.0 - distance as f32,
                DistanceMetric::L2 => 1.0 / (1.0 + distance as f32),
                DistanceMetric::InnerProduct => -(distance as f32),
            };

            results.push(VectorSearchResult {
                id: row.try_get("id").unwrap_or_default(),
                score,
                metadata: metadata_value,
                owner: row.try_get("owner").unwrap_or_default(),
            });
        }

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
// HELPER FUNCTIONS
// =============================================================================

impl PgVectorDriver {
    /// Parse pgvector text format '[0.1,0.2,0.3]' back to Vec<f32>.
    fn parse_pgvector_text(text: &str) -> Vec<f32> {
        let trimmed = text.trim_start_matches('[').trim_end_matches(']');
        if trimmed.is_empty() {
            return vec![];
        }
        trimmed.split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect()
    }
}
