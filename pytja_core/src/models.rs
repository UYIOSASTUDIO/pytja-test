use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use sqlx::FromRow;

// =============================================================================
// --- VECTOR STORE MODELS ---
// =============================================================================

/// Represents a single vector point with its embedding, metadata and optional payload.
/// This is the fundamental unit of storage in the vector subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoint {
    /// Unique identifier for this vector point
    pub id: String,
    /// The embedding vector (f32 for compatibility with all major embedding models)
    pub embedding: Vec<f32>,
    /// Optional JSON metadata payload (filterable)
    pub metadata: Option<serde_json::Value>,
    /// Owner of this vector point (for RBAC enforcement)
    pub owner: String,
    /// Timestamp of creation/last update
    pub created_at: f64,
}

/// Search result returned from similarity queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// The matched vector point ID
    pub id: String,
    /// Similarity score (higher = more similar for cosine, lower = more similar for L2)
    pub score: f32,
    /// Optional metadata payload of the matched point
    pub metadata: Option<serde_json::Value>,
    /// Owner of the matched point
    pub owner: String,
}

/// Configuration for a vector collection (analogous to a table in relational DBs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCollectionConfig {
    /// Collection name (unique within a mount)
    pub name: String,
    /// Dimensionality of vectors stored in this collection
    pub dimension: u32,
    /// Distance metric used for similarity search
    pub distance_metric: DistanceMetric,
    /// Owner/creator of the collection
    pub owner: String,
    /// Timestamp of creation
    pub created_at: f64,
}

/// Supported distance metrics for vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DistanceMetric {
    /// Cosine similarity (normalized dot product) — most common for text embeddings
    Cosine,
    /// L2 / Euclidean distance — common for image embeddings
    L2,
    /// Inner product (dot product) — used by some specialized models
    InnerProduct,
}

impl DistanceMetric {
    /// Returns the pgvector operator for this distance metric.
    pub fn to_pgvector_operator(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "<=>",
            DistanceMetric::L2 => "<->",
            DistanceMetric::InnerProduct => "<#>",
        }
    }

    /// Returns the pgvector index operator class for this metric.
    pub fn to_pgvector_ops(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "vector_cosine_ops",
            DistanceMetric::L2 => "vector_l2_ops",
            DistanceMetric::InnerProduct => "vector_ip_ops",
        }
    }
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::L2 => write!(f, "l2"),
            DistanceMetric::InnerProduct => write!(f, "inner_product"),
        }
    }
}

impl std::str::FromStr for DistanceMetric {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(DistanceMetric::Cosine),
            "l2" | "euclidean" => Ok(DistanceMetric::L2),
            "inner_product" | "ip" | "dot" => Ok(DistanceMetric::InnerProduct),
            _ => Err(format!("Unknown distance metric: '{}'. Use: cosine, l2, inner_product", s)),
        }
    }
}

/// Filter expression for vector metadata queries.
/// Supports basic key-value matching and comparison operators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFilter {
    /// JSON filter expression applied to metadata
    pub filter_json: Option<serde_json::Value>,
    /// Optional owner filter for RBAC
    pub owner_filter: Option<String>,
}

// =============================================================================
// --- EXISTING MODELS ---
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    pub permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct User {
    pub username: String,
    pub public_key: Vec<u8>,
    pub role: String,
    pub is_active: bool,
    pub created_at: f64,
    #[sqlx(default)]
    pub quota_limit: i64,
    #[sqlx(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileNode {
    pub path: String,
    pub name: String,
    pub owner: String,
    pub is_folder: bool,
    
    pub content: Vec<u8>,
    pub blob_id: Option<String>,

    pub size: usize,
    pub lock_pass: Option<String>,
    pub permissions: u8,
    pub created_at: f64,
    pub metadata: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub role: String,
    pub permissions: HashSet<String>,
    pub exp: usize,
    pub sid: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AuditLogEntry {
    pub id: i64,
    pub timestamp: String,
    pub actor: String,
    pub action: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AuditLog {
    pub id: i64,
    pub user_id: String,
    pub action: String,
    pub target: String,
    pub timestamp: f64,
}