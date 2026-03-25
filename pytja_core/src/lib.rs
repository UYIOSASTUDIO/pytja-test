pub mod models;
pub mod error;
pub mod crypto;
pub mod telemetry;
pub mod repo;
pub mod drivers;
pub mod config;
pub mod storage;
pub mod identity;

pub use repo::PytjaRepository;
pub use repo::VectorStore;
pub use drivers::{DriverManager, DatabaseType};
pub use error::PytjaError;
pub use models::{User, FileNode, AuditLogEntry, Role};
pub use models::{VectorPoint, VectorSearchResult, VectorCollectionConfig, VectorFilter, DistanceMetric};
pub use config::AppConfig;
pub use storage::{BlobStorage, FileSystemStorage, S3Storage, StorageType};

#[cfg(target_os = "linux")]
#[no_mangle]
pub extern "C" fn __rust_probestack() {}