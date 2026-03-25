pub mod fs;
pub mod s3;

pub use fs::FileSystemStorage;
pub use s3::S3Storage;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use crate::error::PytjaError;

// Stream Type
pub type ByteStream = BoxStream<'static, Result<Bytes, PytjaError>>;

#[async_trait]
pub trait BlobStorage: Send + Sync {
    async fn put(&self, file_name: &str, stream: ByteStream) -> Result<String, PytjaError>;
    async fn get(&self, key: &str) -> Result<ByteStream, PytjaError>;
    async fn delete(&self, key: &str) -> Result<(), PytjaError>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum StorageType {
    FileSystem,
    S3,
}