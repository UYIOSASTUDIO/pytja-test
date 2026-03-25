use crate::storage::BlobStorage;
use crate::error::PytjaError;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use std::path::Path;
use bytes::Bytes;

pub struct FileSystemStorage {
    base_path: String,
}

impl FileSystemStorage {
    pub async fn new(path: &str) -> Result<Self, PytjaError> {
        fs::create_dir_all(path).await.map_err(|e| PytjaError::System(e.to_string()))?;
        Ok(Self { base_path: path.to_string() })
    }
    
    pub(crate) fn sanitize_path(&self, path: &str) -> Result<std::path::PathBuf, PytjaError> {
        let clean_path = path
            .trim_start_matches('/')
            .trim_start_matches("./")
            .trim_start_matches('\\');

        if clean_path.is_empty() {
            return Err(PytjaError::System("Invalid Path: Filename is empty".into()));
        }
        
        if clean_path.contains("..") {
            return Err(PytjaError::System("Invalid Path: Directory traversal detected".into()));
        }

        Ok(Path::new(&self.base_path).join(clean_path))
    }
}

#[async_trait]
impl BlobStorage for FileSystemStorage {
    async fn put(&self, path: &str, stream: BoxStream<'static, Result<Bytes, PytjaError>>) -> Result<String, PytjaError> {
        // 1. Pfad absichern und Zielverzeichnis vorbereiten
        let full_path = self.sanitize_path(path)?;

        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| PytjaError::System(e.to_string()))?;
        }

        // 2. Datei für den Schreibvorgang erstellen
        let mut file = fs::File::create(&full_path).await.map_err(|e| PytjaError::System(e.to_string()))?;

        // --- START ZERO-COPY STREAMING PIPELINE ---
        // Wir konvertieren den Stream explizit in einen Result-kompatiblen Stream für IO
        let io_stream = stream.map(|res| res.map_err(std::io::Error::from));
        let mut reader = tokio_util::io::StreamReader::new(io_stream);

        tokio::io::copy(&mut reader, &mut file)
            .await
            .map_err(|e| PytjaError::System(format!("Streaming-IO Error: {}", e)))?;
        // --- ENDE ZERO-COPY STREAMING PIPELINE ---

        // Sicherstellen, dass alle Daten physisch geschrieben wurden
        file.flush().await.map_err(|e| PytjaError::System(e.to_string()))?;

        // Relativen Pfad für die Datenbank-Speicherung extrahieren
        let relative_path = full_path.strip_prefix(&self.base_path)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        Ok(relative_path)
    }

    async fn get(&self, blob_id: &str) -> Result<BoxStream<'static, Result<Bytes, PytjaError>>, PytjaError> {
        let full_path = self.sanitize_path(blob_id)?;
        
        if full_path.is_dir() {
            return Err(PytjaError::System("Storage Error: Target is a directory".into()));
        }
        
        let file = fs::File::open(full_path).await.map_err(|e| PytjaError::System(e.to_string()))?;

        let stream = tokio_util::io::ReaderStream::new(file);
        let s = stream.map(|res| {
            res.map_err(|e| PytjaError::System(e.to_string()))});

        Ok(Box::pin(s))
    }

    async fn delete(&self, blob_id: &str) -> Result<(), PytjaError> {
        let full_path = self.sanitize_path(blob_id)?;
        if full_path.exists() {
            fs::remove_file(full_path).await.map_err(|e| PytjaError::System(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::fs;

    #[tokio::test]
    async fn test_path_sanitization_security() {
        let temp_dir = std::env::temp_dir().join("pytja_test_blobs");
        let _ = fs::remove_dir_all(&temp_dir).await;

        let storage = FileSystemStorage::new(temp_dir.to_str().unwrap()).await.unwrap();
        
        let safe = storage.sanitize_path("test.png");
        assert!(safe.is_ok());
        assert!(safe.unwrap().ends_with("pytja_test_blobs/test.png"));

        let attack1 = storage.sanitize_path("../etc/passwd");
        assert!(attack1.is_err(), "Travesal ../ failed to block");
        
        let attack2 = storage.sanitize_path("/var/log/syslog");
        assert!(safe_path_check(&storage, "/root/secret"), "Absolute path escaping detected");

        let _ = fs::remove_dir_all(&temp_dir).await;
    }

    fn safe_path_check(storage: &FileSystemStorage, input: &str) -> bool {
        match storage.sanitize_path(input) {
            Ok(p) => p.starts_with(&storage.base_path),
            Err(_) => true,
        }
    }
}