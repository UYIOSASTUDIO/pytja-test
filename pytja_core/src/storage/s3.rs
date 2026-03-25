use super::{BlobStorage, ByteStream};
use crate::error::PytjaError;
use async_trait::async_trait;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream as S3ByteStream;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use futures::{StreamExt, TryStreamExt};
use uuid::Uuid;
use tracing::{info, debug};
use tokio_util::io::ReaderStream;

const MIN_PART_SIZE: usize = 10 * 1024 * 1024;

pub struct S3Storage {
    client: Client,
    bucket: String,
}

impl S3Storage {
    pub async fn new(bucket: &str, _region: &str) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = Client::new(&config);
        Self { client, bucket: bucket.to_string() }
    }
    
    async fn upload_part(
        &self,
        key: &str,
        upload_id: &str,
        part_number: i32,
        data: Vec<u8>
    ) -> Result<CompletedPart, PytjaError> {
        let body = S3ByteStream::from(data);

        let upload_part_res = self.client.upload_part()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(upload_id)
            .part_number(part_number)
            .body(body)
            .send()
            .await
            .map_err(|e| PytjaError::System(format!("S3 Part Upload Failed: {}", e)))?;

        Ok(CompletedPart::builder()
            .e_tag(upload_part_res.e_tag.unwrap_or_default())
            .part_number(part_number)
            .build())
    }
}

#[async_trait]
impl BlobStorage for S3Storage {
    async fn put(&self, _name: &str, mut stream: ByteStream) -> Result<String, PytjaError> {
        let blob_id = Uuid::new_v4().to_string();

        let create_res = self.client.create_multipart_upload()
            .bucket(&self.bucket)
            .key(&blob_id)
            .send()
            .await
            .map_err(|e| PytjaError::System(format!("S3 Init Upload Failed: {}", e)))?;

        let upload_id = create_res.upload_id.ok_or_else(|| PytjaError::System("No Upload ID from S3".into()))?;

        debug!("Started S3 Multipart Upload for blob: {}", blob_id);

        let mut completed_parts = Vec::new();
        let mut part_number = 1;
        let mut buffer: Vec<u8> = Vec::with_capacity(MIN_PART_SIZE);

        while let Some(chunk_res) = stream.next().await {
            match chunk_res {
                Ok(chunk) => {
                    buffer.extend_from_slice(&chunk);
                    
                    if buffer.len() >= MIN_PART_SIZE {
                        let part_data = std::mem::replace(&mut buffer, Vec::with_capacity(MIN_PART_SIZE));

                        match self.upload_part(&blob_id, &upload_id, part_number, part_data).await {
                            Ok(part) => {
                                completed_parts.push(part);
                                part_number += 1;
                            },
                            Err(e) => {
                                let _ = self.client.abort_multipart_upload()
                                    .bucket(&self.bucket).key(&blob_id).upload_id(&upload_id).send().await;
                                return Err(e);
                            }
                        }
                    }
                },
                Err(e) => {
                    // Stream Fehler -> Abbruch
                    let _ = self.client.abort_multipart_upload()
                        .bucket(&self.bucket).key(&blob_id).upload_id(&upload_id).send().await;
                    return Err(PytjaError::System(format!("Stream error during upload: {}", e)));
                }
            }
        }

        if !buffer.is_empty() {
            match self.upload_part(&blob_id, &upload_id, part_number, buffer).await {
                Ok(part) => completed_parts.push(part),
                Err(e) => {
                    let _ = self.client.abort_multipart_upload()
                        .bucket(&self.bucket).key(&blob_id).upload_id(&upload_id).send().await;
                    return Err(e);
                }
            }
        }

        let completed_upload = CompletedMultipartUpload::builder()
            .set_parts(Some(completed_parts))
            .build();

        self.client.complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&blob_id)
            .upload_id(&upload_id)
            .multipart_upload(completed_upload)
            .send()
            .await
            .map_err(|e| {
                let c = self.client.clone();
                let b = self.bucket.to_string();
                let k = blob_id.to_string();
                let u = upload_id.to_string();

                tokio::spawn(async move {
                    let _ = c.abort_multipart_upload().bucket(b).key(k).upload_id(u).send().await;
                });

                PytjaError::System(e.to_string())
            })?;

        info!("Stored blob {} on S3 (Multipart)", blob_id);
        Ok(blob_id)
    }

    async fn get(&self, key: &str) -> Result<ByteStream, PytjaError> {
        let resp = self.client.get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| PytjaError::NotFound(format!("S3 Download Error: {}", e)))?;

        let reader = resp.body.into_async_read();
        let stream = ReaderStream::new(reader).map_err(PytjaError::IoError);
        Ok(Box::pin(stream))
    }

    async fn delete(&self, key: &str) -> Result<(), PytjaError> {
        self.client.delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| PytjaError::System(e.to_string()))?;
        Ok(())
    }
}