// =============================================================================
// PYTJA VECTOR RPC HANDLERS
// =============================================================================
//
// Server-side implementation of all vector store gRPC endpoints.
// Each handler validates auth/permissions, resolves the vector mount,
// and delegates to the VectorStore trait.
//

use tonic::{Request, Response, Status};
use pytja_proto::pytja::*;
use pytja_core::models::{VectorPoint, VectorFilter, DistanceMetric};
use crate::handlers::service::MyPytjaService;
use std::time::Instant;

impl MyPytjaService {

    /// Resolve a vector store mount by name. Returns an error if not found.
    async fn resolve_vector_store(
        &self,
        mount_name: &str,
    ) -> Result<std::sync::Arc<dyn pytja_core::repo::VectorStore>, Status> {
        self.manager.get_vector_store(mount_name).await
            .ok_or_else(|| Status::not_found(
                format!("Vector store mount '{}' not found. Mount a PgVector database first.", mount_name)
            ))
    }

    // =========================================================================
    // COLLECTION MANAGEMENT
    // =========================================================================

    pub async fn create_vector_collection_impl(
        &self,
        request: Request<CreateVectorCollectionRequest>,
    ) -> Result<Response<AdminActionResponse>, Status> {
        let claims = self.check_permissions(request.metadata(), Some("core:admin:mounts")).await?;
        let req = request.into_inner();

        if req.collection_name.is_empty() {
            return Err(Status::invalid_argument("Collection name cannot be empty."));
        }
        if req.dimension == 0 || req.dimension > 65536 {
            return Err(Status::invalid_argument("Dimension must be between 1 and 65536."));
        }

        let metric: DistanceMetric = req.distance_metric.parse()
            .map_err(|e: String| Status::invalid_argument(e))?;

        let store = self.resolve_vector_store(&req.mount_name).await?;

        store.create_collection(&req.collection_name, req.dimension, metric, &claims.sub)
            .await
            .map_err(|e| match e {
                pytja_core::PytjaError::CollectionAlreadyExists(_) => Status::already_exists(e.to_string()),
                _ => Status::internal(e.to_string()),
            })?;

        // Audit log
        if let Some(primary) = self.manager.get_repo("primary").await {
            let _ = primary.log_action(
                &claims.sub,
                "VECTOR_CREATE_COLLECTION",
                &format!("{}:{}", req.mount_name, req.collection_name),
            ).await;
        }

        Ok(Response::new(AdminActionResponse {
            success: true,
            message: format!(
                "Vector collection '{}' created (dim={}, metric={})",
                req.collection_name, req.dimension, req.distance_metric
            ),
        }))
    }

    pub async fn delete_vector_collection_impl(
        &self,
        request: Request<DeleteVectorCollectionRequest>,
    ) -> Result<Response<AdminActionResponse>, Status> {
        let claims = self.check_permissions(request.metadata(), Some("core:admin:mounts")).await?;
        let req = request.into_inner();

        let store = self.resolve_vector_store(&req.mount_name).await?;

        store.delete_collection(&req.collection_name)
            .await
            .map_err(|e| match e {
                pytja_core::PytjaError::CollectionNotFound(_) => Status::not_found(e.to_string()),
                _ => Status::internal(e.to_string()),
            })?;

        if let Some(primary) = self.manager.get_repo("primary").await {
            let _ = primary.log_action(
                &claims.sub,
                "VECTOR_DELETE_COLLECTION",
                &format!("{}:{}", req.mount_name, req.collection_name),
            ).await;
        }

        Ok(Response::new(AdminActionResponse {
            success: true,
            message: format!("Vector collection '{}' deleted.", req.collection_name),
        }))
    }

    pub async fn list_vector_collections_impl(
        &self,
        request: Request<ListVectorCollectionsRequest>,
    ) -> Result<Response<ListVectorCollectionsResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:fs:read")).await?;
        let req = request.into_inner();

        let store = self.resolve_vector_store(&req.mount_name).await?;
        let collections = store.list_collections().await
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut infos = Vec::new();
        for col in collections {
            let count = store.vector_count(&col.name).await.unwrap_or(0);
            infos.push(VectorCollectionInfo {
                name: col.name,
                dimension: col.dimension,
                distance_metric: col.distance_metric.to_string(),
                owner: col.owner,
                created_at: col.created_at,
                vector_count: count,
            });
        }

        Ok(Response::new(ListVectorCollectionsResponse { collections: infos }))
    }

    // =========================================================================
    // VECTOR UPSERT
    // =========================================================================

    pub async fn vector_upsert_impl(
        &self,
        request: Request<VectorUpsertRequest>,
    ) -> Result<Response<VectorUpsertResponse>, Status> {
        let claims = self.check_permissions(request.metadata(), Some("core:fs:write")).await?;
        let req = request.into_inner();

        if req.points.is_empty() {
            return Err(Status::invalid_argument("No vector points provided."));
        }

        let store = self.resolve_vector_store(&req.mount_name).await?;

        // Convert proto points to domain model
        let now = chrono::Utc::now().timestamp() as f64;
        let points: Vec<VectorPoint> = req.points.into_iter().map(|p| {
            let metadata = if p.metadata_json.is_empty() {
                None
            } else {
                serde_json::from_str(&p.metadata_json).ok()
            };

            VectorPoint {
                id: if p.id.is_empty() { uuid::Uuid::new_v4().to_string() } else { p.id },
                embedding: p.embedding,
                metadata,
                owner: claims.sub.clone(),
                created_at: now,
            }
        }).collect();

        let count = store.vector_upsert(&req.collection_name, points)
            .await
            .map_err(|e| match e {
                pytja_core::PytjaError::CollectionNotFound(_) => Status::not_found(e.to_string()),
                pytja_core::PytjaError::DimensionMismatch { expected, actual } => {
                    Status::invalid_argument(format!("Dimension mismatch: expected {}, got {}", expected, actual))
                },
                _ => Status::internal(e.to_string()),
            })?;

        Ok(Response::new(VectorUpsertResponse {
            success: true,
            upserted_count: count,
            message: format!("{} vectors upserted into '{}'", count, req.collection_name),
        }))
    }

    // =========================================================================
    // VECTOR SEARCH (Similarity)
    // =========================================================================

    pub async fn vector_search_impl(
        &self,
        request: Request<VectorSearchRequest>,
    ) -> Result<Response<VectorSearchResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:fs:read")).await?;
        let req = request.into_inner();

        if req.query_vector.is_empty() {
            return Err(Status::invalid_argument("Query vector cannot be empty."));
        }
        if req.top_k == 0 || req.top_k > 10000 {
            return Err(Status::invalid_argument("top_k must be between 1 and 10000."));
        }

        let store = self.resolve_vector_store(&req.mount_name).await?;

        // Build filter if provided
        let filter = if req.filter_json.is_empty() {
            None
        } else {
            let filter_value: serde_json::Value = serde_json::from_str(&req.filter_json)
                .map_err(|e| Status::invalid_argument(format!("Invalid filter JSON: {}", e)))?;

            Some(VectorFilter {
                filter_json: Some(filter_value),
                owner_filter: None, // RBAC filtering can be added per policy
            })
        };

        let start = Instant::now();

        let results = store.vector_search(
            &req.collection_name,
            req.query_vector,
            req.top_k,
            filter,
        )
            .await
            .map_err(|e| match e {
                pytja_core::PytjaError::CollectionNotFound(_) => Status::not_found(e.to_string()),
                pytja_core::PytjaError::DimensionMismatch { expected, actual } => {
                    Status::invalid_argument(format!("Query vector dimension mismatch: expected {}, got {}", expected, actual))
                },
                _ => Status::internal(e.to_string()),
            })?;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let hits: Vec<VectorSearchHit> = results.into_iter().map(|r| {
            VectorSearchHit {
                id: r.id,
                score: r.score,
                metadata_json: r.metadata.map(|m| m.to_string()).unwrap_or_default(),
                owner: r.owner,
            }
        }).collect();

        let total = hits.len() as u32;

        Ok(Response::new(VectorSearchResponse {
            hits,
            total_hits: total,
            search_time_ms: elapsed_ms,
        }))
    }

    // =========================================================================
    // VECTOR GET (by ID)
    // =========================================================================

    pub async fn vector_get_impl(
        &self,
        request: Request<VectorGetRequest>,
    ) -> Result<Response<VectorGetResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:fs:read")).await?;
        let req = request.into_inner();

        if req.ids.is_empty() {
            return Err(Status::invalid_argument("No vector IDs provided."));
        }

        let store = self.resolve_vector_store(&req.mount_name).await?;

        let points = store.vector_get(&req.collection_name, req.ids)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let proto_points: Vec<VectorPointFull> = points.into_iter().map(|p| {
            VectorPointFull {
                id: p.id,
                embedding: p.embedding,
                metadata_json: p.metadata.map(|m| m.to_string()).unwrap_or_default(),
                owner: p.owner,
                created_at: p.created_at,
            }
        }).collect();

        Ok(Response::new(VectorGetResponse { points: proto_points }))
    }

    // =========================================================================
    // VECTOR DELETE
    // =========================================================================

    pub async fn vector_delete_impl(
        &self,
        request: Request<VectorDeleteRequest>,
    ) -> Result<Response<VectorDeleteResponse>, Status> {
        let claims = self.check_permissions(request.metadata(), Some("core:fs:delete")).await?;
        let req = request.into_inner();

        let store = self.resolve_vector_store(&req.mount_name).await?;

        let deleted = if !req.ids.is_empty() {
            // Delete by IDs
            store.vector_delete(&req.collection_name, req.ids)
                .await
                .map_err(|e| Status::internal(e.to_string()))?
        } else if !req.filter_json.is_empty() {
            // Delete by filter
            let filter_value: serde_json::Value = serde_json::from_str(&req.filter_json)
                .map_err(|e| Status::invalid_argument(format!("Invalid filter JSON: {}", e)))?;

            let filter = VectorFilter {
                filter_json: Some(filter_value),
                owner_filter: None,
            };

            store.vector_delete_by_filter(&req.collection_name, filter)
                .await
                .map_err(|e| Status::internal(e.to_string()))?
        } else {
            return Err(Status::invalid_argument("Provide either IDs or a filter for deletion."));
        };

        // Audit log
        if let Some(primary) = self.manager.get_repo("primary").await {
            let _ = primary.log_action(
                &claims.sub,
                "VECTOR_DELETE",
                &format!("{}:{} ({} deleted)", req.mount_name, req.collection_name, deleted),
            ).await;
        }

        Ok(Response::new(VectorDeleteResponse {
            success: true,
            deleted_count: deleted,
            message: format!("{} vectors deleted from '{}'", deleted, req.collection_name),
        }))
    }

    // =========================================================================
    // VECTOR COUNT
    // =========================================================================

    pub async fn vector_count_impl(
        &self,
        request: Request<VectorCountRequest>,
    ) -> Result<Response<VectorCountResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:fs:read")).await?;
        let req = request.into_inner();

        let store = self.resolve_vector_store(&req.mount_name).await?;

        let count = store.vector_count(&req.collection_name)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(VectorCountResponse { count }))
    }
}