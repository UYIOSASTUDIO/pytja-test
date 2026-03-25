use tonic::{transport::{Server, Identity, ServerTlsConfig}, Request, Response, Status};
use pytja_proto::pytja::pytja_service_server::{PytjaService, PytjaServiceServer};
use pytja_proto::pytja::*;
use pytja_core::{DriverManager, AppConfig, BlobStorage, FileSystemStorage, S3Storage, drivers::DatabaseType};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;
use dotenvy::dotenv;
use std::fs;
use colored::*;

mod session_manager;
mod handlers;
mod plugin_manager;

use crate::session_manager::SessionManager;
use crate::handlers::service::MyPytjaService;
// --- NEU: Import des PluginManagers ---
use crate::plugin_manager::PluginManager;

#[tonic::async_trait]
impl PytjaService for MyPytjaService {
    type DownloadFileStream = ReceiverStream<Result<FileChunk, Status>>;
    type ExecScriptStream = ReceiverStream<Result<ExecResponse, Status>>;
    type StreamServerLogsStream = ReceiverStream<Result<LogStreamEntry, Status>>;

    // --- AUTHENTICATION ---
    async fn ping(&self, r: Request<PingRequest>) -> Result<Response<PingResponse>, Status> { self.ping_impl(r).await }
    async fn get_challenge(&self, r: Request<ChallengeRequest>) -> Result<Response<ChallengeResponse>, Status> { self.get_challenge_impl(r).await }
    async fn login(&self, r: Request<LoginRequest>) -> Result<Response<LoginResponse>, Status> { self.login_impl(r).await }

    // --- FILESYSTEM OPERATIONS ---
    async fn list_directory(&self, r: Request<ListRequest>) -> Result<Response<ListResponse>, Status> { self.list_directory_impl(r).await }
    async fn get_tree(&self, r: Request<TreeRequest>) -> Result<Response<TreeResponse>, Status> { self.get_tree_impl(r).await }
    async fn upload_file(&self, r: Request<tonic::Streaming<UploadRequest>>) -> Result<Response<ActionResponse>, Status> { self.upload_file_impl(r).await }
    async fn download_file(&self, r: Request<DownloadRequest>) -> Result<Response<Self::DownloadFileStream>, Status> { self.download_file_impl(r).await }
    async fn create_node(&self, r: Request<CreateNodeRequest>) -> Result<Response<ActionResponse>, Status> { self.create_node_impl(r).await }
    async fn read_file(&self, r: Request<ReadFileRequest>) -> Result<Response<ReadFileResponse>, Status> { self.read_file_impl(r).await }
    async fn delete_node(&self, r: Request<DeleteNodeRequest>) -> Result<Response<ActionResponse>, Status> { self.delete_node_impl(r).await }
    async fn move_node(&self, r: Request<MoveNodeRequest>) -> Result<Response<ActionResponse>, Status> { self.move_node_impl(r).await }
    async fn copy_node(&self, r: Request<CopyNodeRequest>) -> Result<Response<ActionResponse>, Status> { self.copy_node_impl(r).await }
    async fn change_mode(&self, r: Request<ChangeModeRequest>) -> Result<Response<ActionResponse>, Status> { self.change_mode_impl(r).await }
    async fn chown_node(&self, r: Request<ChownRequest>) -> Result<Response<ActionResponse>, Status> { self.chown_node_impl(r).await }
    async fn lock_node(&self, r: Request<LockRequest>) -> Result<Response<ActionResponse>, Status> { self.lock_node_impl(r).await }
    async fn get_usage(&self, r: Request<UsageRequest>) -> Result<Response<UsageResponse>, Status> { self.get_usage_impl(r).await }
    async fn find_node(&self, r: Request<FindRequest>) -> Result<Response<FindResponse>, Status> { self.find_node_impl(r).await }
    async fn grep_node(&self, r: Request<GrepRequest>) -> Result<Response<GrepResponse>, Status> { self.grep_node_impl(r).await }
    async fn stat_node(&self, r: Request<StatRequest>) -> Result<Response<StatResponse>, Status> { self.stat_node_impl(r).await }
    async fn exec_script(&self, r: Request<ExecRequest>) -> Result<Response<Self::ExecScriptStream>, Status> { self.exec_script_impl(r).await }

    // --- USER ADMINISTRATION ---
    async fn list_users(&self, r: Request<ListUsersRequest>) -> Result<Response<ListUsersResponse>, Status> { self.list_users_impl(r).await }
    async fn register_user(&self, r: Request<RegisterUserRequest>) -> Result<Response<RegisterUserResponse>, Status> { self.register_user_impl(r).await }
    async fn set_user_quota(&self, r: Request<SetQuotaRequest>) -> Result<Response<SetQuotaResponse>, Status> { self.set_user_quota_impl(r).await }
    async fn change_user_role(&self, r: Request<ChangeRoleRequest>) -> Result<Response<ChangeRoleResponse>, Status> { self.change_user_role_impl(r).await }
    async fn kick_user(&self, r: Request<KickUserRequest>) -> Result<Response<ActionResponse>, Status> { self.kick_user_impl(r).await }
    async fn ban_user(&self, r: Request<BanUserRequest>) -> Result<Response<BanUserResponse>, Status> { self.ban_user_impl(r).await }
    async fn get_active_sessions(&self, r: Request<GetSessionsRequest>) -> Result<Response<GetSessionsResponse>, Status> { self.get_active_sessions_impl(r).await }

    // --- RBAC ADMINISTRATION ---
    async fn create_role(&self, r: Request<CreateRoleRequest>) -> Result<Response<AdminActionResponse>, Status> { self.create_role_impl(r).await }
    async fn add_permission(&self, r: Request<AddPermissionRequest>) -> Result<Response<AdminActionResponse>, Status> { self.add_permission_impl(r).await }
    async fn assign_role(&self, r: Request<AssignRoleRequest>) -> Result<Response<AdminActionResponse>, Status> { self.assign_role_impl(r).await }
    async fn list_roles(&self, r: Request<ListRolesRequest>) -> Result<Response<ListRolesResponse>, Status> { self.list_roles_impl(r).await }

    // --- SYSTEM & MOUNTS ---
    async fn get_system_stats(&self, r: Request<SystemStatsRequest>) -> Result<Response<SystemStatsResponse>, Status> { self.get_system_stats_impl(r).await }
    async fn stream_server_logs(&self, r: Request<LogStreamRequest>) -> Result<Response<Self::StreamServerLogsStream>, Status> { self.stream_server_logs_impl(r).await }
    async fn get_audit_logs(&self, r: Request<GetAuditLogsRequest>) -> Result<Response<GetAuditLogsResponse>, Status> { self.get_audit_logs_impl(r).await }
    async fn get_mounts(&self, r: Request<GetMountsRequest>) -> Result<Response<GetMountsResponse>, Status> { self.get_mounts_impl(r).await }
    async fn add_mount(&self, r: Request<AddMountRequest>) -> Result<Response<AdminActionResponse>, Status> { self.add_mount_impl(r).await }
    async fn remove_mount(&self, r: Request<RemoveMountRequest>) -> Result<Response<AdminActionResponse>, Status> { self.remove_mount_impl(r).await }

    // --- INVITE SYSTEM ---
    async fn generate_invite_code(&self, r: Request<GenerateInviteRequest>) -> Result<Response<GenerateInviteResponse>, Status> { self.generate_invite_code_impl(r).await }
    async fn revoke_invite_code(&self, r: Request<RevokeInviteRequest>) -> Result<Response<AdminActionResponse>, Status> { self.revoke_invite_code_impl(r).await }
    async fn list_invite_codes(&self, r: Request<ListInvitesRequest>) -> Result<Response<ListInvitesResponse>, Status> { self.list_invite_codes_impl(r).await }
    async fn read_file_chunk(&self, r: Request<pytja_proto::pytja::ReadChunkRequest>) -> Result<Response<pytja_proto::pytja::ReadChunkResponse>, Status> {
        self.read_file_chunk_impl(r).await
    }
    async fn query_metadata(&self, r: Request<pytja_proto::pytja::QueryMetadataRequest>) -> Result<Response<pytja_proto::pytja::ListResponse>, Status> {
        self.query_metadata_impl(r).await
    }

    // --- VECTOR STORE OPERATIONS ---
    async fn create_vector_collection(&self, r: Request<CreateVectorCollectionRequest>) -> Result<Response<AdminActionResponse>, Status> { self.create_vector_collection_impl(r).await }
    async fn delete_vector_collection(&self, r: Request<DeleteVectorCollectionRequest>) -> Result<Response<AdminActionResponse>, Status> { self.delete_vector_collection_impl(r).await }
    async fn list_vector_collections(&self, r: Request<ListVectorCollectionsRequest>) -> Result<Response<ListVectorCollectionsResponse>, Status> { self.list_vector_collections_impl(r).await }
    async fn vector_upsert(&self, r: Request<VectorUpsertRequest>) -> Result<Response<VectorUpsertResponse>, Status> { self.vector_upsert_impl(r).await }
    async fn vector_search(&self, r: Request<VectorSearchRequest>) -> Result<Response<VectorSearchResponse>, Status> { self.vector_search_impl(r).await }
    async fn vector_get(&self, r: Request<VectorGetRequest>) -> Result<Response<VectorGetResponse>, Status> { self.vector_get_impl(r).await }
    async fn vector_delete(&self, r: Request<VectorDeleteRequest>) -> Result<Response<VectorDeleteResponse>, Status> { self.vector_delete_impl(r).await }
    async fn vector_count(&self, r: Request<VectorCountRequest>) -> Result<Response<VectorCountResponse>, Status> { self.vector_count_impl(r).await }
}

pub async fn start_server() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    colored::control::set_override(true);

    let config = AppConfig::new().expect("CRITICAL: Failed to load configuration");

    let _guard = pytja_core::telemetry::init_telemetry(&config.paths.logs_dir, "pytja_server.log");

    println!("{}", "========================================".green().bold());
    println!("   PYTJA SERVER v1.0       ");
    println!("{}", "========================================".green().bold());

    let manager = Arc::new(DriverManager::new());
    let redis_url = config.redis.as_ref().map(|r| r.url.clone()).unwrap_or_else(|| "redis://127.0.0.1/".to_string());

    println!("Connecting to Session Store (Redis)...");
    let session_mgr = match SessionManager::new(&redis_url).await {
        Ok(mgr) => Arc::new(mgr),
        Err(e) => {
            println!("{}", format!("FATAL: Redis Connection Failed: {}", e).red());
            return Err(e.into());
        }
    };

    manager.load_config(&config.paths.mounts_file).await;

    let db_url = &config.database.primary_url;
    let (db_path, db_type) = if db_url.starts_with("postgres://") || db_url.starts_with("postgresql://") {
        (db_url.as_str(), DatabaseType::Postgres)
    } else if db_url.starts_with("sqlite://") {
        (db_url.strip_prefix("sqlite://").unwrap(), DatabaseType::Sqlite)
    } else {
        panic!("FATAL: Unsupported database URL protocol: {}", db_url);
    };

    println!("Mounting Primary Database ({:?})...", db_type);
    manager.mount("primary", db_path, db_type).await
        .expect("FATAL: Failed to mount primary DB");

    if let Some(repo) = manager.get_repo("primary").await {
        repo.init().await.expect("DB Migration failed");
    } else {
        panic!("FATAL: Primary DB lost immediately after mount!");
    }

    let storage: Arc<dyn BlobStorage> = if config.storage.storage_type == "s3" {
        let bucket = config.storage.s3_bucket.as_deref()
            .expect("CRITICAL: 'storage.s3_bucket' is required in config when storage_type='s3'");
        let region = config.storage.s3_region.as_deref()
            .unwrap_or("us-east-1");

        info!("Using S3 Storage (Bucket: {}, Region: {})", bucket, region);
        Arc::new(S3Storage::new(bucket, region).await)
    } else {
        info!("Using Local Storage at: {}", config.storage.local_path);
        Arc::new(FileSystemStorage::new(&config.storage.local_path).await?)
    };

    let (tx, _rx) = broadcast::channel(100);

    // =========================================================================
    // --- ENTERPRISE WASM PLUGIN MANAGER INITIALIZATION ---
    // =========================================================================
    println!("Initializing WASM Plugin Manager...");
    let plugin_manager = Arc::new(PluginManager::new()?);
    let plugins_dir = std::path::Path::new("plugins");

    if plugins_dir.exists() && plugins_dir.is_dir() {
        let mut entries = tokio::fs::read_dir(plugins_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("wasm") {
                let file_name = path.file_stem().unwrap().to_str().unwrap().to_string();
                let bytes = tokio::fs::read(&path).await?;

                // JIT-Kompilierung und Caching im RAM
                plugin_manager.load_plugin(&file_name, bytes).await?;
                println!("Loaded and cached WASM plugin: {}", file_name);
            }
        }
    } else {
        println!("No 'plugins' directory found. Skipping WASM autoload.");
    }
    // =========================================================================

    // Instanziierung des Pytja Services mit dem neuen PluginManager
    let service = MyPytjaService {
        manager: manager.clone(),
        sessions: session_mgr,
        config: config.clone(),
        storage,
        log_broadcast: tx.clone(),
        plugins: plugin_manager,
    };

    let host = std::env::var("PYTJA_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PYTJA_PORT").unwrap_or_else(|_| "50051".to_string());
    let addr_str = format!("{}:{}", host, port);

    use std::net::ToSocketAddrs;
    let addr = addr_str.to_socket_addrs()
        .expect("CRITICAL: Failed to resolve address")
        .next()
        .expect("CRITICAL: No valid IP address found for host");

    let mut builder = Server::builder()
        .http2_keepalive_interval(Some(std::time::Duration::from_secs(60)))
        .tcp_nodelay(true);

    if let Some(tls_config) = &config.tls {
        if tls_config.enabled {
            println!("{}", "ENABLING TLS/SSL SECURITY".cyan());

            let cert_res = fs::read_to_string(&tls_config.cert_path);
            let key_res = fs::read_to_string(&tls_config.key_path);

            match (cert_res, key_res) {
                (Ok(cert), Ok(key)) => {
                    let identity = Identity::from_pem(cert, key);
                    builder = builder.tls_config(ServerTlsConfig::new().identity(identity))?;
                    println!("TLS Security Active (Dual-Stack Mode).");
                },
                _ => {
                    println!("{}", format!("FATAL: Could not load certs from config paths: {} / {}", tls_config.cert_path, tls_config.key_path).red());
                    return Err("TLS Configuration failed. Server aborted.".into());
                }
            }
        } else {
            println!("TLS Configured but Disabled in config.");
        }
    } else {
        println!("{}", "CRITICAL: NO TLS CONFIG FOUND.".red().bold());
        return Err("Server security policy requires TLS configuration.".into());
    }

    println!("{} {}", "Server listening on".green(), addr);
    println!("----------------------------------------");

    let max_size = 50 * 1024 * 1024;
    let pytja_svc = PytjaServiceServer::new(service)
        .max_decoding_message_size(max_size)
        .max_encoding_message_size(max_size);

    builder
        .add_service(pytja_svc)
        .serve(addr)
        .await?;

    Ok(())
}