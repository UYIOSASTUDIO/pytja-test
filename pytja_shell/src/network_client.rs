use pytja_proto::pytja::pytja_service_client::PytjaServiceClient;
use pytja_proto::pytja::*;
use pytja_proto::pytja::upload_request::Data;
use tonic::transport::{Channel, ClientTlsConfig, Certificate};
use tonic::{Request, Status};
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::{Result, anyhow, Context};
use colored::*;
use std::str::FromStr;
use futures_util::StreamExt;
use std::path::Path;
use tokio::io::AsyncReadExt;

#[derive(Clone)]
pub struct PytjaClient {
    client: Arc<Mutex<PytjaServiceClient<Channel>>>,
    token: Arc<Mutex<Option<String>>>,
    #[allow(dead_code)]
    pub signing_key: Vec<u8>,
    #[allow(dead_code)]
    pub username: String,
    pub e2e_key: [u8; 32],
}

impl PytjaClient {
    pub async fn connect(server_url: String, signing_key: Vec<u8>, username: String, ca_cert_pem: Option<String>, e2e_key: [u8; 32]) -> Result<Self> {
        let mut endpoint = Channel::from_shared(server_url.clone())
            .context("Invalid Server URL")?;

        if server_url.starts_with("https") {
            let mut tls = ClientTlsConfig::new().domain_name("localhost");
            if let Some(pem) = ca_cert_pem {
                let ca = Certificate::from_pem(pem);
                tls = tls.ca_certificate(ca);
            }
            endpoint = endpoint.tls_config(tls)?;
        }

        let channel = endpoint.connect().await
            .context(format!("Failed to connect to {}", server_url))?;

        Ok(Self {
            client: Arc::new(Mutex::new(PytjaServiceClient::new(channel))),
            token: Arc::new(Mutex::new(None)),
            signing_key,
            username,
            e2e_key,
        })
    }

    #[allow(dead_code)]
    pub fn new(_url: &str, _key: Vec<u8>, _user: String) -> Self {
        panic!("Legacy constructor removed. Use PytjaClient::connect() with TLS support.");
    }

    pub async fn set_token(&self, t: &str) {
        let mut lock = self.token.lock().await;
        *lock = Some(t.to_string());
    }

    async fn auth_req<T>(&self, msg: T) -> Request<T> {
        let mut req = Request::new(msg);
        let lock = self.token.lock().await;
        if let Some(token) = &*lock {
            let val = format!("Bearer {}", token);
            if let Ok(meta) = tonic::metadata::MetadataValue::from_str(&val) {
                req.metadata_mut().insert("authorization", meta);
            }
        }
        req
    }

    // --- AUTHENTICATION ---

    pub async fn get_challenge(&self, username: &str) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = Request::new(ChallengeRequest { username: username.to_string() });
        let resp = client.get_challenge(req).await?.into_inner();
        if !resp.user_exists {
            return Err(anyhow!("User not found on server"));
        }
        Ok(resp.challenge)
    }

    pub async fn login(&self, username: &str, challenge: &str, signature: String) -> Result<LoginResponse, Status> {
        let mut client = self.client.lock().await;
        let req = Request::new(LoginRequest {
            username: username.to_string(),
            challenge: challenge.to_string(),
            signature,
        });
        let resp = client.login(req).await?.into_inner();
        Ok(resp)
    }

    pub async fn check_uplink(&self) -> Result<(bool, String)> {
        let mut client = self.client.lock().await;
        match client.ping(Request::new(PingRequest { message: "Ping".into() })).await {
            Ok(r) => {
                // Wir printen hier NICHTS mehr, um die Shell-Animation nicht zu stören
                Ok((true, r.into_inner().server_version))
            },
            Err(_) => Ok((false, "".to_string())),
        }
    }

    // --- FILESYSTEM OPERATIONS ---

    pub async fn list_files(&self, path: &str) -> Result<Vec<FileInfo>> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(ListRequest {
            path: path.to_string(),
            auth_token: "".into(),
            limit: 0,
            offset: 0,
        }).await;
        let resp = client.list_directory(req).await?.into_inner();
        Ok(resp.files)
    }

    pub async fn create_node(&self, path: &str, is_folder: bool, content: Vec<u8>, lock_pass: Option<String>, owner: &str) -> Result<String> {
        let mut client = self.client.lock().await;

        let final_content = if !is_folder && !content.is_empty() {
            pytja_core::crypto::CryptoService::encrypt_e2e(&self.e2e_key, &content)
                .map_err(|e| anyhow!("E2EE Encryption failed: {}", e))?
        } else {
            content
        };

        let req = self.auth_req(CreateNodeRequest {
            path: path.to_string(),
            is_folder,
            owner: owner.to_string(),
            content: final_content,
            lock_password: lock_pass.unwrap_or_default(),
        }).await;
        let resp = client.create_node(req).await?.into_inner();
        if resp.success { Ok(resp.message) } else { Err(anyhow!(resp.message)) }
    }

    pub async fn read_file(&self, path: &str, password: Option<String>) -> Result<(Vec<u8>, String)> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(ReadFileRequest {
            path: path.to_string(),
            password: password.unwrap_or_default()
        }).await;
        let resp = client.read_file(req).await?.into_inner();

        if resp.success {
            let decrypted_content = if !resp.content.is_empty() {
                pytja_core::crypto::CryptoService::decrypt_e2e(&self.e2e_key, &resp.content)
                    .map_err(|e| anyhow!("E2EE Decryption failed (Integrity breach): {}", e))?
            } else {
                resp.content
            };
            Ok((decrypted_content, resp.message))
        } else {
            Err(anyhow!(resp.message))
        }
    }

    pub async fn delete_node(&self, path: &str) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(DeleteNodeRequest { path: path.to_string() }).await;
        let resp = client.delete_node(req).await?.into_inner();
        if resp.success { Ok(resp.message) } else { Err(anyhow!(resp.message)) }
    }

    pub async fn move_node(&self, src: &str, dst: &str) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(MoveNodeRequest { source_path: src.to_string(), dest_path: dst.to_string() }).await;
        let resp = client.move_node(req).await?.into_inner();
        if resp.success { Ok(resp.message) } else { Err(anyhow!(resp.message)) }
    }

    pub async fn copy_node(&self, src: &str, dst: &str, owner: &str) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(CopyNodeRequest { source_path: src.to_string(), dest_path: dst.to_string(), owner: owner.to_string() }).await;
        let resp = client.copy_node(req).await?.into_inner();
        if resp.success { Ok(resp.message) } else { Err(anyhow!(resp.message)) }
    }

    pub async fn change_mode(&self, path: &str, perms: u32) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(ChangeModeRequest { path: path.to_string(), permissions: perms }).await;
        let resp = client.change_mode(req).await?.into_inner();
        if resp.success { Ok(resp.message) } else { Err(anyhow!(resp.message)) }
    }

    pub async fn chown_node(&self, path: &str, owner: &str) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(ChownRequest { path: path.to_string(), new_owner: owner.to_string() }).await;
        let resp = client.chown_node(req).await?.into_inner();
        if resp.success { Ok(resp.message) } else { Err(anyhow!(resp.message)) }
    }

    pub async fn lock_node(&self, path: &str, password: Option<String>) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(LockRequest { path: path.to_string(), password: password.unwrap_or_default() }).await;
        let resp = client.lock_node(req).await?.into_inner();
        if resp.success { Ok(resp.message) } else { Err(anyhow!(resp.message)) }
    }

    pub async fn get_usage(&self, owner: &str) -> Result<u64> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(UsageRequest { owner: owner.to_string() }).await;
        let resp = client.get_usage(req).await?.into_inner();
        Ok(resp.bytes)
    }

    pub async fn find_node(&self, pattern: &str) -> Result<Vec<String>> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(FindRequest { pattern: pattern.to_string() }).await;
        let resp = client.find_node(req).await?.into_inner();
        Ok(resp.paths)
    }

    pub async fn grep_node(&self, pattern: &str) -> Result<Vec<String>> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(GrepRequest { pattern: pattern.to_string() }).await;
        let resp = client.grep_node(req).await?.into_inner();
        Ok(resp.matches)
    }

    pub async fn stat_node(&self, path: &str) -> Result<(bool, bool, bool)> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(StatRequest { path: path.to_string() }).await;
        let resp = client.stat_node(req).await?.into_inner();
        Ok((resp.exists, resp.is_folder, resp.is_locked))
    }

    pub async fn get_tree(&self, root_path: &str) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(TreeRequest { root_path: root_path.to_string() }).await;
        let resp = client.get_tree(req).await?.into_inner();
        Ok(resp.tree_output)
    }

    // --- STREAMING UPLOAD ---

    pub async fn upload_file(&self, local_path: &str, remote_path: &str, lock: Option<String>, owner: &str, metadata_json: Option<String>) -> Result<String> {
        let path = Path::new(local_path);
        if !path.exists() { return Err(anyhow!("File not found")); }

        let file_meta = pytja_proto::pytja::FileMetadata {
            path: remote_path.to_string(),
            owner: owner.to_string(),
            lock_password: lock.unwrap_or_default(),
            is_folder: false,
            metadata: metadata_json,
        };

        let file_path = local_path.to_string();
        let e2e_key = self.e2e_key;

        let outbound = async_stream::stream! {
            yield UploadRequest { data: Some(Data::Metadata(file_meta)) };

            if let Ok(mut file) = tokio::fs::File::open(&file_path).await {
                let mut raw_content = Vec::new();
                if file.read_to_end(&mut raw_content).await.is_ok() {
                    match pytja_core::crypto::CryptoService::encrypt_e2e(&e2e_key, &raw_content) {
                        Ok(encrypted_bytes) => {
                            for chunk in encrypted_bytes.chunks(64 * 1024) {
                                yield UploadRequest { data: Some(Data::Chunk(chunk.to_vec())) };
                            }
                        }
                        Err(e) => eprintln!("[ERROR] E2EE Encryption failed: {}", e),
                    }
                }
            } else {
                eprintln!("[ERROR] Failed to open local file for streaming.");
            }
        };

        let mut request = Request::new(outbound);
        let lock_token = self.token.lock().await;
        if let Some(token) = &*lock_token {
            let val = format!("Bearer {}", token);
            if let Ok(meta) = tonic::metadata::MetadataValue::from_str(&val) {
                request.metadata_mut().insert("authorization", meta);
            }
        }

        let mut client_lock = self.client.lock().await;
        let response = client_lock.upload_file(request).await?.into_inner();
        if response.success { Ok(response.message) } else { Err(anyhow!(response.message)) }
    }

    // --- DOWNLOAD ---

    pub async fn download_file(&self, remote_path: &str, local_path: &str, password: Option<String>) -> Result<String> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(DownloadRequest {
            path: remote_path.to_string(),
            password: password.unwrap_or_default(),
        }).await;

        let mut stream = client.download_file(req).await?.into_inner();

        let mut encrypted_buffer = Vec::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.context("Stream error")?;
            encrypted_buffer.extend_from_slice(&chunk.content);
        }

        let decrypted_bytes = pytja_core::crypto::CryptoService::decrypt_e2e(&self.e2e_key, &encrypted_buffer)
            .context("E2EE Decryption failed (File manipulated or wrong key!)")?;

        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::File::create(local_path).await
            .context("Failed to create local file")?;

        file.write_all(&decrypted_bytes).await?;
        file.flush().await?;

        Ok(format!("Downloaded and decrypted {} bytes to {}", decrypted_bytes.len(), local_path))
    }

    // --- EXEC ---

    pub async fn exec_script(&self, path: &str) -> Result<()> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(ExecRequest { script_path: path.to_string(), args: vec![] }).await;

        let mut stream = client.exec_script(req).await?.into_inner();

        println!("{}", "--- REMOTE OUTPUT START ---".cyan());
        while let Some(resp_result) = stream.next().await {
            match resp_result {
                Ok(resp) => println!("{}", resp.output_line),
                Err(e) => println!("Error in stream: {}", e),
            }
        }
        println!("{}", "--- REMOTE OUTPUT END ---".cyan());
        Ok(())
    }

    pub async fn query_metadata(&self, query: &str) -> Result<Vec<FileInfo>> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(QueryMetadataRequest { query: query.to_string() }).await;
        let resp = client.query_metadata(req).await?.into_inner();
        Ok(resp.files)
    }

    pub async fn get_mounts(&self) -> Result<Vec<MountInfo>> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(GetMountsRequest {}).await;

        let response = client.get_mounts(req).await?.into_inner();
        Ok(response.mounts)
    }

    pub async fn get_system_stats(&self) -> Result<SystemStatsResponse> {
        let mut client = self.client.lock().await;
        let req = self.auth_req(SystemStatsRequest {}).await;

        let response = client.get_system_stats(req).await?.into_inner();
        Ok(response)
    }

}