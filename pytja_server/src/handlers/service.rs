use std::sync::Arc;
use tokio::sync::broadcast;
use pytja_core::{DriverManager, PytjaRepository, AppConfig, BlobStorage, models::Claims};
use crate::session_manager::SessionManager;
use tonic::{Status, metadata::MetadataMap};
use jsonwebtoken::{decode, Validation, DecodingKey};
use std::env;
use pytja_proto::pytja::LogStreamEntry;
use crate::plugin_manager::PluginManager;

pub const DEFAULT_QUOTA_LIMIT: usize = 1024 * 1024 * 1024;

pub struct MyPytjaService {
    pub manager: Arc<DriverManager>,
    pub sessions: Arc<SessionManager>,
    #[allow(dead_code)]
    pub config: AppConfig,
    pub storage: Arc<dyn BlobStorage>,
    pub log_broadcast: broadcast::Sender<LogStreamEntry>,
    pub plugins: Arc<PluginManager>,
}

impl MyPytjaService {
    pub fn get_jwt_secret() -> Vec<u8> {
        match env::var("PYTJA_JWT_SECRET") {
            Ok(s) if !s.is_empty() => s.into_bytes(),
            _ => {
                if cfg!(debug_assertions) {
                    tracing::warn!("UNSAFE: Using default JWT secret. Set PYTJA_JWT_SECRET in .env!");
                    b"pytja_super_secret_key_change_me_in_prod".to_vec()
                } else {
                    panic!("FATAL: PYTJA_JWT_SECRET environment variable is not set!");
                }
            }
        }
    }

    pub async fn check_permissions(&self, meta: &MetadataMap, required_perm: Option<&str>) -> Result<Claims, Status> {
        let token = match meta.get("authorization") {
            Some(t) => t.to_str().map_err(|_| Status::unauthenticated("Invalid Token format"))?,
            None => return Err(Status::unauthenticated("Login required")),
        };

        let token = token.strip_prefix("Bearer ").unwrap_or(token);
        let secret = Self::get_jwt_secret();

        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(&secret),
            &Validation::default(),
        ).map_err(|_| Status::unauthenticated("Invalid Token or Signature"))?;

        if let Some(sid) = &token_data.claims.sid {
            if !self.sessions.is_valid(sid).await {
                return Err(Status::unauthenticated("Session expired or terminated"));
            }
        }

        // --- RATE LIMITING (DDoS Protection) ---
        let is_allowed = self.sessions.check_rate_limit(&token_data.claims.sub, 500)
            .await
            .unwrap_or(true);

        if !is_allowed {
            return Err(Status::resource_exhausted("Rate limit exceeded (Max 500 requests/sec). Please slow down."));
        }

        if let Some(perm) = required_perm {
            let has_perm = token_data.claims.permissions.contains(perm)
                || token_data.claims.permissions.contains("*");
            if !has_perm {
                return Err(Status::permission_denied(format!("Missing permission: '{}'", perm)));
            }
        }

        Ok(token_data.claims)
    }

    pub async fn resolve_repo(&self, full_path: &str) -> Result<(Arc<dyn PytjaRepository>, String), Status> {
        let clean_path = full_path.trim_start_matches('/');
        let mounts = self.manager.list_mounts().await;

        for mount_name in mounts {
            if clean_path == mount_name || clean_path.starts_with(&format!("{}/", mount_name)) {
                let repo = self.manager.get_repo(&mount_name).await
                    .ok_or_else(|| Status::internal(format!("Mount '{}' not found", mount_name)))?;

                let relative_path = if clean_path == mount_name { "/".to_string() }
                else { format!("/{}", &clean_path[mount_name.len() + 1..]) };
                return Ok((repo, relative_path));
            }
        }

        let repo = self.manager.get_repo("primary").await
            .ok_or_else(|| Status::internal("Primary DB connection lost"))?;
        Ok((repo, full_path.to_string()))
    }

    pub async fn get_user_quota_usage(&self, username: &str) -> usize {
        if let Some(bytes) = self.sessions.get_cached_quota(username).await {
            return bytes as usize;
        }
        if let Some(primary) = self.manager.get_repo("primary").await {
            let usage = primary.get_total_usage(username).await.unwrap_or(0);
            self.sessions.set_cached_quota(username, usage as u64).await;
            return usage;
        }
        0
    }
}