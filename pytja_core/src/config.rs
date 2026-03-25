use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use std::env;
use directories::ProjectDirs;

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    pub primary_url: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StorageConfig {
    pub storage_type: String,
    pub local_path: String,
    pub s3_bucket: Option<String>,
    pub s3_region: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RedisConfig {
    pub url: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PathsConfig {
    pub mounts_file: String,
    pub logs_dir: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TlsConfig {
    pub enabled: bool,
    pub cert_path: String,
    pub key_path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub storage: StorageConfig,
    pub redis: Option<RedisConfig>,
    pub paths: PathsConfig,
    pub tls: Option<TlsConfig>,
}

impl AppConfig {
    pub fn new() -> Result<Self, ConfigError> {
        let run_mode = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());
        
        let (default_mounts, default_logs, default_storage) = if let Some(proj_dirs) = ProjectDirs::from("com", "pytja", "server") {
            let data_dir = proj_dirs.data_dir();
            
            let _ = std::fs::create_dir_all(data_dir);
            let _ = std::fs::create_dir_all(data_dir.join("logs"));

            (
                data_dir.join("mounts.json").to_string_lossy().to_string(),
                data_dir.join("logs").to_string_lossy().to_string(),
                data_dir.join("storage_blobs").to_string_lossy().to_string()
            )
        } else {
            ("mounts.json".to_string(), "logs".to_string(), "storage_blobs".to_string())
        };

        let s = Config::builder()
            .set_default("server.host", "0.0.0.0")?
            .set_default("server.port", 50051)?
            .set_default("database.primary_url", "sqlite://pytja.db")?

            .set_default("storage.storage_type", "fs")?
            .set_default("storage.local_path", default_storage)?

            .set_default("paths.mounts_file", default_mounts)?
            .set_default("paths.logs_dir", default_logs)?

            .add_source(File::with_name("config/default").required(false))

            .add_source(File::with_name(&format!("config/{}", run_mode)).required(false))

            .add_source(Environment::with_prefix("PYTJA").separator("__"))

            .build()?;

        s.try_deserialize()
    }
}