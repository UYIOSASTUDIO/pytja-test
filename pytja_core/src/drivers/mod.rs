pub mod sqlite;
pub mod postgres;
pub mod pgvector;
pub mod sqlite_vec;

use crate::repo::{PytjaRepository, VectorStore};
use crate::error::PytjaError;
use std::sync::Arc;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

// Database Support
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DatabaseType {
    Sqlite,
    Postgres,
    MySQL,
    /// PostgreSQL with pgvector extension for vector similarity search.
    /// Connection string should point to a PostgreSQL instance with pgvector installed.
    PgVector,
    /// Embedded SQLite-based vector store with pure Rust similarity search.
    /// Path should be a filesystem path to the SQLite database file.
    /// Ideal for local development, edge deployments, and zero-dependency setups.
    SqliteVec,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MountConfig {
    pub name: String,
    pub path: String,
    pub db_type: DatabaseType,
}

pub struct DriverManager {
    connections: Arc<RwLock<HashMap<String, Arc<dyn PytjaRepository>>>>,
    /// Separate registry for vector store connections
    vector_connections: Arc<RwLock<HashMap<String, Arc<dyn VectorStore>>>>,
    config_cache: Arc<RwLock<Vec<MountConfig>>>,
    config_file_path: Arc<RwLock<String>>,
}

impl Default for DriverManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DriverManager {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            vector_connections: Arc::new(RwLock::new(HashMap::new())),
            config_cache: Arc::new(RwLock::new(Vec::new())),
            config_file_path: Arc::new(RwLock::new("mounts.json".to_string())),
        }
    }

    pub async fn load_config(&self, config_path: &str) {
        info!("Loading configuration from '{}'", config_path);

        {
            let mut p = self.config_file_path.write().await;
            *p = config_path.to_string();
        }

        match fs::read_to_string(config_path).await {
            Ok(content) => {
                match serde_json::from_str::<Vec<MountConfig>>(&content) {
                    Ok(configs) => {
                        info!("Found {} mount definitions.", configs.len());

                        {
                            let mut cache = self.config_cache.write().await;
                            *cache = configs.clone();
                        }

                        for cfg in configs {
                            if let Err(e) = self.mount_internal(&cfg.name, &cfg.path, cfg.db_type.clone(), false).await {
                                error!("Failed to mount database '{}': {}", cfg.name, e);
                            }
                        }
                    },
                    Err(e) => warn!("Could not parse mounts.json: {}", e),
                }
            },
            Err(_) => warn!("No mounts.json found at '{}'. Starting with empty configuration.", config_path),
        }
    }

    pub async fn mount(&self, name: &str, path: &str, db_type: DatabaseType) -> Result<(), PytjaError> {
        self.mount_internal(name, path, db_type, true).await
    }

    async fn mount_internal(&self, name: &str, path: &str, db_type: DatabaseType, save_to_disk: bool) -> Result<(), PytjaError> {
        match db_type {
            DatabaseType::PgVector => {
                // PgVector mounts go into the vector_connections registry
                let driver = pgvector::PgVectorDriver::new(path).await?;
                driver.vector_init().await?;
                let store: Arc<dyn VectorStore> = Arc::new(driver);
                {
                    let mut map = self.vector_connections.write().await;
                    map.insert(name.to_string(), store);
                }
                info!("Mounted vector store '{}' (PgVector)", name);
            },
            DatabaseType::SqliteVec => {
                // SqliteVec mounts go into the vector_connections registry
                let driver = sqlite_vec::SqliteVecDriver::new(path).await?;
                driver.vector_init().await?;
                let store: Arc<dyn VectorStore> = Arc::new(driver);
                {
                    let mut map = self.vector_connections.write().await;
                    map.insert(name.to_string(), store);
                }
                info!("Mounted vector store '{}' (SqliteVec)", name);
            },
            _ => {
                // Standard relational database mounts
                let repo: Arc<dyn PytjaRepository> = match db_type {
                    DatabaseType::Sqlite => {
                        let driver = sqlite::SqliteDriver::new(path).await?;
                        driver.init().await?;
                        Arc::new(driver)
                    },
                    DatabaseType::Postgres => {
                        let driver = postgres::PostgresDriver::new(path).await?;
                        driver.init().await?;
                        Arc::new(driver)
                    },
                    _ => return Err(PytjaError::System("Unsupported DB Type".into())),
                };

                {
                    let mut map = self.connections.write().await;
                    map.insert(name.to_string(), repo);
                }
                info!("Mounted database '{}' ({:?})", name, db_type);
            }
        }

        if save_to_disk {
            self.persist_mount(name, path, db_type).await?;
        }

        Ok(())
    }
    
    async fn persist_mount(&self, name: &str, path: &str, db_type: DatabaseType) -> Result<(), PytjaError> {
        let config_path = {
            let p = self.config_file_path.read().await;
            p.clone()
        };

        let configs_copy;
        {
            let mut cache = self.config_cache.write().await;

            if let Some(existing) = cache.iter_mut().find(|c| c.name == name) {
                existing.path = path.to_string();
                existing.db_type = db_type;
            } else {
                cache.push(MountConfig {
                    name: name.to_string(),
                    path: path.to_string(),
                    db_type,
                });
            }
            configs_copy = cache.clone();
        }
        
        let json = serde_json::to_string_pretty(&configs_copy)
            .map_err(|e| PytjaError::System(format!("Serialization error: {}", e)))?;
        
        let temp_path = format!("{}.tmp", config_path);

        if let Err(e) = fs::write(&temp_path, &json).await {
            return Err(PytjaError::System(format!("Failed to write temp config: {}", e)));
        }

        if let Err(e) = fs::rename(&temp_path, &config_path).await {
            return Err(PytjaError::System(format!("Failed to commit config file: {}", e)));
        }

        info!("Persisted configuration to {}", config_path);
        Ok(())
    }

    pub async fn unmount(&self, name: &str) -> Result<(), PytjaError> {
        let config_path = {
            let p = self.config_file_path.read().await;
            p.clone()
        };

        // Try to remove from relational connections first, then vector connections
        let removed = {
            let mut map = self.connections.write().await;
            map.remove(name).is_some()
        };
        if !removed {
            let mut vmap = self.vector_connections.write().await;
            if vmap.remove(name).is_none() {
                return Err(PytjaError::NotFound(format!("Database '{}' not found", name)));
            }
        }

        let configs_copy;
        {
            let mut cache = self.config_cache.write().await;
            if let Some(pos) = cache.iter().position(|c| c.name == name) {
                cache.remove(pos);
                configs_copy = Some(cache.clone());
            } else {
                configs_copy = None;
            }
        }

        if let Some(cfg) = configs_copy {
            let json = serde_json::to_string_pretty(&cfg)
                .map_err(|e| PytjaError::System(format!("Serialization error: {}", e)))?;

            let temp_path = format!("{}.tmp", config_path);
            let _ = fs::write(&temp_path, &json).await;
            let _ = fs::rename(&temp_path, config_path).await;

            info!("Unmounted '{}' and removed from config.", name);
        }

        Ok(())
    }

    pub async fn get_repo(&self, name: &str) -> Option<Arc<dyn PytjaRepository>> {
        let map = self.connections.read().await;
        map.get(name).cloned()
    }

    pub async fn list_mounts(&self) -> Vec<String> {
        let map = self.connections.read().await;
        map.keys().cloned().collect()
    }

    pub async fn get_mount_configs(&self) -> Vec<MountConfig> {
        let cache = self.config_cache.read().await;
        cache.clone()
    }

    // --- VECTOR STORE ACCESSORS ---

    /// Get a vector store connection by mount name.
    pub async fn get_vector_store(&self, name: &str) -> Option<Arc<dyn VectorStore>> {
        let map = self.vector_connections.read().await;
        map.get(name).cloned()
    }

    /// List all mounted vector store names.
    pub async fn list_vector_mounts(&self) -> Vec<String> {
        let map = self.vector_connections.read().await;
        map.keys().cloned().collect()
    }

    /// Unmount a vector store by name.
    pub async fn unmount_vector(&self, name: &str) -> Result<(), PytjaError> {
        {
            let mut map = self.vector_connections.write().await;
            if map.remove(name).is_none() {
                return Err(PytjaError::NotFound(format!("Vector store '{}' not found", name)));
            }
        }
        info!("Unmounted vector store '{}'", name);
        Ok(())
    }
}