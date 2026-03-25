use crate::repo::PytjaRepository;
use crate::error::PytjaError;
use sqlx::postgres::{PgPool, PgPoolOptions};
use crate::models::{User, FileNode, Role, AuditLog};
use async_trait::async_trait;
use sqlx::Row;

#[derive(Clone)]
pub struct PostgresDriver {
    pool: PgPool,
}

impl PostgresDriver {
    pub async fn new(url: &str) -> Result<Self, PytjaError> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl PytjaRepository for PostgresDriver {
    async fn init(&self) -> Result<(), PytjaError> {
        // Users Table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                public_key BYTEA NOT NULL,
                role TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at DOUBLE PRECISION NOT NULL,
                quota_limit BIGINT NOT NULL DEFAULT 0,
                description TEXT
            )"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        // Roles Table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS roles (
                name TEXT PRIMARY KEY,
                permissions TEXT NOT NULL
            )"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        
        sqlx::query(
            "INSERT INTO roles (name, permissions) VALUES ('admin', '[\"core:fs:read\", \"core:fs:write\", \"core:fs:execute\", \"core:fs:delete\", \"core:admin:users\", \"core:admin:roles\", \"core:admin:system\", \"core:admin:mounts\", \"core:admin:invites\"]')
             ON CONFLICT (name) DO NOTHING"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        // File Nodes Table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS file_nodes (
                path TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner TEXT NOT NULL,
                is_folder BOOLEAN NOT NULL DEFAULT false,
                size BIGINT NOT NULL DEFAULT 0,
                content BYTEA NOT NULL,
                blob_id TEXT,
                lock_pass TEXT,
                permissions INT NOT NULL DEFAULT 0,
                created_at DOUBLE PRECISION NOT NULL,
                metadata TEXT
            )"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        // Audit Logs Table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                timestamp DOUBLE PRECISION NOT NULL
            )"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        // Invite Codes Table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS invite_codes (
                code TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                max_uses INT NOT NULL DEFAULT 0,
                used_count INT NOT NULL DEFAULT 0,
                quota_limit BIGINT NOT NULL DEFAULT 0,
                created_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    async fn get_user(&self, username: &str) -> Result<Option<User>, PytjaError> {
        let row = sqlx::query("SELECT * FROM users WHERE username = $1")
            .bind(username)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if let Some(r) = row {
            Ok(Some(User {
                username: r.try_get("username").unwrap_or_default(),
                public_key: r.try_get("public_key").unwrap_or_default(),
                role: r.try_get("role").unwrap_or_default(),
                is_active: r.try_get("is_active").unwrap_or(true),
                created_at: r.try_get("created_at").unwrap_or(0.0),
                quota_limit: r.try_get("quota_limit").unwrap_or(0),
                description: r.try_get("description").ok(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn user_exists(&self, username: &str) -> Result<bool, PytjaError> {
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM users WHERE username = $1")
            .bind(username)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(count.0 > 0)
    }

    async fn save_user_keys(&self, _username: &str, _public_key: &[u8], _private_key_encrypted: &[u8]) -> Result<(), PytjaError> {
        Ok(())
    }

    async fn list_users(&self) -> Result<Vec<User>, PytjaError> {
        let rows = sqlx::query("SELECT * FROM users").fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.into_iter().map(|r| User {
            username: r.try_get("username").unwrap_or_default(),
            public_key: r.try_get("public_key").unwrap_or_default(),
            role: r.try_get("role").unwrap_or_default(),
            is_active: r.try_get("is_active").unwrap_or(true),
            created_at: r.try_get("created_at").unwrap_or(0.0),
            quota_limit: r.try_get("quota_limit").unwrap_or(0),
            description: r.try_get("description").ok(),
        }).collect())
    }

    async fn create_user(&self, user: &User) -> Result<(), PytjaError> {
        sqlx::query("INSERT INTO users (username, public_key, role, created_at, description, is_active, quota_limit) VALUES ($1, $2, $3, $4, $5, $6, $7)")
            .bind(&user.username)
            .bind(&user.public_key)
            .bind(&user.role)
            .bind(user.created_at)
            .bind(&user.description)
            .bind(user.is_active)
            .bind(user.quota_limit)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn update_user_status(&self, username: &str, is_active: bool, role: &str) -> Result<(), PytjaError> {
        sqlx::query("UPDATE users SET is_active = $1, role = $2 WHERE username = $3")
            .bind(is_active).bind(role).bind(username)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn set_user_quota(&self, _username: &str, _limit: u64) -> Result<(), PytjaError> {
        Err(PytjaError::System("Postgres: set_user_quota not implemented".into()))
    }

    async fn get_user_quota_limit(&self, _username: &str) -> Result<u64, PytjaError> {
        Ok(0)
    }

    // --- Filesystem ---

    async fn save_node(&self, node: &FileNode) -> Result<(), PytjaError> {
        sqlx::query("INSERT INTO file_nodes (path, name, owner, is_folder, size, content, blob_id, lock_pass, permissions, created_at, metadata) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                      ON CONFLICT (path) DO UPDATE SET size=$5, content=$6, blob_id=$7, permissions=$9, metadata=$11")
            .bind(&node.path).bind(&node.name).bind(&node.owner).bind(node.is_folder).bind(node.size as i64)
            .bind(&node.content).bind(&node.blob_id).bind(&node.lock_pass).bind(node.permissions as i32).bind(node.created_at).bind(&node.metadata)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_node(&self, path: &str) -> Result<Option<FileNode>, PytjaError> {
        let row = sqlx::query("SELECT * FROM file_nodes WHERE path = $1")
            .bind(path)
            .fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if let Some(r) = row {
            Ok(Some(FileNode {
                path: r.try_get("path").unwrap_or_default(),
                name: r.try_get("name").unwrap_or_default(),
                owner: r.try_get("owner").unwrap_or_default(),
                is_folder: r.try_get("is_folder").unwrap_or(false),
                size: r.try_get::<i64, _>("size").unwrap_or(0) as usize,
                content: r.try_get("content").unwrap_or_default(),
                blob_id: r.try_get("blob_id").ok(),
                lock_pass: r.try_get("lock_pass").ok(),
                permissions: r.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: r.try_get("created_at").unwrap_or(0.0),
                metadata: r.try_get("metadata").ok(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn list_directory(&self, path: &str) -> Result<Vec<FileNode>, PytjaError> {
        let search = format!("{}/%", path.trim_end_matches('/'));
        let rows = sqlx::query("SELECT * FROM file_nodes WHERE path LIKE $1")
            .bind(search)
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        Ok(rows.iter().map(|r| FileNode {
            path: r.try_get("path").unwrap_or_default(),
            name: r.try_get("name").unwrap_or_default(),
            owner: r.try_get("owner").unwrap_or_default(),
            is_folder: r.try_get("is_folder").unwrap_or(false),
            size: r.try_get::<i64, _>("size").unwrap_or(0) as usize,
            content: vec![], // Optimization: Don't load content on list
            blob_id: r.try_get("blob_id").ok(),
            lock_pass: r.try_get("lock_pass").ok(),
            permissions: r.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
            created_at: r.try_get("created_at").unwrap_or(0.0),
            metadata: r.try_get("metadata").ok(),
        }).collect())
    }

    async fn delete_node_recursive(&self, path: &str) -> Result<(), PytjaError> {
        let search = format!("{}%", path);
        sqlx::query("DELETE FROM file_nodes WHERE path LIKE $1")
            .bind(search)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn move_path(&self, src: &str, dst: &str) -> Result<(), PytjaError> {
        sqlx::query("UPDATE file_nodes SET path = $2 WHERE path = $1")
            .bind(src).bind(dst)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn update_metadata(&self, path: &str, lock_pass: Option<String>, owner: Option<String>) -> Result<(), PytjaError> {
        if let Some(l) = lock_pass {
            sqlx::query("UPDATE file_nodes SET lock_pass = $1 WHERE path = $2")
                .bind(l).bind(path).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        }
        if let Some(o) = owner {
            sqlx::query("UPDATE file_nodes SET owner = $1 WHERE path = $2")
                .bind(o).bind(path).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        }
        Ok(())
    }

    async fn update_permissions(&self, path: &str, perms: u8) -> Result<(), PytjaError> {
        sqlx::query("UPDATE file_nodes SET permissions = $1 WHERE path = $2")
            .bind(perms as i32).bind(path).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn find_nodes(&self, pattern: &str) -> Result<Vec<String>, PytjaError> {
        let rows: Vec<(String,)> = sqlx::query_as("SELECT path FROM file_nodes WHERE name LIKE $1")
            .bind(pattern)
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    async fn get_all_files_content(&self) -> Result<Vec<(String, Vec<u8>)>, PytjaError> {
        let rows: Vec<(String, Vec<u8>)> = sqlx::query_as("SELECT path, content FROM file_nodes WHERE is_folder = false")
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows)
    }

    async fn get_total_usage(&self, owner: &str) -> Result<usize, PytjaError> {
        let row: (i64,) = sqlx::query_as("SELECT COALESCE(SUM(size), 0) FROM file_nodes WHERE owner = $1")
            .bind(owner)
            .fetch_one(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(row.0 as usize)
    }

    // --- RBAC ---

    async fn get_role(&self, name: &str) -> Result<Option<Role>, PytjaError> {
        let row = sqlx::query("SELECT * FROM roles WHERE name = $1")
            .bind(name)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if let Some(r) = row {
            use sqlx::Row;
            let perms_str: String = r.try_get("permissions").unwrap_or_else(|_| "[]".into());
            let permissions: Vec<String> = serde_json::from_str(&perms_str).unwrap_or_default();
            Ok(Some(Role {
                name: r.try_get("name").unwrap_or_default(),
                permissions,
            }))
        } else {
            Ok(None)
        }
    }

    async fn create_role(&self, role: &Role) -> Result<(), PytjaError> {
        let perms_str = serde_json::to_string(&role.permissions).unwrap_or_else(|_| "[]".into());
        sqlx::query("INSERT INTO roles (name, permissions) VALUES ($1, $2)")
            .bind(&role.name).bind(&perms_str)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn update_role_permissions(&self, name: &str, permissions: Vec<String>) -> Result<(), PytjaError> {
        let perms_str = serde_json::to_string(&permissions).unwrap_or_else(|_| "[]".into());
        sqlx::query("UPDATE roles SET permissions = $1 WHERE name = $2")
            .bind(perms_str).bind(name)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn list_roles(&self) -> Result<Vec<Role>, PytjaError> {
        let rows = sqlx::query("SELECT * FROM roles").fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.into_iter().map(|r| {
            use sqlx::Row;
            let perms_str: String = r.try_get("permissions").unwrap_or_else(|_| "[]".into());
            let permissions: Vec<String> = serde_json::from_str(&perms_str).unwrap_or_default();
            Role {
                name: r.try_get("name").unwrap_or_default(),
                permissions,
            }
        }).collect())
    }

    // --- Audit ---

    async fn log_action(&self, user: &str, action: &str, target: &str) -> Result<(), PytjaError> {
        sqlx::query("INSERT INTO audit_logs (actor, action, target, timestamp) VALUES ($1, $2, $3, $4)")
            .bind(user).bind(action).bind(target).bind(chrono::Utc::now().timestamp() as f64)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_audit_logs(&self, _limit: u32, _filter: Option<String>) -> Result<Vec<AuditLog>, PytjaError> {
        Err(PytjaError::System("Postgres: audit logs not implemented".into()))
    }

    // --- INVITE SYSTEM IMPLEMENTATION ---
    async fn create_invite(&self, code: &str, role: &str, max_uses: u32, quota_limit: u64, creator: &str) -> Result<(), PytjaError> {
        sqlx::query("INSERT INTO invite_codes (code, role, max_uses, quota_limit, created_by) VALUES ($1, $2, $3, $4, $5)")
            .bind(code).bind(role).bind(max_uses as i32).bind(quota_limit as i64).bind(creator)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_invite(&self, code: &str) -> Result<Option<(String, u64, u32, u32)>, PytjaError> {
        let row = sqlx::query("SELECT role, quota_limit, max_uses, used_count FROM invite_codes WHERE code = $1")
            .bind(code).fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        if let Some(r) = row {
            use sqlx::Row;
            Ok(Some((
                r.try_get("role").unwrap_or_default(),
                r.try_get::<i64, _>("quota_limit").unwrap_or(0) as u64,
                r.try_get::<i32, _>("max_uses").unwrap_or(0) as u32,
                r.try_get::<i32, _>("used_count").unwrap_or(0) as u32,
            )))
        } else { Ok(None) }
    }

    async fn increment_invite_use(&self, code: &str) -> Result<(), PytjaError> {
        sqlx::query("UPDATE invite_codes SET used_count = used_count + 1 WHERE code = $1")
            .bind(code).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn revoke_invite(&self, code: &str) -> Result<(), PytjaError> {
        sqlx::query("DELETE FROM invite_codes WHERE code = $1")
            .bind(code).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn list_invites(&self) -> Result<Vec<(String, String, u32, u32, String, String)>, PytjaError> {
        let rows = sqlx::query("SELECT * FROM invite_codes ORDER BY created_at DESC")
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.into_iter().map(|r| {
            use sqlx::Row;
            let created_at: String = match r.try_get::<String, _>("created_at") {
                Ok(s) => s,
                Err(_) => "unknown".to_string()
            };
            (
                r.try_get("code").unwrap_or_default(),
                r.try_get("role").unwrap_or_default(),
                r.try_get::<i32, _>("max_uses").unwrap_or(0) as u32,
                r.try_get::<i32, _>("used_count").unwrap_or(0) as u32,
                r.try_get("created_by").unwrap_or_default(),
                created_at
            )
        }).collect())
    }

    // --- SECURE QUERY PUSHDOWN (RBAC) ---
    async fn list_directory_secure(&self, path: &str, username: &str, role: &str) -> Result<Vec<FileNode>, PytjaError> {
        let search = format!("{}/%", path.trim_end_matches('/'));
        let is_admin = role == "admin";

        let rows = sqlx::query("SELECT * FROM file_nodes WHERE path LIKE $1 AND ($2 = true OR permissions > 0 OR owner = $3)")
            .bind(&search).bind(is_admin).bind(username)
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
            use sqlx::Row;
            let p: String = row.try_get("path").unwrap_or_default();
            let relative = p.strip_prefix(path).unwrap_or(&p).trim_start_matches('/');
            if relative.contains('/') { continue; }

            nodes.push(FileNode {
                path: p,
                name: row.try_get("name").unwrap_or_default(),
                owner: row.try_get("owner").unwrap_or_default(),
                is_folder: row.try_get("is_folder").unwrap_or(false),
                content: vec![],
                blob_id: row.try_get::<Option<String>, _>("blob_id").unwrap_or(None).filter(|s| !s.is_empty()),
                size: row.try_get::<i64, _>("size").unwrap_or(0) as usize,
                lock_pass: row.try_get::<Option<String>, _>("lock_pass").unwrap_or(None).filter(|s| !s.is_empty()),
                permissions: row.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: row.try_get::<f64, _>("created_at").unwrap_or(0.0),
                metadata: row.try_get::<Option<String>, _>("metadata").unwrap_or(None),
            });
        }
        Ok(nodes)
    }

    async fn list_recursive_secure(&self, path: &str, username: &str, role: &str) -> Result<Vec<FileNode>, PytjaError> {
        let is_admin = role == "admin";
        let search_pattern = format!("{}/%", path.trim_end_matches('/'));

        let mut query_str = "SELECT * FROM file_nodes WHERE path LIKE $1 AND ($2 = true OR permissions > 0 OR owner = $3)";
        if path == "/" {
            query_str = "SELECT * FROM file_nodes WHERE ($1 = true OR permissions > 0 OR owner = $2)";
        }

        let mut query = sqlx::query(query_str);
        if path != "/" {
            query = query.bind(&search_pattern).bind(is_admin).bind(username);
        } else {
            query = query.bind(is_admin).bind(username);
        }

        let rows = query.fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
            use sqlx::Row;
            nodes.push(FileNode {
                path: row.try_get("path").unwrap_or_default(),
                name: row.try_get("name").unwrap_or_default(),
                owner: row.try_get("owner").unwrap_or_default(),
                is_folder: row.try_get("is_folder").unwrap_or(false),
                content: vec![],
                blob_id: None,
                size: row.try_get::<i64, _>("size").unwrap_or(0) as usize,
                lock_pass: None,
                permissions: row.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: row.try_get::<f64, _>("created_at").unwrap_or(0.0),
                metadata: row.try_get::<Option<String>, _>("metadata").unwrap_or(None),
            });
        }
        Ok(nodes)
    }

    async fn get_node_secure(&self, path: &str, username: &str, role: &str) -> Result<Option<FileNode>, PytjaError> {
        let node = self.get_node(path).await?;
        if let Some(n) = node {
            if role != "admin" && n.permissions == 0 && n.owner != username {
                return Ok(None);
            }
            return Ok(Some(n));
        }
        Ok(None)
    }

    async fn read_node_chunk_secure(&self, path: &str, username: &str, role: &str, offset: usize, size: usize) -> Result<Vec<u8>, PytjaError> {
        let is_admin = role == "admin";
        let pg_offset = offset + 1;

        let row = sqlx::query("SELECT SUBSTRING(content FROM $1::int FOR $2::int) as chunk FROM file_nodes WHERE path = $3 AND ($4 = true OR permissions > 0 OR owner = $5)")
            .bind(pg_offset as i32)
            .bind(size as i32)
            .bind(path)
            .bind(is_admin)
            .bind(username)
            .fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if let Some(r) = row {
            use sqlx::Row;
            let chunk: Vec<u8> = r.try_get("chunk").unwrap_or_default();
            Ok(chunk)
        } else {
            Ok(vec![])
        }
    }

    async fn query_metadata_secure(&self, query: &str, username: &str, role: &str) -> Result<Vec<FileNode>, PytjaError> {
        let is_admin = role == "admin";
        let search = format!("%{}%", query);
        
        let rows = sqlx::query("SELECT * FROM file_nodes WHERE metadata IS NOT NULL AND metadata LIKE $1 AND ($2 = true OR permissions > 0 OR owner = $3)")
            .bind(&search)
            .bind(is_admin)
            .bind(username)
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
            use sqlx::Row;
            nodes.push(FileNode {
                path: row.try_get("path").unwrap_or_default(),
                name: row.try_get("name").unwrap_or_default(),
                owner: row.try_get("owner").unwrap_or_default(),
                is_folder: row.try_get("is_folder").unwrap_or(false),
                content: vec![],
                blob_id: None,
                lock_pass: None,
                size: row.try_get::<i64, _>("size").unwrap_or(0) as usize,
                permissions: row.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: row.try_get::<f64, _>("created_at").unwrap_or(0.0),
                metadata: row.try_get::<Option<String>, _>("metadata").unwrap_or(None),
            });
        }
        Ok(nodes)
    }
}