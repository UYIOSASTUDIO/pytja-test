use crate::repo::PytjaRepository;
use crate::models::{User, FileNode, AuditLog, Role};
use crate::error::PytjaError;
use async_trait::async_trait;
use sqlx::{SqlitePool, Row};
use sqlx::sqlite::SqliteConnectOptions;
use std::str::FromStr;
use tracing::info;

#[derive(Clone)]
pub struct SqliteDriver {
    pool: SqlitePool,
}

impl SqliteDriver {
    pub async fn new(path: &str) -> Result<Self, PytjaError> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| PytjaError::System(e.to_string()))?;
        }

        let conn_str = format!("sqlite://{}", path);
        let options = SqliteConnectOptions::from_str(&conn_str)
            .map_err(|e| PytjaError::System(format!("Connection string error: {}", e)))?
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

        let pool = SqlitePool::connect_with(options).await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl PytjaRepository for SqliteDriver {
    async fn init(&self) -> Result<(), PytjaError> {

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS invite_codes (
        code TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        max_uses INTEGER NOT NULL DEFAULT 1,
        used_count INTEGER NOT NULL DEFAULT 0,
        quota_limit BIGINT NOT NULL DEFAULT 0,
        created_by TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );"
        ).execute(&self.pool).await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                public_key BLOB,
                role TEXT NOT NULL DEFAULT 'guest',
                is_active BOOLEAN NOT NULL DEFAULT 1,
                quota_used BIGINT NOT NULL DEFAULT 0,
                quota_limit BIGINT NOT NULL DEFAULT 0,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS file_nodes (
                path TEXT PRIMARY KEY,
                name TEXT,
                owner TEXT,
                is_folder BOOLEAN,
                size INTEGER,
                content BLOB,
                blob_id TEXT,
                lock_pass TEXT,
                permissions INTEGER DEFAULT 0,
                created_at REAL,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                action TEXT,
                target TEXT,
                timestamp REAL
            );
            CREATE TABLE IF NOT EXISTS roles (
                name TEXT PRIMARY KEY,
                permissions TEXT
            );"
        )
            .execute(&self.pool)
            .await
            .map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        
        let _ = sqlx::query("ALTER TABLE file_nodes ADD COLUMN metadata TEXT;")
            .execute(&self.pool).await;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_files_owner ON file_nodes(owner);").execute(&self.pool).await.ok();

        let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM roles")
            .fetch_one(&self.pool).await.unwrap_or(0);

        if count == 0 {
            let admin_perms = serde_json::to_string(&vec!["*"]).unwrap();
            sqlx::query("INSERT INTO roles (name, permissions) VALUES ('admin', ?)").bind(admin_perms).execute(&self.pool).await.ok();
            let guest_perms = serde_json::to_string(&vec!["core:fs:read"]).unwrap();
            sqlx::query("INSERT INTO roles (name, permissions) VALUES ('guest', ?)").bind(guest_perms).execute(&self.pool).await.ok();
            let user_perms = serde_json::to_string(&vec!["core:fs:read", "core:fs:write", "core:exec"]).unwrap();
            sqlx::query("INSERT INTO roles (name, permissions) VALUES ('user', ?)").bind(user_perms).execute(&self.pool).await.ok();
        }

        info!("SQLite DB initialized (Enterprise Schema)");
        Ok(())
    }

    // --- USER ---
    async fn list_users(&self) -> Result<Vec<User>, PytjaError> {
        let rows = sqlx::query("SELECT * FROM users").fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.iter().map(|r| User {
            username: r.try_get("username").unwrap_or_default(),
            public_key: r.try_get("public_key").unwrap_or_default(),
            role: r.try_get("role").unwrap_or_default(),
            is_active: r.try_get("is_active").unwrap_or(false),
            created_at: r.try_get("created_at").unwrap_or(0.0),
            quota_limit: r.try_get("quota_limit").unwrap_or(0),
            description: r.try_get("description").ok(),
        }).collect())
    }

    async fn create_user(&self, user: &User) -> Result<(), PytjaError> {
        sqlx::query("INSERT INTO users (username, public_key, role, is_active, created_at, quota_limit, description) VALUES (?, ?, ?, ?, ?, ?, ?)")
            .bind(&user.username).bind(&user.public_key).bind(&user.role).bind(user.is_active).bind(user.created_at).bind(user.quota_limit).bind(&user.description)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn set_user_quota(&self, username: &str, limit: u64) -> Result<(), PytjaError> {
        sqlx::query("UPDATE users SET quota_limit = ? WHERE username = ?").bind(limit as i64).bind(username)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_user_quota_limit(&self, username: &str) -> Result<u64, PytjaError> {
        let row: Option<(i64,)> = sqlx::query_as("SELECT quota_limit FROM users WHERE username = ?").bind(username)
            .fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(row.map(|r| r.0 as u64).unwrap_or(0))
    }

    async fn get_user(&self, username: &str) -> Result<Option<User>, PytjaError> {
        let row = sqlx::query("SELECT * FROM users WHERE username = ?").bind(username)
            .fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        if let Some(r) = row {
            Ok(Some(User {
                username: r.try_get("username").unwrap_or_default(),
                public_key: r.try_get("public_key").unwrap_or_default(),
                role: r.try_get("role").unwrap_or_default(),
                is_active: r.try_get("is_active").unwrap_or(false),
                created_at: r.try_get("created_at").unwrap_or(0.0),
                quota_limit: r.try_get("quota_limit").unwrap_or(0),
                description: r.try_get("description").ok(),
            }))
        } else { Ok(None) }
    }

    async fn user_exists(&self, username: &str) -> Result<bool, PytjaError> {
        let result: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM users WHERE username = ?").bind(username)
            .fetch_one(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(result.0 > 0)
    }

    async fn save_user_keys(&self, _: &str, _: &[u8], _: &[u8]) -> Result<(), PytjaError> { Ok(()) }

    async fn update_user_status(&self, username: &str, is_active: bool, role: &str) -> Result<(), PytjaError> {
        sqlx::query("UPDATE users SET is_active = ?, role = ? WHERE username = ?").bind(is_active).bind(role).bind(username)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    // --- ROLE ---
    async fn create_role(&self, role: &Role) -> Result<(), PytjaError> {
        let perms = serde_json::to_string(&role.permissions).unwrap_or("[]".to_string());
        sqlx::query("INSERT OR REPLACE INTO roles (name, permissions) VALUES (?, ?)").bind(&role.name).bind(perms)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_role(&self, name: &str) -> Result<Option<Role>, PytjaError> {
        let row = sqlx::query("SELECT * FROM roles WHERE name = ?").bind(name)
            .fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        if let Some(r) = row {
            let perms_str: String = r.try_get("permissions").unwrap_or("[]".to_string());
            let perms: Vec<String> = serde_json::from_str(&perms_str).unwrap_or_default();
            Ok(Some(Role { name: r.try_get("name").unwrap_or_default(), permissions: perms }))
        } else { Ok(None) }
    }

    async fn list_roles(&self) -> Result<Vec<Role>, PytjaError> {
        let rows = sqlx::query("SELECT * FROM roles").fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.iter().map(|r| {
            let perms_str: String = r.try_get("permissions").unwrap_or("[]".to_string());
            let perms: Vec<String> = serde_json::from_str(&perms_str).unwrap_or_default();
            Role { name: r.try_get("name").unwrap_or_default(), permissions: perms }
        }).collect())
    }

    async fn update_role_permissions(&self, role_name: &str, permissions: Vec<String>) -> Result<(), PytjaError> {
        self.create_role(&Role { name: role_name.to_string(), permissions }).await
    }

    // --- FILE SYSTEM ---
    async fn save_node(&self, node: &FileNode) -> Result<(), PytjaError> {
        sqlx::query("INSERT OR REPLACE INTO file_nodes (path, name, owner, is_folder, size, content, blob_id, lock_pass, permissions, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&node.path).bind(&node.name).bind(&node.owner).bind(node.is_folder).bind(node.size as i64)
            .bind(&node.content).bind(&node.blob_id).bind(&node.lock_pass).bind(node.permissions as i32).bind(node.created_at).bind(&node.metadata)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_node(&self, path: &str) -> Result<Option<FileNode>, PytjaError> {
        let row = sqlx::query("SELECT * FROM file_nodes WHERE path = ?").bind(path)
            .fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        if let Some(row) = row {
            let blob_id: Option<String> = row.try_get::<Option<String>, _>("blob_id").unwrap_or(None).filter(|s| !s.is_empty());
            let lock_pass: Option<String> = row.try_get::<Option<String>, _>("lock_pass").unwrap_or(None).filter(|s| !s.is_empty());

            Ok(Some(FileNode {
                path: row.try_get("path").unwrap_or_default(),
                name: row.try_get("name").unwrap_or_default(),
                owner: row.try_get("owner").unwrap_or_default(),
                is_folder: row.try_get("is_folder").unwrap_or(false),
                content: row.try_get("content").unwrap_or_default(),
                blob_id,
                size: row.try_get::<i64, _>("size").unwrap_or(0) as usize,
                lock_pass,
                permissions: row.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: row.try_get("created_at").unwrap_or(0.0),
                metadata: row.try_get::<Option<String>, _>("metadata").unwrap_or(None),
            }))
        } else { Ok(None) }
    }

    async fn list_directory(&self, path: &str) -> Result<Vec<FileNode>, PytjaError> {
        let search = format!("{}/%", path.trim_end_matches('/'));
        let rows = sqlx::query("SELECT * FROM file_nodes WHERE path LIKE ?").bind(search)
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
            let p: String = row.try_get("path").unwrap_or_default();
            let relative = p.strip_prefix(path).unwrap_or(&p).trim_start_matches('/');
            if relative.contains('/') { continue; }

            let blob_id: Option<String> = row.try_get::<Option<String>, _>("blob_id").unwrap_or(None).filter(|s| !s.is_empty());
            let lock_pass: Option<String> = row.try_get::<Option<String>, _>("lock_pass").unwrap_or(None).filter(|s| !s.is_empty());

            nodes.push(FileNode {
                path: p,
                name: row.try_get("name").unwrap_or_default(),
                owner: row.try_get("owner").unwrap_or_default(),
                is_folder: row.try_get("is_folder").unwrap_or(false),
                content: vec![],
                blob_id,
                size: row.try_get::<i64, _>("size").unwrap_or(0) as usize,
                lock_pass,
                permissions: row.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: row.try_get("created_at").unwrap_or(0.0),
                metadata: row.try_get::<Option<String>, _>("metadata").unwrap_or(None),
            });
        }
        Ok(nodes)
    }
    
    async fn list_recursive(&self, path: &str) -> Result<Vec<FileNode>, PytjaError> {
        let mut query_str = "SELECT * FROM file_nodes WHERE path LIKE ?";
        let search_pattern = format!("{}/%", path.trim_end_matches('/'));
        
        if path == "/" {
            query_str = "SELECT * FROM file_nodes";
        }

        let mut query = sqlx::query(query_str);
        if path != "/" {
            query = query.bind(&search_pattern);
        }

        let rows = query.fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
            let blob_id: Option<String> = row.try_get::<Option<String>, _>("blob_id").unwrap_or(None).filter(|s| !s.is_empty());
            let lock_pass: Option<String> = row.try_get::<Option<String>, _>("lock_pass").unwrap_or(None).filter(|s| !s.is_empty());

            nodes.push(FileNode {
                path: row.try_get("path").unwrap_or_default(),
                name: row.try_get("name").unwrap_or_default(),
                owner: row.try_get("owner").unwrap_or_default(),
                is_folder: row.try_get("is_folder").unwrap_or(false),
                content: vec![],
                blob_id,
                size: row.try_get::<i64, _>("size").unwrap_or(0) as usize,
                lock_pass,
                permissions: row.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: row.try_get("created_at").unwrap_or(0.0),
                metadata: row.try_get::<Option<String>, _>("metadata").unwrap_or(None),
            });
        }
        Ok(nodes)
    }

    async fn delete_node_recursive(&self, path: &str) -> Result<(), PytjaError> {
        let like_pattern = format!("{}/%", path);
        sqlx::query("DELETE FROM file_nodes WHERE path = ? OR path LIKE ?")
            .bind(path).bind(like_pattern).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn move_path(&self, old_path: &str, new_path: &str) -> Result<(), PytjaError> {
        sqlx::query("UPDATE file_nodes SET path = ? || SUBSTR(path, LENGTH(?) + 1) WHERE path = ? OR path LIKE ? || '/%'")
            .bind(new_path).bind(old_path).bind(old_path).bind(old_path).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn update_metadata(&self, path: &str, lock: Option<String>, owner: Option<String>) -> Result<(), PytjaError> {
        if let Some(l) = lock { sqlx::query("UPDATE file_nodes SET lock_pass = ? WHERE path = ?").bind(l).bind(path).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?; }
        if let Some(o) = owner { sqlx::query("UPDATE file_nodes SET owner = ? WHERE path = ?").bind(o).bind(path).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?; }
        Ok(())
    }

    async fn update_permissions(&self, path: &str, permissions: u8) -> Result<(), PytjaError> {
        sqlx::query("UPDATE file_nodes SET permissions = ? WHERE path = ?").bind(permissions as i32).bind(path).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_total_usage(&self, owner: &str) -> Result<usize, PytjaError> {
        let size: Option<i64> = sqlx::query_scalar("SELECT SUM(size) FROM file_nodes WHERE owner = ?").bind(owner).fetch_one(&self.pool).await.ok().flatten();
        Ok(size.unwrap_or(0) as usize)
    }

    async fn find_nodes(&self, pattern: &str) -> Result<Vec<String>, PytjaError> {
        let rows = sqlx::query("SELECT path FROM file_nodes WHERE name LIKE ?").bind(pattern).fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.iter().map(|r| r.try_get("path").unwrap_or_default()).collect())
    }

    async fn get_all_files_content(&self) -> Result<Vec<(String, Vec<u8>)>, PytjaError> {
        let rows = sqlx::query("SELECT path, content FROM file_nodes WHERE is_folder = 0").fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        let mut res = Vec::new();
        for r in rows {
            let path: String = r.try_get("path").unwrap_or_default();
            let content: Vec<u8> = r.try_get("content").unwrap_or_default();
            res.push((path, content));
        }
        Ok(res)
    }

    async fn log_action(&self, actor: &str, action: &str, target: &str) -> Result<(), PytjaError> {
        let now = chrono::Utc::now().timestamp() as f64;
        sqlx::query("INSERT INTO audit_logs (timestamp, user_id, action, target) VALUES (?, ?, ?, ?)").bind(now).bind(actor).bind(action).bind(target)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_audit_logs(&self, limit: u32, user_filter: Option<String>) -> Result<Vec<AuditLog>, PytjaError> {
        let sql = if user_filter.is_some() { format!("SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT {}", limit) }
        else { format!("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT {}", limit) };
        let query = sqlx::query_as::<_, AuditLog>(&sql);
        let query = if let Some(user) = user_filter { query.bind(user) } else { query };
        let result: Vec<AuditLog> = query.fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(result)
    }

    // --- INVITE SYSTEM IMPLEMENTATION ---
    async fn create_invite(&self, code: &str, role: &str, max_uses: u32, quota_limit: u64, creator: &str) -> Result<(), PytjaError> {
        sqlx::query("INSERT INTO invite_codes (code, role, max_uses, quota_limit, created_by) VALUES (?, ?, ?, ?, ?)")
            .bind(code).bind(role).bind(max_uses).bind(quota_limit as i64).bind(creator)
            .execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn get_invite(&self, code: &str) -> Result<Option<(String, u64, u32, u32)>, PytjaError> {
        let row = sqlx::query("SELECT role, quota_limit, max_uses, used_count FROM invite_codes WHERE code = ?")
            .bind(code).fetch_optional(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        if let Some(r) = row {
            Ok(Some((
                r.try_get("role").unwrap_or_default(),
                r.try_get::<i64, _>("quota_limit").unwrap_or(0) as u64,
                r.try_get::<i32, _>("max_uses").unwrap_or(0) as u32,
                r.try_get::<i32, _>("used_count").unwrap_or(0) as u32,
            )))
        } else { Ok(None) }
    }

    async fn increment_invite_use(&self, code: &str) -> Result<(), PytjaError> {
        sqlx::query("UPDATE invite_codes SET used_count = used_count + 1 WHERE code = ?")
            .bind(code).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn revoke_invite(&self, code: &str) -> Result<(), PytjaError> {
        sqlx::query("DELETE FROM invite_codes WHERE code = ?")
            .bind(code).execute(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn list_invites(&self) -> Result<Vec<(String, String, u32, u32, String, String)>, PytjaError> {
        let rows = sqlx::query("SELECT * FROM invite_codes ORDER BY created_at DESC")
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;
        Ok(rows.into_iter().map(|r| {
            (
                r.try_get("code").unwrap_or_default(),
                r.try_get("role").unwrap_or_default(),
                r.try_get::<i32, _>("max_uses").unwrap_or(0) as u32,
                r.try_get::<i32, _>("used_count").unwrap_or(0) as u32,
                r.try_get("created_by").unwrap_or_default(),
                r.try_get::<String, _>("created_at").unwrap_or_default()
            )
        }).collect())
    }

    // --- SECURE QUERY PUSHDOWN (RBAC) ---
    async fn list_directory_secure(&self, path: &str, username: &str, role: &str) -> Result<Vec<FileNode>, PytjaError> {
        let search = format!("{}/%", path.trim_end_matches('/'));
        let is_admin = role == "admin";

        let rows = sqlx::query("SELECT * FROM file_nodes WHERE path LIKE ? AND (? = 1 OR permissions > 0 OR owner = ?)")
            .bind(&search).bind(is_admin).bind(username)
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
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
                created_at: row.try_get("created_at").unwrap_or(0.0),
                metadata: row.try_get::<Option<String>, _>("metadata").unwrap_or(None),
            });
        }
        Ok(nodes)
    }

    async fn list_recursive_secure(&self, path: &str, username: &str, role: &str) -> Result<Vec<FileNode>, PytjaError> {
        let is_admin = role == "admin";
        let search_pattern = format!("{}/%", path.trim_end_matches('/'));

        let mut query_str = "SELECT * FROM file_nodes WHERE path LIKE ? AND (? = 1 OR permissions > 0 OR owner = ?)";
        if path == "/" {
            query_str = "SELECT * FROM file_nodes WHERE (? = 1 OR permissions > 0 OR owner = ?)";
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
                created_at: row.try_get("created_at").unwrap_or(0.0),
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
        
        let sqlite_offset = offset + 1;
        
        let row = sqlx::query("SELECT SUBSTR(content, ?, ?) as chunk FROM file_nodes WHERE path = ? AND (? = 1 OR permissions > 0 OR owner = ?)")
            .bind(sqlite_offset as i64)
            .bind(size as i64)
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

        let rows = sqlx::query("SELECT * FROM file_nodes WHERE metadata IS NOT NULL AND metadata LIKE ? AND (? = 1 OR permissions > 0 OR owner = ?)")
            .bind(&search).bind(is_admin).bind(username)
            .fetch_all(&self.pool).await.map_err(|e| PytjaError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
            nodes.push(FileNode {
                path: row.try_get("path").unwrap_or_default(),
                name: row.try_get("name").unwrap_or_default(),
                owner: row.try_get("owner").unwrap_or_default(),
                is_folder: row.try_get("is_folder").unwrap_or(false),
                content: vec![], blob_id: None, lock_pass: None,
                size: row.try_get::<i64, _>("size").unwrap_or(0) as usize,
                permissions: row.try_get::<i32, _>("permissions").unwrap_or(0) as u8,
                created_at: row.try_get("created_at").unwrap_or(0.0),
                metadata: row.try_get("metadata").ok(),
            });
        }
        Ok(nodes)
    }
}