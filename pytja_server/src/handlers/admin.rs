use tonic::{Request, Response, Status};
use pytja_proto::pytja::*;
use pytja_core::{models::{User, Role}, drivers::DatabaseType};
use crate::handlers::service::MyPytjaService;
use sysinfo::{CpuExt, SystemExt, System};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

impl MyPytjaService {

    // --- USER MANAGEMENT ---

    pub async fn list_users_impl(&self, request: Request<ListUsersRequest>) -> Result<Response<ListUsersResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;
        let users_db = repo.list_users().await.map_err(|e| Status::internal(e.to_string()))?;

        let mut user_list = Vec::new();
        for u in users_db {
            let usage = self.get_user_quota_usage(&u.username).await as u64;
            user_list.push(UserData {
                username: u.username,
                role: u.role,
                is_active: u.is_active,
                quota_used: usage,
                quota_limit: u.quota_limit as u64,
                created_at: chrono::DateTime::from_timestamp(u.created_at as i64, 0)
                    .map(|dt| dt.to_string())
                    .unwrap_or_default(),
            });
        }
        Ok(Response::new(ListUsersResponse { users: user_list }))
    }

    // --- REGISTRATION & INVITES ---

    pub async fn register_user_impl(&self, request: Request<RegisterUserRequest>) -> Result<Response<RegisterUserResponse>, Status> {
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        if repo.user_exists(&req.username).await.unwrap_or(false) {
            return Err(Status::already_exists("User already exists"));
        }

        if req.invite_code.is_empty() {
            return Err(Status::permission_denied("Invite code required for registration."));
        }

        let invite = repo.get_invite(&req.invite_code).await.map_err(|e| Status::internal(e.to_string()))?;
        let (assigned_role, assigned_quota) = match invite {
            Some((role, quota, max_uses, used_count)) => {
                if max_uses > 0 && used_count >= max_uses {
                    return Err(Status::permission_denied("Invite code expired or used maximum number of times."));
                }
                (role, quota)
            },
            None => return Err(Status::permission_denied("Invalid invite code.")),
        };
        
        let new_user = User {
            username: req.username.clone(),
            public_key: req.public_key,
            role: assigned_role,
            is_active: true,
            created_at: chrono::Utc::now().timestamp() as f64,
            quota_limit: assigned_quota as i64,
            description: None,
        };

        repo.create_user(&new_user).await.map_err(|e| Status::internal(e.to_string()))?;

        repo.increment_invite_use(&req.invite_code).await.map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(RegisterUserResponse { success: true, message: "Welcome to Pytja. Registration successful.".into() }))
    }

    pub async fn generate_invite_code_impl(&self, request: Request<GenerateInviteRequest>) -> Result<Response<GenerateInviteResponse>, Status> {
        let claims = self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        let timestamp = chrono::Utc::now().timestamp_subsec_nanos();
        let code = format!("PYTJA-{}-{:X}", req.role.to_uppercase(), timestamp);

        repo.create_invite(&code, &req.role, req.max_uses, req.quota_limit, &claims.sub).await
            .map_err(|e| Status::internal(e.to_string()))?;

        if let Some(primary) = self.manager.get_repo("primary").await {
            let _ = primary.log_action(&claims.sub, "INVITE_GENERATE", &code).await;
        }

        Ok(Response::new(GenerateInviteResponse { success: true, code, message: "Invite generated".into() }))
    }

    pub async fn revoke_invite_code_impl(&self, request: Request<RevokeInviteRequest>) -> Result<Response<AdminActionResponse>, Status> {
        let claims = self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        repo.revoke_invite(&req.code).await.map_err(|e| Status::internal(e.to_string()))?;
        if let Some(primary) = self.manager.get_repo("primary").await {
            let _ = primary.log_action(&claims.sub, "INVITE_REVOKE", &req.code).await;
        }
        Ok(Response::new(AdminActionResponse { success: true, message: "Invite revoked".into() }))
    }

    pub async fn list_invite_codes_impl(&self, request: Request<ListInvitesRequest>) -> Result<Response<ListInvitesResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:invites")).await?;
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        let invites = repo.list_invites().await.map_err(|e| Status::internal(e.to_string()))?;
        let proto_invites = invites.into_iter().map(|(c, r, m, u, by, at)| InviteCodeInfo {
            code: c, role: r, max_uses: m, used_count: u, created_by: by, created_at: at
        }).collect();

        Ok(Response::new(ListInvitesResponse { invites: proto_invites }))
    }

    pub async fn set_user_quota_impl(&self, request: Request<SetQuotaRequest>) -> Result<Response<SetQuotaResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        repo.set_user_quota(&req.username, req.limit_bytes).await.map_err(|e| Status::internal(e.to_string()))?;
        self.sessions.invalidate_quota(&req.username).await;

        Ok(Response::new(SetQuotaResponse { success: true, message: "Quota updated.".into() }))
    }

    pub async fn change_user_role_impl(&self, request: Request<ChangeRoleRequest>) -> Result<Response<ChangeRoleResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        let user = repo.get_user(&req.username).await.map_err(|e| Status::internal(e.to_string()))?
            .ok_or(Status::not_found("User not found"))?;

        repo.update_user_status(&req.username, user.is_active, &req.new_role).await
            .map_err(|e| Status::internal(e.to_string()))?;

        self.sessions.update_session_role(&req.username, &req.new_role).await;

        Ok(Response::new(ChangeRoleResponse { success: true, message: format!("Role changed to {}", req.new_role) }))
    }

    pub async fn kick_user_impl(&self, request: Request<KickUserRequest>) -> Result<Response<ActionResponse>, Status> {
        let claims = self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let req = request.into_inner();

        self.sessions.remove_session(&req.session_id).await;

        if let Some(primary) = self.manager.get_repo("primary").await {
            let _ = primary.log_action(&claims.sub, "KICK", &req.session_id).await;
        }
        Ok(Response::new(ActionResponse { success: true, message: "User session terminated.".into() }))
    }

    pub async fn ban_user_impl(&self, request: Request<BanUserRequest>) -> Result<Response<BanUserResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        let user = repo.get_user(&req.username).await.map_err(|e| Status::internal(e.to_string()))?
            .ok_or(Status::not_found("User not found"))?;

        let new_active_status = !req.ban;
        repo.update_user_status(&req.username, new_active_status, &user.role).await
            .map_err(|e| Status::internal(e.to_string()))?;

        if req.ban {
            self.sessions.clear_user_sessions(&req.username).await;
        }

        let msg = if req.ban { "User banned and sessions terminated." } else { "User unbanned." };
        Ok(Response::new(BanUserResponse { success: true, message: msg.into() }))
    }

    pub async fn get_active_sessions_impl(&self, request: Request<GetSessionsRequest>) -> Result<Response<GetSessionsResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:users")).await?;

        let sessions: Vec<_> = self.sessions.get_all_sessions().await.into_iter().map(|s| SessionInfo {
            session_id: s.session_id,
            username: s.username,
            ip_address: s.ip_address,
            role_level: 0,
            login_time: s.login_time.to_rfc3339(),
            last_activity: s.last_activity.to_rfc3339(),
            role: s.role
        }).collect();

        let total = sessions.len() as i32;
        Ok(Response::new(GetSessionsResponse { sessions, total_active: total }))
    }

    // --- ROLE MANAGEMENT (RBAC) ---

    pub async fn create_role_impl(&self, request: Request<CreateRoleRequest>) -> Result<Response<AdminActionResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:roles")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        repo.create_role(&Role { name: req.name, permissions: vec![] }).await.map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(AdminActionResponse { success: true, message: "Role created".into() }))
    }

    pub async fn add_permission_impl(&self, request: Request<AddPermissionRequest>) -> Result<Response<AdminActionResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:roles")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        if let Some(mut role) = repo.get_role(&req.role_name).await.map_err(|e| Status::internal(e.to_string()))? {
            if !role.permissions.contains(&req.permission) {
                role.permissions.push(req.permission);
                repo.update_role_permissions(&role.name, role.permissions).await.map_err(|e| Status::internal(e.to_string()))?;
            }
            Ok(Response::new(AdminActionResponse { success: true, message: "Permission added".into() }))
        } else {
            Err(Status::not_found("Role not found"))
        }
    }

    pub async fn assign_role_impl(&self, request: Request<AssignRoleRequest>) -> Result<Response<AdminActionResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:users")).await?;
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        if !repo.user_exists(&req.username).await.unwrap_or(false) { return Err(Status::not_found("User not found")); }

        let user = repo.get_user(&req.username).await.unwrap().unwrap();
        repo.update_user_status(&req.username, user.is_active, &req.role_name).await.map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(AdminActionResponse { success: true, message: "Role assigned".into() }))
    }

    pub async fn list_roles_impl(&self, request: Request<ListRolesRequest>) -> Result<Response<ListRolesResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:roles")).await?;
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;
        let roles = repo.list_roles().await.map_err(|e| Status::internal(e.to_string()))?;
        let infos = roles.into_iter().map(|r| RoleInfo { name: r.name, permissions: r.permissions }).collect();
        Ok(Response::new(ListRolesResponse { roles: infos }))
    }

    // --- DATABASE / MOUNTS ---

    pub async fn get_mounts_impl(&self, request: Request<GetMountsRequest>) -> Result<Response<GetMountsResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:mounts")).await?;

        let configs = self.manager.get_mount_configs().await;
        let mut infos = Vec::new();

        for cfg in configs {
            infos.push(MountInfo {
                name: cfg.name.clone(),
                r#type: format!("{:?}", cfg.db_type),
                connection: "Hosted".to_string(),
                is_connected: self.manager.get_repo(&cfg.name).await.is_some(),
            });
        }

        Ok(Response::new(GetMountsResponse { mounts: infos }))
    }

    pub async fn add_mount_impl(&self, request: Request<AddMountRequest>) -> Result<Response<AdminActionResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:sys")).await?;
        let req = request.into_inner();

        let db_type = match req.r#type.as_str() {
            "sqlite" => DatabaseType::Sqlite,
            "postgres" => DatabaseType::Postgres,
            "pgvector" => DatabaseType::PgVector,
            _ => return Err(Status::invalid_argument("Unknown DB Type. Supported: sqlite, postgres, pgvector")),
        };

        self.manager.mount(&req.name, &req.connection_string, db_type).await
            .map_err(|e| Status::internal(format!("Mount failed: {}", e)))?;

        Ok(Response::new(AdminActionResponse { success: true, message: "Database mounted".into() }))
    }

    pub async fn remove_mount_impl(&self, request: Request<RemoveMountRequest>) -> Result<Response<AdminActionResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:sys")).await?;
        let req = request.into_inner();

        if req.name == "primary" {
            return Err(Status::invalid_argument("Cannot unmount primary system database."));
        }

        self.manager.unmount(&req.name).await
            .map_err(|e| Status::internal(format!("Unmount failed: {}", e)))?;

        if let Some(primary) = self.manager.get_repo("primary").await {
            let _ = primary.log_action("admin", "UNMOUNT", &req.name).await;
        }

        Ok(Response::new(AdminActionResponse { success: true, message: format!("Database '{}' unmounted.", req.name) }))
    }

    // --- SYSTEM & LOGS ---

    pub async fn get_system_stats_impl(&self, req: Request<SystemStatsRequest>) -> Result<Response<SystemStatsResponse>, Status> {
        self.check_permissions(req.metadata(), Some("core:admin:system")).await?;

        let mut sys = System::new_all();
        sys.refresh_all();

        let active_sessions = self.sessions.get_all_sessions().await.len() as u64;
        let redis_ok = true;

        Ok(Response::new(SystemStatsResponse {
            cpu_usage_percent: sys.global_cpu_info().cpu_usage() as f64,
            memory_usage_bytes: sys.used_memory(),
            active_sessions,
            active_uploads: 0,
            uptime: format!("{} s", sys.uptime()),
            redis_connected: redis_ok,
        }))
    }

    pub async fn get_audit_logs_impl(&self, request: Request<GetAuditLogsRequest>) -> Result<Response<GetAuditLogsResponse>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:system")).await?;
        let req = request.into_inner();

        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;
        let filter = if req.filter_user.is_empty() { None } else { Some(req.filter_user) };

        let logs_db = repo.get_audit_logs(req.limit, filter).await.map_err(|e| Status::internal(e.to_string()))?;

        let logs_proto = logs_db.into_iter().map(|l| AuditLogEntry {
            timestamp: chrono::DateTime::from_timestamp(l.timestamp as i64, 0)
                .map(|dt| dt.to_string())
                .unwrap_or_default(),
            user: l.user_id,
            action: l.action,
            target: l.target,
        }).collect();

        Ok(Response::new(GetAuditLogsResponse { logs: logs_proto }))
    }

    pub async fn stream_server_logs_impl(&self, request: Request<LogStreamRequest>) -> Result<Response<ReceiverStream<Result<LogStreamEntry, Status>>>, Status> {
        self.check_permissions(request.metadata(), Some("core:admin:system")).await?;

        let mut rx = self.log_broadcast.subscribe();
        let (tx, response_rx) = mpsc::channel(100);

        tokio::spawn(async move {
            while let Ok(msg) = rx.recv().await {
                if tx.send(Ok(msg)).await.is_err() { break; }
            }
        });

        Ok(Response::new(ReceiverStream::new(response_rx)))
    }
}