use pytja_proto::pytja::{self, pytja_service_client::PytjaServiceClient};
use tonic::{transport::Channel, Request};
use std::fs;
use tonic::transport::{ClientTlsConfig, Certificate};
use std::path::PathBuf;
use pytja_core::identity::Identity;
use ed25519_dalek::Signer;

pub struct AdminClient {
    pub client: PytjaServiceClient<Channel>,
    pub token: String,
    pub username: String,
}

impl AdminClient {
    pub async fn connect(url: String) -> anyhow::Result<Self> {
        let mut endpoint = Channel::from_shared(url.clone())?;

        // TLS Integration
        if url.starts_with("https") {
            let possible_paths = vec![
                PathBuf::from("server.crt"),
                PathBuf::from("certs/server.crt"),
                PathBuf::from("../certs/server.crt"),
            ];

            let mut ca_cert = None;
            for p in possible_paths {
                if p.exists() {
                    ca_cert = Some(fs::read_to_string(&p)?);
                    break;
                }
            }

            if let Some(pem) = ca_cert {
                let ca = Certificate::from_pem(pem);
                let tls = ClientTlsConfig::new()
                    .domain_name("localhost")
                    .ca_certificate(ca);

                endpoint = endpoint.tls_config(tls)?;
            } else {
                return Err(anyhow::anyhow!("TLS Certificate 'server.crt' not found. Cannot secure connection."));
            }
        }

        let channel = endpoint.connect().await?;
        let client = PytjaServiceClient::new(channel);

        Ok(Self { client, token: String::new(), username: String::new() })
    }

    pub async fn login_with_identity(&mut self, path: &str) -> anyhow::Result<bool> {
        println!("Authenticating via Identity...");

        let identity = Identity::load_or_prompt(Some(path.to_string()))?;
        self.username = identity.username.clone();

        // Challenge
        let chal_req = Request::new(pytja::ChallengeRequest { username: self.username.clone() });
        let chal_resp: pytja::ChallengeResponse = self.client.get_challenge(chal_req).await?.into_inner();

        if !chal_resp.user_exists {
            return Err(anyhow::anyhow!("User not found on server"));
        }

        // Sign & Login
        use base64::{Engine as _, engine::general_purpose};
        let signature_bytes = identity.keypair.sign(chal_resp.challenge.as_bytes()).to_bytes().to_vec();
        let signature_str = general_purpose::STANDARD.encode(&signature_bytes);

        let login_req = Request::new(pytja::LoginRequest {
            username: self.username.clone(),
            challenge: chal_resp.challenge,
            signature: signature_str,
        });

        let login_resp: pytja::LoginResponse = self.client.login(login_req).await?.into_inner();

        if login_resp.success {
            self.token = login_resp.token;
            Ok(true)
        } else {
            Err(anyhow::anyhow!("Login rejected: {}", login_resp.message))
        }
    }

    pub fn request<T>(&self, msg: T) -> Request<T> {
        let mut req = Request::new(msg);
        if !self.token.is_empty() {
            let auth_value = format!("Bearer {}", self.token);
            if let Ok(val) = auth_value.parse() {
                req.metadata_mut().insert("authorization", val);
            }
        }
        req
    }

    // --- RPC WRAPPERS ---

    pub async fn list_users(&mut self) -> anyhow::Result<Vec<pytja::UserData>> {
        let req = self.request(pytja::ListUsersRequest {});
        let resp: pytja::ListUsersResponse = self.client.list_users(req).await?.into_inner();
        Ok(resp.users)
    }

    #[allow(dead_code)]
    pub async fn register_user(&mut self, username: String, pub_key: Vec<u8>, role: String, quota: u64) -> anyhow::Result<()> {
        let req = self.request(pytja::RegisterUserRequest {
            username,
            public_key: pub_key,
            role,
            quota_limit: quota,
            invite_code: String::new(),
        });
        self.client.register_user(req).await?;
        Ok(())
    }

    pub async fn set_quota(&mut self, username: String, limit: u64) -> anyhow::Result<()> {
        let req = self.request(pytja::SetQuotaRequest { username, limit_bytes: limit });
        self.client.set_user_quota(req).await?;
        Ok(())
    }

    pub async fn get_mounts(&mut self) -> anyhow::Result<Vec<pytja::MountInfo>> {
        let req = self.request(pytja::GetMountsRequest {});
        let resp: pytja::GetMountsResponse = self.client.get_mounts(req).await?.into_inner();
        Ok(resp.mounts)
    }

    pub async fn add_mount(&mut self, name: String, connection_string: String, db_type: String) -> anyhow::Result<()> {
        let req = self.request(pytja::AddMountRequest {
            name, connection_string, r#type: db_type,
        });
        self.client.add_mount(req).await?;
        Ok(())
    }

    pub async fn remove_mount(&mut self, name: String) -> anyhow::Result<()> {
        let req = self.request(pytja::RemoveMountRequest { name });
        self.client.remove_mount(req).await?;
        Ok(())
    }

    pub async fn list_roles(&mut self) -> anyhow::Result<Vec<pytja::RoleInfo>> {
        let req = self.request(pytja::ListRolesRequest {});
        let resp: pytja::ListRolesResponse = self.client.list_roles(req).await?.into_inner();
        Ok(resp.roles)
    }

    pub async fn create_role(&mut self, name: String) -> anyhow::Result<()> {
        let req = self.request(pytja::CreateRoleRequest { name });
        self.client.create_role(req).await?;
        Ok(())
    }

    pub async fn add_permission(&mut self, role_name: String, permission: String) -> anyhow::Result<()> {
        let req = self.request(pytja::AddPermissionRequest { role_name, permission });
        self.client.add_permission(req).await?;
        Ok(())
    }

    pub async fn get_system_stats(&mut self) -> anyhow::Result<pytja::SystemStatsResponse> {
        let req = self.request(pytja::SystemStatsRequest {});
        Ok(self.client.get_system_stats(req).await?.into_inner())
    }

    pub async fn get_audit_logs(&mut self, limit: u32, filter: Option<String>) -> anyhow::Result<Vec<pytja::AuditLogEntry>> {
        let req = self.request(pytja::GetAuditLogsRequest { limit, filter_user: filter.unwrap_or_default() });
        Ok(self.client.get_audit_logs(req).await?.into_inner().logs)
    }

    pub async fn stream_logs(&mut self) -> anyhow::Result<tonic::Streaming<pytja::LogStreamEntry>> {
        let req = self.request(pytja::LogStreamRequest {});
        Ok(self.client.stream_server_logs(req).await?.into_inner())
    }

    // --- INVITE SYSTEM ---

    pub async fn generate_invite(&mut self, role: String, max_uses: u32, quota_limit: u64) -> anyhow::Result<String> {
        let req = self.request(pytja::GenerateInviteRequest {
            role,
            max_uses,
            quota_limit,
        });
        let resp = self.client.generate_invite_code(req).await?.into_inner();
        Ok(resp.code)
    }

    pub async fn list_invites(&mut self) -> anyhow::Result<Vec<pytja::InviteCodeInfo>> {
        let req = self.request(pytja::ListInvitesRequest {});
        let resp = self.client.list_invite_codes(req).await?.into_inner();
        Ok(resp.invites)
    }

    pub async fn revoke_invite(&mut self, code: String) -> anyhow::Result<()> {
        let req = self.request(pytja::RevokeInviteRequest { code });
        self.client.revoke_invite_code(req).await?;
        Ok(())
    }

    // --- USER MANAGEMENT ---

    pub async fn change_user_role(&mut self, username: String, new_role: String) -> anyhow::Result<()> {
        let req = self.request(pytja::ChangeRoleRequest {
            username,
            new_role,
        });
        self.client.change_user_role(req).await?;
        Ok(())
    }
}