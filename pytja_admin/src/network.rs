use anyhow::Result;
use pytja_proto::pytja::pytja_service_client::PytjaServiceClient;
use pytja_proto::pytja::{
    LoginRequest, GetSessionsRequest, ListRolesRequest,
    ChallengeRequest, KickUserRequest, BanUserRequest,
    ChangeRoleRequest, GetMountsRequest, AddMountRequest,
    SessionInfo, RoleInfo, MountInfo
};
use tonic::transport::Channel;
use tonic::Request;
use pytja_core::crypto::CryptoService;
use std::fs;

pub struct AdminClient {
    client: PytjaServiceClient<Channel>,
    token: String,
}

impl AdminClient {
    pub async fn connect(host: String) -> Result<Self> {
        let client = PytjaServiceClient::connect(host).await?;
        Ok(Self { client, token: String::new() })
    }

    pub async fn login(&mut self, username: &str, key_path: &str, password: &str) -> Result<()> {
        // load key & decrypt
        let encrypted_pem = fs::read_to_string(key_path)?;
        let signing_key = CryptoService::decrypt_private_key_local(&encrypted_pem, password)?;

        // get challenge
        let challenge_req = Request::new(ChallengeRequest { username: username.to_string() });
        let challenge_resp = self.client.get_challenge(challenge_req).await?.into_inner();

        // sign
        let signature = CryptoService::sign_message(&signing_key, challenge_resp.challenge.as_bytes());

        // Login
        let login_req = Request::new(LoginRequest {
            username: username.to_string(),
            challenge: challenge_resp.challenge,
            signature,
        });

        let response = self.client.login(login_req).await?.into_inner();

        if response.success {
            self.token = response.token;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Login failed: {}", response.message))
        }
    }

    fn auth_req<T>(&self, msg: T) -> Request<T> {
        let mut req = Request::new(msg);
        if !self.token.is_empty() {
            req.metadata_mut().insert("authorization", self.token.parse().unwrap());
        }
        req
    }

    // --- STANDARD ADMIN ACTIONS ---

    pub async fn get_sessions(&mut self) -> Result<(Vec<SessionInfo>, i32)> {
        let req = self.auth_req(GetSessionsRequest {});
        let resp = self.client.get_active_sessions(req).await?.into_inner();
        Ok((resp.sessions, resp.total_active))
    }

    pub async fn list_roles(&mut self) -> Result<Vec<RoleInfo>> {
        let req = self.auth_req(ListRolesRequest {});
        let resp = self.client.list_roles(req).await?.into_inner();
        Ok(resp.roles)
    }

    pub async fn kick_user(&mut self, session_id: String) -> Result<String> {
        let req = self.auth_req(KickUserRequest {
            session_id,
            reason: "Kicked by Admin Console".to_string()
        });
        let resp = self.client.kick_user(req).await?.into_inner();
        Ok(resp.message)
    }

    pub async fn ban_user(&mut self, username: String, ban: bool) -> Result<String> {
        let req = self.auth_req(BanUserRequest {
            username,
            ban,
            reason: "Admin Action".to_string()
        });
        let resp = self.client.ban_user(req).await?.into_inner();
        Ok(resp.message)
    }

    pub async fn change_role(&mut self, username: String, new_role: String) -> Result<String> {
        let req = self.auth_req(ChangeRoleRequest { username, new_role });
        let resp = self.client.change_user_role(req).await?.into_inner();
        Ok(resp.message)
    }

    pub async fn get_mounts(&mut self) -> Result<Vec<MountInfo>> {
        let req = self.auth_req(GetMountsRequest {});
        let resp = self.client.get_mounts(req).await?.into_inner();
        Ok(resp.mounts)
    }

    pub async fn add_mount(&mut self, name: String, db_type: String, connection: String) -> Result<String> {
        let req = self.auth_req(AddMountRequest {
            name,
            r#type: db_type,
            connection_string: connection,
        });
        let resp = self.client.add_mount(req).await?.into_inner();
        Ok(resp.message)
    }
}