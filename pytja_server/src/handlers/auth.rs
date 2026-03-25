use tonic::{Request, Response, Status};
use pytja_proto::pytja::{LoginRequest, LoginResponse, ChallengeRequest, ChallengeResponse, PingRequest, PingResponse};
use pytja_core::{crypto::CryptoService, models::{Claims, Role}};
use crate::handlers::service::MyPytjaService;
use jsonwebtoken::{encode, Header, EncodingKey};
use std::collections::HashSet;
use tracing::warn;

impl MyPytjaService {
    pub async fn ping_impl(&self, _request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        Ok(Response::new(PingResponse { message: "Pong".into(), server_version: "Pytja Enterprise V3.0".to_string(), is_ready: true }))
    }

    pub async fn get_challenge_impl(&self, request: Request<ChallengeRequest>) -> Result<Response<ChallengeResponse>, Status> {
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;
        let exists = repo.user_exists(&req.username).await.map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ChallengeResponse { challenge: CryptoService::generate_random_challenge(), user_exists: exists }))
    }

    pub async fn login_impl(&self, request: Request<LoginRequest>) -> Result<Response<LoginResponse>, Status> {
        let req = request.into_inner();
        let repo = self.manager.get_repo("primary").await.ok_or(Status::internal("DB Error"))?;

        let user = match repo.get_user(&req.username).await {
            Ok(Some(u)) => u,
            Ok(None) => return Ok(Response::new(LoginResponse { success: false, token: "".into(), message: "User not found".into() })),
            Err(e) => return Err(Status::internal(e.to_string())),
        };

        if !CryptoService::verify_signature(&user.public_key, req.challenge.as_bytes(), &req.signature).unwrap_or(false) {
            warn!("Invalid signature for user {}", req.username);
            return Ok(Response::new(LoginResponse { success: false, token: "".into(), message: "Invalid Signature".into() }));
        }

        let role = if let Some(cached) = self.sessions.get_cached_role(&user.role).await { cached } else {
            let r = repo.get_role(&user.role).await.map_err(|e| Status::internal(e.to_string()))?
                .unwrap_or(Role { name: "guest".into(), permissions: vec![] });
            self.sessions.cache_role(&r).await;
            r
        };

        let mut perms_set = HashSet::new();
        for p in role.permissions { perms_set.insert(p); }

        let expiration = chrono::Utc::now().checked_add_signed(chrono::Duration::minutes(60)).unwrap().timestamp() as usize;
        self.sessions.clear_user_sessions(&user.username).await;

        let session_id = self.sessions.register_session(&user.username, &user.role, "127.0.0.1").await
            .map_err(|e| Status::internal(format!("Redis: {}", e)))?;

        let claims = Claims {
            sub: user.username.clone(),
            role: user.role.clone(),
            permissions: perms_set,
            exp: expiration,
            sid: Some(session_id),
        };

        let secret = Self::get_jwt_secret();
        let token = encode(&Header::default(), &claims, &EncodingKey::from_secret(&secret))
            .map_err(|e| Status::internal(format!("Token error: {}", e)))?;

        Ok(Response::new(LoginResponse { success: true, token, message: "Login successful".into() }))
    }
}