use pytja_core::drivers::sqlite::SqliteDriver;
use pytja_core::drivers::postgres::PostgresDriver;
use pytja_core::repo::PytjaRepository;
use pytja_core::models::User;
use pytja_core::config::AppConfig;
use std::sync::Arc;
use ed25519_dalek::SigningKey;
use rand::{rngs::OsRng, RngCore};
use std::fs;
use std::path::Path;
use base64::{Engine as _, engine::general_purpose};
use dialoguer::{Input, Password};
use aes_gcm::{aead::{Aead, KeyInit}, Aes256Gcm, Nonce};
use pbkdf2::pbkdf2;
use hmac::Hmac;
use sha2::Sha256;

pub async fn start_registrar(output_dir: Option<String>) -> anyhow::Result<()> {
    println!("--- PYTJA IDENTITY REGISTRAR (SECURE V1) ---");

    let config = AppConfig::new().map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;
    let db_url = config.database.primary_url;

    println!("Connecting to Database at: {}", db_url);

    let repo: Arc<dyn PytjaRepository> = if db_url.starts_with("postgres://") || db_url.starts_with("postgresql://") {
        let driver = PostgresDriver::new(&db_url).await?;
        driver.init().await?;
        Arc::new(driver)
    } else if db_url.starts_with("sqlite://") {
        let path = db_url.replace("sqlite://", "");
        let driver = SqliteDriver::new(&path).await?;
        driver.init().await?;
        Arc::new(driver)
    } else {
        return Err(anyhow::anyhow!("Unsupported database URL protocol: {}", db_url));
    };

    let save_dir = output_dir.unwrap_or_else(|| ".".to_string());
    let save_path = Path::new(&save_dir);
    if !save_path.exists() {
        fs::create_dir_all(save_path)?;
    }

    let username: String = Input::new().with_prompt("Username").interact_text()?;

    if repo.user_exists(&username).await.unwrap_or(false) {
        println!("User '{}' already exists in DB.", username);
    }

    let password = Password::new()
        .with_prompt("Identity Password")
        .with_confirmation("Confirm Password", "Mismatch")
        .interact()?;

    println!("Generating keys...");
    let mut csprng = OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let pub_key_bytes = signing_key.verifying_key().to_bytes().to_vec();
    let priv_key_bytes = signing_key.to_bytes();

    let mut salt = [0u8; 16];
    csprng.fill_bytes(&mut salt);

    let mut nonce_bytes = [0u8; 12];
    csprng.fill_bytes(&mut nonce_bytes);

    let mut derived_key = [0u8; 32];
    pbkdf2::<Hmac<Sha256>>(password.as_bytes(), &salt, 100_000, &mut derived_key)
        .expect("Critical error in key derivation");

    let cipher = Aes256Gcm::new(&derived_key.into());
    let nonce = Nonce::from_slice(&nonce_bytes);

    let encrypted_priv = cipher.encrypt(nonce, priv_key_bytes.as_ref())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    let mut payload = Vec::new();
    payload.extend_from_slice(&salt);
    payload.extend_from_slice(&nonce_bytes);
    payload.extend_from_slice(&encrypted_priv);

    let priv_b64 = general_purpose::STANDARD.encode(&payload);
    let pub_b64 = general_purpose::STANDARD.encode(&pub_key_bytes);

    let filename_path = save_path.join(format!("{}.pytja", username));
    let filename = filename_path.to_string_lossy().to_string();

    let content = format!("PYTJA-ID-V2-ENCRYPTED\nUSER:{}\nPRIV:{}\nPUB:{}\nROLE:admin", username, priv_b64, pub_b64);

    fs::write(&filename_path, content)?;
    println!("Identity saved to: {}", filename);

    let user = User {
        username: username.clone(),
        public_key: pub_key_bytes,
        role: "admin".to_string(),
        is_active: true,
        created_at: chrono::Utc::now().timestamp() as f64,
        quota_limit: 0,
        description: Some("Admin User".into()),
    };

    repo.create_user(&user).await.map_err(|e| anyhow::anyhow!("Database Error: {}", e))?;
    println!("User '{}' successfully registered in Enterprise Database.", username);

    Ok(())
}