use anyhow::{Result, anyhow, Context};
use ed25519_dalek::SigningKey;
use std::fs;
use std::path::Path;
use base64::{Engine as _, engine::general_purpose};
use dialoguer::{Password, Input};
use aes_gcm::{aead::{Aead, KeyInit}, Aes256Gcm, Nonce};
use pbkdf2::pbkdf2;
use hmac::Hmac;
use sha2::Sha256;
use colored::*;

pub struct Identity {
    pub username: String,
    pub keypair: SigningKey,
}

impl Identity {
    pub fn load_or_prompt(provided_path: Option<String>) -> Result<Self> {
        let mut path_str = match provided_path {
            Some(p) => p,
            
            None => {
                println!("{}", "No identity path provided via arguments or environment.".yellow());
                Input::<String>::new()
                    .with_prompt("Enter absolute path to your .pytja file (e.g., /Volumes/USB/sandro.pytja)")
                    .interact_text()?
            }
        };
        
        if path_str.starts_with("~/") {
            if let Ok(home) = std::env::var("HOME") {
                path_str = path_str.replacen("~", &home, 1);
            }
        }

        let path = Path::new(&path_str);
        if !path.exists() {
            return Err(anyhow!("Identity file not found at: {}", path_str));
        }

        Self::load(&path_str)
    }

    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Could not open file: {}", path))?;

        let mut username = String::new();
        let mut priv_b64 = String::new();

        for line in content.lines() {
            if let Some(v) = line.strip_prefix("USER:") { username = v.trim().to_string(); }
            if let Some(v) = line.strip_prefix("PRIV:") { priv_b64 = v.trim().to_string(); }
        }

        if username.is_empty() || priv_b64.is_empty() {
            return Err(anyhow!("Invalid .pytja file format"));
        }

        println!("Identity: {} ({})", username, path);
        let password = Password::new().with_prompt("Enter Password").interact()?;
        
        let blob = general_purpose::STANDARD.decode(&priv_b64).context("Base64 decode failed")?;

        if blob.len() < 28 {
            return Err(anyhow!("File corrupted (too short)"));
        }

        let salt = &blob[0..16];
        let nonce_bytes = &blob[16..28];
        let ciphertext = &blob[28..];

        // Decrypt
        let mut derived_key = [0u8; 32];
        pbkdf2::<Hmac<Sha256>>(password.as_bytes(), salt, 100_000, &mut derived_key).expect("PBKDF2 failed");

        let cipher = Aes256Gcm::new(&derived_key.into());
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|_| anyhow!("Decryption failed! Wrong password?"))?;

        if plaintext.len() != 32 {
            return Err(anyhow!("Invalid private key length"));
        }

        let secret: [u8; 32] = plaintext.try_into().unwrap();
        let keypair = SigningKey::from_bytes(&secret);

        Ok(Self { username, keypair })
    }
}