use ed25519_dalek::{SigningKey, VerifyingKey, Signer, Verifier, Signature};
use rand::rngs::OsRng;
use rand::RngCore;
use aes_gcm::{aead::{Aead, AeadCore, KeyInit}, Aes256Gcm, Key, Nonce};
use pbkdf2::pbkdf2;
use hmac::Hmac;
use sha2::{Sha256, Digest};
use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

const SALT: &[u8] = b"pytja_protocol_salt_v2";
const ITERATIONS: u32 = 600_000;

pub struct CryptoService;

impl CryptoService {
    pub fn generate_keypair() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    pub fn pubkey_to_hex(key: &VerifyingKey) -> String {
        hex::encode(key.to_bytes())
    }

    pub fn sign_message(priv_key: &SigningKey, message: &[u8]) -> String {
        let signature = priv_key.sign(message);
        BASE64.encode(signature.to_bytes())
    }
    
    pub fn verify_signature(pub_key_bytes: &[u8], message: &[u8], signature_b64: &str) -> Result<bool> {
        let pub_key = VerifyingKey::from_bytes(pub_key_bytes.try_into().map_err(|_| anyhow!("Invalid PubKey length"))?)?;

        let sig_bytes = BASE64.decode(signature_b64)?;
        let signature = Signature::from_bytes(&sig_bytes.try_into().map_err(|_| anyhow!("Invalid Sig length"))?);

        Ok(pub_key.verify(message, &signature).is_ok())
    }

    pub fn encrypt_private_key_local(priv_key: &SigningKey, password: &str) -> Result<String> {
        let key_bytes = priv_key.to_bytes();

        let mut key = [0u8; 32];
        pbkdf2::<Hmac<Sha256>>(password.as_bytes(), SALT, ITERATIONS, &mut key).unwrap();
        let cipher = Aes256Gcm::new(&key.into());
        
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher.encrypt(&nonce, key_bytes.as_ref()).map_err(|_| anyhow!("Encryption failed"))?;

        let mut combined = nonce.to_vec();
        combined.extend(ciphertext);
        Ok(BASE64.encode(combined))
    }

    pub fn decrypt_private_key_local(encrypted_b64: &str, password: &str) -> Result<SigningKey> {
        let data = BASE64.decode(encrypted_b64)?;
        if data.len() < 12 { return Err(anyhow!("Invalid key file format")); }

        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let mut key = [0u8; 32];
        pbkdf2::<Hmac<Sha256>>(password.as_bytes(), SALT, ITERATIONS, &mut key).unwrap();
        let cipher = Aes256Gcm::new(&key.into());

        let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|_| anyhow!("Wrong Password or Corrupted Key"))?;

        let signing_key = SigningKey::from_bytes(&plaintext.try_into().map_err(|_| anyhow!("Invalid Key Length"))?);
        Ok(signing_key)
    }

    pub fn generate_random_challenge() -> String {
        use rand::RngCore;
        let mut data = [0u8; 32];
        OsRng.fill_bytes(&mut data);
        hex::encode(data)
    }

    pub fn derive_e2e_key(secret_seed: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"PYTJA_ENTERPRISE_E2EE_V1");
        hasher.update(secret_seed);
        let result = hasher.finalize();

        let mut key = [0u8; 32];
        key.copy_from_slice(&result);
        key
    }

    pub fn encrypt_e2e(key: &[u8; 32], plaintext: &[u8]) -> anyhow::Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));

        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let mut ciphertext = cipher.encrypt(nonce, plaintext)
            .map_err(|e| anyhow::anyhow!("AES-256-GCM Encryption failed: {:?}", e))?;

        let mut payload = nonce_bytes.to_vec();
        payload.append(&mut ciphertext);
        Ok(payload)
    }

    pub fn decrypt_e2e(key: &[u8; 32], payload: &[u8]) -> anyhow::Result<Vec<u8>> {
        if payload.len() < 12 {
            anyhow::bail!("E2EE Payload too short (Missing Nonce). File might be corrupt.");
        }

        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce = Nonce::from_slice(&payload[..12]);
        let ciphertext = &payload[12..];

        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("E2EE Decryption failed (Integrity breach or wrong key): {:?}", e))?;

        Ok(plaintext)
    }

    // =========================================================================
    // --- CPU-OFFLOADING (TOKIO SPAWN BLOCKING) ---
    // =========================================================================

    pub async fn encrypt_private_key_local_async(priv_key: SigningKey, password: String) -> anyhow::Result<String> {
        // Wir verschieben die Berechnung auf einen dedizierten CPU-Thread
        tokio::task::spawn_blocking(move || {
            Self::encrypt_private_key_local(&priv_key, &password)
        })
            .await
            .map_err(|e| anyhow::anyhow!("CPU Task Panicked: {}", e))?
    }

    pub async fn decrypt_private_key_local_async(encrypted_b64: String, password: String) -> anyhow::Result<SigningKey> {
        tokio::task::spawn_blocking(move || {
            Self::decrypt_private_key_local(&encrypted_b64, &password)
        })
            .await
            .map_err(|e| anyhow::anyhow!("CPU Task Panicked: {}", e))?
    }

    pub async fn encrypt_e2e_async(key: [u8; 32], plaintext: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        tokio::task::spawn_blocking(move || {
            Self::encrypt_e2e(&key, &plaintext)
        })
            .await
            .map_err(|e| anyhow::anyhow!("CPU Task Panicked: {}", e))?
    }

    pub async fn decrypt_e2e_async(key: [u8; 32], payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        tokio::task::spawn_blocking(move || {
            Self::decrypt_e2e(&key, &payload)
        })
            .await
            .map_err(|e| anyhow::anyhow!("CPU Task Panicked: {}", e))?
    }
}