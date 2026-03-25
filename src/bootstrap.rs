use anyhow::{Result, anyhow};
use std::net::TcpStream;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use std::fs;

pub async fn run_enterprise_wizard() -> Result<()> {
    println!("--- PYTJA ENTERPRISE INITIALIZATION ---");

    print!("Checking session store (Redis) availability... ");
    if TcpStream::connect("127.0.0.1:6379").is_err() {
        println!("FAILED\n");
        println!("CRITICAL: Redis is required but not running.");
        println!("Please install and start Redis to continue.");
        println!("macOS: brew install redis && brew services start redis");
        println!("Linux: sudo apt install redis-server && sudo systemctl start redis");
        return Err(anyhow!("System requirement not met: Redis missing"));
    }
    println!("OK");

    let config_path = Path::new("config/default.toml");
    let identity_file = if !config_path.exists() {
        println!("First run detected. Provisioning local infrastructure...");
        provision_infrastructure().await?
    } else {
        println!("Existing configuration found.");
        find_local_identity()
    };

    print!("Starting Pytja Enterprise Backend... ");
    let current_exe = std::env::current_exe()?;
    let mut server_process = Command::new(&current_exe)
        .arg("server")
        .spawn()
        .map_err(|e| anyhow!("Failed to spawn server daemon: {}", e))?;

    wait_for_port(50051, 30).await?;
    println!("ONLINE");

    println!("Handing over to interactive shell...\n");
    let shell_result = pytja_shell::start_shell(identity_file).await;

    println!("Shutting down backend server...");
    let _ = server_process.kill();
    let _ = server_process.wait();

    shell_result
}

async fn provision_infrastructure() -> Result<Option<String>> {
    fs::create_dir_all("config")?;
    fs::create_dir_all("certs")?;
    fs::create_dir_all("data/storage")?;
    fs::create_dir_all("logs")?;

    println!("Generating zero-trust TLS certificates...");

    // --- ENTERPRISE FIX: LibreSSL (macOS) Compatible SAN Injection ---
    let cert_conf = r#"
[req]
default_bits = 4096
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_ext

[dn]
C = DE
CN = localhost

[v3_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
"#;
    fs::write("cert.conf", cert_conf.trim())?;

    let openssl_status = Command::new("openssl")
        .args(&[
            "req", "-x509", "-nodes", "-days", "365", "-newkey", "rsa:4096",
            "-keyout", "certs/server.key",
            "-out", "certs/server.crt",
            "-config", "cert.conf"
        ])
        .output()?;

    let _ = fs::remove_file("cert.conf");

    if !openssl_status.status.success() {
        let stderr = String::from_utf8_lossy(&openssl_status.stderr);
        return Err(anyhow!("Failed to generate certificates: {}", stderr));
    }

    let toml_content = r#"
[server]
host = "127.0.0.1"
port = 50051

[database]
primary_url = "sqlite://data/pytja.db"

[storage]
storage_type = "fs"
local_path = "data/storage"

[tls]
enabled = true
cert_path = "certs/server.crt"
key_path = "certs/server.key"

[paths]
logs_dir = "logs"
mounts_file = "data/mounts.json"
"#;
    fs::write("config/default.toml", toml_content.trim())?;

    println!("Provisioning cryptographic session keys...");
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let jwt_secret = format!("pytja_enterprise_auto_key_{}", timestamp);
    let env_content = format!("PYTJA_JWT_SECRET={}\n", jwt_secret);
    fs::write(".env", env_content)?;

    println!("\nInfrastructure ready. Proceeding with Administrator setup:");
    pytja_registrar::start_registrar(Some(".".into())).await?;

    Ok(find_local_identity())
}

async fn wait_for_port(port: u16, max_retries: usize) -> Result<()> {
    let addr = format!("127.0.0.1:{}", port);
    for _ in 0..max_retries {
        if TcpStream::connect(&addr).is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(anyhow!("Server did not start in time on port {}", port))
}

fn find_local_identity() -> Option<String> {
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "pytja" {
                    return Some(entry.path().to_string_lossy().into_owned());
                }
            }
        }
    }
    None
}