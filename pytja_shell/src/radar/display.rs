use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use super::models::RadarPermission;

pub async fn handle_display_request(
    req: &Value,
    allowed_perms: &[RadarPermission],
    plugin_name: &str,
    registry: Arc<Mutex<HashMap<String, String>>>,
) -> String {
    if !allowed_perms.contains(&RadarPermission::DisplayUi) {
        return r#"{"status": "error", "message": "403 Forbidden: Missing display_ui permission"}"#.to_string();
    }

    let method = req["method"].as_str().unwrap_or("");

    if method == "render" {
        let html = req["params"]["html"].as_str().unwrap_or("<h1>Empty Render</h1>");
        registry.lock().await.insert(plugin_name.to_string(), html.to_string());
        r#"{"status": "success", "data": {"message": "UI rendered successfully"}}"#.to_string()
    } else {
        r#"{"status": "error", "message": "Method not implemented in display module"}"#.to_string()
    }
}

pub async fn run_ui_server(registry: Arc<Mutex<HashMap<String, String>>>) {
    if let Ok(listener) = TcpListener::bind("127.0.0.1:8080").await {
        println!("[RADAR] Headless UI Server listening on http://127.0.0.1:8080/ui/");
        while let Ok((mut socket, _)) = listener.accept().await {
            let reg = registry.clone();
            tokio::spawn(async move {
                let mut buf = [0; 1024];
                if let Ok(n) = socket.read(&mut buf).await {
                    let req_str = String::from_utf8_lossy(&buf[..n]);
                    if req_str.starts_with("GET /ui/") {
                        let parts: Vec<&str> = req_str.split_whitespace().collect();
                        if parts.len() > 1 {
                            let plugin = parts[1].trim_start_matches("/ui/");
                            let html = {
                                let lock = reg.lock().await;
                                lock.get(plugin).cloned().unwrap_or_else(|| "<h1>404 - Plugin UI not found</h1>".to_string())
                            };
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                                html.len(),
                                html
                            );
                            let _ = socket.write_all(response.as_bytes()).await;
                        }
                    }
                }
            });
        }
    }
}