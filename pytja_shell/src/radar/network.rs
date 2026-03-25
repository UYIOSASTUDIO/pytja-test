use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use std::collections::HashMap;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use super::models::RadarPermission;

pub type SocketMap = Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>;

pub async fn handle_network_request(
    req: &Value,
    allowed_perms: &[RadarPermission],
    plugin_tx: mpsc::Sender<String>,
    sockets: SocketMap,
) -> String {
    if !allowed_perms.contains(&RadarPermission::NetworkTcp) {
        return r#"{"status": "error", "message": "403 Forbidden: Missing network_tcp permission"}"#.to_string();
    }

    let method = req["method"].as_str().unwrap_or("");

    // --- HTTP/REST ---
    if method == "send" || method == "http" {
        let url = req["params"]["url"].as_str().unwrap_or("");
        let method_http = req["params"]["method"].as_str().unwrap_or("GET");
        let body_opt = req["params"]["body"].as_str();

        let client_http = reqwest::Client::new();
        let mut request_builder = match method_http {
            "POST" => client_http.post(url),
            "PUT" => client_http.put(url),
            _ => client_http.get(url),
        };

        if let Some(b) = body_opt {
            request_builder = request_builder.body(b.to_string());
        }

        match request_builder.send().await {
            Ok(resp) => {
                let status_code = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                let res_json = serde_json::json!({
                    "status": "success",
                    "data": { "status_code": status_code, "body": body_text }
                });
                res_json.to_string()
            },
            Err(e) => {
                let res_json = serde_json::json!({ "status": "error", "message": e.to_string() });
                res_json.to_string()
            }
        }
    }
    // --- WEBSOCKET CONNECT ---
    else if method == "ws_connect" {
        let url = req["params"]["url"].as_str().unwrap_or("");
        let socket_id = req["params"]["id"].as_str().unwrap_or("");

        match connect_async(url).await {
            Ok((ws_stream, _)) => {
                let (mut write, mut read) = ws_stream.split();

                let (ws_tx, mut ws_rx) = mpsc::channel::<String>(100);
                sockets.lock().await.insert(socket_id.to_string(), ws_tx);

                // Outbound (Plugin -> Internet)
                tokio::spawn(async move {
                    while let Some(msg) = ws_rx.recv().await {
                        if write.send(Message::Text(msg)).await.is_err() { break; }
                    }
                });

                // Inbound (Internet -> Plugin Inbox)
                let p_tx = plugin_tx.clone();
                let s_id = socket_id.to_string();
                tokio::spawn(async move {
                    while let Some(msg) = read.next().await {
                        if let Ok(Message::Text(text)) = msg {
                            let event = format!("WS_MSG:{}:{}", s_id, text);
                            let _ = p_tx.send(event).await;
                        }
                    }
                });

                r#"{"status": "success", "data": {"message": "WebSocket connected asynchronously"}}"#.to_string()
            }
            Err(e) => {
                let res_json = serde_json::json!({ "status": "error", "message": e.to_string() });
                res_json.to_string()
            }
        }
    }
    // --- WEBSOCKET SEND ---
    else if method == "ws_send" {
        let socket_id = req["params"]["id"].as_str().unwrap_or("");
        let data = req["params"]["data"].as_str().unwrap_or("");

        if let Some(sender) = sockets.lock().await.get(socket_id) {
            match sender.send(data.to_string()).await {
                Ok(_) => r#"{"status": "success"}"#.to_string(),
                Err(_) => r#"{"status": "error", "message": "Socket connection lost"}"#.to_string()
            }
        } else {
            r#"{"status": "error", "message": "Socket ID not found or disconnected"}"#.to_string()
        }
    }
    else {
        r#"{"status": "error", "message": "Method not implemented in network module"}"#.to_string()
    }
}