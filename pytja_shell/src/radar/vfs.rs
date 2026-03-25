use serde_json::Value;
use crate::network_client::PytjaClient;
use super::models::RadarPermission;

pub async fn handle_vfs_request(
    req: &Value,
    client: &PytjaClient,
    allowed_perms: &[RadarPermission],
    owner_id: &str,
) -> String {
    let method = req["method"].as_str().unwrap_or("");

    if method == "list_dir" {
        if !allowed_perms.contains(&RadarPermission::FsRead) {
            return r#"{"status": "error", "message": "403 Forbidden: Missing fs_read permission"}"#.to_string();
        }

        let target_path = req["params"]["path"].as_str().unwrap_or("/");

        match client.list_files(target_path).await {
            Ok(items) => {
                let json_items: Vec<String> = items.iter().map(|i| {
                    format!(r#"{{"name": "{}", "is_folder": {}, "size": {}}}"#, i.name, i.is_folder, i.size)
                }).collect();
                format!(r#"{{"status": "success", "data": {{"items": [{}]}}}}"#, json_items.join(", "))
            },
            Err(e) => {
                let safe_err = e.to_string().replace("\"", "\\\"");
                format!(r#"{{"status": "error", "message": "{}"}}"#, safe_err)
            }
        }
    } else if method == "write_file" {
        if !allowed_perms.contains(&RadarPermission::FsWrite) {
            return r#"{"status": "error", "message": "403 Forbidden: Missing fs_write permission"}"#.to_string();
        }

        let target_path = req["params"]["path"].as_str().unwrap_or("");
        let content_str = req["params"]["content"].as_str().unwrap_or("");
        let content_bytes = content_str.as_bytes().to_vec();

        match client.create_node(target_path, false, content_bytes, None, owner_id).await {
            Ok(_) => r#"{"status": "success", "data": {"message": "File written successfully"}}"#.to_string(),
            Err(e) => {
                let safe_err = e.to_string().replace("\"", "\\\"");
                format!(r#"{{"status": "error", "message": "{}"}}"#, safe_err)
            }
        }
    // --- NEU: ENTERPRISE APPEND LOGIC ---
    } else if method == "write_append" {
        if !allowed_perms.contains(&RadarPermission::FsWrite) {
            return r#"{"status": "error", "message": "403 Forbidden: Missing fs_write permission"}"#.to_string();
        }

        let target_path = req["params"]["path"].as_str().unwrap_or("");
        let content_str = req["params"]["content"].as_str().unwrap_or("");

        // 1. Lese die bestehende Datei (falls vorhanden)
        let mut final_content = Vec::new();
        if let Ok((existing_data, _)) = client.read_file(target_path, None).await {
            final_content.extend(existing_data);
        }

        // 2. Hänge die neuen Daten an
        final_content.extend(content_str.as_bytes());

        // 3. Speichere die Datei zurück in das VFS
        match client.create_node(target_path, false, final_content, None, owner_id).await {
            Ok(_) => r#"{"status": "success", "data": {"message": "File appended successfully"}}"#.to_string(),
            Err(e) => {
                let safe_err = e.to_string().replace("\"", "\\\"");
                format!(r#"{{"status": "error", "message": "{}"}}"#, safe_err)
            }
        }
    } else {
        r#"{"status": "error", "message": "Method not implemented in VFS module"}"#.to_string()
    }
}