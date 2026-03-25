mod abi;

/// System lifecycle and alarms
pub mod host {
    use crate::abi;

    pub fn ping() {
        abi::ping_watchdog();
    }

    pub fn alarm(message: &str) -> Result<(), String> {
        abi::send_ipc("host", "alarm", serde_json::json!({ "message": message }))?;
        Ok(())
    }
}

/// Native Desktop Window (Sidecar API)
pub mod window {
    use crate::abi;

    pub fn create(title: &str, html: &str, width: f64, height: f64) -> Result<(), String> {
        abi::send_ipc("window", "create", serde_json::json!({
            "title": title,
            "html": html,
            "width": width,
            "height": height
        }))?;
        Ok(())
    }

    pub fn emit(payload: serde_json::Value) -> Result<(), String> {
        abi::send_ipc("window", "emit", serde_json::json!({
            "payload": payload
        }))?;
        Ok(())
    }
}

/// Virtual Filesystem (E2EE)
pub mod vfs {
    use crate::abi;

    pub fn write(path: &str, content: &str) -> Result<(), String> {
        abi::send_ipc("vfs", "write", serde_json::json!({
            "path": path,
            "content": content
        }))?;
        Ok(())
    }

    pub fn read(path: &str) -> Result<String, String> {
        let res = abi::send_ipc("vfs", "read", serde_json::json!({ "path": path }))?;
        Ok(res["data"].as_str().unwrap_or("").to_string())
    }
}

/// Network requests through the host proxy
pub mod network {
    use crate::abi;

    pub fn get(url: &str) -> Result<String, String> {
        let res = abi::send_ipc("network", "fetch", serde_json::json!({
            "method": "GET",
            "url": url
        }))?;
        Ok(res["body"].as_str().unwrap_or("").to_string())
    }
}