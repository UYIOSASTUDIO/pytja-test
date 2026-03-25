use serde_json::json;
use std::thread;
use std::time::Duration;

// -----------------------------------------------------------------------------
// FFI BINDINGS (RADAR ABI)
// -----------------------------------------------------------------------------
#[link(wasm_import_module = "radar_abi")]
extern "C" {
    fn host_log_status(code: i32);
    fn host_heartbeat();
    fn host_ipc_request(req_ptr: i32, req_len: i32, res_ptr: i32, res_cap: i32) -> i32;
}

// -----------------------------------------------------------------------------
// SAFE RUST WRAPPERS
// -----------------------------------------------------------------------------
struct RadarCore;

impl RadarCore {
    pub fn heartbeat() {
        unsafe { host_heartbeat() };
    }

    pub fn log_status(code: i32) {
        unsafe { host_log_status(code) };
    }

    pub fn send_ipc(payload: &serde_json::Value) -> Result<serde_json::Value, String> {
        let req_string = payload.to_string();
        let req_bytes = req_string.as_bytes();

        let mut res_buffer = vec![0u8; 65536];

        let result_len = unsafe {
            host_ipc_request(
                req_bytes.as_ptr() as i32,
                req_bytes.len() as i32,
                res_buffer.as_mut_ptr() as i32,
                res_buffer.capacity() as i32,
            )
        };

        if result_len < 0 {
            return Err(format!("IPC Request failed or buffer too small. Code: {}", result_len));
        }

        let res_string = String::from_utf8_lossy(&res_buffer[..result_len as usize]);
        serde_json::from_str(&res_string).map_err(|e| e.to_string())
    }

    pub fn alarm(message: &str) {
        let _ = Self::send_ipc(&json!({
            "module": "host",
            "method": "alarm",
            "params": { "message": message }
        }));
    }
}

// -----------------------------------------------------------------------------
// BUSINESS LOGIC
// -----------------------------------------------------------------------------
fn main() {
    RadarCore::alarm("Omni-Agent initialized. Booting subsystems...");

    let html_dashboard = r#"
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                /* Grundlayout: Weißer Hintergrund, volle Höhe, Flexbox für Zentrierung */
                body {
                    margin: 0;
                    padding: 0;
                    background-color: #ffffff;
                    color: #1a1a1a;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;

                    /* ENTERPRISE UX FIX: Verhindert das Markieren von Text und den I-Beam Cursor */
                    -webkit-user-select: none;
                    user-select: none;
                    cursor: default;
                }

                /* Die unsichtbare Drag-Bar für die macOS-Ampel-Buttons */
                .titlebar {
                    height: 20px;
                    width: 100%;
                    background: transparent;

                    /* ENTERPRISE UX FIX: Offene Hand beim Hovern */
                    cursor: grab;
                }

                /* Geschlossene Hand, wenn die Maustaste gedrückt wird */
                .titlebar:active {
                    cursor: grabbing;
                }

                /* Zentrierter Inhaltsbereich */
                .content {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                }

                h1 {
                    font-size: 3.5rem;
                    margin-bottom: 30px;
                    font-weight: 700;
                    letter-spacing: -1.5px;
                }

                /* Modernes Button-Design */
                .btn {
                    background-color: #000000;
                    color: #ffffff;
                    padding: 14px 28px;
                    border: none;
                    border-radius: 8px;
                    font-size: 1.1rem;
                    font-weight: 600;
                    cursor: pointer;
                    text-decoration: none;
                    transition: transform 0.2s, background-color 0.2s;
                }

                .btn:hover {
                    background-color: #333333;
                    transform: scale(1.02);
                }
            </style>
        </head>
        <body>
            <div class="titlebar"></div>

            <div class="content">
                <h1>Demo plugin</h1>
                <a href="https://pytja.com" target="_blank" class="btn">Visit Website</a>
            </div>

            <script>
                // Fängt den Mausklick auf die unsichtbare Titelleiste ab
                document.querySelector('.titlebar').addEventListener('mousedown', (e) => {
                    // Prüfen, ob es ein Linksklick ist (button 0)
                    if (e.button === 0) {
                        // Sendet den Befehl an den Rust-Host
                        window.ipc.postMessage('DRAG_WINDOW');
                    }
                });
            </script>
        </body>
        </html>
    "#;

    let ui_res = RadarCore::send_ipc(&json!({
        "module": "window",
        "method": "create",
        "params": {
            "title": "Pytja Omni-Agent",
            "html": html_dashboard,
            "width": 600,
            "height": 400
        }
    }));

    if ui_res.is_err() {
        RadarCore::log_status(101);
        return;
    }

    RadarCore::alarm("UI deployed. Entering main telemetry loop.");

    let mut iteration = 0;

    loop {
        RadarCore::heartbeat();

        let net_res = RadarCore::send_ipc(&json!({
            "module": "network",
            "method": "http",
            "params": {
                "method": "GET",
                "url": "https://api.coindesk.com/v1/bpi/currentprice/USD.json"
            }
        }));

        let mut display_text = format!("Iteration: {}", iteration);

        if let Ok(response) = net_res {
            if response["status"] == "success" {
                let body_str = response["data"]["body"].as_str().unwrap_or("{}");
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(body_str) {
                    if let Some(price) = parsed["bpi"]["USD"]["rate"].as_str() {
                        display_text = format!("BTC Price: {} USD (Update {})", price, iteration);
                    }
                }
            }
        }

        let _ = RadarCore::send_ipc(&json!({
            "module": "window",
            "method": "emit",
            "params": {
                "payload": display_text
            }
        }));

        let log_entry = format!("[LOG] {}\n", display_text);
        let _ = RadarCore::send_ipc(&json!({
            "module": "vfs",
            "method": "write_append", 
            "params": {
                "path": "/omni_telemetry.log",
                "content": log_entry
            }
        }));

        thread::sleep(Duration::from_secs(10));
        iteration += 1;
    }
}