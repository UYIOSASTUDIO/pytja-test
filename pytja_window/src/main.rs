use wry::application::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    dpi::LogicalSize,
};
use wry::webview::WebViewBuilder;
use serde::Deserialize;
use std::env;
use std::io::{self, BufRead};
use std::thread;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

#[cfg(target_os = "macos")]
use wry::application::platform::macos::WindowBuilderExtMacOS;

#[derive(Deserialize)]
struct WindowConfig {
    plugin_id: String,
    title: String,
    html_b64: String,
    width: f64,
    height: f64,
}

#[derive(Debug)]
enum UserEvent {
    IncomingData(String),
    Shutdown,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("Missing window configuration payload");
    }

    let config: WindowConfig = serde_json::from_str(&args[1])
        .map_err(|e| anyhow::anyhow!("Invalid payload: {}", e))?;

    let html_content = String::from_utf8(BASE64.decode(&config.html_b64)?)?;
    let window_title = format!("{} [Agent: {}]", config.title, config.plugin_id.to_uppercase());

    let event_loop = EventLoop::<UserEvent>::with_user_event();
    let proxy = event_loop.create_proxy();

    // Asynchroner Thread zum Lesen von stdin (IPC vom Pytja-Host)
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(content) = line {
                let _ = proxy.send_event(UserEvent::IncomingData(content));
            }
        }
        let _ = proxy.send_event(UserEvent::Shutdown);
    });

    // 1. Window Builder initialisieren (noch NICHT builden)
    let mut window_builder = WindowBuilder::new()
        .with_title(window_title)
        .with_inner_size(LogicalSize::new(config.width, config.height));

    // --- ENTERPRISE FRAMELESS DESIGN (macOS) ---
    // Hier sagen wir macOS, dass die Titelleiste verschwinden soll
    // und der Web-Inhalt bis ganz nach oben (unter die Ampel-Buttons) rutschen darf.
    #[cfg(target_os = "macos")]
    {
        window_builder = window_builder
            .with_titlebar_transparent(true)
            .with_fullsize_content_view(true)
            .with_title_hidden(true);
    }

    // 2. Jetzt das modifizierte Fenster bauen
    let window = window_builder.build(&event_loop)?;

    let plugin_id_clone = config.plugin_id.clone();

    let webview = WebViewBuilder::new(window)?
        .with_html(&html_content)?
        .with_ipc_handler(move |window, string_payload| {
            // --- ENTERPRISE FIX: Native OS Window Dragging ---
            if string_payload == "DRAG_WINDOW" {
                let _ = window.drag_window();
                return;
            }

            let event_json = serde_json::json!({
                "source_plugin": plugin_id_clone,
                "event_data": string_payload
            });
            println!("PYTJA_IPC_EVENT:{}", event_json);
        })
        .build()?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::UserEvent(UserEvent::IncomingData(js_payload)) => {
                // Daten vom Pytja-Host in das Frontend injizieren
                let script = format!("window.dispatchEvent(new CustomEvent('pytja_host_event', {{ detail: {} }}));", js_payload);
                let _ = webview.evaluate_script(&script);
            }
            Event::UserEvent(UserEvent::Shutdown) => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
                println!("PYTJA_IPC_EVENT:{{\"source_plugin\": \"{}\", \"event_data\": \"WINDOW_CLOSED\"}}", config.plugin_id);
            }
            _ => {}
        }
    });
}