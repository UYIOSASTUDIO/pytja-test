use crate::client::AdminClient;
use dialoguer::{theme::ColorfulTheme, Select, Input};
use comfy_table::{Table, presets::UTF8_FULL, Cell, Color, Attribute};
use console::Term;
use std::time::Duration;
use crossterm::event::{self, Event, KeyCode};

pub async fn show(client: &mut AdminClient) -> anyhow::Result<()> {
    loop {
        Term::stdout().clear_screen()?;
        println!("MODULE: SYSTEM INTELLIGENCE & HEALTH");
        println!("------------------------------------");

        let items = vec![
            "1. Live Dashboard (Auto-Refresh)",
            "2. Audit Trail (Security Log)",
            "3. Live Server Log Stream (Tail)",
            "4. Back to Main Menu"
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Action")
            .items(&items)
            .default(0)
            .interact()?;

        match selection {
            0 => live_dashboard(client).await?,
            1 => show_audit_log(client).await?,
            2 => stream_logs(client).await?,
            3 => break,
            _ => {}
        }
    }
    Ok(())
}

async fn live_dashboard(client: &mut AdminClient) -> anyhow::Result<()> {
    println!("Starting Dashboard... (Press 'q' or 'ESC' to return)");

    loop {
        let stats = client.get_system_stats().await?;
        Term::stdout().clear_screen()?;

        println!("PYTJA LIVE MONITOR");
        println!("==================");

        println!("CPU Usage:      {:.1}%", stats.cpu_usage_percent);
        println!("RAM Usage:      {} MB", stats.memory_usage_bytes / 1024 / 1024);
        println!("Uptime:         {}", stats.uptime);
        println!("------------------");
        println!("Active Sessions: {}", stats.active_sessions);
        println!("Redis Status:    {}", if stats.redis_connected { "[OK] Connected" } else { "[FAIL] Error" });
        println!("\nPress 'q' or 'ESC' to return to menu.");

        // Nicht-blockierendes Polling: Wartet bis zu 2 Sekunden auf Eingaben
        if event::poll(Duration::from_secs(2))? {
            if let Event::Key(key_event) = event::read()? {
                if key_event.code == KeyCode::Char('q') || key_event.code == KeyCode::Esc {
                    break;
                }
            }
        }
    }
    Ok(())
}

async fn show_audit_log(client: &mut AdminClient) -> anyhow::Result<()> {
    let limit: u32 = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Number of entries to fetch")
        .default(50)
        .interact()?;

    let filter_input: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Filter by user (leave empty for all)")
        .allow_empty(true)
        .interact_text()?;

    let filter = if filter_input.trim().is_empty() {
        None
    } else {
        Some(filter_input.trim().to_string())
    };

    let logs = client.get_audit_logs(limit, filter).await?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Time", "User", "Action", "Target"]);

    for log in logs {
        let action_color = match log.action.as_str() {
            "DELETE" | "BAN" | "KICK" => Color::Red,
            "UPLOAD" | "CREATE" => Color::Green,
            _ => Color::White,
        };

        table.add_row(vec![
            Cell::new(&log.timestamp),
            Cell::new(&log.user).add_attribute(Attribute::Bold),
            Cell::new(&log.action).fg(action_color),
            Cell::new(&log.target),
        ]);
    }
    println!("{}", table);
    println!("\nPress Enter to return...");
    let _ = std::io::stdin().read_line(&mut String::new());
    Ok(())
}

async fn stream_logs(client: &mut AdminClient) -> anyhow::Result<()> {
    println!("Connecting to Log Stream... (Ctrl+C to exit)");
    let mut stream = client.stream_logs().await?;

    while let Some(log_res) = stream.message().await? {
        println!("[{}] {} | {}", log_res.timestamp, log_res.level, log_res.message);
    }
    Ok(())
}