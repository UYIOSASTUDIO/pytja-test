mod client;
mod menus;
mod utils;

use dialoguer::{theme::ColorfulTheme, Select, Input};
use console::Term;
use client::AdminClient;

pub async fn start_admin() -> anyhow::Result<()> {
    // establish connection
    let mut client = AdminClient::connect("https://localhost:50051".to_string()).await
        .expect("Could not connect to Pytja Server. Is it running or is the cert missing?");

    // Login-Screen & Authentication
    Term::stdout().clear_screen()?;
    println!("╔══════════════════════════════════════════╗");
    println!("║      PYTJA COMMAND CENTER (PCC) v1.0     ║");
    println!("║      Enterprise Admin Interface          ║");
    println!("╚══════════════════════════════════════════╝");
    println!();

    let identity_path: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Path to Admin Identity File (.pytja)")
        .default("./identity/admin.pytja".into())
        .interact_text()?;

    println!("Authenticating...");
    
    match client.login_with_identity(&identity_path).await {
        Ok(_) => {
            println!("Login successful as '{}'. Welcome, Admin.", client.username);
            std::thread::sleep(std::time::Duration::from_millis(1000));
        },
        Err(e) => {
            println!("Authentication failed: {}", e);
            return Ok(());
        }
    }

    // Main Loop
    loop {
        Term::stdout().clear_screen()?;

        println!("╔══════════════════════════════════════════╗");
        println!("║      PYTJA COMMAND CENTER (PCC) v1.0     ║");
        println!("║      User: {:<29} ║", client.username);
        println!("╚══════════════════════════════════════════╝");
        println!();

        let items = vec![
            "1. User & Identity Management",
            "2. Database & Mounts",
            "3. Role Based Access Control (RBAC)",
            "4. System Health & Logs",
            "0. Exit"
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select Module")
            .default(0)
            .items(&items)
            .interact()?;

        match selection {
            0 => menus::users::show(&mut client).await?,
            1 => menus::databases::show(&mut client).await?,
            2 => menus::rbac::show(&mut client).await?,
            3 => menus::system::show(&mut client).await?,
            4 => break,
            _ => {}
        }
    }

    println!("Session terminated. Goodbye.");
    Ok(())
}
