use crate::client::AdminClient;
use dialoguer::{theme::ColorfulTheme, Select, Input, Confirm};
use comfy_table::{Table, presets::UTF8_FULL, Cell, Color, Attribute};
use console::Term;

pub async fn show(client: &mut AdminClient) -> anyhow::Result<()> {
    loop {
        Term::stdout().clear_screen()?;
        println!("MODULE: DATABASE & MOUNT CONTROL");
        println!("--------------------------------");

        let items = vec![
            "1. List Mounted Databases (Status)",
            "2. Mount New Database (Hot-Plug)",
            "3. Unmount Database",
            "4. Back to Main Menu"
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Action")
            .items(&items)
            .default(0)
            .interact()?;

        match selection {
            0 => list_mounts(client).await?,
            1 => mount_wizard(client).await?,
            2 => unmount_wizard(client).await?,
            3 => break,
            _ => {}
        }

        println!("\nPress Enter to continue...");
        let _ = std::io::stdin().read_line(&mut String::new());
    }
    Ok(())
}

async fn list_mounts(client: &mut AdminClient) -> anyhow::Result<()> {
    println!("Fetching mount status...");
    let mounts = client.get_mounts().await?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Name", "Type", "Connection Info", "Status"]);

    for m in mounts {
        let status_cell = if m.is_connected {
            Cell::new("ONLINE").fg(Color::Green).add_attribute(Attribute::Bold)
        } else {
            Cell::new("OFFLINE").fg(Color::Red).add_attribute(Attribute::Bold)
        };

        let type_display = if m.name == "primary" {
            format!("{} (SYSTEM)", m.r#type)
        } else {
            m.r#type.clone()
        };

        table.add_row(vec![
            Cell::new(&m.name).add_attribute(Attribute::Bold),
            Cell::new(type_display),
            Cell::new(&m.connection),
            status_cell,
        ]);
    }

    println!("{}", table);
    Ok(())
}

async fn mount_wizard(client: &mut AdminClient) -> anyhow::Result<()> {
    println!("\n--- MOUNT NEW DATABASE ---");
    println!("This will attach a new database to the virtual file system in real-time.\n");

    let name: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Mount Name (e.g. 'archive', 'projects')")
        .validate_with(|input: &String| -> Result<(), &str> {
            if input.chars().all(|c| c.is_alphanumeric() || c == '_') { Ok(()) }
            else { Err("Name must be alphanumeric (a-z, 0-9, _)") }
        })
        .interact_text()?;

    // Check gegen "primary" verhindern
    if name == "primary" {
        println!("Error: 'primary' is reserved for the system database.");
        return Ok(());
    }

    let db_types = vec!["SQLite (Local File)", "PostgreSQL (Remote Server)"];
    let type_selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Database Type")
        .items(&db_types)
        .default(0)
        .interact()?;

    let (db_type_str, conn_string) = match type_selection {
        0 => {
            // SQLite
            let path: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Absolute Path to .db file")
                .default("./data_storage/extra.db".into())
                .interact_text()?;
            // Automatisch sqlite:// Prefix hinzufügen, falls vergessen
            let conn = if path.starts_with("sqlite://") { path } else { format!("sqlite://{}", path) };
            ("sqlite", conn)
        },
        1 => {
            // Postgres
            let url: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Connection URL (postgres://user:pass@host:port/db)")
                .interact_text()?;
            ("postgres", url)
        },
        _ => return Ok(()),
    };

    println!("Attempting to mount '{}' via {}...", name, db_type_str);

    match client.add_mount(name.clone(), conn_string, db_type_str.to_string()).await {
        Ok(_) => println!("Successfully mounted '{}'. It is now accessible via /{}", name, name),
        Err(e) => println!("Mount failed: {}", e),
    }

    Ok(())
}

async fn unmount_wizard(client: &mut AdminClient) -> anyhow::Result<()> {
    let mounts = client.get_mounts().await?;

    // Filter primary raus, das darf man nicht unmounten
    let mount_names: Vec<String> = mounts.iter()
        .filter(|m| m.name != "primary")
        .map(|m| m.name.clone())
        .collect();

    if mount_names.is_empty() {
        println!("No additional mounts found to remove.");
        return Ok(());
    }

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Mount to Detach")
        .items(&mount_names)
        .interact()?;

    let target = &mount_names[selection];

    if Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt(format!("Are you sure you want to unmount '{}'? This will close active connections.", target))
        .interact()?
    {
        match client.remove_mount(target.clone()).await {
            Ok(_) => println!("'{}' has been unmounted.", target),
            Err(e) => println!("Error unmounting: {}", e), // Falls Server "Unimplemented" wirft (was aktuell der Fall ist, siehe unten)
        }
    } else {
        println!("Operation cancelled.");
    }

    Ok(())
}