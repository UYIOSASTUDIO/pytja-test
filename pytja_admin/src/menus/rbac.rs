use crate::client::AdminClient;
use dialoguer::{theme::ColorfulTheme, Select, Input, MultiSelect, Confirm};
use comfy_table::{Table, presets::UTF8_FULL, Cell, Color, Attribute};
use console::Term;

pub async fn show(client: &mut AdminClient) -> anyhow::Result<()> {
    loop {
        Term::stdout().clear_screen()?;
        println!("MODULE: ROLE BASED ACCESS CONTROL (RBAC)");
        println!("--------------------------------------");

        let items = vec![
            "1. List Roles & Permissions",
            "2. Create New Role",
            "3. Add Permission to Role",
            "4. Back to Main Menu"
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Action")
            .items(&items)
            .default(0)
            .interact()?;

        match selection {
            0 => list_roles(client).await?,
            1 => create_role(client).await?,
            2 => add_permission_wizard(client).await?,
            3 => break,
            _ => {}
        }

        println!("\nPress Enter to continue...");
        let _ = std::io::stdin().read_line(&mut String::new());
    }
    Ok(())
}

async fn list_roles(client: &mut AdminClient) -> anyhow::Result<()> {
    let roles = client.list_roles().await?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Role Name", "Permissions (Scopes)"]);

    for role in roles {
        let perms_display = if role.permissions.is_empty() {
            "No permissions".to_string()
        } else {
            role.permissions.join(", ")
        };

        let name_cell = if role.name == "admin" {
            Cell::new(&role.name).fg(Color::Red).add_attribute(Attribute::Bold)
        } else {
            Cell::new(&role.name).fg(Color::Cyan).add_attribute(Attribute::Bold)
        };

        table.add_row(vec![
            name_cell,
            Cell::new(perms_display),
        ]);
    }

    println!("{}", table);
    Ok(())
}

async fn create_role(client: &mut AdminClient) -> anyhow::Result<()> {
    println!("\n--- CREATE NEW ROLE ---");
    let name: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Role Name (e.g. 'auditor')")
        .interact_text()?;

    match client.create_role(name.clone()).await {
        Ok(_) => println!("Role '{}' created. You can now add permissions to it.", name),
        Err(e) => println!("Error: {}", e),
    }
    Ok(())
}

async fn add_permission_wizard(client: &mut AdminClient) -> anyhow::Result<()> {
    // Choose role
    let roles = client.list_roles().await?;
    if roles.is_empty() {
        println!("No roles available.");
        return Ok(());
    }

    let role_names: Vec<String> = roles.iter().map(|r| r.name.clone()).collect();
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Target Role")
        .items(&role_names)
        .interact()?;

    let target_role = &role_names[selection];

    // Choose permission
    let standard_perms = vec![
        "core:fs:read",   // read
        "core:fs:write",  // write (Upload, Download)
        "core:exec",      // execute script
        "core:admin:read", // sea admin infos
        "core:admin:sys",  // manage mounts
        "core:admin:users",// manage users
        "core:admin:roles" // manage roles
    ];

    println!("\nAvailable Standard Permissions:");
    for (i, p) in standard_perms.iter().enumerate() {
        println!("  {}. {}", i + 1, p);
    }
    println!("  C. Custom Input\n");

    let _perm_input: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter Permission (Type exact string or custom)")
        .interact_text()?;

    // REDO UI for Permission Selection:
    let use_list = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Select from standard list?")
        .default(true)
        .interact()?;

    let permissions_to_add = if use_list {
        let chosen_indices = MultiSelect::with_theme(&ColorfulTheme::default())
            .with_prompt("Select Permissions (Space to select, Enter to confirm)")
            .items(&standard_perms)
            .interact()?;

        chosen_indices.into_iter().map(|i| standard_perms[i].to_string()).collect::<Vec<String>>()
    } else {
        let custom: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Enter Custom Permission String")
            .interact_text()?;
        vec![custom]
    };

    println!("Adding {} permissions to '{}'...", permissions_to_add.len(), target_role);

    for p in permissions_to_add {
        match client.add_permission(target_role.clone(), p.clone()).await {
            Ok(_) => println!("  [+] Added '{}'", p),
            Err(e) => println!("  [-] Failed to add '{}': {}", p, e),
        }
    }

    println!("Done.");
    Ok(())
}