use crate::client::AdminClient;
use dialoguer::{theme::ColorfulTheme, Select, Input, Confirm};
use comfy_table::{Table, presets::UTF8_FULL, Cell, Color as TColor};
use console::Term;
use colored::*;

pub async fn show(client: &mut AdminClient) -> anyhow::Result<()> {
    loop {
        Term::stdout().clear_screen()?;
        println!("╔══════════════════════════════════════════╗");
        println!("║      USER & IDENTITY MANAGEMENT          ║");
        println!("╚══════════════════════════════════════════╝");
        println!("💡 Tip: Press 'Esc' or 'q' in any menu to cancel instantly.\n");

        let items = vec![
            "1. List All Users (Table View)",
            "2. Edit User (Change Role, Quota, Ban)",
            "3. Generate Invite Code (Onboarding)",
            "4. List Active Invite Codes",
            "5. Revoke Invite Code",
            "0. Back to Main Menu"
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select Action")
            .default(0)
            .items(&items)
            .interact_opt()?;

        match selection {
            Some(0) => list_users_table(client).await?,
            Some(1) => {
                if !edit_user(client).await? {
                    return Err(anyhow::anyhow!("Session invalidated due to own role change. Please restart the admin tool."));
                }
            },
            Some(2) => generate_invite(client).await?,
            Some(3) => list_invites(client).await?,
            Some(4) => revoke_invite(client).await?,
            Some(5) | None => break,
            _ => {}
        }
    }
    Ok(())
}

async fn list_users_table(client: &mut AdminClient) -> anyhow::Result<()> {
    println!("\nFetching users from server...");
    let users = client.list_users().await?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Username", "Role", "Active", "Quota Used", "Quota Limit", "Created At"]);

    for u in users {
        let active_cell = if u.is_active { Cell::new("Yes").fg(TColor::Green) } else { Cell::new("Banned").fg(TColor::Red) };
        let limit_str = if u.quota_limit == 0 { "Default".to_string() } else { format_bytes(u.quota_limit) };

        table.add_row(vec![
            Cell::new(&u.username).add_attribute(comfy_table::Attribute::Bold),
            Cell::new(&u.role),
            active_cell,
            Cell::new(format_bytes(u.quota_used)),
            Cell::new(limit_str),
            Cell::new(&u.created_at),
        ]);
    }

    println!("{}", table);
    println!("\nPress Enter to continue...");
    let mut _buf = String::new();
    std::io::stdin().read_line(&mut _buf)?;
    Ok(())
}

async fn edit_user(client: &mut AdminClient) -> anyhow::Result<bool> {
    let users = client.list_users().await?;
    if users.is_empty() {
        println!("{}", "No users found.".yellow());
        std::thread::sleep(std::time::Duration::from_secs(2));
        return Ok(true);
    }

    let user_names: Vec<String> = users.iter().map(|u| u.username.clone()).collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select User to edit")
        .default(0)
        .items(&user_names)
        .interact_opt()?;

    let selected_idx = match selection {
        Some(idx) => idx,
        None => return Ok(true),
    };

    let selected_user = &users[selected_idx];
    let action_items = vec!["Change Role", "Change Quota"];

    let action_sel = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(format!("Action for {}", selected_user.username.cyan()))
        .default(0)
        .items(&action_items)
        .interact_opt()?;

    let action_idx = match action_sel {
        Some(idx) => idx,
        None => return Ok(true),
    };

    match action_idx {
        0 => {
            let roles = vec!["admin", "user", "guest"];
            let role_sel = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select new Role")
                .default(0)
                .items(&roles)
                .interact_opt()?;

            let role_idx = match role_sel {
                Some(idx) => idx,
                None => return Ok(true),
            };

            let new_role = roles[role_idx].to_string();
            client.change_user_role(selected_user.username.clone(), new_role.clone()).await?;
            println!("{} is now {}.", selected_user.username.green(), new_role.yellow());
            
            if selected_user.username == client.username {
                println!("{}", "\nSECURITY LOCKOUT: You changed your own role.".red().bold());
                println!("Your current session has been terminated to apply the new permissions.");
                std::thread::sleep(std::time::Duration::from_secs(4));
                return Ok(false);
            }
        },
        1 => {
            let current_gb = selected_user.quota_limit as f64 / 1024.0 / 1024.0 / 1024.0;
            let new_quota_str: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt(format!("New Quota Limit in GB (Current: {:.2}, 'q' = cancel)", current_gb))
                .default(current_gb.to_string())
                .interact()?;

            if new_quota_str.trim().eq_ignore_ascii_case("q") { return Ok(true); }

            let new_quota_gb: f64 = new_quota_str.parse().unwrap_or(current_gb);
            let new_quota_bytes = (new_quota_gb * 1024.0 * 1024.0 * 1024.0) as u64;
            client.set_quota(selected_user.username.clone(), new_quota_bytes).await?;
            println!("Quota updated.");
        },
        _ => {}
    }

    std::thread::sleep(std::time::Duration::from_secs(2));
    Ok(true)
}

async fn generate_invite(client: &mut AdminClient) -> anyhow::Result<()> {
    let roles = vec!["admin", "user", "guest"];
    let role_sel = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Assign Role for this Code")
        .default(2)
        .items(&roles)
        .interact_opt()?;

    let role_idx = match role_sel {
        Some(idx) => idx,
        None => return Ok(()),
    };
    let selected_role = roles[role_idx].to_string();

    let max_uses_str: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Maximum Uses (0 = unlimited, 'q' = cancel)")
        .default("1".to_string())
        .interact()?;

    if max_uses_str.trim().eq_ignore_ascii_case("q") { return Ok(()); }
    let max_uses: u32 = max_uses_str.parse().unwrap_or(1);

    let quota_str: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Initial Quota Limit in GB ('q' = cancel)")
        .default("1.0".to_string())
        .interact()?;

    if quota_str.trim().eq_ignore_ascii_case("q") { return Ok(()); }
    let quota_gb: f64 = quota_str.parse().unwrap_or(1.0);
    let quota_bytes = (quota_gb * 1024.0 * 1024.0 * 1024.0) as u64;

    match client.generate_invite(selected_role.clone(), max_uses, quota_bytes).await {
        Ok(code) => {
            println!("\n{}", "INVITE CODE GENERATED SUCCESSFULLY".green().bold());
            println!("Share this code with the user: {}", code.cyan().bold());
            println!("Role: {}, Max Uses: {}", selected_role, max_uses);
        },
        Err(e) => println!("{}", format!("Failed: {}", e).red()),
    }

    println!("\nPress Enter to continue...");
    let mut _buf = String::new();
    std::io::stdin().read_line(&mut _buf)?;
    Ok(())
}

async fn list_invites(client: &mut AdminClient) -> anyhow::Result<()> {
    println!("\nFetching Invite Codes...");
    let invites = client.list_invites().await?;

    if invites.is_empty() {
        println!("{}", "No active invite codes found.".yellow());
    } else {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec!["Code", "Role", "Uses", "Created By", "Created At"]);

        for inv in invites {
            let uses_str = if inv.max_uses == 0 { format!("{}/∞", inv.used_count) } else { format!("{}/{}", inv.used_count, inv.max_uses) };
            table.add_row(vec![
                Cell::new(&inv.code).fg(TColor::Cyan).add_attribute(comfy_table::Attribute::Bold),
                Cell::new(&inv.role),
                Cell::new(uses_str),
                Cell::new(&inv.created_by),
                Cell::new(&inv.created_at),
            ]);
        }
        println!("{}", table);
    }

    println!("\nPress Enter to continue...");
    let mut _buf = String::new();
    std::io::stdin().read_line(&mut _buf)?;
    Ok(())
}

async fn revoke_invite(client: &mut AdminClient) -> anyhow::Result<()> {
    let code: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter Invite Code to revoke ('q' to cancel)")
        .interact()?;

    if code.trim().eq_ignore_ascii_case("q") { return Ok(()); }

    let confirm = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Are you sure?")
        .interact_opt()?;

    if let Some(true) = confirm {
        client.revoke_invite(code).await?;
        println!("{}", "Code revoked.".green());
    }

    std::thread::sleep(std::time::Duration::from_secs(1));
    Ok(())
}

fn format_bytes(b: u64) -> String {
    const UNIT: u64 = 1024;
    if b < UNIT { return format!("{} B", b); }
    let div = UNIT as f64;
    let exp = (b as f64).ln() / div.ln();
    let pre = "KMGTPE".chars().nth(exp as usize - 1).unwrap_or('?');
    format!("{:.1} {}B", (b as f64) / div.powi(exp as i32), pre)
}