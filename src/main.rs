use clap::{Parser, Subcommand};

mod bootstrap;

#[derive(Parser)]
#[command(name = "pytja", version = "2.0", about = "Pytja Enterprise Platform", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Server,
    Shell {
        #[arg(short, long, env = "PYTJA_IDENTITY_PATH")]
        identity: Option<String>,
    },
    Registrar {
        #[arg(short, long)]
        output: Option<String>,
    },
    Admin,
    Bootstrap,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Server) => {
            pytja_server::start_server().await.map_err(|e| anyhow::anyhow!("Server crashed: {}", e))?;
        }
        Some(Commands::Shell { identity }) => {
            pytja_shell::start_shell(identity).await?;
        }
        Some(Commands::Registrar { output }) => {
            pytja_registrar::start_registrar(output).await?;
        }
        Some(Commands::Admin) => {
            pytja_admin::start_admin().await?;

        }
        Some(Commands::Bootstrap) => {
            bootstrap::run_enterprise_wizard().await?;
        }
        None => {
            bootstrap::run_enterprise_wizard().await?;
        }
    }

    Ok(())
}