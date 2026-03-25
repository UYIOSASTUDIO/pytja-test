use crate::vfs::VirtualFileSystem;
use crate::radar::RadarEngine;
use crate::network_client::PytjaClient;
use rustyline::{Editor, Context};
use rustyline::completion::{Completer, extract_word};
use rustyline::hint::Hinter;
use rustyline::highlight::Highlighter;
use rustyline::validate::{Validator, ValidationResult, ValidationContext};
use rustyline::Helper;
use rustyline::history::DefaultHistory;
use rustyline::error::ReadlineError;
use rustyline::config::Configurer;
use colored::*;
use std::io::{self, Write};
use std::str;
use pytja_core::FileNode;
use chrono::{DateTime, Local};
use tokio::sync::{Mutex, mpsc};
use std::sync::Arc;
use pytja_proto::FileInfo;
use directories::ProjectDirs;
use walkdir::WalkDir;
use std::path::Path;
use tracing::{info, warn, error};

pub struct PytjaHelper {
    commands: Vec<String>,
    plugins: Vec<String>,
}

impl Completer for PytjaHelper {
    type Candidate = String;

    fn complete(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> rustyline::Result<(usize, Vec<String>)> {
        let (start, word) = extract_word(line, pos, None, |c| c == ' ' || c == '\t');
        let mut matches = Vec::new();

        if start == 0 {
            for cmd in &self.commands {
                if cmd.starts_with(word) {
                    matches.push(cmd.clone());
                }
            }
            for plugin in &self.plugins {
                if plugin.starts_with(word) {
                    matches.push(plugin.clone());
                }
            }
        } else {
            if line.starts_with("daemon start ") || line.starts_with("daemon stop ") || line.starts_with("ui open ") {
                for plugin in &self.plugins {
                    if plugin.starts_with(word) {
                        matches.push(plugin.clone());
                    }
                }
            }
        }
        Ok((start, matches))
    }
}

impl Hinter for PytjaHelper {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> { None }
}

impl Highlighter for PytjaHelper {}

impl Validator for PytjaHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None))
    }
    fn validate_while_typing(&self) -> bool { false }
}

impl Helper for PytjaHelper {}

pub struct Terminal {
    vfs: Arc<Mutex<VirtualFileSystem>>,
    username: String,
    pub radar_engine: RadarEngine,
    client: PytjaClient,
    alarm_rx: mpsc::Receiver<String>,
    alerts: Vec<String>,
    unread_alerts: usize,
    current_path: String,
}

impl Terminal {
    pub fn new(
        vfs: Arc<tokio::sync::Mutex<VirtualFileSystem>>,
        username: String,
        radar_engine: RadarEngine,
        client: PytjaClient,
        alarm_rx: tokio::sync::mpsc::Receiver<String>
    ) -> Self {
        Self {
            vfs, username, radar_engine, client, alarm_rx,
            alerts: Vec::new(),
            unread_alerts: 0,
            current_path: "/".to_string(),
        }
    }

    pub async fn start(&mut self) -> anyhow::Result<()> {
        self.print_banner();
        let plugin_names: Vec<String> = self.radar_engine.get_manifests().into_iter().map(|m| m.name).collect();

        let helper = PytjaHelper {
            commands: vec![
                "exit".into(), "help".into(), "clear".into(), "whoami".into(),
                "ls".into(), "ll".into(), "cd".into(), "mkdir".into(), "touch".into(),
                "cp".into(), "mv".into(), "rm".into(), "nano".into(), "cat".into(),
                "upload".into(), "download".into(), "exec".into(), "chmod".into(),
                "chown".into(), "lock".into(), "tree".into(), "stat".into(),
                "find".into(), "grep".into(), "du".into(), "query".into(),
                "plugins".into(), "mounts".into(), "daemon".into(), "ui".into()
            ],
            plugins: plugin_names,
        };

        let mut rl = Editor::<PytjaHelper, DefaultHistory>::new()?;
        rl.set_helper(Some(helper));
        rl.set_completion_type(rustyline::CompletionType::List);

        let history_path = if let Some(proj_dirs) = ProjectDirs::from("com", "pytja", "shell") {
            let data_dir = proj_dirs.data_dir();
            std::fs::create_dir_all(data_dir).ok();
            Some(data_dir.join("history.txt"))
        } else {
            None
        };

        if let Some(ref path) = history_path {
            let _ = rl.load_history(path);
        }

        // --- AUTOSTART SEQUENCE ---
        let manifests = self.radar_engine.get_manifests();
        let mut autostart_count = 0;

        for manifest in manifests {
            if manifest.autostart {
                println!("[RADAR] Auto-booting background service: {}", manifest.name.cyan());
                let client_clone = self.client.clone();
                if let Err(e) = self.radar_engine.start_daemon(&manifest.name, vec![], client_clone) {
                    println!("{} Failed to autostart daemon '{}': {}", "[ERROR]".red().bold(), manifest.name, e);
                } else {
                    autostart_count += 1;
                }
            }
        }
        if autostart_count > 0 {
            println!("{} {} autonomous agent(s) booted successfully.\n", "[OK]".green().bold(), autostart_count);
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        loop {
            while let Ok(msg) = self.alarm_rx.try_recv() {
                self.alerts.push(msg);
                self.unread_alerts += 1;
            }

            let prompt = if self.unread_alerts > 0 {
                format!(
                    "┌──({}@pytja)-[{}]-[\x1b[31m!\x1b[0m {} ALERTS]\n└─$ ",
                    self.username.red(),
                    self.current_path.blue(),
                    self.unread_alerts
                )
            } else {
                format!(
                    "┌──({}@pytja)-[{}]\n└─$ ",
                    self.username.red(),
                    self.current_path.blue()
                )
            };

            let readline = rl.readline(&prompt);
            match readline {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    let _ = rl.add_history_entry(line);

                    let commands: Vec<&str> = line.split("&&").collect();
                    for cmd_str in commands {
                        let cmd_trimmed = cmd_str.trim();

                        if cmd_trimmed == "dmesg" || cmd_trimmed == "alerts" {
                            println!("\n\x1b[36m--- SYSTEM EVENT LOG (UNREAD: {}) ---\x1b[0m", self.unread_alerts);
                            if self.alerts.is_empty() {
                                println!("No system events recorded.");
                            } else {
                                for (i, alert) in self.alerts.iter().enumerate() {
                                    println!("[{:03}] {}", i + 1, alert);
                                }
                            }
                            println!("\x1b[36m--------------------------------------\x1b[0m\n");
                            self.unread_alerts = 0;
                            continue;
                        }

                        if !self.dispatch_command(cmd_trimmed).await {
                            if let Some(ref path) = history_path {
                                let _ = rl.save_history(path);
                            }
                            return Ok(());
                        }
                    }
                },
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C");
                    break;
                },
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D");
                    break;
                },
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }

        if let Some(ref path) = history_path {
            let _ = rl.save_history(path);
        }
        Ok(())
    }

    async fn dispatch_command(&mut self, cmd_input: &str) -> bool {
        let parts: Vec<&str> = cmd_input.split_whitespace().collect();
        if parts.is_empty() { return true; }
        let cmd = parts[0];
        let args = parts[1..].to_vec();

        match cmd {
            "exit" => return self.handle_exit().await,
            "help" => self.handle_help(),
            "clear" => self.print_banner(),
            "whoami" => println!("{}", self.username.green().bold()),
            "ls" | "ll" => self.handle_ls(args).await,
            "cd" => self.handle_cd(args).await,
            "mkdir" => self.handle_mkdir(args).await,
            "touch" => self.handle_touch(args).await,
            "cp" => self.handle_cp(args).await,
            "mv" => self.handle_mv(args).await,
            "rm" => self.handle_rm(args).await,
            "nano" => self.handle_nano(args).await,
            "cat" => self.handle_cat(args).await,
            "upload" => self.handle_upload(args).await,
            "download" => self.handle_download(args).await,
            "exec" => self.handle_exec(args).await,
            "chmod" => self.handle_chmod(args).await,
            "chown" => self.handle_chown(args).await,
            "lock" => self.handle_lock(args).await,
            "tree" => self.handle_tree(args).await,
            "stat" => self.handle_stat(args).await,
            "find" => self.handle_find(args).await,
            "grep" => self.handle_grep(args).await,
            "du" => self.handle_du(args).await,
            "query" => self.handle_query(args).await,
            "plugins" => self.handle_plugins(),
            "mounts" => self.handle_mounts(args).await,
            "daemon" => self.handle_daemon(args).await,
            _ => {
                if self.radar_engine.has_plugin(cmd) {
                    println!("{} Radar Enclave: {}", "Launching".cyan(), cmd.bold());
                    self.execute_radar_plugin(cmd, args).await;
                } else {
                    println!("Command not found: {}", cmd);
                }
            }
        }
        true
    }

    // --- COMMAND HANDLERS ---

    async fn handle_exit(&mut self) -> bool {
        println!("Shutting down OS subsystems...");
        println!("Encrypt Files...");
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        println!("Connection terminated.");
        false
    }

    fn handle_help(&self) {
        println!("\n{}", "PYTJA SHELL MANUAL".white().bold());
        println!("{}", "=".repeat(60));
        println!("\n{}", "[ FILE OPERATIONS ]".cyan());
        println!("{:<10} : {:<30}", "ls", "List [-a] [-s DATE/SIZE/NAME] [-r]");
        println!("{:<10} : {:<30}", "cd", "Change directory");
        println!("{:<10} : {:<30}", "mkdir", "Create dir [-lock]");
        println!("{:<10} : {:<30}", "touch", "Create file [-lock]");
        println!("{:<10} : {:<30}", "cp", "Copy file <src> <dst>");
        println!("{:<10} : {:<30}", "mv", "Move/Rename <src> <dst>");
        println!("{:<10} : {:<30}", "rm", "Delete file/folder");
        println!("{:<10} : {:<30}", "nano", "Edit file");
        println!("{:<10} : {:<30}", "cat", "Read file");
        println!("\n{}", "[ SECURITY & PERMISSIONS ]".cyan());
        println!("{:<10} : {:<30}", "chmod", "Change Mode <0|1|2> <file>");
        println!("{:<10} : {:<30}", "chown", "Change Owner <user> <file>");
        println!("{:<10} : {:<30}", "lock", "Set/Change Password <file>");
        println!("\n{}", "[ INTELLIGENCE ]".cyan());
        println!("{:<10} : {:<30}", "tree", "Show structure [path]");
        println!("{:<10} : {:<30}", "stat", "Show node details <file>");
        println!("{:<10} : {:<30}", "find", "Find by name <pattern>");
        println!("{:<10} : {:<30}", "grep", "Search content <pattern>");
        println!("{:<10} : {:<30}", "du", "Disk usage / Quota");
        println!("{:<10} : {:<30}", "mounts", "Show DB mounts [--status]");
        println!("\n{}", "[ NETWORK ]".cyan());
        println!("{:<10} : {:<30}", "upload", "Import from Host [-lock]");
        println!("{:<10} : {:<30}", "download", "Export to Host");
        println!("{:<10} : {:<30}", "plugins", "Show active plugins & perms");
        println!("{}", "=".repeat(60));
    }

    async fn handle_ls(&self, args: Vec<&str>) {
        let show_hidden = args.contains(&"-a") || args.contains(&"-sh");
        let reverse = args.contains(&"-r");
        let mut sort_by = "DATE";

        if let Some(idx) = args.iter().position(|&x| x == "-s") {
            if idx + 1 < args.len() { sort_by = args[idx + 1]; }
        }

        let current_path = self.current_path.clone();

        match self.client.list_files(&current_path).await {
            Ok(items) => {
                let mut visible_items: Vec<&FileInfo> = items.iter()
                    .filter(|item| show_hidden || !item.name.starts_with('.'))
                    .collect();

                match sort_by.to_uppercase().as_str() {
                    "NAME" => visible_items.sort_by(|a, b| if reverse { b.name.to_lowercase().cmp(&a.name.to_lowercase()) } else { a.name.to_lowercase().cmp(&b.name.to_lowercase()) }),
                    "SIZE" => visible_items.sort_by(|a, b| if reverse { a.size.cmp(&b.size) } else { b.size.cmp(&a.size) }),
                    "TYPE" => visible_items.sort_by(|a, b| if reverse { a.is_folder.cmp(&b.is_folder).then(b.name.cmp(&a.name)) } else { b.is_folder.cmp(&a.is_folder).then(a.name.cmp(&b.name)) }),
                    "OWNER" => visible_items.sort_by(|a, b| if reverse { b.owner.cmp(&a.owner) } else { a.owner.cmp(&b.owner) }),
                    _ => visible_items.sort_by(|a, b| if reverse { a.created_at.partial_cmp(&b.created_at).unwrap() } else { b.created_at.partial_cmp(&a.created_at).unwrap() }),
                }

                println!("{:<6} {:<8} {:<10} {:<15} {:<18} NAME", "TYPE", "PERM", "SIZE", "OWNER", "DATE");
                println!("{}", "-".repeat(75));

                for item in &visible_items {
                    let type_str = if item.is_folder { "DIR" } else { "FILE" };
                    let color_name = if item.is_folder { item.name.blue() } else { item.name.green() };
                    let size_str = if item.is_folder { "---".to_string() } else { self.format_size(item.size) };
                    let date_str = self.format_date(item.created_at);
                    let perm_str = match item.permissions {
                        0 => "PRIV".red(),
                        1 => "PUB-R".yellow(),
                        2 => "PUB-W".green(),
                        _ => "???".dimmed(),
                    };
                    println!("{:<6} {:<8} {:<10} {:<15} {:<18} {}", type_str, perm_str, size_str, item.owner, date_str, color_name);
                }
                println!("\n[TOTAL: {} (REMOTE)]", visible_items.len());
            },
            Err(e) => println!("Server Error: {}", e.to_string().red()),
        }
    }

    async fn handle_cd(&mut self, args: Vec<&str>) {
        // ENTERPRISE FIX: "cd" ohne Argumente springt professionell ins Root-Verzeichnis
        let target = if args.is_empty() { "/" } else { args[0] };

        let new_path = if target == ".." {
            if self.current_path == "/" {
                "/".to_string()
            } else {
                let path = Path::new(&self.current_path);
                path.parent().unwrap_or(Path::new("/")).to_string_lossy().to_string()
            }
        } else if target.starts_with('/') {
            target.to_string()
        } else if self.current_path == "/" {
            format!("/{}", target)
        } else {
            format!("{}/{}", self.current_path, target)
        };

        match self.client.stat_node(&new_path).await {
            Ok((exists, is_folder, is_locked)) => {
                if !exists { println!("{} Directory not found.", "Error:".red()); return; }
                if !is_folder { println!("{} Not a directory.", "Error:".red()); return; }
                if is_locked { let _ = self.ask_password("Locked Directory. Enter Password: "); }

                self.current_path = new_path;
                self.vfs.lock().await.current_path = self.current_path.clone();
            },
            Err(e) => println!("{} {}", "Server Error:".red(), e),
        }
    }

    async fn handle_mkdir(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: mkdir <name> [-lock]"); return; }
        let name = args[0];
        let mut lock_pass = None;

        if args.contains(&"-lock") {
            let p1 = self.ask_password("Set Password: ");
            let p2 = self.ask_password("Confirm Password: ");
            if p1 != p2 { println!("{}", "Passwords do not match.".red()); return; }
            if !p1.is_empty() { lock_pass = Some(p1); }
        }

        let full_path = self.resolve_path(name).await;
        match self.client.create_node(&full_path, true, vec![], lock_pass, &self.username).await {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_touch(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: touch <name> [content] [-lock]"); return; }
        let mut name = args[0].to_string();
        if !name.contains('.') { name.push_str(".txt"); }

        let mut lock_pass = None;
        let mut content_parts = Vec::new();

        for arg in &args[1..] {
            if *arg == "-lock" {
                let p1 = self.ask_password("Set Password: ");
                let p2 = self.ask_password("Confirm Password: ");
                if p1 != p2 { println!("{}", "Passwords do not match.".red()); return; }
                if !p1.is_empty() { lock_pass = Some(p1); }
            } else { content_parts.push(*arg); }
        }

        let content_str = content_parts.join(" ");
        let content_bytes = content_str.trim_matches('"').trim_matches('\'').as_bytes().to_vec();
        let full_path = self.resolve_path(&name).await;

        match self.client.create_node(&full_path, false, content_bytes, lock_pass, &self.username).await {
            Ok(_) => println!("{}", "File created.".green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_cp(&self, args: Vec<&str>) {
        if args.len() < 2 { println!("Usage: cp <source> <dest>"); return; }
        let src_path = self.resolve_path(args[0]).await;
        let dst_path = self.resolve_path(args[1]).await;

        match self.client.copy_node(&src_path, &dst_path, &self.username).await {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_mv(&self, args: Vec<&str>) {
        if args.len() < 2 { println!("Usage: mv <source> <dest>"); return; }
        let src_path = self.resolve_path(args[0]).await;
        let dst_path = self.resolve_path(args[1]).await;

        match self.client.move_node(&src_path, &dst_path).await {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_rm(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: rm <name>"); return; }
        let full_path = self.resolve_path(args[0]).await;

        match self.client.delete_node(&full_path).await {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_nano(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: nano <file>"); return; }
        let path = self.resolve_path(args[0]).await;

        let vfs_guard = self.vfs.lock().await;
        if let Some(db) = vfs_guard.get_db().await {
            if let Ok(Some(node)) = db.get_node(&path).await {
                if !self.check_lock(&node) { return; }
            }
        }
        drop(vfs_guard);

        if let Err(e) = self.vfs.lock().await.edit_file(args[0]).await {
            self.handle_error("Context", e);
        }
    }

    async fn handle_cat(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: cat <name>"); return; }
        let full_path = self.resolve_path(args[0]).await;

        match self.client.stat_node(&full_path).await {
            Ok((exists, is_folder, _)) => {
                if !exists {
                    println!("cat: {}: No such file or directory", args[0]);
                    return;
                }
                if is_folder {
                    println!("cat: {}: Is a directory", args[0].blue());
                    return;
                }
            },
            Err(e) => {
                self.handle_error("File Check Failed", e);
                return;
            }
        }

        match self.client.read_file(&full_path, None).await {
            Ok((content, _)) => self.print_file_content(&content),
            Err(e) => {
                if e.to_string().contains("Password") {
                    let pass = self.ask_password("Locked File. Password: ");
                    match self.client.read_file(&full_path, Some(pass)).await {
                        Ok((content, _)) => self.print_file_content(&content),
                        Err(e2) => println!("{}", e2.to_string().red()),
                    }
                } else {
                    self.handle_error("Context", e);
                }
            }
        }
    }

    async fn handle_upload(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: upload <local> [remote]"); return; }
        let local_path = Path::new(args[0]);

        if !local_path.exists() {
            self.handle_error("Upload Error", "Local path does not exist");
            return;
        }

        let remote_base = if args.len() > 1 { args[1].to_string() } else {
            let name = local_path.file_name().unwrap_or_default().to_string_lossy();
            let current_path = self.vfs.lock().await.get_cwd().to_string();
            if current_path == "/" { format!("/{}", name) } else { format!("{}/{}", current_path, name) }
        };

        info!("Starting upload: {:?} -> {}", local_path, remote_base);

        if local_path.is_dir() {
            println!("Recursive Upload: {} -> {}", local_path.display(), remote_base);
            if let Err(e) = self.client.create_node(&remote_base, true, vec![], None, &self.username).await {
                self.handle_error("Mkdir Remote Root", e);
            }

            for entry in WalkDir::new(local_path).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path == local_path { continue; }

                let rel = path.strip_prefix(local_path).unwrap();
                let remote = format!("{}/{}", remote_base, rel.to_string_lossy()).replace("//", "/");

                if path.is_dir() {
                    if let Err(e) = self.client.create_node(&remote, true, vec![], None, &self.username).await {
                        warn!("Failed to create dir {}: {}", remote, e);
                        println!("{} {} ({})", "SKIP".yellow(), remote, e);
                    }
                } else {
                    print!("Uploading {}... ", rel.to_string_lossy());
                    match self.client.upload_file(path.to_str().unwrap(), &remote, None, &self.username, None).await {
                        Ok(_) => println!("{}", "OK".green()),
                        Err(e) => {
                            println!("{}", "FAIL".red());
                            error!("Upload failed for {}: {:?}", remote, e);
                        }
                    }
                }
            }
        } else {
            let lock_pass = if args.contains(&"-lock") { Some(self.ask_password("Set Password: ")) } else { None };
            println!("Uploading {}...", local_path.display());

            match self.client.upload_file(local_path.to_str().unwrap(), &remote_base, lock_pass, &self.username, None).await {
                Ok(_) => {
                    info!("Upload success: {}", remote_base);
                    println!("{}", "Upload complete.".green());
                },
                Err(e) => self.handle_error("Upload failed", e),
            }
        }
    }

    async fn handle_download(&self, args: Vec<&str>) {
        if args.len() < 2 { println!("Usage: download <remote> <local>"); return; }
        let full_remote = self.resolve_path(args[0]).await;
        let local_path = Path::new(args[1]);

        info!("Starting download: {} -> {:?}", full_remote, local_path);

        match self.client.stat_node(&full_remote).await {
            Ok((exists, is_folder, _)) => {
                if !exists {
                    self.handle_error("Download", "Remote path not found");
                    return;
                }

                if is_folder {
                    println!("Recursive Download: {} -> {}", full_remote, local_path.display());
                    if let Err(e) = std::fs::create_dir_all(local_path) {
                        self.handle_error("Local FS Error", e);
                        return;
                    }

                    let mut stack = vec![(full_remote, local_path.to_path_buf())];
                    while let Some((r, l)) = stack.pop() {
                        match self.client.list_files(&r).await {
                            Ok(files) => {
                                for f in files {
                                    let c_r = if r == "/" { format!("/{}", f.name) } else { format!("{}/{}", r, f.name) };
                                    let c_l = l.join(&f.name);
                                    if f.is_folder {
                                        std::fs::create_dir_all(&c_l).ok();
                                        stack.push((c_r, c_l));
                                    } else {
                                        print!("Downloading {}... ", f.name);
                                        match self.client.download_file(&c_r, c_l.to_str().unwrap(), None).await {
                                            Ok(_) => println!("{}", "OK".green()),
                                            Err(e) => {
                                                println!("{}", "FAIL".red());
                                                error!("Download fail {}: {:?}", c_r, e);
                                            }
                                        }
                                    }
                                }
                            },
                            Err(e) => self.handle_error(&format!("List {}", r), e),
                        }
                    }
                } else {
                    println!("Downloading {}...", full_remote);
                    match self.client.download_file(&full_remote, local_path.to_str().unwrap(), None).await {
                        Ok(_) => println!("{}", "Done.".green()),
                        Err(e) => self.handle_error("Download failed", e),
                    }
                }
            },
            Err(e) => self.handle_error("Stat failed", e),
        }
    }

    async fn handle_exec(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: exec <script.py>"); return; }
        let path = self.resolve_path(args[0]).await;
        println!("{}", "[!] EXECUTING REMOTE KERNEL...".yellow());
        if let Err(e) = self.client.exec_script(&path).await {
            self.handle_error("Context", e);
        }
    }

    async fn handle_chmod(&self, args: Vec<&str>) {
        if args.len() < 2 { println!("Usage: chmod <0|1|2> <file>"); return; }
        let perm_val: i32 = args[0].parse().unwrap_or(-1);
        let path = self.resolve_path(args[1]).await;
        if !(0..=2).contains(&perm_val) { println!("Invalid mode."); return; }
        match self.client.change_mode(&path, perm_val as u32).await {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_chown(&self, args: Vec<&str>) {
        if args.len() < 2 { println!("Usage: chown <new_owner> <file>"); return; }
        let path = self.resolve_path(args[1]).await;
        match self.client.chown_node(&path, args[0]).await {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_lock(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: lock <file>"); return; }
        let path = self.resolve_path(args[0]).await;
        let p1 = self.ask_password("Enter new Password (empty to unlock): ");
        if !p1.is_empty() {
            let p2 = self.ask_password("Confirm: ");
            if p1 != p2 { println!("Mismatch."); return; }
        }
        let password_opt = if p1.is_empty() { None } else { Some(p1) };
        match self.client.lock_node(&path, password_opt).await {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_tree(&self, args: Vec<&str>) {
        let path = if args.is_empty() { "" } else { args[0] };
        let mut full_path = self.resolve_path(path).await;

        if full_path.ends_with("/.") {
            full_path = full_path.trim_end_matches('.').trim_end_matches('/').to_string();
        }
        if full_path.is_empty() {
            full_path = "/".to_string();
        }

        match self.client.get_tree(&full_path).await {
            Ok(tree) => println!("{}", tree),
            Err(e) => self.handle_error("Tree Fetch Error", e),
        }
    }

    async fn handle_stat(&self, args: Vec<&str>) {
        if args.is_empty() { return; }
        let full_path = self.resolve_path(args[0]).await;

        match self.client.stat_node(&full_path).await {
            Ok((exists, is_folder, is_locked)) => {
                println!("{}", "--- NODE STATUS ---".yellow());
                println!("Path:   {}", full_path);
                println!("Exists: {}", exists);
                println!("Type:   {}", if is_folder { "Directory" } else { "File" });
                println!("Locked: {}", if is_locked { "YES" } else { "NO" });
            },
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_find(&self, args: Vec<&str>) {
        if args.is_empty() { return; }
        match self.client.find_node(args[0]).await {
            Ok(paths) => {
                println!("Found {} matches:", paths.len());
                for p in paths { println!(" - {}", p.cyan()); }
            },
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_grep(&self, args: Vec<&str>) {
        if args.is_empty() { return; }
        match self.client.grep_node(args[0]).await {
            Ok(matches) => {
                println!("Found content in {} files:", matches.len());
                for m in matches { println!(" - {}", m.green()); }
            },
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn handle_du(&self, _args: Vec<&str>) {
        match self.client.get_usage(&self.username).await {
            Ok(bytes) => {
                let mb = bytes as f64 / 1024.0 / 1024.0;
                println!("Usage: {:.2} MB", mb);
            },
            Err(e) => self.handle_error("Context", e),
        }
    }

    async fn resolve_path(&self, input: &str) -> String {
        self.vfs.lock().await.resolve_path(input)
    }

    fn print_file_content(&self, content: &[u8]) {
        println!("\n{}", "--- BEGIN MESSAGE ---".cyan());
        if let Ok(s) = str::from_utf8(content) { println!("{}", s); }
        else { println!("{}", "[BINARY DATA]".red()); }
        println!("{}\n", "--- END MESSAGE ---".cyan());
    }

    fn format_size(&self, size: u64) -> String {
        const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
        let mut size = size as f64;
        let mut unit_idx = 0;
        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }
        format!("{:.2} {}", size, UNITS[unit_idx])
    }

    fn print_banner(&self) {
        print!("\x1B[2J\x1B[1;1H");
        println!("{}", r#"
                        __
                       /\ \__   __
         _____   __  __\ \ ,_\ /\_\     __
        /\ '__`\/\ \/\ \\ \ \/ \/\ \  /'__`\
        \ \ \L\ \ \ \_\ \\ \ \_ \ \ \/\ \L\.\_
         \ \ ,__/\/`____ \\ \__\_\ \ \ \__/.\_\
          \ \ \/  `/___/> \\/__/\ \_\ \/__/\/_/
           \ \_\     /\___/    \ \____/
            \/_/     \/__/      \/___/
        "#.white().bold());
        println!("        PYTJA V1.0 // USER: {}", self.username);
    }

    fn ask_password(&self, prompt: &str) -> String {
        print!("{}", prompt);
        io::stdout().flush().unwrap();
        rpassword::read_password().unwrap_or_default()
    }

    fn check_lock(&self, node: &FileNode) -> bool {
        if let Some(ref real_pass) = node.lock_pass {
            let input = self.ask_password(&format!("🔒 Enter Password for '{}': ", node.name));
            if input.trim() != real_pass {
                println!("{}", "ACCESS DENIED.".red());
                return false;
            }
        }
        true
    }

    fn format_date(&self, timestamp: f64) -> String {
        let seconds = timestamp as i64;
        if let Some(dt_utc) = DateTime::from_timestamp(seconds, 0) {
            let dt_local: DateTime<Local> = dt_utc.with_timezone(&Local);
            dt_local.format("%Y-%m-%d %H:%M").to_string()
        } else {
            "Unknown".to_string()
        }
    }

    fn handle_error(&self, context: &str, e: impl std::fmt::Display + std::fmt::Debug) {
        error!(target: "shell", "{}: {:?}", context, e);
        println!("{} {}", format!("{}:", context).red().bold(), e);
    }

    async fn handle_query(&self, args: Vec<&str>) {
        if args.is_empty() { println!("Usage: query <search_term>"); return; }
        let search_term = args.join(" ");

        println!("{} Searching metadata for '{}'...", "🔍".cyan(), search_term);
        match self.client.query_metadata(&search_term).await {
            Ok(items) => {
                if items.is_empty() {
                    println!("No files found matching the metadata query.");
                } else {
                    println!("Found {} files:", items.len());
                    for item in items {
                        let icon = if item.is_folder { "📁" } else { "📄" };
                        println!(" {} {} (Owner: {}, Size: {})", icon, item.name.green(), item.owner, self.format_size(item.size));
                    }
                }
            },
            Err(e) => self.handle_error("Query Error", e),
        }
    }

    fn handle_plugins(&self) {
        let manifests = self.radar_engine.get_manifests();

        if manifests.is_empty() {
            println!("\nNo plugins loaded.\n");
            return;
        }

        println!("\n{:<20} {:<10} DESCRIPTION", "NAME", "VERSION");
        println!("{}", "-".repeat(60));

        for manifest in manifests.iter() {
            let name_padded = format!("{:<20}", manifest.name);
            let version_padded = format!("{:<10}", manifest.version);

            println!("{} {} {}", name_padded.green().bold(), version_padded.yellow(), manifest.description.dimmed());
        }
        println!("\n[TOTAL: {} RADAR PLUGINS]\n", manifests.len());
    }

    async fn execute_radar_plugin(&mut self, cmd: &str, args: Vec<&str>) {
        let mut input_data = None;
        let mut filename = String::new();

        let mut i = 0;
        while i < args.len() {
            if args[i] == "--input" && i + 1 < args.len() {
                let remote_path = self.resolve_path(args[i+1]).await;
                println!("[RADAR] Streaming '{}' directly into MemFS...", remote_path);

                match self.client.read_file(&remote_path, None).await {
                    Ok((bytes, _meta)) => {
                        filename = std::path::Path::new(&remote_path).file_name().unwrap_or_default().to_string_lossy().to_string();
                        input_data = Some(bytes);
                    },
                    Err(e) => {
                        self.handle_error("Radar MemFS Stream", e);
                        return;
                    }
                }
                break;
            }
            i += 1;
        }

        let args_owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let input_tuple = if let Some(bytes) = input_data {
            Some((filename, bytes))
        } else {
            None
        };

        match self.radar_engine.execute_ephemeral(cmd, args_owned, input_tuple).await {
            Ok((msg, output_files)) => {
                println!("[OK] {}", msg);

                if !output_files.is_empty() {
                    let mut synced = 0;
                    let current_path = self.current_path.clone();
                    for (name, bytes) in output_files {
                        let remote_path = if current_path == "/" {
                            format!("/{}", name)
                        } else {
                            format!("{}/{}", current_path, name)
                        };

                        match self.client.create_node(&remote_path, false, bytes, None, &self.username).await {
                            Ok(_) => {
                                println!("[SYNC] Uploaded output '{}' to Pytja Server.", name);
                                synced += 1;
                            },
                            Err(e) => self.handle_error(&format!("Sync failed for {}", name), e),
                        }
                    }
                    if synced > 0 {
                        println!("[OK] {} Output files successfully exported.", synced);
                    }
                }
            },
            Err(e) => self.handle_error("Radar Enclave Crash", e),
        }
    }

    async fn handle_daemon(&mut self, args: Vec<&str>) {
        if args.is_empty() {
            println!("Usage: daemon <start|stop|ls> [plugin_name]");
            return;
        }

        match args[0] {
            "start" => {
                if args.len() < 2 {
                    println!("Usage: daemon start <plugin_name>");
                    return;
                }
                let plugin = args[1];
                let p_args = args[2..].iter().map(|s| s.to_string()).collect();

                match self.radar_engine.start_daemon(plugin, p_args, self.client.clone()) {
                    Ok(_) => println!("[OK] Daemon '{}' launched in background.", plugin),
                    Err(e) => self.handle_error("Daemon Boot Failure", e),
                }
            },
            "stop" => {
                if args.len() < 2 {
                    println!("Usage: daemon stop <plugin_name>");
                    return;
                }
                match self.radar_engine.kill_daemon(args[1]) {
                    Ok(_) => println!("Daemon '{}' stopped.", args[1]),
                    Err(e) => self.handle_error("Daemon Stop", e),
                }
            },
            "ls" => {
                let daemons = self.radar_engine.list_daemons();
                if daemons.is_empty() {
                    println!("No active daemons running.");
                    return;
                }
                println!("\n{:<25} STATUS", "DAEMON NAME");
                println!("{}", "-".repeat(40));
                for d in daemons {
                    println!("{:<25} RUNNING", d);
                }
                println!();
            },
            "logs" => {
                if args.len() < 2 {
                    println!("Usage: daemon logs <plugin_name>");
                    return;
                }
                match self.radar_engine.get_daemon_logs(args[1]).await {
                    Ok(logs) => {
                        println!("\n{}", format!("--- LOGS: {} ---", args[1]).cyan().bold());
                        println!("{}", logs);
                    },
                    Err(e) => self.handle_error("Daemon Logs", e),
                }
            },
            "send" => {
                if args.len() < 3 {
                    println!("Usage: daemon send <plugin_name> <payload>");
                    return;
                }
                let plugin = args[1];
                let payload = args[2..].join(" ");

                match self.radar_engine.send_to_daemon(plugin, payload).await {
                    Ok(_) => println!("[OK] Message dispatched to daemon '{}'.", plugin),
                    Err(e) => self.handle_error("Daemon C2 Bus", e),
                }
            },
            "kill" => {
                if args.len() < 2 {
                    println!("Usage: daemon kill <plugin_name>");
                    return;
                }
                match self.radar_engine.kill_daemon(args[1]) {
                    Ok(_) => println!("{} Daemon '{}' was forcefully terminated.", "[OK]".green().bold(), args[1]),
                    Err(e) => self.handle_error("Daemon Kill", e),
                }
            },
            _ => println!("Invalid daemon command. Use start, stop, or ls."),
        }
    }

    async fn handle_mounts(&self, args: Vec<&str>) {
        let detailed = args.contains(&"--status");

        match self.client.get_mounts().await {
            Ok(mounts) => {
                println!("\n{:<15} {:<15} {:<20} {:<10}", "MOUNT NAME", "TYPE", "CONNECTION", "STATUS");
                println!("{}", "-".repeat(65));

                for m in mounts {
                    let status = if m.is_connected { "ONLINE".green() } else { "OFFLINE".red() };
                    let conn = if detailed { m.connection } else { "*** HIDDEN ***".dimmed().to_string() };
                    println!("{:<15} {:<15} {:<20} {:<10}", m.name.cyan(), m.r#type, conn, status);
                }

                if detailed {
                    match self.client.get_system_stats().await {
                        Ok(stats) => {
                            println!("\n{}", "[ SYSTEM PERFORMANCE & HEALTH ]".cyan().bold());
                            println!("{}", "-".repeat(65));
                            println!("{:<20}: {:.2}%", "CPU Usage", stats.cpu_usage_percent);
                            println!("{:<20}: {}", "RAM Usage", self.format_size(stats.memory_usage_bytes));
                            println!("{:<20}: {}", "Active Sessions", stats.active_sessions);
                            println!("{:<20}: {}", "Server Uptime", stats.uptime);

                            let redis_status = if stats.redis_connected { "ONLINE".green() } else { "OFFLINE".red() };
                            println!("{:<20}: {}", "Redis Cache", redis_status);
                        },
                        Err(e) => self.handle_error("Stats Fetch Error", e),
                    }
                }
                println!();
            },
            Err(e) => self.handle_error("Mounts Fetch Error", e),
        }
    }
}