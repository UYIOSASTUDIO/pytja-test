__
                       /\ \__   __
         _____   __  __\ \ ,_\ /\_\     __
        /\ '__`\/\ \/\ \\ \ \/ \/\ \  /'__`\
        \ \ \L\ \ \ \_\ \\ \ \_ \ \ \/\ \L\.\_
         \ \ ,__/\/`____ \\ \__\_\ \ \ \__/.\_\
          \ \ \/  `/___/> \\/__/\ \_\ \/__/\/_/
           \ \_\     /\___/    \ \____/
            \/_/     \/__/      \/___/
            
PYTJA PLATFORM - DEPLOYMENT GUIDE
================================================================================

Welcome to the Pytja release. This binary is compiled for maximum
performance and features an automated deployment wizard for macOS, Linux,
and Windows.

Pytja is a deterministic, WASM-powered data engine and plugin architecture
engineered for bare-metal performance and zero-trust isolation.

[PRE-RELEASE ARCHITECTURE]
Pytja is currently in active early development. While the core WASM engine
and sandbox isolation are functional, the system has not yet been certified
for mission-critical production workloads. Please test thoroughly in sandboxed
environments before deploying to live infrastructure.

1. SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------
Pytja requires a local Redis cache for session state management.

macOS installation via Homebrew:
  brew install redis
  brew services start redis

Linux installation (Debian/Ubuntu):
  sudo apt-get update
  sudo apt-get install redis-server
  sudo systemctl enable --now redis-server

Windows installation (via Docker):
  docker run --name pytja-redis -p 6379:6379 -d redis

  (Alternatively, install WSL2 via "wsl --install" and use
  the Linux instructions above).


2. EXECUTION AND PROVISIONING
--------------------------------------------------------------------------------
Execute the binary directly from your terminal. The application utilizes
Zero-Touch Provisioning to establish your environment securely.

For macOS and Linux:
Grant execution permissions before running the binary.
  chmod +x pytja-<os-version>

(macOS only) Remove the quarantine attribute if blocked by Gatekeeper:
  xattr -d com.apple.quarantine pytja-macos

Execute the application:
  ./pytja-<os-version>

For Windows:
Open PowerShell or Command Prompt and execute the provided .exe file:
  .\pytja-windows.exe

The bootstrap wizard will automatically configure your local network binding, 
generate TLS certificates, and initialize the SQLite database. You will be 
prompted to create your primary administrative identity during this phase.


3. DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
Upon successful initialization, the application will manage the following local 
directories in your current working folder:

- /certs  : Contains the generated Zero-Trust TLS certificates.
- /config : Contains the default.toml configuration file.
- /data   : Houses the SQLite database (pytja.db) and local storage blobs.
- /logs   : Contains detailed telemetry and operational logs.
- .env    : Stores the uniquely generated cryptographic JWT secret.


4. OPERATIONAL HANDOFF
--------------------------------------------------------------------------------
After completing the identity setup, the backend server will detach as a 
background daemon, and your terminal will seamlessly transition into the 
interactive Pytja Shell. 

To terminate the session and the background daemon, simply type "exit" within 
the shell.