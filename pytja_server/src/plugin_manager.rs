use wasmtime::{Config, Engine, Linker, Module, Store};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use anyhow::{Result, anyhow};

#[derive(Clone)]
pub struct PluginManager {
    engine: Engine,
    module_cache: Arc<RwLock<HashMap<String, Module>>>,
}

impl PluginManager {
    pub fn new() -> Result<Self> {
        let mut config = Config::new();
        config.cranelift_opt_level(wasmtime::OptLevel::SpeedAndSize);
        // config.async_support(false); <--- ENTFERNT, da Feature deaktiviert ist

        // Explizites Error-Mapping für anyhow
        let engine = Engine::new(&config)
            .map_err(|e| anyhow!("Failed to initialize Wasmtime Engine: {}", e))?;

        Ok(Self {
            engine,
            module_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn load_plugin(&self, plugin_name: &str, wasm_bytes: Vec<u8>) -> Result<()> {
        let engine_clone = self.engine.clone();

        let module = tokio::task::spawn_blocking(move || {
            // Explizites Error-Mapping innerhalb des Worker-Threads
            Module::new(&engine_clone, &wasm_bytes)
                .map_err(|e| anyhow!("WASM Compilation error: {}", e))
        })
            .await
            .map_err(|e| anyhow!("Task Panicked during WASM compilation: {}", e))??; // Doppel-? ist hier korrekt (1x für JoinError, 1x für ModuleError)

        let mut cache = self.module_cache.write().await;
        cache.insert(plugin_name.to_string(), module);

        Ok(())
    }

    // =========================================================================
    // --- FAST INSTANTIATION & ABI EXECUTION (Run Many) ---
    // =========================================================================

    pub async fn execute_plugin(&self, plugin_name: &str, input_data: Vec<u8>) -> Result<Vec<u8>> {
        let module = {
            let cache = self.module_cache.read().await;
            cache.get(plugin_name)
                .cloned()
                .ok_or_else(|| anyhow!("Plugin '{}' not found in cache", plugin_name))?
        };

        let engine = self.engine.clone();

        tokio::task::spawn_blocking(move || {
            let mut store = Store::new(&engine, ());
            let linker = Linker::new(&engine);

            // Zero-Overhead Instanziierung
            let instance = linker.instantiate(&mut store, &module)
                .map_err(|e| anyhow!("WASM Instantiation failed: {}", e))?;

            // 1. WASM Exports abrufen (Memory und Funktionen)
            let memory = instance.get_memory(&mut store, "memory")
                .ok_or_else(|| anyhow!("WASM module must export 'memory'"))?;

            let alloc_func = instance.get_typed_func::<u32, u32>(&mut store, "alloc")
                .map_err(|_| anyhow!("WASM module must export 'alloc(u32) -> u32'"))?;

            let process_func = instance.get_typed_func::<(u32, u32), u64>(&mut store, "process")
                .map_err(|_| anyhow!("WASM module must export 'process(u32, u32) -> u64'"))?;

            // 2. Speicher in der Sandbox allozieren
            let input_len = input_data.len() as u32;
            let input_ptr = alloc_func.call(&mut store, input_len)
                .map_err(|e| anyhow!("Failed to allocate memory in WASM: {}", e))?;

            // 3. Daten vom Host (Server) in die WASM-Sandbox kopieren
            memory.write(&mut store, input_ptr as usize, &input_data)
                .map_err(|e| anyhow!("Failed to write to WASM memory: {}", e))?;

            // 4. KI- oder Datenverarbeitungs-Logik ausführen
            // Die WASM-Funktion gibt einen u64 zurück (die oberen 32 Bit sind der Pointer, die unteren 32 Bit die Länge)
            let result_packed = process_func.call(&mut store, (input_ptr, input_len))
                .map_err(|e| anyhow!("Plugin execution failed: {}", e))?;

            let result_ptr = (result_packed >> 32) as u32;
            let result_len = (result_packed & 0xFFFFFFFF) as u32;

            // 5. Ergebnis aus der Sandbox zurück zum Host kopieren
            let mut output_data = vec![0u8; result_len as usize];
            memory.read(&mut store, result_ptr as usize, &mut output_data)
                .map_err(|e| anyhow!("Failed to read from WASM memory: {}", e))?;

            // Optional: Hier könnte man noch eine `dealloc` Funktion aufrufen,
            // da die Instance aber nach diesem Block zerstört wird, wird der Speicher ohnehin freigegeben.

            Ok(output_data)
        })
            .await
            .map_err(|e| anyhow!("Task Panicked during WASM execution: {}", e))?
    }
}