use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_appender::non_blocking::WorkerGuard;
use std::path::Path;

pub fn init_telemetry(log_dir: &str, file_name: &str) -> WorkerGuard {
    if !Path::new(log_dir).exists() {
        let _ = std::fs::create_dir_all(log_dir);
    }

    let file_appender = tracing_appender::rolling::daily(log_dir, file_name);

    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .json() // Strukturiertes JSON
        .with_file(true)
        .with_line_number(true)
        .with_target(false)
        .with_current_span(false)
        .with_span_list(false);

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info,pytja_core=debug")) // Core debuggen wir genauer
        .with(file_layer)
        .init();

    guard
}