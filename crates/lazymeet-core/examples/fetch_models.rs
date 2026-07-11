//! Download the configured model set headlessly (CI benchmarking):
//!   cargo run --release -p lazymeet-core --example fetch_models
//! Exits 0 once every model the pipeline needs is on disk.

use std::sync::Arc;

use lazymeet_core::{models, Ctx};

struct PrintEvents;

impl lazymeet_core::events::Events for PrintEvents {
    fn emit(&self, event: &str, payload: serde_json::Value) {
        if event == "models-progress" {
            println!("{payload}");
        }
    }
}

fn main() {
    let ctx = Ctx::from_env(Arc::new(PrintEvents));
    ctx.paths.ensure_dirs().expect("data dirs");
    println!("models dir: {}", ctx.paths.models_dir.display());
    models::ensure_models_async(&ctx);
    // ~3.2 GB total; a stuck download should fail the CI job, not hang it.
    for _ in 0..600 {
        if models::is_ready(&ctx) {
            println!("all models ready");
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    eprintln!("timed out waiting for model downloads");
    std::process::exit(1);
}
