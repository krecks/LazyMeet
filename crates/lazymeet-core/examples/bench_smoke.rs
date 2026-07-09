//! Drive the built-in hardware benchmark headless (what the app's gauge
//! button runs) and print each progress event:
//!   cargo run --release -p lazymeet-core --example bench_smoke

use std::sync::atomic::Ordering;
use std::sync::Arc;

use lazymeet_core::events::Events;
use lazymeet_core::{bench, Ctx};

struct PrintEvents;

impl Events for PrintEvents {
    fn emit(&self, event: &str, payload: serde_json::Value) {
        if event == "benchmark-progress" {
            println!("{payload}");
        }
    }
}

fn main() {
    let ctx = Ctx::from_env(Arc::new(PrintEvents));
    ctx.paths.ensure_dirs().expect("data dirs");
    bench::start(&ctx).expect("benchmark should start");
    while ctx.bench_running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}
