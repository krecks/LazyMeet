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
    let mut ctx = Ctx::from_env(Arc::new(PrintEvents));
    if let Some(p) = std::env::var_os("LAZYMEET_FFMPEG") {
        Arc::get_mut(&mut ctx).unwrap().ffmpeg = Some(p.into());
    }
    ctx.paths.ensure_dirs().expect("data dirs");
    let source = std::env::args().nth(1).map(Into::into);
    bench::start(&ctx, source).expect("benchmark should start");
    while ctx.bench_running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}
