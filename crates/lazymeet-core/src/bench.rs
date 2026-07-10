//! Built-in hardware benchmark: measures the pipeline stages on every
//! backend available on this machine (CPU vs CoreML for the ONNX models,
//! GPU vs CPU for the summarizer LLM) so users can verify the defaults on
//! their own hardware. Results stream to the UI as `benchmark-progress`
//! events; one run at a time, and never while a meeting is processing.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{ApiError, ApiResult};
use crate::Ctx;

/// Kick off a benchmark in a background thread. Progress arrives as
/// `benchmark-progress` events: `{status: "running", step, rows}` after each
/// measurement, then `{status: "done", rows}` or `{status: "error", message}`.
pub fn start(ctx: &Arc<Ctx>) -> ApiResult<Value> {
    if ctx.fake {
        return Err(ApiError::conflict("benchmark is not available in FAKE mode"));
    }
    if !crate::models::is_ready(ctx) {
        return Err(ApiError::conflict("models are still downloading"));
    }
    if crate::jobs::is_busy(ctx) {
        return Err(ApiError::conflict(
            "a recording is being processed — run the benchmark once it finishes",
        ));
    }
    if ctx.bench_running.swap(true, Ordering::SeqCst) {
        return Err(ApiError::conflict("a benchmark is already running"));
    }

    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&ctx)));
        if let Err(_panic) = result {
            ctx.events.emit(
                "benchmark-progress",
                json!({"status": "error", "message": "benchmark crashed"}),
            );
        }
        ctx.bench_running.store(false, Ordering::SeqCst);
    });
    Ok(json!({"started": true}))
}

#[cfg(feature = "native-pipeline")]
fn run(ctx: &Arc<Ctx>) {
    let threads = ctx.config.read().unwrap().num_threads() as i32;
    let models_dir = ctx.paths.models_dir.clone();
    let mut rows: Vec<Value> = Vec::new();

    let emit = |rows: &Vec<Value>, step: &str| {
        ctx.events.emit(
            "benchmark-progress",
            json!({"status": "running", "step": step, "rows": rows}),
        );
    };

    // ONNX stages per provider. CoreML only exists on macOS.
    let mut providers = vec![("cpu", "CPU")];
    if cfg!(target_os = "macos") {
        providers.push(("coreml", "CoreML (ANE)"));
    }
    for (provider, label) in providers {
        emit(&rows, &format!("ONNX models on {label}…"));
        match crate::pipeline::native::bench_onnx(&models_dir, threads, provider) {
            Ok(stage_rows) => {
                for (stage, secs, audio_secs) in stage_rows {
                    rows.push(json!({
                        "stage": stage,
                        "backend": label,
                        "seconds": (secs * 10.0).round() / 10.0,
                        "speed": format!("{:.1}x realtime", audio_secs / secs),
                    }));
                }
            }
            Err(e) => rows.push(json!({
                "stage": "ONNX models",
                "backend": label,
                "error": e,
            })),
        }
        emit(&rows, "");
    }

    // Summarizer LLM: GPU first (default), then CPU. On builds without a GPU
    // backend (Windows/Linux) the "GPU" load silently runs on the CPU, so
    // only offer both variants where Metal exists.
    let mut llm_variants = vec![(false, "CPU")];
    if cfg!(target_os = "macos") {
        llm_variants.insert(0, (true, "Metal GPU"));
    }
    for (gpu, label) in llm_variants {
        emit(&rows, &format!("Summarizer LLM on {label}…"));
        match crate::pipeline::native::bench_llm(&models_dir, threads, gpu, 384) {
            Ok((load_s, tok_per_s)) => rows.push(json!({
                "stage": "Summarizer LLM",
                "backend": label,
                "seconds": (load_s * 10.0).round() / 10.0,
                "speed": format!("{tok_per_s:.1} tokens/s"),
            })),
            Err(e) => rows.push(json!({
                "stage": "Summarizer LLM",
                "backend": label,
                "error": e,
            })),
        }
        emit(&rows, "");
    }

    ctx.events.emit("benchmark-progress", json!({"status": "done", "rows": rows}));
}

#[cfg(not(feature = "native-pipeline"))]
fn run(ctx: &Arc<Ctx>) {
    ctx.events.emit(
        "benchmark-progress",
        json!({"status": "error", "message": "this build has no native pipeline"}),
    );
}
