//! Built-in hardware benchmark: measures the pipeline stages on every
//! backend available on this machine (CPU vs CoreML for the ONNX models,
//! GPU vs CPU for the summarizer LLM) so users can verify the defaults on
//! their own hardware. Results stream to the UI as `benchmark-progress`
//! events; one run at a time, and never while a meeting is processing.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{ApiError, ApiResult};
use crate::Ctx;

/// Kick off a benchmark in a background thread. `source` is an optional
/// user-picked recording; None benchmarks the deterministic synthetic audio.
/// Progress arrives as `benchmark-progress` events: `{status: "running",
/// step, rows}` after each measurement, then `{status: "done", rows,
/// recommendation, current}` or `{status: "error", message}`.
pub fn start(ctx: &Arc<Ctx>, source: Option<PathBuf>) -> ApiResult<Value> {
    if let Some(path) = &source {
        if !path.is_file() {
            return Err(ApiError::invalid("file not found"));
        }
    }
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
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&ctx, source)));
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
fn run(ctx: &Arc<Ctx>, source: Option<PathBuf>) {
    let (threads, diar_model) = {
        let config = ctx.config.read().unwrap();
        (config.num_threads() as i32, config.diar_model())
    };
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let models_dir = ctx.paths.models_dir.clone();
    let mut rows: Vec<Value> = Vec::new();

    let source_label = source
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "synthetic audio".to_string());

    let emit = |rows: &Vec<Value>, step: &str| {
        ctx.events.emit(
            "benchmark-progress",
            json!({
                "status": "running",
                "step": step,
                "rows": rows,
                "threads": threads,
                "cores": cores,
                "source": source_label,
            }),
        );
    };

    // Benchmark audio: the user's recording (first 300 s, decoded locally)
    // or the deterministic synthetic buffer.
    emit(&rows, "Preparing audio…");
    let samples = match &source {
        Some(path) => {
            let Some(ffmpeg) = ctx.ffmpeg.as_deref() else {
                ctx.events.emit(
                    "benchmark-progress",
                    json!({"status": "error", "message": "ffmpeg is not available"}),
                );
                return;
            };
            match crate::pipeline::native::bench_audio_from_file(ffmpeg, path, &ctx.paths.data_dir)
            {
                Ok(samples) => samples,
                Err(e) => {
                    ctx.events.emit(
                        "benchmark-progress",
                        json!({"status": "error", "message": format!("could not decode the recording: {e}")}),
                    );
                    return;
                }
            }
        }
        None => crate::pipeline::native::bench_audio_synth(),
    };

    // ONNX stages per provider this build offers (currently the CPU only).
    for provider in crate::config::onnx_provider_options() {
        let label = match provider {
            "coreml" => "CoreML (ANE)",
            _ => "CPU",
        };
        emit(&rows, &format!("ONNX models on {label}…"));
        match crate::pipeline::native::bench_onnx(
            &models_dir,
            threads,
            provider,
            &diar_model,
            &samples,
        ) {
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
        // A 2048-token transcript (~10 minutes of meeting) and a 384-token
        // answer: the time column is reading plus answering, which is what a
        // summary costs; load time is excluded (it mostly measures the disk).
        match crate::pipeline::native::bench_llm(&models_dir, threads, gpu, 2048, 384) {
            Ok(b) => rows.push(json!({
                "stage": "Summarizer LLM",
                "backend": label,
                "seconds": ((b.prefill_secs + b.generate_secs) * 10.0).round() / 10.0,
                "speed": format!(
                    "reads {:.0} tokens/s, writes {:.1} tokens/s",
                    b.prompt_tokens as f64 / b.prefill_secs,
                    b.generated_tokens as f64 / b.generate_secs,
                ),
            })),
            Err(e) => rows.push(json!({
                "stage": "Summarizer LLM",
                "backend": label,
                "error": e,
            })),
        }
        emit(&rows, "");
    }

    // Fastest backend per stage → a ready-to-apply config recommendation.
    let fastest_onnx = |stage: &str| -> &'static str {
        let best = rows.iter().filter(|r| r["stage"] == stage && r.get("error").is_none()).min_by(
            |a, b| {
                let a = a["seconds"].as_f64().unwrap_or(f64::MAX);
                let b = b["seconds"].as_f64().unwrap_or(f64::MAX);
                a.total_cmp(&b)
            },
        );
        match best.and_then(|r| r["backend"].as_str()) {
            Some("CoreML (ANE)") => "coreml",
            _ => "cpu",
        }
    };
    // The LLM winner is the faster summary workload (reading + answering).
    let fastest_llm_gpu = rows
        .iter()
        .filter(|r| r["stage"] == "Summarizer LLM" && r.get("error").is_none())
        .min_by(|a, b| {
            let a = a["seconds"].as_f64().unwrap_or(f64::MAX);
            let b = b["seconds"].as_f64().unwrap_or(f64::MAX);
            a.total_cmp(&b)
        })
        .map(|r| r["backend"] == "Metal GPU")
        .unwrap_or(true);
    let recommendation = json!({
        "ASR_PROVIDER": fastest_onnx("Transcription"),
        "DIAR_PROVIDER": fastest_onnx("Diarization"),
        "SPEAKER_PROVIDER": fastest_onnx("Speaker ID"),
        "SUMMARY_PROVIDER": if fastest_llm_gpu { "gpu" } else { "cpu" },
    });
    let current = {
        let config = ctx.config.read().unwrap();
        json!({
            "ASR_PROVIDER": config.asr_provider(),
            "DIAR_PROVIDER": config.diar_provider(),
            "SPEAKER_PROVIDER": config.speaker_provider(),
            "SUMMARY_PROVIDER": config.summary_provider(),
        })
    };

    ctx.events.emit(
        "benchmark-progress",
        json!({
            "status": "done",
            "rows": rows,
            "threads": threads,
            "cores": cores,
            "source": source_label,
            "recommendation": recommendation,
            "current": current,
        }),
    );
}

#[cfg(not(feature = "native-pipeline"))]
fn run(ctx: &Arc<Ctx>, _source: Option<PathBuf>) {
    ctx.events.emit(
        "benchmark-progress",
        json!({"status": "error", "message": "this build has no native pipeline"}),
    );
}
