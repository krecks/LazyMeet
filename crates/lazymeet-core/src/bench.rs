//! Built-in benchmark: runs the selected models over five minutes of audio
//! the way a meeting runs them, and estimates how long an hour of meeting
//! takes on this machine. The summary is timed on every backend this build
//! offers (Metal GPU and CPU on macOS), which is the one choice left to
//! recommend. Results stream to the UI as `benchmark-progress` events; one
//! run at a time, and meetings wait while it runs.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{ApiError, ApiResult};
use crate::Ctx;

/// Kick off a benchmark in a background thread. `source` is an optional
/// user-picked recording; None benchmarks the deterministic synthetic audio.
/// Progress arrives as `benchmark-progress` events: `{status: "running",
/// step, rows, models}` after each measurement, then `{status: "done", rows,
/// models, estimate, recommendation, current}` or `{status: "error",
/// message}`. `estimate` (seconds per hour of meeting) is null when a stage
/// could not be timed.
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
    if let Some(reason) = crate::jobs::claim_for_benchmark(ctx) {
        return Err(ApiError::conflict(reason));
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
        crate::jobs::release_from_benchmark(&ctx);
    });
    Ok(json!({"started": true}))
}

/// Transcript tokens per hour of meeting: 5987 tokens for a 28.5-minute
/// two-person meeting (Qwen 3.5 tokenizer), scaled to 60 minutes.
#[cfg(feature = "native-pipeline")]
const TOKENS_PER_HOUR: f64 = 12_600.0;
/// A typical summary answer (summary + takeaways + action items), tokens.
#[cfg(feature = "native-pipeline")]
const ANSWER_TOKENS: f64 = 600.0;

#[cfg(feature = "native-pipeline")]
fn run(ctx: &Arc<Ctx>, source: Option<PathBuf>) {
    use crate::config::{self, Config};
    use crate::models;
    use crate::pipeline::{native, PipelineCtx};
    use std::time::Instant;

    // One settings snapshot for the whole run: the models and threads a
    // meeting started now would use.
    let config = Config::resolve(&ctx.paths);
    let threads = config.num_threads() as i32;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let models_dir = ctx.paths.models_dir.clone();
    let pctx = PipelineCtx {
        fake: false,
        config: &config,
        models_dir: &models_dir,
        ffmpeg: ctx.ffmpeg.clone(),
    };

    let asr = models::effective_asr_entry(&models_dir, false, &config.asr_model());
    // As diarize() resolves it: no meeting count, so the env default.
    let diar = models::effective_diar_entry(
        &models_dir,
        false,
        &config.diar_model(),
        config::env_num_speakers(),
    );
    let llm = models::effective_llm_entry(&models_dir, false, &config.summary_model());
    let llm_path = models_dir.join(&llm.path);
    let (diar_name, llm_name) = (diar.short_name(), llm.short_name());
    let used = json!({"asr": asr.short_name(), "diar": diar_name, "summary": llm_name});

    let source_label = source
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "synthetic audio".to_string());
    let mut rows: Vec<Value> = Vec::new();
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
                "models": used,
            }),
        );
    };
    let fail = |message: String| {
        ctx.events.emit("benchmark-progress", json!({"status": "error", "message": message}));
    };
    let round1 = |secs: f64| (secs * 10.0).round() / 10.0;

    emit(&rows, "Preparing audio…");
    let (wav, audio_secs) = match native::bench_wav(&pctx, source.as_deref(), &ctx.paths.data_dir) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };
    let realtime = |secs: f64| format!("{:.1}x realtime", audio_secs / secs);

    // Transcription and speaker separation, exactly as a meeting runs them:
    // concurrently, with the meeting's thread split, VAD included. Models
    // that write text token by token spend nothing on noise, so they are
    // timed only on a real recording.
    let token_by_token = asr.kind != "nemo_transducer";
    let mut per_audio_sec = None; // transcription + separation wall time per audio second
    let mut turns = Vec::new();
    if source.is_none() && token_by_token {
        rows.push(json!({
            "stage": "Transcription + speaker separation",
            "backend": "CPU",
            "model": asr.short_name(),
            "error": format!(
                "{} is timed on one of your recordings only — on synthetic audio its speed means nothing",
                asr.short_name()
            ),
        }));
    } else {
        emit(&rows, "Transcribing and separating speakers…");
        // One row: separation's threads join transcription once it is done,
        // so both finish together.
        let t = Instant::now();
        match native::transcribe_and_diarize(&pctx, &wav, -1, None, || {}) {
            Ok((_, found)) => {
                let total = t.elapsed().as_secs_f64();
                turns = found;
                per_audio_sec = Some(total / audio_secs);
                rows.push(json!({
                    "stage": "Transcription + speaker separation",
                    "backend": "CPU",
                    "model": format!("{} + {diar_name}", asr.short_name()),
                    "seconds": round1(total),
                    "speed": realtime(total),
                }));
            }
            Err(e) => rows.push(json!({
                "stage": "Transcription + speaker separation",
                "backend": "CPU",
                "model": asr.short_name(),
                "error": e,
            })),
        }
    }
    emit(&rows, "");

    // Voice matching: one voiceprint per separated speaker.
    let mut voice_secs = 0.0;
    if !turns.is_empty() {
        emit(&rows, "Matching voices…");
        let t = Instant::now();
        match native::compute_voiceprints(&pctx, &wav, &turns) {
            Ok(prints) => {
                voice_secs = t.elapsed().as_secs_f64();
                rows.push(json!({
                    "stage": "Voice matching",
                    "backend": "CPU",
                    "model": "TitaNet",
                    "seconds": round1(voice_secs),
                    "speed": match prints.len() {
                        1 => "1 speaker".to_string(),
                        n => format!("{n} speakers"),
                    },
                }));
            }
            Err(e) => rows.push(json!({"stage": "Voice matching", "backend": "CPU", "error": e})),
        }
        emit(&rows, "");
    }
    let _ = std::fs::remove_file(&wav);

    // Summary: the selected model on each backend this build offers (Metal
    // GPU and CPU on macOS). A 2048-token transcript and a 384-token answer
    // keep the run short; reading and writing speed scale to a meeting.
    let current = config.summary_provider();
    let mut summary: Vec<(String, f64)> = Vec::new(); // (provider, secs per meeting hour)
                                                      // GPU (the default where it exists) first.
    let mut providers = config::summary_provider_options();
    providers.sort_by_key(|p| *p != "gpu");
    for provider in providers {
        let label = if provider == "gpu" { "GPU" } else { "CPU" };
        emit(&rows, &format!("Summarizing on the {label}…"));
        match native::bench_llm(&llm_path, threads, provider == "gpu", 2048, 384) {
            Ok(b) => {
                let read = b.prompt_tokens as f64 / b.prefill_secs;
                let write = b.generated_tokens as f64 / b.generate_secs;
                summary
                    .push((provider.to_string(), TOKENS_PER_HOUR / read + ANSWER_TOKENS / write));
                rows.push(json!({
                    "stage": "Summary",
                    "backend": label,
                    "provider": provider,
                    "model": llm_name,
                    "seconds": round1(b.prefill_secs + b.generate_secs),
                    "speed": format!("reads {read:.0} tokens/s, writes {write:.1} tokens/s"),
                }));
            }
            Err(e) => rows.push(json!({
                "stage": "Summary",
                "backend": label,
                "provider": provider,
                "model": llm_name,
                "error": e,
            })),
        }
        emit(&rows, "");
    }

    let recommendation = recommend(&summary);
    // One hour of meeting with the current settings.
    let summary_secs = summary.iter().find(|(p, _)| *p == current).map(|(_, s)| *s);
    let estimate = match (per_audio_sec, summary_secs) {
        (Some(rate), Some(summary_secs)) => {
            let transcription = rate * 3600.0;
            json!({
                "per_hour_secs": (transcription + voice_secs + summary_secs).round(),
                "transcription_secs": transcription.round(),
                "voice_secs": voice_secs.round(),
                "summary_secs": summary_secs.round(),
            })
        }
        _ => Value::Null,
    };

    ctx.events.emit(
        "benchmark-progress",
        json!({
            "status": "done",
            "rows": rows,
            "threads": threads,
            "cores": cores,
            "source": source_label,
            "models": used,
            "estimate": estimate,
            "recommendation": recommendation,
            "current": {"SUMMARY_PROVIDER": current},
        }),
    );
}

/// The summary backend to recommend from `(provider, seconds)` timings: the
/// fastest, and only when at least two backends measured — a single or
/// failed measurement is no choice.
#[cfg_attr(not(feature = "native-pipeline"), allow(dead_code))]
fn recommend(summary: &[(String, f64)]) -> Value {
    if summary.len() < 2 {
        return json!({});
    }
    let fastest = summary.iter().min_by(|a, b| a.1.total_cmp(&b.1)).map(|(p, _)| p.as_str());
    json!({"SUMMARY_PROVIDER": fastest})
}

#[cfg(not(feature = "native-pipeline"))]
fn run(ctx: &Arc<Ctx>, _source: Option<PathBuf>) {
    ctx.events.emit(
        "benchmark-progress",
        json!({"status": "error", "message": "this build has no native pipeline"}),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommends_only_a_measured_choice() {
        assert_eq!(recommend(&[]), json!({}));
        // One backend (Windows/Linux, or the other one failed): no choice.
        assert_eq!(recommend(&[("cpu".into(), 90.0)]), json!({}));
        assert_eq!(
            recommend(&[("gpu".into(), 45.0), ("cpu".into(), 160.0)]),
            json!({"SUMMARY_PROVIDER": "gpu"})
        );
        assert_eq!(
            recommend(&[("gpu".into(), 200.0), ("cpu".into(), 160.0)]),
            json!({"SUMMARY_PROVIDER": "cpu"})
        );
    }
}
