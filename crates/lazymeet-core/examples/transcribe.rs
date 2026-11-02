//! Run the ASR stage alone — for comparing transcription models.
//!
//! Recording mode (VAD + worker pool, exactly the pipeline's path):
//!   cargo run --release -p lazymeet-core --example transcribe -- <wav> \
//!       [--models-dir DIR] [--model ID] [--language CODE] [--threads N]
//!   → prints the segments as JSON, timing on stderr.
//!
//! Clip mode (`--clips LIST`, one 16 kHz WAV path per line): decodes each
//! clip whole and prints one `{"wav", "text"}` JSON object per line — for
//! word-error-rate evaluation on labelled test sets such as FLEURS.

use std::path::PathBuf;
use std::time::Instant;

use lazymeet_core::config::{Config, Paths};
use lazymeet_core::pipeline::{native, PipelineCtx};
use serde_json::json;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut wav: Option<PathBuf> = None;
    let mut clips: Option<PathBuf> = None;
    let mut models_dir = Paths::from_env().models_dir;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut threads: i32 = 7;
    while let Some(arg) = args.next() {
        let mut value = || args.next().expect("missing value");
        match arg.as_str() {
            "--models-dir" => models_dir = value().into(),
            "--model" => model = Some(value()),
            "--language" => language = Some(value()),
            "--threads" => threads = value().parse::<i32>().expect("number").max(1),
            "--clips" => clips = Some(value().into()),
            other => wav = Some(other.into()),
        }
    }

    // Default settings, not the user's config.json.
    let mut config = Config::resolve(&Paths::at("/nonexistent"));
    if let Some(model) = &model {
        config.override_value("ASR_MODEL", json!(model));
    }
    if let Some(language) = &language {
        config.override_value("ASR_LANGUAGE", json!(language));
    }
    let ctx = PipelineCtx { fake: false, config: &config, models_dir: &models_dir, ffmpeg: None };

    if let Some(list) = clips {
        let paths: Vec<PathBuf> = std::fs::read_to_string(list)
            .expect("clip list")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(PathBuf::from)
            .collect();
        let t = Instant::now();
        let texts = native::transcribe_clips(&ctx, &paths, threads as usize).expect("transcribe");
        eprintln!("{} clips in {:.1}s", paths.len(), t.elapsed().as_secs_f64());
        for (p, text) in paths.iter().zip(texts) {
            println!("{}", json!({"wav": p, "text": text}));
        }
        return;
    }

    let wav = wav.expect("usage: transcribe <wav> [options] | --clips LIST");
    let t = Instant::now();
    let segments = native::transcribe(&ctx, &wav, None, threads).expect("transcribe");
    eprintln!("{} segments in {:.1}s", segments.len(), t.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string(&segments).unwrap());
}
