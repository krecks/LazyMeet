//! Diarize one 16 kHz mono WAV with the pipeline's diarization stage and
//! print the speaker turns as JSON — for comparing diarization engines on the
//! same recording:
//!   cargo run --release -p lazymeet-core --example diarize -- <wav> \
//!       [--models-dir DIR] [--model nemotron-3-diarization|pyannote-3.0] \
//!       [--num-speakers N] [--threads N]
//!
//! `--onnx FILE --dump-probs OUT` instead runs the Nemotron network in FILE
//! (e.g. the fp32 export) and writes its raw frame probabilities to OUT as
//! little-endian f32, `frames × 8` — for checking against the Hugging Face
//! transformers reference.

use std::path::PathBuf;
use std::time::Instant;

use lazymeet_core::config::{Config, Paths};
use lazymeet_core::pipeline::{native, nemotron, PipelineCtx};
use serde_json::json;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut wav: Option<PathBuf> = None;
    let mut models_dir = Paths::from_env().models_dir;
    let mut model: Option<String> = None;
    let mut num_speakers: i64 = -1;
    let mut threads: i32 = 4;
    let mut onnx: Option<PathBuf> = None;
    let mut dump: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        let mut value = || args.next().expect("missing value");
        match arg.as_str() {
            "--models-dir" => models_dir = value().into(),
            "--model" => model = Some(value()),
            "--num-speakers" => num_speakers = value().parse().expect("number"),
            "--threads" => threads = value().parse().expect("number"),
            "--onnx" => onnx = Some(value().into()),
            "--dump-probs" => dump = Some(value().into()),
            other => wav = Some(other.into()),
        }
    }
    let wav = wav.expect("usage: diarize <wav> [options]");

    if let (Some(onnx), Some(dump)) = (onnx, dump) {
        let samples: Vec<f32> = hound::WavReader::open(&wav)
            .expect("wav")
            .samples::<i16>()
            .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
            .collect();
        let t = Instant::now();
        let probs = nemotron::Nemotron::load(&onnx, threads)
            .and_then(|mut m| m.probabilities(&samples))
            .expect("nemotron");
        eprintln!(
            "{} frames in {:.1}s",
            probs.len() / nemotron::SPEAKERS,
            t.elapsed().as_secs_f64()
        );
        let bytes: Vec<u8> = probs.iter().flat_map(|p| p.to_le_bytes()).collect();
        std::fs::write(dump, bytes).expect("write probs");
        return;
    }

    // Default settings, not the user's config.json.
    let mut config = Config::resolve(&Paths::at("/nonexistent"));
    if let Some(model) = model {
        config.override_value("DIAR_MODEL", json!(model));
    }
    let ctx = PipelineCtx { fake: false, config: &config, models_dir: &models_dir, ffmpeg: None };
    let t = Instant::now();
    let turns = native::diarize(&ctx, &wav, num_speakers, None, threads).expect("diarize");
    let speakers: std::collections::BTreeSet<&str> =
        turns.iter().map(|t| t.speaker.as_str()).collect();
    eprintln!(
        "{} turns, {} speakers in {:.1}s",
        turns.len(),
        speakers.len(),
        t.elapsed().as_secs_f64()
    );
    println!("{}", serde_json::to_string(&turns).unwrap());
}
