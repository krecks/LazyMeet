//! Process one recording end-to-end from the terminal and print the meeting
//! JSON — the parity/verification tool for the Rust pipeline.
//!
//! Usage:
//!   lazymeet-cli <recording> [--data-dir DIR] [--models-dir DIR]
//!                [--num-speakers N] [--ffmpeg PATH]
//!
//! Defaults to the real app data dir (and its downloaded models), so results
//! are directly comparable with what the app produces. Stage transitions are
//! printed as they happen.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lazymeet_core::config::Paths;
use lazymeet_core::events::Events;
use lazymeet_core::{api, models, Ctx};

/// Prints each stage transition with elapsed seconds since the previous one,
/// so per-stage durations can be compared across runs (providers, threads…).
struct StderrEvents(std::sync::Mutex<Instant>);

impl Events for StderrEvents {
    fn emit(&self, event: &str, payload: serde_json::Value) {
        if event == "meeting-updated" {
            let mut last = self.0.lock().unwrap();
            let elapsed = last.elapsed();
            *last = Instant::now();
            eprintln!(
                "[{:>12}] +{:6.2}s {}",
                payload["status"].as_str().unwrap_or("?"),
                elapsed.as_secs_f64(),
                payload["id"]
            );
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut recording: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut models_dir: Option<PathBuf> = None;
    let mut ffmpeg: Option<PathBuf> = None;
    let mut num_speakers: i64 = -1;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => data_dir = args.next().map(PathBuf::from),
            "--models-dir" => models_dir = args.next().map(PathBuf::from),
            "--ffmpeg" => ffmpeg = args.next().map(PathBuf::from),
            "--num-speakers" => {
                num_speakers = args.next().and_then(|v| v.parse().ok()).unwrap_or(-1)
            }
            other if recording.is_none() => recording = Some(PathBuf::from(other)),
            other => {
                eprintln!("unexpected argument: {other}");
                std::process::exit(2);
            }
        }
    }
    let Some(recording) = recording else {
        eprintln!(
            "usage: lazymeet-cli <recording> [--data-dir DIR] [--models-dir DIR] \
             [--num-speakers N] [--ffmpeg PATH]"
        );
        std::process::exit(2);
    };

    let mut paths = match data_dir {
        Some(dir) => Paths::at(dir),
        None => Paths::from_env(),
    };
    if let Some(dir) = models_dir {
        paths.models_dir = dir;
    }
    let ffmpeg = ffmpeg.or_else(find_ffmpeg);
    eprintln!("data:   {}", paths.data_dir.display());
    eprintln!("models: {}", paths.models_dir.display());
    eprintln!("ffmpeg: {}", ffmpeg.as_deref().map(|p| p.display().to_string()).unwrap_or("NOT FOUND".into()));

    let mut ctx = Ctx::new(
        paths,
        lazymeet_core::config::fake_mode_from_env(),
        Arc::new(StderrEvents(std::sync::Mutex::new(Instant::now()))),
    );
    Arc::get_mut(&mut ctx).unwrap().ffmpeg = ffmpeg;
    ctx.paths.ensure_dirs().expect("could not create data dirs");
    if !models::is_ready(&ctx) {
        eprintln!("models are missing under {} — run the app once or pass --models-dir", ctx.paths.models_dir.display());
        std::process::exit(1);
    }

    let started = Instant::now();
    let res = api::create_meeting(&ctx, &recording, num_speakers, None, None).unwrap_or_else(|e| {
        eprintln!("create_meeting failed: {e}");
        std::process::exit(1);
    });
    let id = res["id"].as_str().unwrap().to_string();

    loop {
        let meeting = api::get_meeting(&ctx, &id).expect("meeting exists");
        let status = meeting["status"].as_str().unwrap_or("");
        if status == "done" || status == "error" {
            eprintln!("finished in {:.1?}", started.elapsed());
            println!("{}", serde_json::to_string_pretty(&meeting).unwrap());
            std::process::exit(if status == "done" { 0 } else { 1 });
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Dev convenience: prefer a bundled/env ffmpeg, else PATH, else the
/// imageio-ffmpeg binary inside the repo venv (same one Python used).
fn find_ffmpeg() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("LAZYMEET_FFMPEG") {
        return Some(PathBuf::from(p));
    }
    if let Ok(output) = std::process::Command::new("which").arg("ffmpeg").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    None
}
