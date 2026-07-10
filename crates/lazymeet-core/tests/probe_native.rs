//! probe_media against the real bundled ffmpeg: exact duration from the
//! decoded sample count, peak buckets that reflect where the audio is loud.
#![cfg(feature = "native-pipeline")]

mod common;

use std::io::Write;

use common::TestApp;
use lazymeet_core::config::Config;
use lazymeet_core::pipeline::{self, PipelineCtx};

/// 2 s of 16 kHz mono 16-bit WAV: silence with a 0.5-amplitude tone burst in
/// the middle (0.8–1.2 s).
fn write_test_wav(path: &std::path::Path) {
    const RATE: u32 = 16000;
    let samples: Vec<i16> = (0..RATE * 2)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            if (0.8..1.2).contains(&t) {
                (16384.0 * (t * 440.0 * std::f64::consts::TAU).sin()) as i16
            } else {
                0
            }
        })
        .collect();
    let data_len = (samples.len() * 2) as u32;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(36 + data_len).to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&RATE.to_le_bytes()).unwrap();
    f.write_all(&(RATE * 2).to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_len.to_le_bytes()).unwrap();
    for s in samples {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
}

fn bundled_ffmpeg() -> Option<std::path::PathBuf> {
    let triple = format!(
        "{}-{}",
        std::env::consts::ARCH,
        match std::env::consts::OS {
            "macos" => "apple-darwin",
            "windows" => "pc-windows-msvc",
            _ => "unknown-linux-gnu",
        }
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src-tauri/binaries")
        .join(format!("ffmpeg-{triple}{}", std::env::consts::EXE_SUFFIX));
    path.is_file().then_some(path)
}

#[test]
fn probe_reads_duration_and_peaks_through_real_ffmpeg() {
    let Some(ffmpeg) = bundled_ffmpeg() else {
        eprintln!("bundled ffmpeg not present for this platform — skipping");
        return;
    };
    let app = TestApp::fake(); // only used for paths/config
    let wav = app.tmp.path().join("probe.wav");
    write_test_wav(&wav);

    let config = Config::resolve(&app.ctx.paths);
    let pctx = PipelineCtx {
        fake: false, // force the native probe despite the fake test app
        config: &config,
        models_dir: &app.ctx.paths.models_dir,
        ffmpeg: Some(ffmpeg),
    };
    let probe = pipeline::probe_media(&pctx, &wav).expect("probe");

    // Duration comes from the decoded sample count.
    assert!((probe.duration - 2.0).abs() < 0.05, "duration = {}", probe.duration);
    // 2 s at one bucket per 100 ms.
    assert_eq!(probe.peaks.len(), 20);
    // The tone burst shows up where it is, silence stays silent.
    assert!(probe.peaks[0] < 0.05, "leading silence loud: {}", probe.peaks[0]);
    let max = probe.peaks.iter().cloned().fold(0.0f32, f32::max);
    assert!(max > 0.3, "burst not visible in peaks: {max}");

    // A file with no decodable audio errors instead of returning zeros.
    let bogus = app.tmp.path().join("not-media.bin");
    std::fs::write(&bogus, b"definitely not audio").unwrap();
    let err = pipeline::probe_media(&pctx, &bogus).unwrap_err();
    assert!(err.contains("ffmpeg failed"), "unexpected error: {err}");
}
