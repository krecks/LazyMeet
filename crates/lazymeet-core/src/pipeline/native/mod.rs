//! Real ML pipeline: ffmpeg (bundled binary) → sherpa-onnx VAD + ASR →
//! diarization (Nemotron via onnxruntime, or classic pyannote in sherpa-onnx)
//! → speaker embeddings → llama.cpp summarization.
//!
//! Port of `backend/pipeline/{audio,transcribe,diarize,identify,summarize}.py`.
//! Models load per job and drop at the end — matching the memory behavior of
//! the old per-meeting worker process.
//!
//! This module owns ffmpeg I/O and the pipeline entry points; the heavy stages
//! live in submodules: `asr` (VAD + recognition), `diarize` (Nemotron /
//! pyannote), `llm` (summarization) and `bench` (benchmark primitives).

use std::num::NonZeroU32;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Condvar, Mutex, OnceLock};

use serde_json::{Map, Value};

use sherpa_onnx::{
    FastClusteringConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    OfflineTransducerModelConfig, OfflineWhisperModelConfig, SileroVadModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, VadModelConfig,
    VoiceActivityDetector,
};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
// `Special` is deprecated upstream but still the only token_to_bytes API.
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use super::nemotron::{self, Nemotron};
use super::{
    remap_turns, round3, MediaProbe, MergedSegment, PipelineCtx, Segment, Summary, Turn,
    Voiceprints,
};
use crate::config;
use crate::models::{resolve_asr_at, resolve_diar_at, AsrPaths};

mod asr;
mod bench;
mod diarize;
mod llm;

use asr::{load_wav, Transcription};
use diarize::{new_extractor, speaker_embedding};

pub use asr::transcribe;
pub use bench::{bench_audio_from_file, bench_audio_synth, bench_llm, bench_onnx, LlmBench};
pub use diarize::diarize;
pub use llm::summarize;

const SAMPLE_RATE: u32 = 16000;

// --------------------------------------------------------------------------
// ffmpeg (port of audio.py)
// --------------------------------------------------------------------------

fn ffmpeg_cmd(ctx: &PipelineCtx) -> Result<Command, String> {
    let path = ctx.ffmpeg.as_ref().ok_or("ffmpeg binary not configured")?;
    // `mut` is only used on Windows (cfg-gated below).
    #[allow(unused_mut)]
    let mut cmd = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    Ok(cmd)
}

fn run_ffmpeg(mut cmd: Command) -> Result<(), String> {
    let output = cmd.output().map_err(|e| format!("ffmpeg failed to start: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.lines().rev().take(4).collect::<Vec<_>>().join(" | ");
        return Err(format!("ffmpeg failed ({}): {tail}", output.status));
    }
    Ok(())
}

/// Decode any recording to a 16 kHz mono PCM WAV — and, when `playback` is
/// given, the 96 kbps AAC playback track in the same invocation (one decode
/// of the source instead of a second full pass at the end of the run).
pub fn extract_audio(
    ctx: &PipelineCtx,
    src: &Path,
    out_wav: &Path,
    playback: Option<&Path>,
) -> Result<(), String> {
    let build = |with_playback: bool| -> Result<Command, String> {
        let mut cmd = ffmpeg_cmd(ctx)?;
        cmd.arg("-y")
            .arg("-i")
            .arg(src)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "wav"])
            .arg(out_wav);
        if with_playback {
            if let Some(playback) = playback {
                cmd.args(["-vn", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"])
                    .arg(playback);
            }
        }
        Ok(cmd)
    };
    if playback.is_some() {
        // If the combined pass fails (e.g. an exotic container the AAC muxer
        // chokes on), fall back to the WAV alone — the meeting must not fail
        // over the optional playback track (retried at the end of the run).
        if run_ffmpeg(build(true)?).is_ok() {
            return Ok(());
        }
        if let Some(playback) = playback {
            let _ = std::fs::remove_file(playback);
        }
    }
    run_ffmpeg(build(false)?)
}

/// Extract the audio track to an AAC .m4a for playback. 96 kbps is transparent
/// for speech; +faststart moves the index up front so the webview can seek.
pub fn extract_playback_audio(ctx: &PipelineCtx, src: &Path, out_m4a: &Path) -> Result<(), String> {
    let mut cmd = ffmpeg_cmd(ctx)?;
    cmd.arg("-y")
        .arg("-i")
        .arg(src)
        .args(["-vn", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"])
        .arg(out_m4a);
    run_ffmpeg(cmd)
}

// --------------------------------------------------------------------------
// Media probe (duration + waveform peaks for the prep screen)
// --------------------------------------------------------------------------

/// Probe decode rate — waveform peaks don't need speech-band fidelity.
const PROBE_RATE: u32 = 8000;
/// One fine peak bucket per 100 ms of audio while streaming.
const FINE_BUCKET_SAMPLES: usize = 800;
/// At most this many peaks are handed to the UI — 100 ms resolution up to an
/// hour of audio, so the prep screen's zoom stays sharp on long recordings.
const MAX_PEAKS: usize = 36_000;

/// Duration + waveform peaks from a single ffmpeg decode. The duration comes
/// from the decoded sample count — exact, and no ffprobe is bundled. The PCM
/// is streamed (a 2 h recording decodes to ~115 MB) and folded into 100 ms
/// peak buckets on the fly, then down to at most [`MAX_PEAKS`] values.
pub fn probe_media(ctx: &PipelineCtx, src: &Path) -> Result<MediaProbe, String> {
    use std::io::Read;
    use std::process::Stdio;

    let mut cmd = ffmpeg_cmd(ctx)?;
    cmd.args(["-v", "error"])
        .arg("-i")
        .arg(src)
        .args(["-vn", "-ac", "1", "-ar"])
        .arg(PROBE_RATE.to_string())
        .args(["-f", "s16le", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("ffmpeg failed to start: {e}"))?;

    // Drain stderr on its own thread so a chatty ffmpeg can't deadlock.
    let mut stderr = child.stderr.take().expect("piped stderr");
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr.read_to_string(&mut buf);
        buf
    });

    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut fine: Vec<f32> = Vec::new();
    let mut peak = 0.0f32;
    let mut in_bucket = 0usize;
    let mut total_samples = 0u64;
    let mut push = |s: i16| {
        let v = (s as f32 / 32768.0).abs();
        if v > peak {
            peak = v;
        }
        in_bucket += 1;
        total_samples += 1;
        if in_bucket == FINE_BUCKET_SAMPLES {
            fine.push(peak);
            peak = 0.0;
            in_bucket = 0;
        }
    };
    let mut carry: Option<u8> = None; // odd-length read: half an i16
    let mut buf = [0u8; 65536];
    loop {
        let n = match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stderr_thread.join();
                return Err(format!("ffmpeg read failed: {e}"));
            }
        };
        let mut chunk = &buf[..n];
        if let Some(lo) = carry.take() {
            push(i16::from_le_bytes([lo, chunk[0]]));
            chunk = &chunk[1..];
        }
        let mut pairs = chunk.chunks_exact(2);
        for pair in &mut pairs {
            push(i16::from_le_bytes([pair[0], pair[1]]));
        }
        if let [lo] = pairs.remainder() {
            carry = Some(*lo);
        }
    }
    if in_bucket > 0 {
        fine.push(peak);
    }

    let status = child.wait().map_err(|e| format!("ffmpeg failed: {e}"))?;
    let stderr_text = stderr_thread.join().unwrap_or_default();
    if !status.success() {
        let tail: String = stderr_text.lines().rev().take(4).collect::<Vec<_>>().join(" | ");
        return Err(format!("ffmpeg failed ({status}): {tail}"));
    }
    if total_samples == 0 {
        return Err("no audio track found in this file".into());
    }
    let duration = total_samples as f64 / PROBE_RATE as f64;
    let group = fine.len().div_ceil(MAX_PEAKS).max(1);
    let peaks = fine
        .chunks(group)
        // 3 decimals is plenty for drawing and keeps the JSON payload small.
        .map(|c| (c.iter().copied().fold(0.0, f32::max) * 1000.0).round() / 1000.0)
        .collect();
    Ok(MediaProbe { duration, peaks })
}

// --------------------------------------------------------------------------
// VAD + ASR (port of transcribe.py)
// --------------------------------------------------------------------------

fn path_str(p: &Path) -> Option<String> {
    Some(p.to_string_lossy().into_owned())
}

/// Transcription and diarization read the same WAV with disjoint models, so
/// they run concurrently (see `super::transcribe_and_diarize`).
///
/// Diarization gets a third of the thread budget: Nemotron is most efficient
/// per core at low thread counts (2 threads take 77 s on a 28.5-minute
/// recording, 4 take 49 s, 7 take 59 s on an M3 Pro), and transcription is
/// the longer stage. When diarization finishes, its threads join the
/// transcription workers. The classic pyannote diarizer is far slower and
/// keeps half the budget.
pub fn transcribe_and_diarize(
    ctx: &PipelineCtx,
    wav: &Path,
    num_speakers: i64,
    ranges: Option<&[(f64, f64)]>,
    on_transcribed: impl FnOnce(),
) -> Result<(Vec<Segment>, Vec<Turn>), String> {
    let threads = ctx.config.num_threads().max(1) as usize;
    let diar_threads = match super::diar_engine(ctx, num_speakers) {
        "nemotron" => ((threads + 1) / 3).max(1),
        _ => threads.div_ceil(2).max(1),
    };
    let (transcription, cut) = Transcription::start(ctx, wav, ranges)?;
    let transcription = &transcription;
    std::thread::scope(|s| {
        s.spawn(cut);
        let diar = s.spawn(move || {
            let turns = diarize(ctx, wav, num_speakers, ranges, diar_threads as i32);
            std::thread::scope(|s| {
                for _ in 0..diar_threads {
                    s.spawn(|| transcription.work());
                }
            });
            turns
        });
        let segments = transcription.finish(threads.saturating_sub(diar_threads));
        on_transcribed();
        let turns = diar.join().unwrap_or_else(|_| Err("diarization panicked".into()));
        Ok((segments, turns?))
    })
}

/// One averaged voiceprint per diarized speaker, from its longest turns.
pub fn compute_voiceprints(
    ctx: &PipelineCtx,
    wav: &Path,
    turns: &[Turn],
) -> Result<Voiceprints, String> {
    let samples = load_wav(wav)?;
    let extractor = new_extractor(
        ctx.models_dir,
        &ctx.config.speaker_provider(),
        ctx.config.num_threads() as i32,
    )?;

    let mut speakers: Vec<&str> = Vec::new();
    for t in turns {
        if !speakers.contains(&t.speaker.as_str()) {
            speakers.push(&t.speaker);
        }
    }

    let mut voiceprints = Voiceprints::new();
    for spk in speakers {
        // Use each speaker's longest turns (more reliable embedding).
        let mut spk_turns: Vec<&Turn> = turns.iter().filter(|t| t.speaker == spk).collect();
        spk_turns.sort_by(|a, b| (b.end - b.start).total_cmp(&(a.end - a.start)));
        let mut embs: Vec<Vec<f32>> = Vec::new();
        for turn in spk_turns.iter().take(5) {
            let a = (turn.start * SAMPLE_RATE as f64) as usize;
            let b = ((turn.end * SAMPLE_RATE as f64) as usize).min(samples.len());
            if b.saturating_sub(a) < (SAMPLE_RATE as f64 * 0.3) as usize {
                continue; // skip < 0.3s
            }
            if let Some(e) = speaker_embedding(&extractor, &samples[a..b]) {
                embs.push(e);
            }
        }
        if !embs.is_empty() {
            let dim = embs[0].len();
            let mut mean = vec![0.0f64; dim];
            for e in &embs {
                for (m, v) in mean.iter_mut().zip(e) {
                    *m += *v as f64;
                }
            }
            for m in &mut mean {
                *m /= embs.len() as f64;
            }
            voiceprints.insert(spk.to_string(), mean);
        }
    }
    Ok(voiceprints)
}

#[cfg(test)]
mod tests {
    use super::llm::detect_language;

    #[test]
    fn detects_meeting_language() {
        let de = "Guten Morgen zusammen, fangen wir mit dem Protokoll der letzten Sitzung an \
            und gehen danach die offenen Punkte durch. Wer übernimmt die Abstimmung mit dem \
            Kunden bis Ende der Woche?";
        assert_eq!(detect_language(&de.repeat(3)), Some("de"));
        let en = "Good morning everyone, let's start with the minutes from last time and \
            then go through the open items. Who takes the customer follow-up by Friday?";
        assert_eq!(detect_language(&en.repeat(3)), Some("en"));
    }

    #[test]
    fn unreliable_detection_returns_none() {
        assert_eq!(detect_language("ok"), None);
    }

    /// Real-model smoke test for the language fix — needs the downloaded models:
    /// cargo test -p lazymeet-core --lib german_meeting -- --ignored --nocapture
    #[test]
    #[ignore = "runs the real summarizer; needs models on disk"]
    fn german_meeting_gets_german_notes() {
        let real_models = crate::config::Paths::from_env().models_dir;
        // Default config (SUMMARY_LANGUAGE=auto), untouched by the user's config.json.
        let paths = crate::config::Paths::at("/nonexistent");
        let config = crate::config::Config::resolve(&paths);
        let ctx = crate::pipeline::PipelineCtx {
            fake: false,
            config: &config,
            models_dir: &real_models,
            ffmpeg: None,
        };
        let lines = [
            "Guten Morgen zusammen. Lasst uns mit dem Statusbericht anfangen.",
            "Der Prototyp ist fertig, aber die Tests laufen noch bis Freitag.",
            "Dann übernehme ich die Abstimmung mit dem Kunden und schicke die Einladung raus.",
            "Wir müssen außerdem entscheiden, ob wir das Release auf nächste Woche verschieben.",
            "Einverstanden, wir verschieben es. Markus aktualisiert den Zeitplan bis Montag.",
        ];
        let segments: Vec<crate::pipeline::MergedSegment> = lines
            .iter()
            .enumerate()
            .map(|(i, text)| crate::pipeline::MergedSegment {
                start: i as f64 * 5.0,
                end: i as f64 * 5.0 + 4.0,
                speaker: format!("spk{}", i % 2),
                text: text.to_string(),
            })
            .collect();
        let mut speakers = serde_json::Map::new();
        speakers.insert("spk0".into(), serde_json::json!("Anna"));
        speakers.insert("spk1".into(), serde_json::json!("Markus"));

        let summary = super::summarize(&ctx, &segments, &speakers).unwrap();
        println!(
            "summary: {}\ntakeaways: {:#?}\naction items: {:#?}",
            summary.summary, summary.takeaways, summary.action_items
        );
        let text = format!("{} {}", summary.summary, summary.takeaways.join(" "));
        assert_eq!(detect_language(&text), Some("de"), "notes are not German: {text}");
    }
}
