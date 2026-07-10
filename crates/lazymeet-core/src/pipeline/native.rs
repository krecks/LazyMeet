//! Real ML pipeline: ffmpeg (bundled binary) → sherpa-onnx VAD + ASR →
//! diarization → speaker embeddings → llama.cpp summarization.
//!
//! Port of `backend/pipeline/{audio,transcribe,diarize,identify,summarize}.py`
//! on top of the crate-validated spike (`spike/src/main.rs`). Models load per
//! job and drop at the end — matching the memory behavior of the old
//! per-meeting worker process.

use std::num::NonZeroU32;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use serde_json::{Map, Value};

use sherpa_rs::diarize::{Diarize, DiarizeConfig};
use sherpa_rs::read_audio_file;
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};

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

use super::{
    remap_turns, round3, MediaProbe, MergedSegment, PipelineCtx, Segment, Summary, Turn,
    Voiceprints,
};
use crate::config;
use crate::models::resolve_asr_at;

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

enum Recognizer {
    Transducer(TransducerRecognizer),
    Whisper(WhisperRecognizer),
}

impl Recognizer {
    fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> String {
        match self {
            Recognizer::Transducer(r) => r.transcribe(sample_rate, samples),
            Recognizer::Whisper(r) => r.transcribe(sample_rate, samples).text,
        }
    }
}

fn load_recognizer(ctx: &PipelineCtx) -> Result<Recognizer, String> {
    let spec = resolve_asr_at(ctx.models_dir, ctx.fake, &ctx.config.asr_model());
    let threads = ctx.config.num_threads() as i32;
    let provider = ctx.config.asr_provider();
    let language = match ctx.config.asr_language().as_str() {
        "auto" => String::new(), // "" lets Whisper detect the language itself
        code => code.to_string(),
    };
    if spec.kind == "whisper" {
        WhisperRecognizer::new(WhisperConfig {
            encoder: spec.encoder.to_string_lossy().into_owned(),
            decoder: spec.decoder.to_string_lossy().into_owned(),
            tokens: spec.tokens.to_string_lossy().into_owned(),
            language,
            bpe_vocab: None,
            tail_paddings: None,
            provider: Some(provider.clone()),
            num_threads: Some(threads),
            debug: false,
        })
        .map(Recognizer::Whisper)
        .map_err(|e| format!("whisper load: {e}"))
    } else {
        // Parakeet (nemo_transducer) — multilingual, always auto-detects.
        TransducerRecognizer::new(TransducerConfig {
            encoder: spec.encoder.to_string_lossy().into_owned(),
            decoder: spec.decoder.to_string_lossy().into_owned(),
            joiner: spec.joiner.to_string_lossy().into_owned(),
            tokens: spec.tokens.to_string_lossy().into_owned(),
            model_type: "nemo_transducer".into(),
            num_threads: threads,
            sample_rate: SAMPLE_RATE as i32,
            feature_dim: 80,
            decoding_method: "greedy_search".into(),
            provider: Some(provider),
            ..Default::default()
        })
        .map(Recognizer::Transducer)
        .map_err(|e| format!("parakeet load: {e}"))
    }
}

fn load_wav(wav: &Path) -> Result<Vec<f32>, String> {
    let (samples, sr) = read_audio_file(&wav.to_string_lossy()).map_err(|e| e.to_string())?;
    if sr != SAMPLE_RATE {
        return Err(format!("expected 16 kHz mono wav, got {sr} Hz"));
    }
    Ok(samples)
}

/// Seconds (original-media time) → clamped, non-empty sample index spans into
/// the loaded WAV. A range past the end of the audio clamps to it.
fn sample_ranges(ranges: &[(f64, f64)], len: usize) -> Vec<(usize, usize)> {
    ranges
        .iter()
        .map(|&(s, e)| {
            let a = ((s * SAMPLE_RATE as f64) as usize).min(len);
            let b = ((e * SAMPLE_RATE as f64).ceil() as usize).min(len);
            (a, b)
        })
        .filter(|(a, b)| b > a)
        .collect()
}

fn new_vad(ctx: &PipelineCtx) -> Result<SileroVad, String> {
    SileroVad::new(
        SileroVadConfig {
            model: ctx.models_dir.join("silero_vad.onnx").to_string_lossy().into_owned(),
            // sherpa-onnx defaults (Python left them implicit).
            threshold: 0.5,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            max_speech_duration: 20.0,
            sample_rate: SAMPLE_RATE,
            window_size: 512,
            provider: Some(ctx.config.asr_provider()),
            num_threads: Some(ctx.config.num_threads() as i32),
            debug: false,
        },
        100.0,
    )
    .map_err(|e| format!("vad load: {e}"))
}

/// Feed one contiguous sample span through VAD + recognizer, appending
/// segments stamped in original-media time (`offset` = span start, seconds).
fn transcribe_span(
    vad: &mut SileroVad,
    recognizer: &mut Recognizer,
    samples: &[f32],
    offset: f64,
    out: &mut Vec<Segment>,
) {
    let drain = |vad: &mut SileroVad, recognizer: &mut Recognizer, out: &mut Vec<Segment>| {
        while !vad.is_empty() {
            let seg = vad.front();
            let start = offset + seg.start as f64 / SAMPLE_RATE as f64;
            let end = start + seg.samples.len() as f64 / SAMPLE_RATE as f64;
            let text = recognizer.transcribe(SAMPLE_RATE, &seg.samples);
            let text = text.trim();
            if !text.is_empty() {
                out.push(Segment {
                    start: round3(start),
                    end: round3(end),
                    text: text.to_string(),
                });
            }
            vad.pop();
        }
    };
    for chunk in samples.chunks(512) {
        vad.accept_waveform(chunk.to_vec());
        drain(vad, recognizer, out);
    }
    vad.flush();
    drain(vad, recognizer, out);
}

/// The ASR models are utterance-level, so Silero VAD cuts the audio into
/// speech segments first; the timing comes from the VAD. With `ranges`, only
/// the selected spans are fed — each through a fresh VAD (sherpa's VAD has no
/// reset API, and a fresh one makes the sample counter unambiguous) — so no
/// utterance can straddle a splice.
pub fn transcribe(
    ctx: &PipelineCtx,
    wav: &Path,
    ranges: Option<&[(f64, f64)]>,
) -> Result<Vec<Segment>, String> {
    let samples = load_wav(wav)?;
    let mut recognizer = load_recognizer(ctx)?;
    let spans = match ranges {
        Some(rs) => sample_ranges(rs, samples.len()),
        None => vec![(0, samples.len())],
    };
    let mut segments: Vec<Segment> = Vec::new();
    for (a, b) in spans {
        let mut vad = new_vad(ctx)?;
        transcribe_span(
            &mut vad,
            &mut recognizer,
            &samples[a..b],
            a as f64 / SAMPLE_RATE as f64,
            &mut segments,
        );
    }
    Ok(segments)
}

// --------------------------------------------------------------------------
// Diarization (port of diarize.py)
// --------------------------------------------------------------------------

pub fn diarize(
    ctx: &PipelineCtx,
    wav: &Path,
    num_speakers: i64,
    ranges: Option<&[(f64, f64)]>,
) -> Result<Vec<Turn>, String> {
    // A meeting-supplied count (>0) wins; otherwise the env default (-1 auto).
    let n = if num_speakers > 0 { num_speakers } else { config::env_num_speakers() };
    let mut diarizer = Diarize::new(
        ctx.models_dir.join("diarization_segmentation.onnx"),
        ctx.models_dir.join("speaker_embedding.onnx"),
        DiarizeConfig {
            // num_clusters > 0 forces an exact speaker count; -1 lets the
            // threshold decide (higher => fewer speakers).
            num_clusters: Some(n as i32),
            threshold: Some(ctx.config.diar_threshold() as f32),
            // smooth out sub-second fragments that would explode into
            // hundreds of spurious speakers
            min_duration_on: Some(0.3),
            min_duration_off: Some(0.5),
            provider: Some(ctx.config.diar_provider()),
            debug: false,
        },
    )
    .map_err(|e| format!("diarizer load: {e}"))?;

    let samples = load_wav(wav)?;
    // With ranges, the diarizer runs once over the spliced selection (speaker
    // clustering must see all spans together); the concat-time turns are then
    // remapped — splitting at splice boundaries — back to original time.
    let (input, remap) = match ranges {
        None => (samples, None),
        Some(rs) => {
            let spans = sample_ranges(rs, samples.len());
            let mut concat = Vec::with_capacity(spans.iter().map(|(a, b)| b - a).sum());
            for &(a, b) in &spans {
                concat.extend_from_slice(&samples[a..b]);
            }
            if concat.is_empty() {
                return Ok(Vec::new());
            }
            // Remap against the *clamped* spans so a range past the end of
            // the audio maps consistently with what was actually fed.
            let secs: Vec<(f64, f64)> = spans
                .iter()
                .map(|&(a, b)| (a as f64 / SAMPLE_RATE as f64, b as f64 / SAMPLE_RATE as f64))
                .collect();
            (concat, Some(secs))
        }
    };
    let mut turns: Vec<Turn> = diarizer
        .compute(input, None)
        .map_err(|e| format!("diarize: {e}"))?
        .into_iter()
        .map(|s| Turn {
            start: round3(s.start as f64),
            end: round3(s.end as f64),
            speaker: format!("SPEAKER_{:02}", s.speaker),
        })
        .collect();
    if let Some(secs) = remap {
        turns = remap_turns(&secs, &turns);
    }
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    Ok(turns)
}

// --------------------------------------------------------------------------
// Voiceprints (port of identify.py's extractor half)
// --------------------------------------------------------------------------

/// One averaged voiceprint per diarized speaker, from its longest turns.
pub fn compute_voiceprints(
    ctx: &PipelineCtx,
    wav: &Path,
    turns: &[Turn],
) -> Result<Voiceprints, String> {
    let samples = load_wav(wav)?;
    let mut extractor = EmbeddingExtractor::new(ExtractorConfig {
        model: ctx.models_dir.join("speaker_embedding.onnx").to_string_lossy().into_owned(),
        provider: Some(ctx.config.speaker_provider()),
        num_threads: Some(ctx.config.num_threads() as usize),
        debug: false,
    })
    .map_err(|e| format!("extractor load: {e}"))?;

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
            if let Ok(e) = extractor.compute_speaker_embedding(samples[a..b].to_vec(), SAMPLE_RATE)
            {
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

// --------------------------------------------------------------------------
// Summarization (port of summarize.py)
// --------------------------------------------------------------------------

/// Tokens reserved for the model's response plus prompt scaffolding.
const RESERVE: usize = 2560;

/// Fixed response-format instruction, appended after the (user-editable)
/// SUMMARY_PROMPT so a customized prompt can't break summary parsing.
const FORMAT: &str = "Respond with ONLY a JSON object of the form \
    {\"summary\": str, \"takeaways\": [str], \"action_items\": [str]}.";

const NOTES_SYSTEM: &str = "You are a meeting assistant. Summarize this portion of a meeting transcript \
    into concise bullet notes capturing decisions, topics, and any action items. \
    Return plain text bullets only.";

fn language_name<'a>(code: &'a str) -> &'a str {
    super::LANGUAGE_NAMES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, n)| *n)
        .unwrap_or(code)
}

/// Detect the transcript's language, restricted to the languages the
/// summarizer can be instructed in. None when detection isn't reliable
/// (very short or heavily mixed-language transcripts).
fn detect_language(text: &str) -> Option<&'static str> {
    use whatlang::Lang;
    const LANGS: &[(Lang, &str)] = &[
        (Lang::Eng, "en"),
        (Lang::Deu, "de"),
        (Lang::Fra, "fr"),
        (Lang::Spa, "es"),
        (Lang::Ita, "it"),
        (Lang::Por, "pt"),
        (Lang::Nld, "nl"),
        (Lang::Pol, "pl"),
        (Lang::Rus, "ru"),
        (Lang::Ukr, "uk"),
        (Lang::Tur, "tr"),
        (Lang::Cmn, "zh"),
        (Lang::Jpn, "ja"),
        (Lang::Kor, "ko"),
    ];
    // A few thousand characters are plenty for a confident call.
    let sample: String = text.chars().take(4000).collect();
    let detector = whatlang::Detector::with_allowlist(LANGS.iter().map(|(l, _)| *l).collect());
    let info = detector.detect(&sample)?;
    if !info.is_reliable() {
        return None;
    }
    LANGS.iter().find(|(l, _)| *l == info.lang()).map(|(_, code)| *code)
}

/// The language order given to the summarizer. An explicit setting wins;
/// "auto" detects the language from the transcript text. Naming the language
/// outright ("Write everything in German.") is what small models actually
/// follow — the former "same language as the meeting" phrasing made them
/// drift into English on non-English meetings.
fn lang_instruction(ctx: &PipelineCtx, transcript: &str) -> String {
    let code = ctx.config.summary_language();
    let code = if !code.is_empty() && code != "auto" {
        Some(code)
    } else {
        detect_language(transcript).map(str::to_string)
    };
    match code {
        Some(code) => format!("Write everything in {}.", language_name(&code)),
        None => "Write in the same language the meeting was held in.".into(),
    }
}

fn llama_backend() -> Result<&'static LlamaBackend, String> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| e.clone())
}

struct Llm {
    model: LlamaModel,
    n_ctx: u32,
    threads: i32,
}

impl Llm {
    fn load(ctx: &PipelineCtx) -> Result<Self, String> {
        let backend = llama_backend()?;
        let path = crate::models::resolve_llm_at(ctx.models_dir, ctx.fake, &ctx.config.summary_model());
        // Offload all layers to the GPU (Metal on Apple Silicon) when built
        // with GPU support; if that load fails (driver/VRAM), fall back to a
        // CPU-only load rather than failing the meeting. SUMMARY_PROVIDER
        // "cpu" must set n_gpu_layers to 0 explicitly — llama.cpp's *default*
        // on Metal builds is full offload.
        let cpu = || LlamaModelParams::default().with_n_gpu_layers(0);
        let model = if ctx.config.use_gpu() {
            let gpu = LlamaModelParams::default().with_n_gpu_layers(1_000_000);
            LlamaModel::load_from_file(backend, &path, &gpu)
                .or_else(|_| LlamaModel::load_from_file(backend, &path, &cpu()))
        } else {
            LlamaModel::load_from_file(backend, &path, &cpu())
        }
        .map_err(|e| format!("llm load: {e}"))?;
        Ok(Self {
            model,
            n_ctx: config::llm_n_ctx(),
            threads: ctx.config.num_threads() as i32,
        })
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.model.str_to_token(text, AddBos::Never).map(|t| t.len()).unwrap_or(0)
    }

    fn detokenize(&self, tokens: &[LlamaToken]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            #[allow(deprecated)]
            if let Ok(b) = self.model.token_to_bytes(t, Special::Tokenize) {
                bytes.extend(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn truncate_to_tokens(&self, text: &str, budget: usize) -> String {
        match self.model.str_to_token(text, AddBos::Never) {
            Ok(tokens) if tokens.len() > budget => self.detokenize(&tokens[..budget]),
            _ => text.to_string(),
        }
    }

    /// One chat completion: apply the model's template, prefill an empty
    /// think-block (the C-API template matcher doesn't disable Qwen thinking),
    /// decode greedily at temp 0.2.
    fn chat(&self, system: &str, user: &str, max_tokens: usize) -> Result<String, String> {
        let backend = llama_backend()?;
        let template = self
            .model
            .chat_template(None)
            .map_err(|e| format!("chat template: {e}"))?;
        let mut prompt = self
            .model
            .apply_chat_template(
                &template,
                &[
                    LlamaChatMessage::new("system".into(), system.into())
                        .map_err(|e| e.to_string())?,
                    LlamaChatMessage::new("user".into(), user.into())
                        .map_err(|e| e.to_string())?,
                ],
                true,
            )
            .map_err(|e| e.to_string())?;
        // Qwen's jinja template inserts an empty think block when thinking is
        // disabled; prefill it or the model burns the budget on reasoning.
        prompt.push_str("<think>\n\n</think>\n\n");

        let tokens = self.model.str_to_token(&prompt, AddBos::Always).map_err(|e| e.to_string())?;
        let n_batch = (tokens.len() + max_tokens + 32).max(512) as u32;
        let mut ctx = self
            .model
            .new_context(
                backend,
                LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(self.n_ctx))
                    .with_n_batch(n_batch)
                    .with_n_threads(self.threads)
                    .with_n_threads_batch(self.threads),
            )
            .map_err(|e| format!("llm context: {e}"))?;

        let mut batch = LlamaBatch::new(n_batch as usize, 1);
        let last_idx = tokens.len() as i32 - 1;
        for (i, token) in (0i32..).zip(&tokens) {
            batch.add(*token, i, &[0], i == last_idx).map_err(|e| e.to_string())?;
        }
        ctx.decode(&mut batch).map_err(|e| e.to_string())?;

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.2), LlamaSampler::dist(42)]);
        let mut out: Vec<u8> = Vec::new();
        let mut n_cur = batch.n_tokens();
        let mut n_gen = 0usize;
        loop {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) || n_gen >= max_tokens {
                break;
            }
            #[allow(deprecated)]
            if let Ok(bytes) = self.model.token_to_bytes(token, Special::Tokenize) {
                out.extend(bytes);
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true).map_err(|e| e.to_string())?;
            n_cur += 1;
            n_gen += 1;
            ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

fn transcript_lines(segments: &[MergedSegment], speakers: &Map<String, Value>) -> Vec<String> {
    segments
        .iter()
        .map(|seg| {
            let name = speakers
                .get(&seg.speaker)
                .and_then(|v| v.as_str())
                .unwrap_or(&seg.speaker);
            format!("{name}: {}", seg.text)
        })
        .collect()
}

fn chunk_lines(llm: &Llm, lines: &[String], budget: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut cur_tok = 0usize;
    for line in lines {
        let t = llm.count_tokens(line) + 1;
        if !cur.is_empty() && cur_tok + t > budget {
            chunks.push(cur.join("\n"));
            cur.clear();
            cur_tok = 0;
        }
        cur.push(line);
        cur_tok += t;
    }
    if !cur.is_empty() {
        chunks.push(cur.join("\n"));
    }
    chunks
}

/// Drop <think>…</think> blocks some Qwen variants emit before the answer.
fn strip_thinking(content: &str) -> String {
    let re = regex::Regex::new(r"(?s)<think>.*?</think>").unwrap();
    re.replace_all(content, "").trim().to_string()
}

fn parse_summary(content: &str) -> Summary {
    let content = strip_thinking(content);
    let candidate = match (content.find('{'), content.rfind('}')) {
        (Some(a), Some(b)) if b > a => &content[a..=b],
        _ => content.as_str(),
    };
    let data: Option<Value> = serde_json::from_str(&content)
        .ok()
        .or_else(|| serde_json::from_str(candidate).ok());
    let Some(Value::Object(data)) = data else {
        return Summary { summary: content.trim().to_string(), takeaways: vec![], action_items: vec![] };
    };
    let strings = |key: &str| -> Vec<String> {
        data.get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .map(|x| match x.as_str() {
                        Some(s) => s.to_string(),
                        None => x.to_string(),
                    })
                    .filter(|s| !s.trim().is_empty())
                    .collect()
            })
            .unwrap_or_default()
    };
    Summary {
        summary: data
            .get("summary")
            .map(|v| match v.as_str() {
                Some(s) => s.trim().to_string(),
                None => v.to_string(),
            })
            .unwrap_or_default(),
        takeaways: strings("takeaways"),
        action_items: strings("action_items"),
    }
}

// --------------------------------------------------------------------------
// Built-in benchmark (explicit backends, driven by bench.rs)
// --------------------------------------------------------------------------

/// Deterministic pseudo-speech: band-limited noise with an amplitude wobble.
/// Content doesn't matter for timing — the models process every frame either
/// way — but it must not be pure silence or digital full-scale noise.
fn synth_audio(seconds: f64) -> Vec<f32> {
    let n = (seconds * SAMPLE_RATE as f64) as usize;
    let mut state: u32 = 0x1234_5678;
    let mut prev = 0.0f32;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let white = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
            prev = 0.8 * prev + 0.2 * white; // crude low-pass ≈ speech band
            let wobble = 1.0 + 0.5 * (i as f32 * 3.0 / SAMPLE_RATE as f32).sin();
            (prev * wobble * 0.6).clamp(-1.0, 1.0)
        })
        .collect()
}

/// 300 s (5 min) of synthetic benchmark audio (the default source).
pub fn bench_audio_synth() -> Vec<f32> {
    synth_audio(300.0)
}

/// Decode (up to) the first 300 s (5 min) of a user-picked recording to
/// benchmark samples. Decoding happens locally via the bundled ffmpeg; the
/// scratch WAV is deleted immediately after reading.
pub fn bench_audio_from_file(
    ffmpeg: &Path,
    src: &Path,
    scratch_dir: &Path,
) -> Result<Vec<f32>, String> {
    let wav = scratch_dir.join(".bench-audio.wav");
    #[allow(unused_mut)]
    let mut cmd = Command::new(ffmpeg);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.arg("-y")
        .arg("-i")
        .arg(src)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-t", "300", "-f", "wav"])
        .arg(&wav);
    run_ffmpeg(cmd)?;
    let result = load_wav(&wav);
    let _ = std::fs::remove_file(&wav);
    let samples = result?;
    if samples.len() < SAMPLE_RATE as usize {
        return Err("the recording has less than one second of audio".into());
    }
    Ok(samples)
}

/// Time the three ONNX stages on the given execution provider over `samples`
/// (16 kHz mono). Returns (stage name, seconds, seconds of audio processed).
pub fn bench_onnx(
    models_dir: &Path,
    threads: i32,
    provider: &str,
    samples: &[f32],
) -> Result<Vec<(String, f64, f64)>, String> {
    use std::time::Instant;
    let mut rows = Vec::new();
    let sr = SAMPLE_RATE as usize;

    // ASR: default model, up to 120 s through the recognizer (no VAD — the
    // recognizer cost is what differs between providers).
    let spec = crate::models::resolve_asr_at(models_dir, false, crate::models::ASR_DEFAULT);
    let mut recognizer = TransducerRecognizer::new(TransducerConfig {
        encoder: spec.encoder.to_string_lossy().into_owned(),
        decoder: spec.decoder.to_string_lossy().into_owned(),
        joiner: spec.joiner.to_string_lossy().into_owned(),
        tokens: spec.tokens.to_string_lossy().into_owned(),
        model_type: "nemo_transducer".into(),
        num_threads: threads,
        sample_rate: SAMPLE_RATE as i32,
        feature_dim: 80,
        decoding_method: "greedy_search".into(),
        provider: Some(provider.into()),
        ..Default::default()
    })
    .map_err(|e| format!("ASR load ({provider}): {e}"))?;
    let asr = &samples[..samples.len().min(120 * sr)];
    // One warmup pass (first call pays one-time graph setup), then measure.
    let _ = recognizer.transcribe(SAMPLE_RATE, &asr[..sr.min(asr.len())]);
    let t = Instant::now();
    for chunk in asr.chunks(10 * sr) {
        let _ = recognizer.transcribe(SAMPLE_RATE, chunk);
    }
    rows.push(("Transcription".to_string(), t.elapsed().as_secs_f64(), asr.len() as f64 / sr as f64));
    drop(recognizer);

    // Diarization: up to 300 s.
    let mut diarizer = Diarize::new(
        models_dir.join("diarization_segmentation.onnx"),
        models_dir.join("speaker_embedding.onnx"),
        DiarizeConfig {
            num_clusters: Some(-1),
            threshold: Some(0.7),
            min_duration_on: Some(0.3),
            min_duration_off: Some(0.5),
            provider: Some(provider.into()),
            debug: false,
        },
    )
    .map_err(|e| format!("diarizer load ({provider}): {e}"))?;
    let diar = samples[..samples.len().min(300 * sr)].to_vec();
    let diar_secs = diar.len() as f64 / sr as f64;
    let t = Instant::now();
    let _ = diarizer.compute(diar, None);
    rows.push(("Diarization".to_string(), t.elapsed().as_secs_f64(), diar_secs));
    drop(diarizer);

    // Speaker embeddings: 40 × 5 s windows (matches how voiceprints are made).
    let mut extractor = EmbeddingExtractor::new(ExtractorConfig {
        model: models_dir.join("speaker_embedding.onnx").to_string_lossy().into_owned(),
        provider: Some(provider.into()),
        num_threads: Some(threads as usize),
        debug: false,
    })
    .map_err(|e| format!("extractor load ({provider}): {e}"))?;
    // A 5 s window from the middle of the source (whole clip if shorter).
    let win_len = (5 * sr).min(samples.len());
    let start = (samples.len() - win_len) / 2;
    let window = samples[start..start + win_len].to_vec();
    let _ = extractor.compute_speaker_embedding(window.clone(), SAMPLE_RATE);
    let t = Instant::now();
    for _ in 0..40 {
        let _ = extractor.compute_speaker_embedding(window.clone(), SAMPLE_RATE);
    }
    let spk_secs = 40.0 * win_len as f64 / sr as f64;
    rows.push(("Speaker ID".to_string(), t.elapsed().as_secs_f64(), spk_secs));

    Ok(rows)
}

/// Time the summarizer LLM: (model load seconds, generated tokens/second).
pub fn bench_llm(
    models_dir: &Path,
    threads: i32,
    gpu: bool,
    n_tokens: usize,
) -> Result<(f64, f64), String> {
    use std::time::Instant;
    let backend = llama_backend()?;
    let path = crate::models::resolve_llm_at(models_dir, false, crate::models::LLM_DEFAULT);
    let params = if gpu {
        LlamaModelParams::default().with_n_gpu_layers(1_000_000)
    } else {
        LlamaModelParams::default().with_n_gpu_layers(0)
    };
    let t = Instant::now();
    let model =
        LlamaModel::load_from_file(backend, &path, &params).map_err(|e| format!("llm load: {e}"))?;
    let load_s = t.elapsed().as_secs_f64();

    let llm = Llm { model, n_ctx: 2048, threads };
    let t = Instant::now();
    let out = llm.chat(
        "You are a meeting assistant.",
        "Write a long, detailed paragraph about effective meeting notes.",
        n_tokens,
    )?;
    let gen_s = t.elapsed().as_secs_f64();
    // chat() stops at EOG or the cap; count what actually came out.
    let produced = llm.count_tokens(&out).max(1);
    Ok((load_s, produced as f64 / gen_s))
}

/// Long meetings are chunked to fit the context window (map-reduce: summarize
/// each chunk to notes, then summarize the notes). The final answer is strict
/// JSON parsed defensively.
pub fn summarize(
    ctx: &PipelineCtx,
    segments: &[MergedSegment],
    speakers: &Map<String, Value>,
) -> Result<Summary, String> {
    let llm = Llm::load(ctx)?;
    let lines = transcript_lines(segments, speakers);
    let budget = (llm.n_ctx as usize).saturating_sub(RESERVE).max(1024);
    let full = lines.join("\n");
    let lang = lang_instruction(ctx, &full);
    let custom = ctx.config.summary_prompt();
    let task = custom.trim();
    let task = if task.is_empty() { config::DEFAULT_SUMMARY_PROMPT } else { task };

    // The language order is repeated at the END of every user message: small
    // models weigh trailing instructions far more than the system prompt.
    let body = if llm.count_tokens(&full) <= budget {
        format!("Transcript:\n{full}\n\n{lang}")
    } else {
        // Map-reduce for long meetings.
        let mut notes = Vec::new();
        for chunk in chunk_lines(&llm, &lines, budget) {
            let note =
                llm.chat(&format!("{NOTES_SYSTEM} {lang}"), &format!("{chunk}\n\n{lang}"), 512)?;
            notes.push(strip_thinking(&note));
        }
        let combined = llm.truncate_to_tokens(&notes.join("\n"), budget);
        format!("Meeting notes:\n{combined}\n\n{lang}")
    };
    let response = llm.chat(&format!("{task} {FORMAT} {lang}"), &body, 1024)?;
    Ok(parse_summary(&response))
}

#[cfg(test)]
mod tests {
    use super::detect_language;

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
