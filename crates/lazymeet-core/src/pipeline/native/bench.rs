//! Built-in benchmark primitives (driven by `bench.rs`).

use super::asr::{load_recognizer, load_wav};
use super::llm::{llama_backend, Llm, FORMAT};
use super::*;

/// Deterministic pseudo-speech: band-limited noise with an amplitude wobble.
/// Parakeet and the speaker models spend the same compute on it as on
/// speech; models that write text token by token (Whisper, Qwen3-ASR,
/// Cohere) don't, so `bench.rs` times those only on a real recording. It
/// must not be pure silence or digital full-scale noise.
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

/// Write the benchmark audio to `scratch_dir` as the 16 kHz mono WAV the
/// pipeline reads: up to the first 300 s (5 min) of a user-picked recording
/// (decoded locally via the bundled ffmpeg), or the synthetic buffer when
/// `src` is None. Returns the WAV (the caller deletes it) and its seconds.
pub fn bench_wav(
    ffmpeg: Option<&Path>,
    src: Option<&Path>,
    scratch_dir: &Path,
) -> Result<(std::path::PathBuf, f64), String> {
    let wav = scratch_dir.join(".bench-audio.wav");
    match src {
        Some(src) => {
            let ffmpeg = ffmpeg.ok_or("ffmpeg is not available")?;
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
            run_ffmpeg(cmd).map_err(|e| format!("could not decode the recording: {e}"))?;
        }
        None => {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut out = hound::WavWriter::create(&wav, spec).map_err(|e| e.to_string())?;
            for s in synth_audio(300.0) {
                out.write_sample((s * i16::MAX as f32) as i16).map_err(|e| e.to_string())?;
            }
            out.finalize().map_err(|e| e.to_string())?;
        }
    }
    let samples = hound::WavReader::open(&wav).map_err(|e| e.to_string())?.duration();
    if samples < SAMPLE_RATE {
        let _ = std::fs::remove_file(&wav);
        return Err("the recording has less than one second of audio".into());
    }
    Ok((wav, samples as f64 / SAMPLE_RATE as f64))
}

/// One summarizer benchmark run (see `bench_llm`).
pub struct LlmBench {
    pub load_secs: f64,
    pub prompt_tokens: usize,
    pub prefill_secs: f64,
    pub generated_tokens: usize,
    pub generate_secs: f64,
}

/// Time the summarizer LLM on a summary-shaped workload: the real summary
/// instructions over a fixed `prompt_tokens`-token transcript, answering with
/// up to `answer_tokens` tokens. Reading the transcript (prefill) dominates a
/// real summary and is where GPU and CPU differ most — on an M3 Pro a
/// 28-minute meeting's summary takes 26 s on Metal and 105 s on the CPU,
/// while generation alone is only ~1.5x apart — so both halves are measured.
pub fn bench_llm(
    path: &Path,
    threads: i32,
    gpu: bool,
    prompt_tokens: usize,
    answer_tokens: usize,
) -> Result<LlmBench, String> {
    use std::time::Instant;
    let backend = llama_backend()?;
    let params = if gpu {
        LlamaModelParams::default().with_n_gpu_layers(1_000_000)
    } else {
        LlamaModelParams::default().with_n_gpu_layers(0)
    };
    let t = Instant::now();
    let model =
        LlamaModel::load_from_file(backend, path, &params).map_err(|e| format!("llm load: {e}"))?;
    let load_secs = t.elapsed().as_secs_f64();

    let n_ctx = (prompt_tokens + answer_tokens + 512) as u32;
    let llm = Llm { model, n_ctx, threads };
    let lines = [
        "Alice: We agreed to move the rollout to Thursday so QA can sign off first.",
        "Bob: The migration script still needs a dry run against the staging database.",
        "Carol: I can own the dry run, but I need read access to the replica by Monday.",
        "Alice: Fine. Bob, can you also update the runbook with the rollback steps?",
        "Bob: Will do. Should we invite support to the go/no-go call on Wednesday?",
        "Carol: Yes, and let's send customers the maintenance notice two days ahead.",
    ];
    let mut transcript = String::new();
    while llm.count_tokens(&transcript) < prompt_tokens {
        for line in lines {
            transcript.push_str(line);
            transcript.push('\n');
        }
    }
    let transcript = llm.truncate_to_tokens(&transcript, prompt_tokens);
    let (_, timing) = llm.chat_timed(
        &format!("{} {FORMAT}", config::DEFAULT_SUMMARY_PROMPT),
        &format!("Transcript:\n{transcript}"),
        answer_tokens,
    )?;
    Ok(LlmBench {
        load_secs,
        prompt_tokens: timing.prompt_tokens,
        prefill_secs: timing.prefill_secs,
        generated_tokens: timing.generated_tokens.max(1),
        generate_secs: timing.generate_secs,
    })
}

/// Decode whole 16 kHz WAV clips (no VAD) with `workers` one-thread workers
/// sharing one recognizer, in input order — for word-error-rate evaluation
/// on labelled test sets.
pub fn transcribe_clips(
    ctx: &PipelineCtx,
    clips: &[std::path::PathBuf],
    workers: usize,
) -> Result<Vec<String>, String> {
    let recognizer = load_recognizer(ctx, 1)?;
    par_map(clips, workers, |clip| {
        let samples = load_wav(clip).map_err(|e| format!("{}: {e}", clip.display()))?;
        Ok(recognizer.transcribe(&samples).trim().to_string())
    })
    .into_iter()
    .collect()
}
