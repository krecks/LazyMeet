//! Built-in benchmark primitives (explicit backends, driven by `bench.rs`).

use super::asr::{load_wav, recognizer_config, Recognizer};
use super::diarize::{new_extractor, pyannote_diarizer, speaker_embedding};
use super::llm::{llama_backend, Llm, FORMAT};
use super::*;

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
/// (16 kHz mono), diarizing with `diar_model` (the one the settings select).
/// Returns (stage name, seconds, seconds of audio processed).
pub fn bench_onnx(
    models_dir: &Path,
    threads: i32,
    provider: &str,
    diar_model: &str,
    samples: &[f32],
) -> Result<Vec<(String, f64, f64)>, String> {
    use std::time::Instant;
    let mut rows = Vec::new();
    let sr = SAMPLE_RATE as usize;

    // ASR: default model, up to 300 s in 10 s pieces (no VAD — the
    // recognizer cost is what differs between providers), decoded the way a
    // meeting is: `threads` one-thread workers sharing one recognizer.
    let spec = crate::models::resolve_asr_at(models_dir, false, crate::models::ASR_DEFAULT);
    let inner = OfflineRecognizer::create(&recognizer_config(&spec, "", provider, 1))
        .ok_or_else(|| format!("ASR load ({provider}) failed"))?;
    let recognizer = Recognizer { inner, language: None };
    let asr = &samples[..samples.len().min(300 * sr)];
    // One warmup pass (first call pays one-time graph setup), then measure.
    let _ = recognizer.transcribe(&asr[..sr.min(asr.len())]);
    let pieces: Vec<&[f32]> = asr.chunks(10 * sr).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                while let Some(piece) =
                    pieces.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                {
                    let _ = recognizer.transcribe(piece);
                }
            });
        }
    });
    rows.push((
        "Transcription".to_string(),
        t.elapsed().as_secs_f64(),
        asr.len() as f64 / sr as f64,
    ));
    drop(recognizer);

    // Diarization: up to 300 s.
    let diar = &samples[..samples.len().min(300 * sr)];
    let diar_secs = diar.len() as f64 / sr as f64;
    let (engine, model) = resolve_diar_at(models_dir, false, diar_model, -1);
    let t = Instant::now();
    if engine == "nemotron" {
        let mut diarizer =
            Nemotron::load(&model, threads).map_err(|e| format!("diarizer load: {e}"))?;
        let _ = diarizer.probabilities(diar);
    } else {
        let diarizer = pyannote_diarizer(&model, models_dir, -1, 0.7, provider, threads)
            .map_err(|e| format!("diarizer load ({provider}): {e}"))?;
        let _ = diarizer.process(diar);
    }
    rows.push(("Diarization".to_string(), t.elapsed().as_secs_f64(), diar_secs));

    // Speaker embeddings: 40 × 5 s windows (matches how voiceprints are made).
    let extractor = new_extractor(models_dir, provider, threads)
        .map_err(|e| format!("extractor load ({provider}): {e}"))?;
    // A 5 s window from the middle of the source (whole clip if shorter).
    let win_len = (5 * sr).min(samples.len());
    let start = (samples.len() - win_len) / 2;
    let window = &samples[start..start + win_len];
    let _ = speaker_embedding(&extractor, window);
    let t = Instant::now();
    for _ in 0..40 {
        let _ = speaker_embedding(&extractor, window);
    }
    let spk_secs = 40.0 * win_len as f64 / sr as f64;
    rows.push(("Speaker ID".to_string(), t.elapsed().as_secs_f64(), spk_secs));

    Ok(rows)
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
    models_dir: &Path,
    threads: i32,
    gpu: bool,
    prompt_tokens: usize,
    answer_tokens: usize,
) -> Result<LlmBench, String> {
    use std::time::Instant;
    let backend = llama_backend()?;
    let path = crate::models::resolve_llm_at(models_dir, false, crate::models::LLM_DEFAULT);
    let params = if gpu {
        LlamaModelParams::default().with_n_gpu_layers(1_000_000)
    } else {
        LlamaModelParams::default().with_n_gpu_layers(0)
    };
    let t = Instant::now();
    let model = LlamaModel::load_from_file(backend, &path, &params)
        .map_err(|e| format!("llm load: {e}"))?;
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
