//! VAD + offline ASR: one recognizer for every model family, plus the
//! multi-threaded `Transcription` worker pool (port of `transcribe.py`).

use super::*;

/// One offline recognizer for every ASR family; `language` is the per-stream
/// hint for Qwen3-ASR (Whisper takes its hint in the model config).
pub(super) struct Recognizer {
    pub(super) inner: OfflineRecognizer,
    pub(super) language: Option<&'static str>,
}

impl Recognizer {
    /// Safe to call from several threads at once: every call decodes its own
    /// stream, and sherpa-onnx's recognizers keep no per-decode state.
    pub(super) fn transcribe(&self, samples: &[f32]) -> String {
        let stream = self.inner.create_stream();
        if let Some(language) = self.language {
            stream.set_option("language", language);
        }
        stream.accept_waveform(SAMPLE_RATE as i32, samples);
        self.inner.decode(&stream);
        stream.get_result().map(|r| r.text).unwrap_or_default()
    }
}

/// The recognizer config for one ASR model on disk.
pub(super) fn recognizer_config(
    spec: &AsrPaths,
    language: &str,
    provider: &str,
    threads: i32,
) -> OfflineRecognizerConfig {
    let mut config = OfflineRecognizerConfig::default();
    let model = &mut config.model_config;
    model.num_threads = threads;
    model.provider = Some(provider.to_string());
    match spec.kind {
        "whisper" => {
            model.whisper = OfflineWhisperModelConfig {
                encoder: path_str(&spec.encoder),
                decoder: path_str(&spec.decoder),
                // "" lets Whisper detect the language itself.
                language: Some(language.to_string()),
                task: Some("transcribe".into()),
                ..Default::default()
            };
            model.tokens = path_str(&spec.tokens);
            config.feat_config.feature_dim = 512;
        }
        "qwen3_asr" => {
            model.qwen3_asr = OfflineQwen3ASRModelConfig {
                conv_frontend: path_str(&spec.conv_frontend),
                encoder: path_str(&spec.encoder),
                decoder: path_str(&spec.decoder),
                tokenizer: path_str(&spec.tokenizer),
                // A 20 s VAD segment of fast speech runs past the default 128.
                max_new_tokens: 256,
                ..Default::default()
            };
            model.tokens = Some(String::new());
        }
        _ => {
            // Parakeet (nemo_transducer) — multilingual, always auto-detects.
            model.transducer = OfflineTransducerModelConfig {
                encoder: path_str(&spec.encoder),
                decoder: path_str(&spec.decoder),
                joiner: path_str(&spec.joiner),
            };
            model.tokens = path_str(&spec.tokens);
            model.model_type = Some("nemo_transducer".into());
        }
    }
    config.decoding_method = Some("greedy_search".into());
    config
}

fn load_recognizer(ctx: &PipelineCtx, threads: i32) -> Result<Recognizer, String> {
    let spec = resolve_asr_at(ctx.models_dir, ctx.fake, &ctx.config.asr_model());
    let code = ctx.config.asr_language();
    let code = if code == "auto" { "" } else { code.as_str() };
    let config = recognizer_config(&spec, code, &ctx.config.asr_provider(), threads);
    let inner = OfflineRecognizer::create(&config)
        .ok_or_else(|| format!("could not load the {} transcription model", spec.kind))?;
    // Qwen3-ASR takes the hint as a language name ("German"), not a code.
    // Ukrainian is not among its languages, so that hint is left to its
    // own detection rather than forced.
    let language = match spec.kind {
        "qwen3_asr" if !code.is_empty() && code != "uk" => {
            super::super::LANGUAGE_NAMES.iter().find(|(c, _)| *c == code).map(|(_, name)| *name)
        }
        _ => None,
    };
    Ok(Recognizer { inner, language })
}

/// 16-bit PCM WAV → samples scaled by 1/32767, as the pipeline always has
/// (voiceprints saved in the speaker library were computed this way).
pub(super) fn load_wav(wav: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(wav).map_err(|e| e.to_string())?;
    let sr = reader.spec().sample_rate;
    if sr != SAMPLE_RATE {
        return Err(format!("expected 16 kHz mono wav, got {sr} Hz"));
    }
    reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32).map_err(|e| e.to_string()))
        .collect()
}

/// Seconds (original-media time) → clamped, non-empty sample index spans into
/// the loaded WAV. A range past the end of the audio clamps to it.
pub(super) fn sample_ranges(ranges: &[(f64, f64)], len: usize) -> Vec<(usize, usize)> {
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

fn new_vad(ctx: &PipelineCtx) -> Result<VoiceActivityDetector, String> {
    let config = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: path_str(&ctx.models_dir.join("silero_vad.onnx")),
            // sherpa-onnx defaults (Python left them implicit).
            threshold: 0.5,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            max_speech_duration: 20.0,
            window_size: 512,
        },
        sample_rate: SAMPLE_RATE as i32,
        // Silero runs on 32 ms windows — too small to split across cores;
        // extra threads only add synchronization (5.3 s vs 4.2 s at 7 vs 1
        // for a 28.5-minute recording).
        num_threads: 1,
        provider: Some(ctx.config.asr_provider()),
        ..Default::default()
    };
    VoiceActivityDetector::create(&config, 100.0)
        .ok_or_else(|| "could not load the VAD model".into())
}

/// One VAD-cut speech segment waiting for the recognizer; `start` is in
/// original-media seconds.
struct Utterance {
    start: f64,
    samples: Vec<f32>,
}

/// Feed one contiguous sample span through VAD, queueing its utterances
/// stamped in original-media time (`offset` = span start, seconds). Returns
/// false once nobody is listening any more.
fn cut_span(
    vad: &VoiceActivityDetector,
    samples: &[f32],
    offset: f64,
    tx: &SyncSender<Utterance>,
) -> bool {
    let drain = || {
        while let Some(seg) = vad.front() {
            let utterance = Utterance {
                start: offset + seg.start() as f64 / SAMPLE_RATE as f64,
                samples: seg.samples().to_vec(),
            };
            vad.pop();
            if tx.send(utterance).is_err() {
                return false;
            }
        }
        true
    };
    for chunk in samples.chunks(512) {
        vad.accept_waveform(chunk);
        if !drain() {
            return false;
        }
    }
    vad.flush();
    drain()
}

/// A transcription in progress. The ASR models are utterance-level, so
/// Silero VAD cuts the audio into speech segments on its own thread, and any
/// number of workers (`work`) decode them concurrently with the one shared
/// recognizer.
///
/// Workers decode with a single ONNX thread each: segments are independent,
/// and one-thread-per-segment parallelism scales where onnxruntime's
/// intra-op threading does not — Parakeet on a 28.5-minute recording took
/// 142 s at 4 threads, 147 s at 7, and 67 s as 8 one-thread workers (M3
/// Pro), with identical output. Whisper and Qwen3-ASR gain the same way.
pub struct Transcription {
    recognizer: Recognizer,
    queue: Mutex<Receiver<Utterance>>,
    /// Utterances taken off the queue whose decode hasn't finished yet.
    busy: Mutex<usize>,
    idle: Condvar,
    segments: Mutex<Vec<Segment>>,
}

impl Transcription {
    /// Load the recognizer and VAD; returns the transcription plus the VAD
    /// job, which the caller runs on its own thread to feed the workers.
    /// With `ranges`, only the selected spans are fed — each through a fresh
    /// VAD (sherpa's VAD has no reset API, and a fresh one makes the sample
    /// counter unambiguous) — so no utterance can straddle a splice.
    pub fn start(
        ctx: &PipelineCtx,
        wav: &Path,
        ranges: Option<&[(f64, f64)]>,
    ) -> Result<(Self, impl FnOnce() + Send + 'static), String> {
        let samples = load_wav(wav)?;
        let recognizer = load_recognizer(ctx, 1)?;
        let spans = match ranges {
            Some(rs) => sample_ranges(rs, samples.len()),
            None => vec![(0, samples.len())],
        };
        let vads = spans.iter().map(|_| new_vad(ctx)).collect::<Result<Vec<_>, _>>()?;
        // Bounded so the VAD, which runs far ahead of the recognizers, doesn't
        // hold a second copy of all the speech in memory.
        let (tx, rx) = mpsc::sync_channel(64);
        let cut = move || {
            for (vad, (a, b)) in vads.iter().zip(spans) {
                if !cut_span(vad, &samples[a..b], a as f64 / SAMPLE_RATE as f64, &tx) {
                    return;
                }
            }
        };
        let transcription = Self {
            recognizer,
            queue: Mutex::new(rx),
            busy: Mutex::new(0),
            idle: Condvar::new(),
            segments: Mutex::new(Vec::new()),
        };
        Ok((transcription, cut))
    }

    /// Decode queued utterances until the VAD is done and the queue is empty.
    /// Workers can join at any time.
    pub fn work(&self) {
        loop {
            let utterance = {
                let queue = self.queue.lock().unwrap();
                match queue.recv() {
                    // Counted while the queue is still locked, so once the
                    // queue reports empty-and-closed every taken utterance
                    // is visible in `busy`.
                    Ok(u) => {
                        *self.busy.lock().unwrap() += 1;
                        u
                    }
                    Err(_) => return,
                }
            };
            let text = self.recognizer.transcribe(&utterance.samples);
            let text = text.trim();
            if !text.is_empty() {
                let end = utterance.start + utterance.samples.len() as f64 / SAMPLE_RATE as f64;
                self.segments.lock().unwrap().push(Segment {
                    start: round3(utterance.start),
                    end: round3(end),
                    text: text.to_string(),
                });
            }
            let mut busy = self.busy.lock().unwrap();
            *busy -= 1;
            if *busy == 0 {
                self.idle.notify_all();
            }
        }
    }

    /// Run `workers` workers to completion, wait out any that joined from
    /// elsewhere, and return the segments in time order.
    pub fn finish(&self, workers: usize) -> Vec<Segment> {
        std::thread::scope(|s| {
            for _ in 0..workers.max(1) {
                s.spawn(|| self.work());
            }
        });
        let mut busy = self.busy.lock().unwrap();
        while *busy > 0 {
            busy = self.idle.wait(busy).unwrap();
        }
        drop(busy);
        let mut segments = std::mem::take(&mut *self.segments.lock().unwrap());
        segments.sort_by(|a, b| a.start.total_cmp(&b.start));
        segments
    }
}

/// Transcribe with `threads` concurrent workers (see `Transcription`).
pub fn transcribe(
    ctx: &PipelineCtx,
    wav: &Path,
    ranges: Option<&[(f64, f64)]>,
    threads: i32,
) -> Result<Vec<Segment>, String> {
    let (transcription, cut) = Transcription::start(ctx, wav, ranges)?;
    std::thread::scope(|s| {
        s.spawn(cut);
        Ok(transcription.finish(threads as usize))
    })
}
