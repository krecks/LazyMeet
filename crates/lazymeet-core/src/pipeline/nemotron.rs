//! NVIDIA's Nemotron 3 Diarization (a Streaming Sortformer), run offline on
//! the onnxruntime that sherpa-onnx links in.
//!
//! The model gives every 10 ms frame a probability for each of up to eight
//! speakers, whose channels are numbered in the order the voices are first
//! heard. The int8 ONNX export (onnx-community) holds no state: one call
//! encodes one chunk of log-mel features together with the embeddings of
//! earlier frames — the Arrival-Order Speaker Cache and a FIFO of the latest
//! frames — and returns the chunk's logits plus its embeddings for the cache.
//! The features, the chunk loop and the cache policy are here: a step-by-step
//! port of the offline forward of `Nemotron3DiarizationForAudioFrameClassification`
//! and of `Nemotron3DiarizationSpeakerCache` in Hugging Face transformers
//! (Apache-2.0), cross-checked against the MIT-licensed Rust port in
//! jankeesvw/omarchy-meeting-recorder.

use std::path::Path;
use std::sync::Arc;

use ort::session::Session;
use ort::value::Tensor;
use realfft::{RealFftPlanner, RealToComplex};

// Feature extractor (`preprocessor_config.json`).
const HOP: usize = 160;
const N_FFT: usize = 512;
const WIN: usize = 400;
const BINS: usize = N_FFT / 2 + 1;
const MELS: usize = 128;
const SAMPLE_RATE: f64 = 16_000.0;
const PREEMPHASIS: f32 = 0.97;
const LOG_GUARD: f32 = 1.0 / (1u32 << 24) as f32; // 2^-24

// Model (`config.json`): the offline chunking and FIFO sizes at the top
// level, the speaker-cache policy from `streaming_config`.
pub const SPEAKERS: usize = 8;
const HIDDEN: usize = 512;
const SUBSAMPLING: usize = 8;
const CHUNK_LEN: usize = 340;
const RIGHT_CONTEXT: usize = 40;
const FIFO_LEN: usize = 40;
const UPDATE_PERIOD: usize = 300;
const CACHE_LEN: usize = 264;
const SILENCE_PER_SPEAKER: usize = 1;
const SCORE_THRESHOLD: f32 = 0.25;
const LATEST_FRAMES_BOOST: f32 = 0.05;
const MIN_POSITIVE_RATE: f32 = 0.5;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;

/// Seconds per output frame (one spectrogram hop).
pub const FRAME_SECS: f64 = HOP as f64 / SAMPLE_RATE;

pub struct Nemotron {
    session: Session,
    /// Learned embedding for the cache's silence slots; the model returns it
    /// with every call.
    silence: Vec<f32>,
}

fn ort_err(e: impl std::fmt::Display) -> String {
    format!("speaker diarization model: {e}")
}

impl Nemotron {
    /// Runs on the CPU — the only execution provider the static onnxruntime
    /// build offers (see `config::onnx_provider_options`).
    pub fn load(model: &Path, threads: i32) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(ort_err)?
            .with_intra_threads(threads.max(1) as usize)
            .map_err(ort_err)?
            .commit_from_file(model)
            .map_err(ort_err)?;
        Ok(Self { session, silence: Vec::new() })
    }

    /// Speaker probabilities for every 10 ms frame of `samples` (16 kHz
    /// mono): `frames × SPEAKERS`, row-major. `frames` is `1 + len / 160`,
    /// the frame count of a centered STFT.
    pub fn probabilities(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        let mel = LogMel::new(samples);
        let steps = mel.frames.div_ceil(SUBSAMPLING); // encoder frames
        let mut cache = SpeakerCache::default();
        let mut logits: Vec<f32> = Vec::with_capacity(steps * SUBSAMPLING * SPEAKERS);
        let mut features = Vec::new();

        for start in (0..steps).step_by(CHUNK_LEN) {
            let end = (start + CHUNK_LEN).min(steps);
            let lookahead_end = (end + RIGHT_CONTEXT).min(steps);
            let chunk_rows = lookahead_end - start; // encoder frames fed

            // The chunk's features, zero-padded to whole encoder frames — the
            // same padding the full-recording feature stacking would add.
            features.clear();
            mel.rows(start * SUBSAMPLING, lookahead_end * SUBSAMPLING, &mut features);

            let cached = cache.cached();
            let cached_rows = cached.len() / HIDDEN;
            // Cached frames are always valid; a chunk frame is valid when its
            // first spectrogram frame is (`attention_mask[:, ::8]`).
            let mask: Vec<bool> = std::iter::repeat(true).take(cached_rows)
                .chain((start..lookahead_end).map(|e| e * SUBSAMPLING < mel.valid))
                .collect();

            let outputs = self
                .session
                .run(ort::inputs![
                    "input_features" => Tensor::from_array(
                        ([1usize, chunk_rows * SUBSAMPLING, MELS], features.clone())
                    ).map_err(ort_err)?,
                    "cached_embeds" => Tensor::from_array(
                        ([1usize, cached_rows, HIDDEN], cached.clone())
                    ).map_err(ort_err)?,
                    "attention_mask" => Tensor::from_array(
                        ([1usize, mask.len()], mask.iter().map(|&m| m as i64).collect::<Vec<_>>())
                    ).map_err(ort_err)?,
                ])
                .map_err(ort_err)?;
            let (_, step_logits) = outputs["logits"].try_extract_tensor::<f32>().map_err(ort_err)?;
            let (_, chunk_embeds) =
                outputs["chunk_embeds"].try_extract_tensor::<f32>().map_err(ort_err)?;
            if step_logits.len() != mask.len() * SUBSAMPLING * SPEAKERS
                || chunk_embeds.len() != chunk_rows * HIDDEN
            {
                return Err(ort_err("unexpected output shape"));
            }
            if self.silence.is_empty() {
                let (_, silence) =
                    outputs["silence_embeds"].try_extract_tensor::<f32>().map_err(ort_err)?;
                self.silence = silence.to_vec();
            }

            let kept = cached_rows * SUBSAMPLING * SPEAKERS..(cached_rows + end - start) * SUBSAMPLING * SPEAKERS;
            logits.extend_from_slice(&step_logits[kept]);

            let mut input = cached;
            input.extend_from_slice(chunk_embeds);
            cache.update(&input, step_logits, &mask, end - start, &self.silence);
        }

        // The last encoder frame may be feature-stacking padding.
        logits.truncate(mel.frames * SPEAKERS);
        Ok(logits.into_iter().map(sigmoid).collect())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// --------------------------------------------------------------------------
// Log-mel features (`NemotronAsrStreamingFeatureExtractor`, center=True)
// --------------------------------------------------------------------------

/// Pre-emphasis, `torch.stft(center=True, pad_mode="constant")` with a
/// 400-sample symmetric Hann window centered in 512, power spectrum, Slaney
/// mel filterbank, `log(mel + 2^-24)`. Never normalized; frames from
/// `valid` on are zeroed like the extractor's attention mask does.
struct LogMel<'a> {
    samples: &'a [f32],
    window: Vec<f32>,
    /// Per mel band: its first FFT bin and its (non-zero) weights from there.
    bands: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn RealToComplex<f32>>,
    /// Spectrogram frames: `1 + len / HOP`.
    frames: usize,
    /// Frames backed by audio: `len / HOP`.
    valid: usize,
}

impl<'a> LogMel<'a> {
    fn new(samples: &'a [f32]) -> Self {
        let offset = (N_FFT - WIN) / 2;
        let window = (0..N_FFT)
            .map(|i| {
                if (offset..offset + WIN).contains(&i) {
                    let k = (i - offset) as f32;
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * k / (WIN - 1) as f32).cos()
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            samples,
            window,
            bands: slaney_mel_bands(),
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT),
            frames: 1 + samples.len() / HOP,
            valid: samples.len() / HOP,
        }
    }

    /// Sample `i` of the pre-emphasized signal, zero-padded by `N_FFT / 2`
    /// on both sides.
    fn padded(&self, i: usize) -> f32 {
        match i.checked_sub(N_FFT / 2) {
            Some(0) if !self.samples.is_empty() => self.samples[0],
            Some(j) if j < self.samples.len() => {
                self.samples[j] - PREEMPHASIS * self.samples[j - 1]
            }
            _ => 0.0,
        }
    }

    /// Append the features of frames `from..to` (`(to - from) × MELS`).
    fn rows(&self, from: usize, to: usize, out: &mut Vec<f32>) {
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut power = [0.0f32; BINS];
        for frame in from..to {
            if frame >= self.valid {
                out.extend(std::iter::repeat(0.0).take(MELS));
                continue;
            }
            let at = frame * HOP;
            for (i, v) in input.iter_mut().enumerate() {
                *v = self.padded(at + i) * self.window[i];
            }
            self.fft.process(&mut input, &mut spectrum).expect("buffers from the plan");
            for (p, c) in power.iter_mut().zip(&spectrum) {
                *p = c.norm_sqr();
            }
            out.extend(self.bands.iter().map(|(first, weights)| {
                let energy: f32 = weights.iter().zip(&power[*first..]).map(|(w, p)| w * p).sum();
                (energy + LOG_GUARD).ln()
            }));
        }
    }
}

/// `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000,
/// norm="slaney")`, stored sparsely: each band is a narrow triangle.
fn slaney_mel_bands() -> Vec<(usize, Vec<f32>)> {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let logstep = 6.4f64.ln() / 27.0;
    let hz_to_mel = |hz: f64| {
        if hz >= MIN_LOG_HZ {
            MIN_LOG_HZ / F_SP + (hz / MIN_LOG_HZ).ln() / logstep
        } else {
            hz / F_SP
        }
    };
    let mel_to_hz = |mel: f64| {
        let min_log_mel = MIN_LOG_HZ / F_SP;
        if mel >= min_log_mel {
            MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
        } else {
            F_SP * mel
        }
    };
    let top = hz_to_mel(SAMPLE_RATE / 2.0);
    let points: Vec<f64> =
        (0..MELS + 2).map(|i| mel_to_hz(top * i as f64 / (MELS + 1) as f64)).collect();
    (0..MELS)
        .map(|m| {
            let (lower, center, upper) = (points[m], points[m + 1], points[m + 2]);
            let norm = 2.0 / (upper - lower);
            let weights: Vec<f32> = (0..BINS)
                .map(|k| {
                    let hz = k as f64 * SAMPLE_RATE / N_FFT as f64;
                    let rising = (hz - lower) / (center - lower);
                    let falling = (upper - hz) / (upper - center);
                    (rising.min(falling).max(0.0) * norm) as f32
                })
                .collect();
            let first = weights.iter().position(|w| *w > 0.0).unwrap_or(0);
            let last = weights.iter().rposition(|w| *w > 0.0).unwrap_or(first);
            (first, weights[first..=last].to_vec())
        })
        .collect()
}

// --------------------------------------------------------------------------
// Arrival-Order Speaker Cache + FIFO (`Nemotron3DiarizationSpeakerCache`)
// --------------------------------------------------------------------------

#[derive(Default)]
struct SpeakerCache {
    /// Cache frames × HIDDEN.
    embeds: Vec<f32>,
    /// Cache frames × SPEAKERS — only consulted once the cache is compressed
    /// (before that, every step re-estimates the plain frames' probabilities).
    probs: Vec<f32>,
    /// FIFO frames × HIDDEN.
    fifo: Vec<f32>,
    compressed: bool,
}

impl SpeakerCache {
    /// What the next chunk attends to: the cache, then the FIFO.
    fn cached(&self) -> Vec<f32> {
        let mut all = Vec::with_capacity(self.embeds.len() + self.fifo.len());
        all.extend_from_slice(&self.embeds);
        all.extend_from_slice(&self.fifo);
        all
    }

    /// Push a processed chunk to the FIFO; when it overflows, move at least
    /// `UPDATE_PERIOD` of its oldest frames to the cache, and compress the
    /// cache once it outgrows `CACHE_LEN`. `input` is the step's encoder
    /// input (cache, FIFO, chunk, look-ahead), `logits` its logits at the
    /// spectrogram rate, `mask` its valid frames; only the `chunk_rows`
    /// chunk frames join the FIFO — the look-ahead is fed again next step.
    fn update(&mut self, input: &[f32], logits: &[f32], mask: &[bool], chunk_rows: usize, silence: &[f32]) {
        let cache_rows = self.embeds.len() / HIDDEN;
        let fifo_rows = self.fifo.len() / HIDDEN;
        let probs = pool_probs(logits, mask);

        let chunk_start = cache_rows + fifo_rows;
        let mut fifo = std::mem::take(&mut self.fifo);
        fifo.extend_from_slice(&input[chunk_start * HIDDEN..(chunk_start + chunk_rows) * HIDDEN]);
        let fifo_rows = fifo.len() / HIDDEN;

        let popped = if fifo_rows <= FIFO_LEN {
            0
        } else {
            UPDATE_PERIOD.max(fifo_rows - FIFO_LEN).min(fifo_rows)
        };
        if popped > 0 {
            let fifo_probs = &probs[cache_rows * SPEAKERS..(cache_rows + fifo_rows) * SPEAKERS];
            // A compressed cache is out of frame order, so the probabilities
            // stored with it are the only valid ones for its frames.
            let mut cache_probs = if self.compressed {
                self.probs[..cache_rows * SPEAKERS].to_vec()
            } else {
                probs[..cache_rows * SPEAKERS].to_vec()
            };
            let mut cache_embeds = std::mem::take(&mut self.embeds);
            cache_embeds.extend_from_slice(&fifo[..popped * HIDDEN]);
            cache_probs.extend_from_slice(&fifo_probs[..popped * SPEAKERS]);
            fifo.drain(..popped * HIDDEN);
            if cache_embeds.len() / HIDDEN > CACHE_LEN {
                (cache_embeds, cache_probs) = compress(&cache_embeds, &cache_probs, silence);
                self.compressed = true;
            }
            self.embeds = cache_embeds;
            self.probs = cache_probs;
        }
        self.fifo = fifo;
    }
}

/// Sigmoid of the 10 ms logits averaged per encoder frame, zero on masked
/// frames: `encoder frames × SPEAKERS`.
fn pool_probs(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let rows = logits.len() / (SUBSAMPLING * SPEAKERS);
    let mut probs = vec![0.0f32; rows * SPEAKERS];
    for (row, valid) in mask.iter().enumerate().take(rows) {
        if !valid {
            continue;
        }
        for k in 0..SUBSAMPLING {
            let at = (row * SUBSAMPLING + k) * SPEAKERS;
            for s in 0..SPEAKERS {
                probs[row * SPEAKERS + s] += sigmoid(logits[at + s]) / SUBSAMPLING as f32;
            }
        }
    }
    probs
}

/// Per-speaker budget of cache frames, excluding its silence slots.
const BUDGET: usize = CACHE_LEN / SPEAKERS - SILENCE_PER_SPEAKER;

/// How telling each frame is of each speaker: finite only for frames where
/// that speaker is active, and — for a speaker with enough clearly positive
/// frames — only for those.
fn frame_scores(probs: &[f32], frames: usize) -> Vec<f32> {
    let min_positive = (BUDGET as f32 * MIN_POSITIVE_RATE).floor() as usize;
    let log_half = 0.5f32.ln();
    let mut scores = vec![0.0f32; frames * SPEAKERS];
    for f in 0..frames {
        let row = &probs[f * SPEAKERS..(f + 1) * SPEAKERS];
        let log_complements: Vec<f32> =
            row.iter().map(|p| (1.0 - p).max(SCORE_THRESHOLD).ln()).collect();
        let sum: f32 = log_complements.iter().sum();
        for s in 0..SPEAKERS {
            scores[f * SPEAKERS + s] = if row[s] > 0.5 {
                row[s].max(SCORE_THRESHOLD).ln() - log_complements[s] + sum - log_half
            } else {
                f32::NEG_INFINITY
            };
        }
    }
    for s in 0..SPEAKERS {
        let positive = (0..frames).filter(|f| scores[f * SPEAKERS + s] > 0.0).count();
        if positive >= min_positive {
            for f in 0..frames {
                let v = &mut scores[f * SPEAKERS + s];
                if v.is_finite() && *v <= 0.0 {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }
    scores
}

/// Add `amount` to each speaker's `count` highest scores.
fn boost(scores: &mut [f32], frames: usize, count: usize, amount: f32) {
    let count = count.min(frames);
    let mut order: Vec<usize> = (0..frames).collect();
    for s in 0..SPEAKERS {
        order.sort_by(|a, b| scores[b * SPEAKERS + s].total_cmp(&scores[a * SPEAKERS + s]));
        for f in &order[..count] {
            scores[f * SPEAKERS + s] += amount;
        }
    }
}

/// Keep the `CACHE_LEN` most telling frames, grouped by speaker and in their
/// original order within a speaker, each speaker's group closed by a slot of
/// learned silence; slots nobody earns become silence too.
fn compress(embeds: &[f32], probs: &[f32], silence: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let frames = embeds.len() / HIDDEN;
    let mut scores = frame_scores(probs, frames);
    // Frames past the cache capacity are the ones just popped from the FIFO.
    for v in &mut scores[CACHE_LEN * SPEAKERS..] {
        *v += LATEST_FRAMES_BOOST;
    }
    let log_half = 0.5f32.ln();
    boost(&mut scores, frames, (BUDGET as f32 * STRONG_BOOST_RATE).floor() as usize, -2.0 * log_half);
    boost(&mut scores, frames, (BUDGET as f32 * WEAK_BOOST_RATE).floor() as usize, -log_half);

    // Speaker-major flat index over the frames plus the silence slots, which
    // always win (score +inf).
    let scored = frames + SILENCE_PER_SPEAKER;
    let mut flat: Vec<(f32, usize)> = (0..SPEAKERS)
        .flat_map(|s| (0..scored).map(move |f| (s, f)))
        .map(|(s, f)| {
            let v = if f < frames { scores[f * SPEAKERS + s] } else { f32::INFINITY };
            (v, s * scored + f)
        })
        .collect();
    flat.sort_by(|a, b| b.0.total_cmp(&a.0));
    let sentinel = scored * SPEAKERS;
    let mut picked: Vec<usize> = flat[..CACHE_LEN]
        .iter()
        .map(|&(v, i)| if v == f32::NEG_INFINITY { sentinel } else { i })
        .collect();
    picked.sort_unstable();

    let mut out_embeds = Vec::with_capacity(CACHE_LEN * HIDDEN);
    let mut out_probs = Vec::with_capacity(CACHE_LEN * SPEAKERS);
    for i in picked {
        let frame = if i == sentinel { frames } else { (i % scored).min(frames) };
        if frame < frames {
            out_embeds.extend_from_slice(&embeds[frame * HIDDEN..(frame + 1) * HIDDEN]);
            out_probs.extend_from_slice(&probs[frame * SPEAKERS..(frame + 1) * SPEAKERS]);
        } else {
            out_embeds.extend_from_slice(silence);
            out_probs.extend(std::iter::repeat(0.0).take(SPEAKERS));
        }
    }
    (out_embeds, out_probs)
}

// --------------------------------------------------------------------------
// Probabilities → speaker turns
// --------------------------------------------------------------------------

/// Speech runs shorter than this are dropped (seconds).
const MIN_DURATION_ON: f64 = 0.3;
/// Pauses within one speaker shorter than this are bridged (seconds).
const MIN_DURATION_OFF: f64 = 0.5;

/// One speaker turn in seconds; `speaker` counts from 0 in order of first
/// appearance.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

/// Turn frame probabilities (`frames × SPEAKERS`) into speaker turns: a
/// speaker is active above 0.5; short pauses are bridged and blips dropped
/// with the same durations the classic diarizer uses. A positive
/// `num_speakers` keeps only that many speakers (the ones with the most
/// speech) and gives the others' turns to the speaker of the nearest kept
/// turn. Overlapping speech yields overlapping turns.
pub fn turns(probs: &[f32], num_speakers: i64) -> Vec<SpeakerTurn> {
    let frames = probs.len() / SPEAKERS;
    let mut raw: Vec<(f64, f64, usize)> = Vec::new();
    for s in 0..SPEAKERS {
        let mut runs: Vec<(f64, f64)> = Vec::new();
        let mut start: Option<usize> = None;
        for f in 0..=frames {
            let on = f < frames && probs[f * SPEAKERS + s] > 0.5;
            match (on, start) {
                (true, None) => start = Some(f),
                (false, Some(from)) => {
                    push_bridged(&mut runs, from as f64 * FRAME_SECS, f as f64 * FRAME_SECS);
                    start = None;
                }
                _ => {}
            }
        }
        raw.extend(
            runs.into_iter()
                .filter(|(a, b)| b - a >= MIN_DURATION_ON - 1e-9)
                .map(|(a, b)| (a, b, s)),
        );
    }
    if num_speakers > 0 {
        raw = keep_largest(raw, num_speakers as usize);
    }
    renumber(raw)
}

/// Append a run, merging it into the previous one across a short pause.
fn push_bridged(runs: &mut Vec<(f64, f64)>, from: f64, to: f64) {
    match runs.last_mut() {
        Some(last) if from - last.1 < MIN_DURATION_OFF - 1e-9 => last.1 = last.1.max(to),
        _ => runs.push((from, to)),
    }
}

/// Keep the `n` speakers with the most speech; every other turn goes to the
/// speaker of the nearest kept turn, and each speaker's turns are merged
/// again across short pauses.
fn keep_largest(raw: Vec<(f64, f64, usize)>, n: usize) -> Vec<(f64, f64, usize)> {
    let mut spoken = [0.0f64; SPEAKERS];
    for &(a, b, s) in &raw {
        spoken[s] += b - a;
    }
    let mut ranked: Vec<usize> = (0..SPEAKERS).filter(|&s| spoken[s] > 0.0).collect();
    ranked.sort_by(|a, b| spoken[*b].total_cmp(&spoken[*a]).then(a.cmp(b)));
    let kept: Vec<usize> = ranked.into_iter().take(n.max(1)).collect();
    let anchors: Vec<(f64, f64, usize)> =
        raw.iter().copied().filter(|(_, _, s)| kept.contains(s)).collect();
    if anchors.is_empty() {
        return raw;
    }
    let mut out: Vec<(f64, f64, usize)> = raw
        .into_iter()
        .map(|(a, b, s)| {
            if kept.contains(&s) {
                return (a, b, s);
            }
            let middle = (a + b) / 2.0;
            let distance = |t: &&(f64, f64, usize)| {
                if middle < t.0 {
                    t.0 - middle
                } else {
                    (middle - t.1).max(0.0)
                }
            };
            let nearest = anchors.iter().min_by(|x, y| distance(x).total_cmp(&distance(y)));
            (a, b, nearest.map_or(s, |t| t.2))
        })
        .collect();
    out.sort_by(|x, y| x.2.cmp(&y.2).then(x.0.total_cmp(&y.0)));
    let mut merged: Vec<(f64, f64, usize)> = Vec::with_capacity(out.len());
    for (a, b, s) in out {
        match merged.last_mut() {
            Some(last) if last.2 == s && a - last.1 < MIN_DURATION_OFF - 1e-9 => {
                last.1 = last.1.max(b)
            }
            _ => merged.push((a, b, s)),
        }
    }
    merged
}

/// Sort by start and number the speakers in order of first appearance.
fn renumber(mut raw: Vec<(f64, f64, usize)>) -> Vec<SpeakerTurn> {
    raw.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.2.cmp(&y.2)));
    let mut order: Vec<usize> = Vec::new();
    raw.into_iter()
        .map(|(start, end, s)| {
            let speaker = order.iter().position(|o| *o == s).unwrap_or_else(|| {
                order.push(s);
                order.len() - 1
            });
            SpeakerTurn { start, end, speaker }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `frames × SPEAKERS` probabilities with the given speaker active over
    /// the given frame ranges.
    fn probs(frames: usize, active: &[(usize, std::ops::Range<usize>)]) -> Vec<f32> {
        let mut p = vec![0.0f32; frames * SPEAKERS];
        for (s, range) in active {
            for f in range.clone() {
                p[f * SPEAKERS + s] = 0.9;
            }
        }
        p
    }

    #[test]
    fn short_pauses_are_bridged_and_blips_dropped() {
        // Speaker 0: 0–1 s, a 0.3 s pause, 1.3–2 s (bridged into one turn);
        // speaker 1: a 0.2 s blip at 5 s (dropped) and 3–4 s.
        let p = probs(600, &[(0, 0..100), (0, 130..200), (1, 500..520), (1, 300..400)]);
        let t = turns(&p, -1);
        assert_eq!(
            t,
            vec![
                SpeakerTurn { start: 0.0, end: 2.0, speaker: 0 },
                SpeakerTurn { start: 3.0, end: 4.0, speaker: 1 },
            ]
        );
    }

    #[test]
    fn speakers_are_numbered_by_first_appearance() {
        // Channel 5 speaks first, then channel 2.
        let p = probs(500, &[(5, 0..100), (2, 150..250), (5, 300..400)]);
        let speakers: Vec<usize> = turns(&p, -1).iter().map(|t| t.speaker).collect();
        assert_eq!(speakers, vec![0, 1, 0]);
    }

    #[test]
    fn a_fixed_count_keeps_the_largest_speakers() {
        // Two main speakers and a third with little speech right after
        // speaker 1's turn: asked for two, the third folds into speaker 1.
        let p = probs(1200, &[(0, 0..400), (1, 450..850), (2, 870..950)]);
        let t = turns(&p, 2);
        assert_eq!(
            t,
            vec![
                SpeakerTurn { start: 0.0, end: 4.0, speaker: 0 },
                SpeakerTurn { start: 4.5, end: 9.5, speaker: 1 },
            ]
        );
    }

    #[test]
    fn overlapping_speech_gives_overlapping_turns() {
        let p = probs(400, &[(0, 0..300), (1, 200..400)]);
        let t = turns(&p, -1);
        assert_eq!(t.len(), 2);
        assert!(t[1].start < t[0].end);
    }

    #[test]
    fn mel_bands_match_librosa() {
        let bands = slaney_mel_bands();
        assert_eq!(bands.len(), MELS);
        // Band starts rise with frequency and every band has weight.
        assert!(bands.windows(2).all(|w| w[0].0 <= w[1].0));
        assert!(bands.iter().all(|(_, w)| w.iter().any(|v| *v > 0.0)));
        // Reference values from librosa.filters.mel(sr=16000, n_fft=512,
        // n_mels=128, fmin=0, fmax=8000, norm="slaney"): (band, first
        // non-zero bin, its leading weights, non-zero count).
        let reference: [(usize, usize, &[f32], usize); 3] = [
            (1, 1, &[0.014389008, 0.013988535], 2),
            (60, 49, &[0.011569846, 0.019461744], 2),
            (127, 244, &[4.764134e-5, 0.00093941996, 0.0018311986], 12),
        ];
        for (band, first, weights, len) in reference {
            let (start, w) = &bands[band];
            assert_eq!((*start, w.len()), (first, len), "band {band}");
            for (got, want) in w.iter().zip(weights) {
                assert!((got - want).abs() < 1e-8, "band {band}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn features_are_zero_past_the_audio() {
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mel = LogMel::new(&samples);
        assert_eq!((mel.frames, mel.valid), (11, 10));
        let mut out = Vec::new();
        mel.rows(8, 16, &mut out);
        assert_eq!(out.len(), 8 * MELS);
        // Frames 8, 9 carry audio; 10.. (past `valid`) are zero rows.
        assert!(out[..2 * MELS].iter().all(|v| v.is_finite() && *v != 0.0));
        assert!(out[2 * MELS..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn compress_keeps_cache_length_with_silence_slots() {
        let frames = CACHE_LEN + 40;
        let embeds: Vec<f32> = (0..frames * HIDDEN).map(|i| (i / HIDDEN) as f32).collect();
        // Speaker 0 active on the first half, speaker 1 on the second.
        let mut p = vec![0.0f32; frames * SPEAKERS];
        for f in 0..frames {
            p[f * SPEAKERS + usize::from(f >= frames / 2)] = 0.95;
        }
        let silence = vec![-1.0f32; HIDDEN];
        let (e, pr) = compress(&embeds, &p, &silence);
        assert_eq!(e.len(), CACHE_LEN * HIDDEN);
        assert_eq!(pr.len(), CACHE_LEN * SPEAKERS);
        let rows: Vec<f32> = e.chunks(HIDDEN).map(|r| r[0]).collect();
        // Speaker 0's frames come first, in order, then its silence slot,
        // then speaker 1's frames in order.
        let first_silence = rows.iter().position(|v| *v == -1.0).unwrap();
        assert!(rows[..first_silence].windows(2).all(|w| w[0] < w[1]));
        assert!(rows[..first_silence].iter().all(|v| (*v as usize) < frames / 2));
        assert!(rows[first_silence + 1] as usize >= frames / 2);
    }
}
