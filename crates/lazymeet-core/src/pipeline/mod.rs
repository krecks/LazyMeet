//! The processing pipeline: audio extraction, ASR, diarization, speaker
//! identification, merge, summarization.
//!
//! The pure-math parts (cosine, consolidation, matching, merge) are shared by
//! FAKE and native modes and are ports of `backend/pipeline/identify.py` /
//! `merge.py`. Heavy ML stages live in `native` (feature `native-pipeline`);
//! FAKE mode swaps them for the deterministic stubs in `fake`.

pub mod fake;
#[cfg(feature = "native-pipeline")]
pub mod native;
#[cfg(feature = "native-pipeline")]
pub mod nemotron;

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::Config;

/// One transcribed utterance (before speaker assignment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// One diarization turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

/// A transcript segment with its speaker attached (field order matches the
/// JSON the Python backend wrote: start, end, speaker, text).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub summary: String,
    pub takeaways: Vec<String>,
    pub action_items: Vec<String>,
}

pub type Voiceprints = BTreeMap<String, Vec<f64>>;

/// Duration + downsampled waveform peaks for the pre-transcription prep
/// screen (`probe_media`).
#[derive(Debug, Clone, Serialize)]
pub struct MediaProbe {
    pub duration: f64,
    pub peaks: Vec<f32>,
}

// --------------------------------------------------------------------------
// Time-range selection (transcribe only parts of a recording)
// --------------------------------------------------------------------------

/// A user-selected span of the recording, in seconds of original-media time.
pub type TimeRange = (f64, f64);

/// Ranges shorter than this are accidental clicks and are dropped.
const MIN_RANGE_SECS: f64 = 0.1;
/// Neighbors closer than this merge into one range.
const MERGE_GAP_SECS: f64 = 0.01;

pub fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// Clean up user-supplied ranges: clamp negative starts to zero, drop
/// non-finite / inverted / sub-100 ms entries, sort, and merge overlapping or
/// near-touching neighbors. Ends past the recording are left alone — the true
/// duration isn't known here and the pipeline clamps to the decoded samples.
pub fn normalize_ranges(raw: &[TimeRange]) -> Result<Vec<TimeRange>, String> {
    let mut ranges: Vec<TimeRange> = raw
        .iter()
        .filter(|(s, e)| s.is_finite() && e.is_finite())
        .map(|&(s, e)| (s.max(0.0), e))
        .filter(|(s, e)| e - s >= MIN_RANGE_SECS)
        .collect();
    ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<TimeRange> = Vec::new();
    for (s, e) in ranges {
        match merged.last_mut() {
            Some(last) if s <= last.1 + MERGE_GAP_SECS => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    if merged.is_empty() {
        return Err("no usable time ranges selected".into());
    }
    Ok(merged)
}

/// Map diarization turns from the concatenated-selection timeline back to
/// original-media time. `ranges` must be normalized (sorted, disjoint). A turn
/// that straddles a splice boundary is split at it — turns carry no text, so
/// the split is lossless.
pub fn remap_turns(ranges: &[TimeRange], turns: &[Turn]) -> Vec<Turn> {
    // cum[k] = where range k starts on the concatenated timeline.
    let mut cum = Vec::with_capacity(ranges.len());
    let mut total = 0.0;
    for (s, e) in ranges {
        cum.push(total);
        total += e - s;
    }
    let mut out = Vec::new();
    for turn in turns {
        for (k, &(s, e)) in ranges.iter().enumerate() {
            let lo = turn.start.max(cum[k]);
            let hi = turn.end.min(cum[k] + (e - s));
            if hi - lo > 1e-6 {
                out.push(Turn {
                    start: round3(s + lo - cum[k]),
                    end: round3(s + hi - cum[k]),
                    speaker: turn.speaker.clone(),
                });
            }
        }
    }
    out.sort_by(|a, b| a.start.total_cmp(&b.start));
    out
}

/// Language names the summarizer can be instructed in; the same codes are
/// valid Whisper language hints ("auto" is added at the API layer).
pub const LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
];

pub struct PipelineCtx<'a> {
    pub fake: bool,
    pub config: &'a Config,
    pub models_dir: &'a Path,
    /// Absolute path of the bundled ffmpeg binary (native mode only).
    pub ffmpeg: Option<std::path::PathBuf>,
}

// --------------------------------------------------------------------------
// Stage dispatch (fake vs native)
// --------------------------------------------------------------------------

/// Decode the upload to the 16 kHz mono WAV the ASR needs and — when
/// `playback` is given — the AAC playback track in the same pass, so a large
/// video is read once instead of again at the end of the run.
pub fn extract_audio(
    ctx: &PipelineCtx,
    src: &Path,
    out_wav: &Path,
    playback: Option<&Path>,
) -> Result<(), String> {
    if let Some(parent) = out_wav.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if ctx.fake {
        fake::write_silent_wav(out_wav).map_err(|e| e.to_string())?;
        if let Some(playback) = playback {
            std::fs::copy(src, playback).map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    native_only(ctx, |_| native_impl::extract_audio(ctx, src, out_wav, playback))
}

pub fn extract_playback_audio(ctx: &PipelineCtx, src: &Path, out_m4a: &Path) -> Result<(), String> {
    if let Some(parent) = out_m4a.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if ctx.fake {
        std::fs::copy(src, out_m4a).map_err(|e| e.to_string())?;
        return Ok(());
    }
    native_only(ctx, |_| native_impl::extract_playback_audio(ctx, src, out_m4a))
}

/// `ranges` restricts transcription to the selected spans (None = whole
/// file); timestamps stay in original-media time either way.
pub fn transcribe(
    ctx: &PipelineCtx,
    wav: &Path,
    ranges: Option<&[TimeRange]>,
) -> Result<Vec<Segment>, String> {
    if ctx.fake {
        return Ok(fake::transcribe(ranges));
    }
    native_only(ctx, |_| native_impl::transcribe(ctx, wav, ranges, ctx.config.num_threads() as i32))
}

pub fn diarize(
    ctx: &PipelineCtx,
    wav: &Path,
    num_speakers: i64,
    ranges: Option<&[TimeRange]>,
) -> Result<Vec<Turn>, String> {
    if ctx.fake {
        return Ok(fake::diarize(ranges));
    }
    native_only(ctx, |_| {
        native_impl::diarize(ctx, wav, num_speakers, ranges, ctx.config.num_threads() as i32)
    })
}

/// The diarization engine a meeting runs on ("nemotron" | "pyannote"): the
/// selected model, the classic one for more speakers than Nemotron follows,
/// the default while a selection is still downloading. A meeting-supplied
/// speaker count (>0) wins over the env default.
pub fn diar_engine(ctx: &PipelineCtx, num_speakers: i64) -> &'static str {
    let n = if num_speakers > 0 { num_speakers } else { crate::config::env_num_speakers() };
    crate::models::resolve_diar_at(ctx.models_dir, ctx.fake, &ctx.config.diar_model(), n).0
}

/// Transcription and diarization read the same WAV with disjoint models, so
/// the native pipeline runs them concurrently. `on_transcribed` fires on the
/// calling thread the moment transcription finishes, letting the caller
/// advance the visible status to "diarizing" — the stage order the UI sees
/// is unchanged, the diarizing phase just shrinks to whatever remains after
/// transcription. A transcription failure wins over a diarization one,
/// matching the old sequential order.
pub fn transcribe_and_diarize(
    ctx: &PipelineCtx,
    wav: &Path,
    num_speakers: i64,
    ranges: Option<&[TimeRange]>,
    on_transcribed: impl FnOnce(),
) -> Result<(Vec<Segment>, Vec<Turn>), String> {
    if ctx.fake {
        let segments = fake::transcribe(ranges);
        on_transcribed();
        return Ok((segments, fake::diarize(ranges)));
    }
    native_only(ctx, |_| {
        // Each stage gets half the thread budget, rounded up: they contend
        // for the same cores, and two half-budget stages side by side finish
        // well ahead of two full-budget ones fighting over the machine
        // (190 s vs 260 s for a 28.5-minute recording on an 8P+2E M1 Pro).
        let half = (ctx.config.num_threads() as i32 + 1) / 2;
        std::thread::scope(|s| {
            let diar = s.spawn(|| native_impl::diarize(ctx, wav, num_speakers, ranges, half));
            let segments = native_impl::transcribe(ctx, wav, ranges, half);
            if segments.is_ok() {
                on_transcribed();
            }
            let turns = diar.join().unwrap_or_else(|_| Err("diarization panicked".into()));
            Ok((segments?, turns?))
        })
    })
}

/// Duration + waveform peaks for the prep screen, from one ffmpeg decode.
pub fn probe_media(ctx: &PipelineCtx, src: &Path) -> Result<MediaProbe, String> {
    if ctx.fake {
        return Ok(fake::probe());
    }
    native_only(ctx, |_| native_impl::probe_media(ctx, src))
}

pub fn compute_voiceprints(
    ctx: &PipelineCtx,
    wav: &Path,
    turns: &[Turn],
) -> Result<Voiceprints, String> {
    if ctx.fake {
        return Ok(fake::voiceprints(turns));
    }
    native_only(ctx, |_| native_impl::compute_voiceprints(ctx, wav, turns))
}

pub fn summarize(
    ctx: &PipelineCtx,
    segments: &[MergedSegment],
    speakers: &Map<String, Value>,
) -> Result<Summary, String> {
    if ctx.fake {
        return Ok(fake::summary());
    }
    native_only(ctx, |_| native_impl::summarize(ctx, segments, speakers))
}

#[cfg(feature = "native-pipeline")]
use native as native_impl;

#[cfg(feature = "native-pipeline")]
fn native_only<T>(
    _ctx: &PipelineCtx,
    f: impl FnOnce(&PipelineCtx) -> Result<T, String>,
) -> Result<T, String> {
    f(_ctx)
}

#[cfg(not(feature = "native-pipeline"))]
mod native_impl {
    // Never called: native_only() errors first when the feature is off.
    use super::*;
    pub fn extract_audio(
        _: &PipelineCtx,
        _: &Path,
        _: &Path,
        _: Option<&Path>,
    ) -> Result<(), String> {
        unreachable!()
    }
    pub fn extract_playback_audio(_: &PipelineCtx, _: &Path, _: &Path) -> Result<(), String> {
        unreachable!()
    }
    pub fn transcribe(
        _: &PipelineCtx,
        _: &Path,
        _: Option<&[TimeRange]>,
        _: i32,
    ) -> Result<Vec<Segment>, String> {
        unreachable!()
    }
    pub fn diarize(
        _: &PipelineCtx,
        _: &Path,
        _: i64,
        _: Option<&[TimeRange]>,
        _: i32,
    ) -> Result<Vec<Turn>, String> {
        unreachable!()
    }
    pub fn probe_media(_: &PipelineCtx, _: &Path) -> Result<MediaProbe, String> {
        unreachable!()
    }
    pub fn compute_voiceprints(
        _: &PipelineCtx,
        _: &Path,
        _: &[Turn],
    ) -> Result<Voiceprints, String> {
        unreachable!()
    }
    pub fn summarize(
        _: &PipelineCtx,
        _: &[MergedSegment],
        _: &Map<String, Value>,
    ) -> Result<Summary, String> {
        unreachable!()
    }
}

#[cfg(not(feature = "native-pipeline"))]
fn native_only<T>(
    _ctx: &PipelineCtx,
    _f: impl FnOnce(&PipelineCtx) -> Result<T, String>,
) -> Result<T, String> {
    Err("the native pipeline was not compiled in (feature `native-pipeline`)".into())
}

// --------------------------------------------------------------------------
// Speaker identification math (mode-independent; port of identify.py)
// --------------------------------------------------------------------------

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return -1.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return -1.0;
    }
    dot / (na * nb)
}

/// A short speaker folds into a main speaker above this similarity.
const TINY_MERGE_FLOOR: f64 = 0.4;
/// Below the floor and shorter than this => dropped as noise.
const NOISE_MAX_SECONDS: f64 = 5.0;

/// Collapse over-split diarization labels into real speakers.
///
/// (1) greedily merge labels whose long-window voiceprints match at
/// `merge_threshold`, folding the shorter speaker into the longer one; (2)
/// fold remaining sub-`min_speaker_seconds` speakers into the closest main
/// speaker; (3) drop tiny fragments that resemble no one — their transcript
/// segments reattach to the nearest surviving turn in `assign_speakers`.
pub fn consolidate_speakers(
    turns: &[Turn],
    voiceprints: &Voiceprints,
    merge_threshold: f64,
    min_speaker_seconds: f64,
) -> (Vec<Turn>, Voiceprints) {
    let mut dur: BTreeMap<String, f64> = BTreeMap::new();
    for t in turns {
        *dur.entry(t.speaker.clone()).or_insert(0.0) += t.end - t.start;
    }

    let mut vp = voiceprints.clone();
    let mut alias: BTreeMap<String, String> = BTreeMap::new(); // folded -> surviving

    fn fold(
        alias: &mut BTreeMap<String, String>,
        dur: &mut BTreeMap<String, f64>,
        vp: &mut Voiceprints,
        loser: &str,
        winner: &str,
    ) {
        for v in alias.values_mut() {
            if v == loser {
                *v = winner.to_string();
            }
        }
        alias.insert(loser.to_string(), winner.to_string());
        let lost = dur.remove(loser).unwrap_or(0.0);
        *dur.entry(winner.to_string()).or_insert(0.0) += lost;
        vp.remove(loser);
    }

    // Stage 1: merge matching voiceprints, highest similarity first. Survivors
    // keep their voiceprints, so pair scores never change and one sorted pass
    // over the precomputed pairs is exact greedy merging.
    let spks: Vec<String> = vp.keys().cloned().collect(); // sorted (BTreeMap)
    let mut pairs: Vec<(f64, String, String)> = Vec::new();
    for (i, a) in spks.iter().enumerate() {
        for b in &spks[i + 1..] {
            pairs.push((cosine(&vp[a], &vp[b]), a.clone(), b.clone()));
        }
    }
    // Python: sorted(..., reverse=True) over (score, a, b) tuples.
    pairs.sort_by(|x, y| {
        y.0.total_cmp(&x.0).then_with(|| y.1.cmp(&x.1)).then_with(|| y.2.cmp(&x.2))
    });
    for (score, a, b) in &pairs {
        if *score < merge_threshold {
            break;
        }
        if alias.contains_key(a) || alias.contains_key(b) {
            continue;
        }
        let (loser, winner) =
            if dur.get(a).copied().unwrap_or(0.0) < dur.get(b).copied().unwrap_or(0.0) {
                (a, b)
            } else {
                (b, a)
            };
        fold(&mut alias, &mut dur, &mut vp, loser, winner);
    }

    // Stages 2+3: deal with what's left below min_speaker_seconds. Without any
    // main speaker (e.g. a very short clip) everything is kept as-is.
    let cores: Vec<String> = vp
        .keys()
        .filter(|s| dur.get(*s).copied().unwrap_or(0.0) >= min_speaker_seconds)
        .cloned()
        .collect();
    let mut dropped: std::collections::BTreeSet<String> = Default::default();
    if !cores.is_empty() {
        let smalls: Vec<String> = vp.keys().filter(|s| !cores.contains(s)).cloned().collect();
        for spk in smalls {
            // max(cores, key=cosine) — first max wins ties, cores in order.
            let mut best: Option<(&String, f64)> = None;
            for c in &cores {
                let score = cosine(&vp[&spk], &vp[c]);
                if best.map(|(_, s)| score > s).unwrap_or(true) {
                    best = Some((c, score));
                }
            }
            let (best_core, best_score) = best.unwrap();
            let best_core = best_core.clone();
            if best_score >= TINY_MERGE_FLOOR {
                fold(&mut alias, &mut dur, &mut vp, &spk, &best_core);
            } else if dur.get(&spk).copied().unwrap_or(0.0) < NOISE_MAX_SECONDS {
                dropped.insert(spk.clone());
                vp.remove(&spk);
            }
        }
        // Labels with no usable voiceprint at all can't be compared to anyone.
        let all_labels: std::collections::BTreeSet<String> =
            turns.iter().map(|t| t.speaker.clone()).collect();
        for spk in all_labels {
            if !vp.contains_key(&spk)
                && !alias.contains_key(&spk)
                && !dropped.contains(&spk)
                && dur.get(&spk).copied().unwrap_or(0.0) < NOISE_MAX_SECONDS
            {
                dropped.insert(spk);
            }
        }
    }

    let mut out = Vec::new();
    for t in turns {
        let spk = alias.get(&t.speaker).cloned().unwrap_or_else(|| t.speaker.clone());
        if dropped.contains(&spk) {
            continue;
        }
        out.push(Turn { start: t.start, end: t.end, speaker: spk });
    }
    (out, vp)
}

/// Map diarized ids to display names using the saved speaker library.
///
/// Returns `(speakers, recognized)` where `speakers` maps every diarized id to
/// a display name (saved name if matched, otherwise "Speaker N") and
/// `recognized` lists the ids that were auto-matched.
pub fn match_speakers(
    voiceprints: &Voiceprints,
    library: &Map<String, Value>,
    threshold: f64,
) -> (Map<String, Value>, Vec<String>) {
    let mut speakers = Map::new();
    let mut recognized = Vec::new();

    // Stable "Speaker N" numbering from the sorted diarized ids (BTreeMap keys).
    for (idx, (spk, emb)) in voiceprints.iter().enumerate() {
        let mut best_name: Option<&str> = None;
        let mut best_score = -1.0f64;
        for (name, entry) in library {
            let lib_emb: Vec<f64> = entry
                .get("embedding")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default();
            let score = cosine(emb, &lib_emb);
            if score > best_score {
                best_name = Some(name);
                best_score = score;
            }
        }
        match best_name {
            Some(name) if best_score >= threshold => {
                speakers.insert(spk.clone(), serde_json::json!(name));
                recognized.push(spk.clone());
            }
            _ => {
                speakers.insert(spk.clone(), serde_json::json!(format!("Speaker {}", idx + 1)));
            }
        }
    }
    (speakers, recognized)
}

// --------------------------------------------------------------------------
// Merge (port of merge.py)
// --------------------------------------------------------------------------

fn overlap(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

/// Attach a speaker to each transcript segment by maximum temporal overlap.
pub fn assign_speakers(segments: &[Segment], turns: &[Turn]) -> Vec<MergedSegment> {
    let mut merged = Vec::with_capacity(segments.len());
    for seg in segments {
        let mut best_speaker: Option<&str> = None;
        let mut best_ov = 0.0f64;
        for turn in turns {
            let ov = overlap(seg.start, seg.end, turn.start, turn.end);
            if ov > best_ov {
                best_speaker = Some(&turn.speaker);
                best_ov = ov;
            }
        }
        if best_speaker.is_none() && !turns.is_empty() {
            // No overlap (e.g. boundary segment): fall back to nearest turn.
            let mid = (seg.start + seg.end) / 2.0;
            let mut best: Option<(&Turn, f64)> = None;
            for t in turns {
                let d = ((t.start + t.end) / 2.0 - mid).abs();
                if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                    best = Some((t, d));
                }
            }
            best_speaker = best.map(|(t, _)| t.speaker.as_str());
        }
        merged.push(MergedSegment {
            start: seg.start,
            end: seg.end,
            speaker: best_speaker.filter(|s| !s.is_empty()).unwrap_or("SPEAKER_00").to_string(),
            text: seg.text.clone(),
        });
    }
    merged
}
