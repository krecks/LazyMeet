//! Unit tests for the diarization consolidation pass (port of
//! test_consolidate.py; no models needed).

mod common;

use lazymeet_core::pipeline::{consolidate_speakers, cosine, fake, Turn, Voiceprints};

// backend/config.py defaults, what the worker passes in.
const MERGE_THRESHOLD: f64 = 0.6;
const MIN_SPEAKER_SECONDS: f64 = 15.0;

fn turn(spk: &str, start: f64, dur: f64) -> Turn {
    Turn { start, end: start + dur, speaker: spk.to_string() }
}

fn vp(entries: &[(&str, &[f64])]) -> Voiceprints {
    entries.iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect()
}

#[test]
fn oversplit_labels_merge_to_real_speakers() {
    // A and A2 are the same voice; B is distinct; N is sub-second noise.
    let vp = vp(&[
        ("A", &[1.0, 0.0, 0.0]),
        ("A2", &[0.95, 0.1, 0.0]),
        ("B", &[0.0, 1.0, 0.0]),
        ("N", &[0.0, 0.0, 1.0]),
    ]);
    let turns = vec![
        turn("A", 0.0, 100.0),
        turn("B", 100.0, 50.0),
        turn("A2", 150.0, 10.0),
        turn("N", 160.0, 1.0),
    ];
    let (out_turns, out_vp) =
        consolidate_speakers(&turns, &vp, MERGE_THRESHOLD, MIN_SPEAKER_SECONDS);
    let keys: Vec<&str> = out_vp.keys().map(String::as_str).collect();
    assert_eq!(keys, ["A", "B"]);
    let speakers: Vec<&str> = out_turns.iter().map(|t| t.speaker.as_str()).collect();
    assert_eq!(speakers, ["A", "B", "A"]);
}

#[test]
fn tiny_speaker_folds_into_closest_main_speaker() {
    // cos(C, A) = 0.5 — below the merge threshold but above the tiny floor.
    let vp = vp(&[("A", &[1.0, 0.0, 0.0]), ("B", &[0.0, 1.0, 0.0]), ("C", &[0.5, 0.0, 0.8660254])]);
    let turns = vec![turn("A", 0.0, 60.0), turn("B", 60.0, 60.0), turn("C", 120.0, 8.0)];
    let (out_turns, out_vp) =
        consolidate_speakers(&turns, &vp, MERGE_THRESHOLD, MIN_SPEAKER_SECONDS);
    let keys: Vec<&str> = out_vp.keys().map(String::as_str).collect();
    assert_eq!(keys, ["A", "B"]);
    let speakers: Vec<&str> = out_turns.iter().map(|t| t.speaker.as_str()).collect();
    assert_eq!(speakers, ["A", "B", "A"]);
}

#[test]
fn quiet_distinct_speaker_survives() {
    // 10 s of a genuinely different voice must not be folded or dropped.
    let vp = vp(&[("A", &[1.0, 0.0, 0.0]), ("B", &[0.0, 1.0, 0.0])]);
    let turns = vec![turn("A", 0.0, 60.0), turn("B", 60.0, 10.0)];
    let (out_turns, out_vp) =
        consolidate_speakers(&turns, &vp, MERGE_THRESHOLD, MIN_SPEAKER_SECONDS);
    assert_eq!(out_vp.len(), 2);
    assert_eq!(out_turns.len(), 2);
}

#[test]
fn short_clip_without_main_speaker_kept_as_is() {
    // Both under MIN_SPEAKER_SECONDS: no main speaker, nothing gets touched.
    let vp = vp(&[("A", &[1.0, 0.0, 0.0]), ("B", &[0.0, 1.0, 0.0])]);
    let turns = vec![turn("A", 0.0, 8.0), turn("B", 8.0, 6.0)];
    let (out_turns, out_vp) =
        consolidate_speakers(&turns, &vp, MERGE_THRESHOLD, MIN_SPEAKER_SECONDS);
    assert_eq!(out_vp.len(), 2);
    assert_eq!(out_turns.len(), 2);
}

#[test]
fn fake_embeddings_stay_distinct() {
    // The FAKE pipeline's two speakers must survive consolidation.
    let sim = cosine(&fake::embedding("SPEAKER_00"), &fake::embedding("SPEAKER_01"));
    assert!(sim < MERGE_THRESHOLD, "cos = {sim}");
}
