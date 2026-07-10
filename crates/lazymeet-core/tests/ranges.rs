//! Time-range selection: normalization, concat→original remapping, and the
//! end-to-end contract that a meeting created with ranges only transcribes
//! the selected spans (timestamps staying in original-media time).

mod common;

use common::TestApp;
use lazymeet_core::api;
use lazymeet_core::pipeline::{normalize_ranges, remap_turns, Turn};

fn turn(start: f64, end: f64, speaker: &str) -> Turn {
    Turn { start, end, speaker: speaker.into() }
}

// --------------------------------------------------------------------------
// normalize_ranges
// --------------------------------------------------------------------------

#[test]
fn normalize_sorts_and_merges_overlaps() {
    let got = normalize_ranges(&[(4.0, 9.0), (1.0, 5.0)]).unwrap();
    assert_eq!(got, vec![(1.0, 9.0)]);
}

#[test]
fn normalize_merges_near_touching_neighbors() {
    // 0.005 s apart — closer than the 0.01 s merge gap.
    let got = normalize_ranges(&[(0.0, 2.0), (2.005, 4.0)]).unwrap();
    assert_eq!(got, vec![(0.0, 4.0)]);
    // A real gap survives.
    let got = normalize_ranges(&[(0.0, 2.0), (3.0, 4.0)]).unwrap();
    assert_eq!(got, vec![(0.0, 2.0), (3.0, 4.0)]);
}

#[test]
fn normalize_clamps_negative_starts() {
    let got = normalize_ranges(&[(-1.0, 2.0)]).unwrap();
    assert_eq!(got, vec![(0.0, 2.0)]);
}

#[test]
fn normalize_drops_degenerate_ranges() {
    // Empty, inverted, sub-100ms, and non-finite entries all go; the good one stays.
    let got = normalize_ranges(&[
        (5.0, 5.0),
        (7.0, 3.0),
        (1.0, 1.05),
        (f64::NAN, 2.0),
        (0.0, f64::INFINITY),
        (10.0, 12.0),
    ])
    .unwrap();
    assert_eq!(got, vec![(10.0, 12.0)]);
}

#[test]
fn normalize_errors_when_nothing_survives() {
    assert!(normalize_ranges(&[]).is_err());
    assert!(normalize_ranges(&[(5.0, 5.0), (9.0, 2.0)]).is_err());
}

// --------------------------------------------------------------------------
// remap_turns (concatenated-selection time → original-media time)
// --------------------------------------------------------------------------

#[test]
fn remap_splits_turn_straddling_a_splice() {
    // Selection: 10–20 and 40–50 (concat timeline: 0–10 then 10–20).
    // A turn 5–15 in concat time crosses the splice at concat 10.
    let ranges = [(10.0, 20.0), (40.0, 50.0)];
    let got = remap_turns(&ranges, &[turn(5.0, 15.0, "SPEAKER_00")]);
    assert_eq!(got.len(), 2);
    assert_eq!((got[0].start, got[0].end, got[0].speaker.as_str()), (15.0, 20.0, "SPEAKER_00"));
    assert_eq!((got[1].start, got[1].end, got[1].speaker.as_str()), (40.0, 45.0, "SPEAKER_00"));
}

#[test]
fn remap_offsets_turn_inside_one_range() {
    let ranges = [(10.0, 20.0), (40.0, 50.0)];
    // Concat 12–18 lies entirely in the second range (concat 10–20).
    let got = remap_turns(&ranges, &[turn(12.0, 18.0, "SPEAKER_01")]);
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].start, got[0].end), (42.0, 48.0));
}

#[test]
fn remap_clamps_past_selected_length_and_sorts() {
    let ranges = [(10.0, 20.0), (40.0, 50.0)];
    // End beyond the 20 s of selected audio clamps to the last range's end;
    // a later-emitted early turn still comes out sorted.
    let got = remap_turns(&ranges, &[turn(15.0, 99.0, "A"), turn(0.0, 2.0, "B")]);
    assert_eq!(got.len(), 2);
    assert_eq!((got[0].start, got[0].end, got[0].speaker.as_str()), (10.0, 12.0, "B"));
    assert_eq!((got[1].start, got[1].end, got[1].speaker.as_str()), (45.0, 50.0, "A"));
}

// --------------------------------------------------------------------------
// End-to-end over the FAKE pipeline
// --------------------------------------------------------------------------

#[test]
fn ranged_meeting_transcribes_only_the_selection() {
    let app = TestApp::fake();
    let id = app.upload_ranges("clip.mp4", b"fake video bytes", &[(3.4, 12.0)]);
    let meeting = app.wait_terminal(&id);
    assert_eq!(meeting["status"], "done", "error: {:?}", meeting["error"]);

    // The selection round-trips on the record.
    assert_eq!(meeting["ranges"], serde_json::json!([[3.4, 12.0]]));

    // Every transcript segment lies inside the selection, in original time.
    let segments = meeting["segments"].as_array().unwrap();
    assert!(!segments.is_empty());
    for seg in segments {
        let start = seg["start"].as_f64().unwrap();
        let end = seg["end"].as_f64().unwrap();
        assert!(start >= 3.4 - 1e-6 && end <= 12.0 + 1e-6, "segment out of range: {seg}");
    }
    // Content outside the selection didn't make it in.
    let text = segments.iter().map(|s| s["text"].as_str().unwrap()).collect::<Vec<_>>().join(" ");
    assert!(!text.contains("Hi everyone"), "pre-selection utterance leaked: {text}");
    assert!(text.contains("transcription pipeline status"));

    // Speakers were still diarized and named.
    assert!(!meeting["speakers"].as_object().unwrap().is_empty());
}

#[test]
fn empty_selection_is_a_422() {
    let app = TestApp::fake();
    let src = app.tmp.path().join("incoming");
    std::fs::create_dir_all(&src).unwrap();
    let path = src.join("clip.mp4");
    std::fs::write(&path, b"fake").unwrap();
    let err = api::create_meeting(&app.ctx, &path, -1, Some(&[(5.0, 5.0)]), None).unwrap_err();
    assert_eq!(err.code, 422);
}

#[test]
fn ranges_are_normalized_before_storage() {
    let app = TestApp::fake();
    let id = app.upload_ranges("norm.mp4", b"fake", &[(-1.0, 2.0), (1.0, 4.0)]);
    let meeting = app.wait_terminal(&id);
    // Clamped to 0 and merged into one span.
    assert_eq!(meeting["ranges"], serde_json::json!([[0.0, 4.0]]));
}

#[test]
fn no_ranges_keeps_todays_behavior() {
    let app = TestApp::fake();
    let id = app.upload("full.mp4", b"fake");
    let meeting = app.wait_terminal(&id);
    assert_eq!(meeting["status"], "done");
    assert_eq!(meeting["ranges"], serde_json::Value::Null);
    // The canonical fake transcript is intact — all five utterances.
    let segments = meeting["segments"].as_array().unwrap();
    assert_eq!(segments.len(), 5);
    assert_eq!(segments[0]["start"].as_f64().unwrap(), 0.0);
    assert_eq!(segments[4]["end"].as_f64().unwrap(), 19.0);
}
