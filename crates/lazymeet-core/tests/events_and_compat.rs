//! New coverage the Python suite couldn't have: event-sequence assertions and
//! compatibility with records the Python backend wrote to disk.

mod common;

use common::{status_events, TestApp};
use lazymeet_core::{api, store};
use serde_json::json;

#[test]
fn worker_emits_stage_events_in_order() {
    let app = TestApp::fake();
    let mid = app.upload("events.mp4", b"fake video bytes");
    let meeting = app.wait_terminal(&mid);
    assert_eq!(meeting["status"], "done");

    assert_eq!(
        status_events(&app.events, &mid),
        vec!["extracting", "transcribing", "diarizing", "identifying", "summarizing", "done"]
    );
    // Terminal events carry the full public meeting; intermediates don't.
    let payloads = app.events.payloads("meeting-updated");
    let done = payloads.iter().find(|p| p["status"] == "done").unwrap();
    assert_eq!(done["meeting"]["id"].as_str(), Some(mid.as_str()));
    assert!(done["meeting"].get("voiceprints").is_none());
    let first = payloads.iter().find(|p| p["status"] == "extracting").unwrap();
    assert!(first.get("meeting").is_none());
}

#[test]
fn python_written_meeting_loads_identically() {
    // A meeting record produced by the real Python backend (fixture) must be
    // readable, publicly identical (internals stripped), editable, and its
    // unknown internal fields must survive a rewrite untouched.
    let app = TestApp::fake();
    let fixture = include_str!("fixtures/python_meeting.json");
    let record: serde_json::Value = serde_json::from_str(fixture).unwrap();
    let mid = record["id"].as_str().unwrap();
    std::fs::write(app.ctx.paths.meetings_dir().join(format!("{mid}.json")), fixture).unwrap();

    let meeting = api::get_meeting(&app.ctx, mid).unwrap();
    assert_eq!(meeting["title"], "standup.mp4");
    assert_eq!(meeting["status"], "done");
    assert_eq!(meeting["speakers"], json!({"SPEAKER_00": "Anna", "SPEAKER_01": "Speaker 2"}));
    assert_eq!(meeting["recognized"], json!(["SPEAKER_00"]));
    assert_eq!(meeting["segments"].as_array().unwrap().len(), 2);
    // Python-internal fields are stripped from the public view...
    assert!(meeting.get("voiceprints").is_none());
    assert!(meeting.get("worker_pid").is_none());
    assert_eq!(meeting["has_audio"], false); // no upload on disk

    // ...but survive an edit round-trip in the stored record.
    api::patch_meeting(&app.ctx, mid, &common::changes("title", json!("Ported Standup"))).unwrap();
    let stored = store::get_meeting(&app.ctx.paths, &app.ctx.locks, mid).unwrap();
    assert_eq!(stored["worker_pid"], 12345);
    assert_eq!(stored["voiceprints"]["SPEAKER_00"], json!([0.1, 0.2]));
    assert_eq!(stored["title"], "Ported Standup");
}

#[test]
fn python_written_speaker_library_matches() {
    // speakers.json written by Python: entries keep working for matching and
    // enrolment (sample append + centroid recompute).
    let app = TestApp::fake();
    std::fs::write(app.ctx.paths.speakers_file(), include_str!("fixtures/python_speakers.json"))
        .unwrap();

    let names = store::get_speaker_names(&app.ctx.paths, &app.ctx.locks);
    assert_eq!(names, ["Anna"]);
    let lib = store::load_speaker_library(&app.ctx.paths, &app.ctx.locks);
    assert_eq!(lib["Anna"]["embedding"], json!([0.1, 0.2]));

    // Enrol another sample — running mean over both.
    store::upsert_speaker(&app.ctx.paths, &app.ctx.locks, "Anna", &[0.3, 0.4]);
    let lib = store::load_speaker_library(&app.ctx.paths, &app.ctx.locks);
    let emb: Vec<f64> =
        lib["Anna"]["embedding"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!((emb[0] - 0.2).abs() < 1e-9 && (emb[1] - 0.3).abs() < 1e-9);
    assert_eq!(lib["Anna"]["samples"].as_array().unwrap().len(), 2);
}

#[test]
fn created_at_sorts_against_python_timestamps() {
    // Python wrote isoformat() with "+00:00"; new records must interleave
    // correctly in the string-sorted meeting list.
    let app = TestApp::fake();
    let fixture = include_str!("fixtures/python_meeting.json");
    let record: serde_json::Value = serde_json::from_str(fixture).unwrap();
    let mid = record["id"].as_str().unwrap();
    std::fs::write(app.ctx.paths.meetings_dir().join(format!("{mid}.json")), fixture).unwrap();
    let ours =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "new.mp4", -1, None, None).unwrap();

    // Same timestamp format (microseconds + +00:00 offset)...
    let py_ts = record["created_at"].as_str().unwrap();
    let rs_ts = ours["created_at"].as_str().unwrap();
    let shape = |ts: &str| (ts.len(), ts.contains('T'), ts.ends_with("+00:00"), ts.contains('.'));
    assert_eq!(shape(py_ts), shape(rs_ts), "python: {py_ts} rust: {rs_ts}");

    // ...and the new record (created later) lists first.
    let listed = api::list_meetings(&app.ctx);
    assert_eq!(listed[0]["id"], ours["id"]);
    assert_eq!(listed[1]["id"].as_str(), Some(mid));
}
