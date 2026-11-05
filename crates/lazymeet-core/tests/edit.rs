//! Editing operations: meeting patch, segment patch, title plumbing
//! (port of test_edit.py).

mod common;

use common::{changes, obj, TestApp};
use lazymeet_core::{api, store};
use serde_json::json;

fn process_meeting(app: &TestApp, name: &str) -> serde_json::Value {
    let mid = app.upload(name, b"fake video bytes");
    let meeting = app.wait_terminal(&mid);
    assert_eq!(meeting["status"], "done", "{:?}", meeting["error"]);
    meeting
}

// --------------------------------------------------------------------------
// Title
// --------------------------------------------------------------------------

#[test]
fn title_rename_flows_everywhere() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap();
    // processing names it from the notes (see tests/titles.rs)
    assert_eq!(meeting["title"], "Transcription pipeline sync");

    let updated =
        api::patch_meeting(&app.ctx, mid, &changes("title", json!("  Weekly Standup  "))).unwrap();
    assert_eq!(updated["title"], "Weekly Standup");
    assert_eq!(updated["filename"], "edit.mp4"); // original stays immutable

    let listed = api::list_meetings(&app.ctx);
    let entry = listed.as_array().unwrap().iter().find(|m| m["id"].as_str() == Some(mid)).unwrap();
    assert_eq!(entry["title"], "Weekly Standup");
    // the sidebar can sort by length and show how many people spoke
    let last_end = meeting["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["end"].as_f64().unwrap())
        .fold(0.0, f64::max);
    assert_eq!(entry["duration"].as_f64(), Some(last_end));
    assert!(entry["speaker_count"].as_u64().unwrap() >= 1);

    let (md, filename) = api::export_meeting(&app.ctx, mid, "md").unwrap();
    let md = String::from_utf8(md).unwrap();
    assert!(md.starts_with("# Weekly Standup"));
    assert_eq!(filename, "Weekly-Standup.md");
}

#[test]
fn title_rejects_empty_and_too_long() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap();
    let err = api::patch_meeting(&app.ctx, mid, &changes("title", json!("   "))).unwrap_err();
    assert_eq!(err.code, 422);
    let long = "x".repeat(201);
    let err = api::patch_meeting(&app.ctx, mid, &changes("title", json!(long))).unwrap_err();
    assert_eq!(err.code, 422);
}

#[test]
fn title_editable_while_processing() {
    let app = TestApp::fake();
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "busy.mp4", -1, None, None).unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        changes("status", json!("transcribing")),
    )
    .unwrap();
    let updated =
        api::patch_meeting(&app.ctx, mid, &changes("title", json!("Renamed early"))).unwrap();
    assert_eq!(updated["title"], "Renamed early");
}

#[test]
fn old_records_without_title_get_fallback() {
    let app = TestApp::fake();
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "legacy.mp4", -1, None, None)
            .unwrap();
    let mid = record["id"].as_str().unwrap();
    // Simulate a record from before titles existed.
    let path = app.ctx.paths.meetings_dir().join(format!("{mid}.json"));
    let mut data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    data.as_object_mut().unwrap().remove("title");
    std::fs::write(&path, serde_json::to_string(&data).unwrap()).unwrap();

    assert_eq!(api::get_meeting(&app.ctx, mid).unwrap()["title"], "legacy.mp4");
    let listed = api::list_meetings(&app.ctx);
    let entry = listed.as_array().unwrap().iter().find(|m| m["id"].as_str() == Some(mid)).unwrap();
    assert_eq!(entry["title"], "legacy.mp4");
}

// --------------------------------------------------------------------------
// Summary / takeaways / action items
// --------------------------------------------------------------------------

#[test]
fn content_edit_on_done_meeting() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap();
    let data = api::patch_meeting(
        &app.ctx,
        mid,
        &obj(&[
            ("summary", json!("We agreed on the plan.")),
            ("summary_long", json!("Long version.")),
            ("takeaways", json!(["Ship it", "  ", "", "Test it  "])),
            (
                "action_items",
                json!([
                    {"short": "Alice: deploy", "detail": "After the freeze, on Friday."}
                ]),
            ),
        ]),
    )
    .unwrap();
    assert_eq!(data["summary"], "We agreed on the plan.");
    assert_eq!(data["summary_long"], "Long version.");
    // empties dropped, strings upgraded to the {short, detail} shape
    assert_eq!(
        data["takeaways"],
        json!([
            {"short": "Ship it", "detail": ""},
            {"short": "Test it", "detail": ""}
        ])
    );
    assert_eq!(
        data["action_items"],
        json!([{"short": "Alice: deploy", "detail": "After the freeze, on Friday."}])
    );

    let (md, _) = api::export_meeting(&app.ctx, mid, "md").unwrap();
    let md = String::from_utf8(md).unwrap();
    assert!(md.contains("We agreed on the plan."));
    assert!(md.contains("Long version."));
    assert!(md.contains("- [ ] Alice: deploy\n  After the freeze, on Friday."));
}

#[test]
fn content_edit_rejects_bad_summary_values() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap();
    let before = api::get_meeting(&app.ctx, mid).unwrap();

    // A malformed value must not be coerced into "" and wipe the note.
    for bad in [json!(42), json!(null), json!(["a"]), json!({"short": "a"})] {
        for key in ["summary", "summary_long"] {
            let err = api::patch_meeting(&app.ctx, mid, &changes(key, bad.clone())).unwrap_err();
            assert_eq!(err.code, 422, "{key} = {bad} should be rejected");
        }
    }
    // Too long is rejected too; the notes are untouched either way.
    let huge = "x".repeat(20_001);
    let err = api::patch_meeting(&app.ctx, mid, &changes("summary_long", json!(huge))).unwrap_err();
    assert_eq!(err.code, 422);
    assert_eq!(api::get_meeting(&app.ctx, mid).unwrap()["summary"], before["summary"]);

    // The cap itself is fine.
    let ok = "y".repeat(20_000);
    let updated = api::patch_meeting(&app.ctx, mid, &changes("summary_long", json!(ok))).unwrap();
    assert_eq!(updated["summary_long"].as_str().unwrap().len(), 20_000);
}

#[test]
fn content_edit_blocked_while_processing_or_error() {
    let app = TestApp::fake();
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "locked.mp4", -1, None, None)
            .unwrap();
    let mid = record["id"].as_str().unwrap();
    for status in ["queued", "summarizing", "error"] {
        store::update_meeting(
            &app.ctx.paths,
            &app.ctx.locks,
            mid,
            false,
            changes("status", json!(status)),
        )
        .unwrap();
        let err =
            api::patch_meeting(&app.ctx, mid, &changes("summary", json!("nope"))).unwrap_err();
        assert_eq!(err.code, 409, "status {status}");
    }
}

#[test]
fn patch_with_no_fields_is_422() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let err =
        api::patch_meeting(&app.ctx, meeting["id"].as_str().unwrap(), &serde_json::Map::new())
            .unwrap_err();
    assert_eq!(err.code, 422);
}

// --------------------------------------------------------------------------
// Segments
// --------------------------------------------------------------------------

#[test]
fn segment_text_edit() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap();
    let updated =
        api::patch_segment(&app.ctx, mid, 0, &changes("text", json!("Corrected words."))).unwrap();
    assert_eq!(updated["segments"][0]["text"], "Corrected words.");
    // other segments untouched
    assert_eq!(
        updated["segments"].as_array().unwrap()[1..],
        meeting["segments"].as_array().unwrap()[1..]
    );
}

#[test]
fn segment_speaker_reassignment() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap();
    let ids: Vec<String> = meeting["speakers"].as_object().unwrap().keys().cloned().collect();
    assert!(ids.len() >= 2, "fake pipeline should produce two speakers");
    let other =
        if meeting["segments"][0]["speaker"].as_str() == Some(&ids[0]) { &ids[1] } else { &ids[0] };

    let updated = api::patch_segment(&app.ctx, mid, 0, &changes("speaker", json!(other))).unwrap();
    assert_eq!(updated["segments"][0]["speaker"].as_str(), Some(other.as_str()));
    // speakers map and rename history stay intact
    assert_eq!(updated["speakers"], meeting["speakers"]);
}

#[test]
fn segment_edit_error_cases() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap();
    let err = api::patch_segment(&app.ctx, mid, 999, &changes("text", json!("x"))).unwrap_err();
    assert_eq!(err.code, 404);
    let err =
        api::patch_segment(&app.ctx, mid, 0, &changes("speaker", json!("SPEAKER_99"))).unwrap_err();
    assert_eq!(err.code, 422);
    let err = api::patch_segment(&app.ctx, mid, 0, &changes("text", json!("   "))).unwrap_err();
    assert_eq!(err.code, 422);
    let err = api::patch_segment(&app.ctx, mid, 0, &serde_json::Map::new()).unwrap_err();
    assert_eq!(err.code, 422);
    let err = api::patch_segment(&app.ctx, "nope", 0, &changes("text", json!("x"))).unwrap_err();
    assert_eq!(err.code, 404);
}

#[test]
fn segment_edit_blocked_while_processing() {
    let app = TestApp::fake();
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "busyseg.mp4", -1, None, None)
            .unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[
            ("status", json!("summarizing")),
            ("segments", json!([{"start": 0, "end": 1, "speaker": "s0", "text": "hi"}])),
            ("speakers", json!({"s0": "Speaker 1"})),
        ]),
    )
    .unwrap();
    let err = api::patch_segment(&app.ctx, mid, 0, &changes("text", json!("x"))).unwrap_err();
    assert_eq!(err.code, 409);
}

#[test]
fn concurrent_segment_edits_both_survive() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "edit.mp4");
    let mid = meeting["id"].as_str().unwrap().to_string();
    assert!(meeting["segments"].as_array().unwrap().len() >= 2);

    let mut handles = Vec::new();
    for i in [0usize, 1usize] {
        let ctx = app.ctx.clone();
        let mid = mid.clone();
        handles.push(std::thread::spawn(move || {
            store::update_segment(
                &ctx.paths,
                &ctx.locks,
                &mid,
                i,
                Some(&format!("edit {i}")),
                None,
            )
            .unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let record = store::get_meeting(&app.ctx.paths, &app.ctx.locks, &mid).unwrap();
    assert_eq!(record["segments"][0]["text"], "edit 0");
    assert_eq!(record["segments"][1]["text"], "edit 1");
}
