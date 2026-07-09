//! End-to-end smoke tests over the FAKE pipeline (port of test_smoke.py).

mod common;

use common::{changes, obj, status_events, TestApp};
use lazymeet_core::{api, jobs, store, Ctx};
use serde_json::json;

#[test]
fn full_lifecycle() {
    let app = TestApp::fake();
    let mid = app.upload("standup.mp4", b"fake video bytes");

    let meeting = app.wait_terminal(&mid);
    assert_eq!(meeting["status"], "done", "{:?}", meeting["error"]);
    assert!(!meeting["segments"].as_array().unwrap().is_empty());
    assert!(!meeting["speakers"].as_object().unwrap().is_empty());
    assert!(!meeting["summary"].as_str().unwrap().is_empty());
    assert!(meeting.get("voiceprints").is_none());
    assert!(meeting.get("worker_pid").is_none());

    // The audio is kept (KEEP_AUDIO default) and served for playback. Video
    // uploads are reduced to their extracted audio track; the video is gone.
    assert_eq!(meeting["has_audio"], true);
    let audio = api::get_meeting_audio_path(&app.ctx, &mid).unwrap();
    let audio_path = audio["path"].as_str().unwrap();
    assert_eq!(std::fs::read(audio_path).unwrap(), b"fake video bytes");
    assert_eq!(audio["filename"], "standup.m4a");
    let upload_dir = app.uploads_dir().join(&mid);
    assert!(upload_dir.join("playback.m4a").is_file());
    assert!(!upload_dir.join("standup.mp4").exists());

    let speaker_id = meeting["speakers"].as_object().unwrap().keys().next().unwrap().clone();
    let mut names = serde_json::Map::new();
    names.insert(speaker_id.clone(), json!("Alice"));
    let updated = api::rename_speakers(&app.ctx, &mid, &names).unwrap();
    assert_eq!(updated["speakers"][&speaker_id], "Alice");
    assert!(api::get_speakers(&app.ctx)
        .as_array()
        .unwrap()
        .contains(&json!("Alice")));

    api::patch_meeting(&app.ctx, &mid, &changes("title", json!("Daily Standup"))).unwrap();

    let (md, filename) = api::export_meeting(&app.ctx, &mid).unwrap();
    assert!(md.starts_with("# Daily Standup"));
    assert!(md.contains("Alice"));
    assert_eq!(filename, "Daily-Standup.md");

    api::delete_meeting(&app.ctx, &mid).unwrap();
    assert_eq!(api::get_meeting(&app.ctx, &mid).unwrap_err().code, 404);
    api::delete_speaker(&app.ctx, "Alice").unwrap();
}

#[test]
fn audio_upload_kept_unchanged() {
    // An audio-only upload is already just audio — no extraction, no deletion.
    let app = TestApp::fake();
    let mid = app.upload("standup.m4a", b"fake audio bytes");
    let meeting = app.wait_terminal(&mid);
    assert_eq!(meeting["status"], "done", "{:?}", meeting["error"]);
    assert_eq!(meeting["has_audio"], true);

    let upload_dir = app.uploads_dir().join(&mid);
    assert!(upload_dir.join("standup.m4a").is_file());
    let record = store::get_meeting(&app.ctx.paths, &app.ctx.locks, &mid).unwrap();
    assert_eq!(
        store::audio_path(&app.ctx.paths, &record),
        Some(upload_dir.join("standup.m4a"))
    );
    let audio = api::get_meeting_audio_path(&app.ctx, &mid).unwrap();
    assert_eq!(std::fs::read(audio["path"].as_str().unwrap()).unwrap(), b"fake audio bytes");

    api::delete_meeting(&app.ctx, &mid).unwrap();
}

#[test]
fn worker_write_preserves_user_renames() {
    let app = TestApp::fake();
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "clobber.mp4", -1).unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        changes("speakers", json!({"spk_0": "Alice"})),
    )
    .unwrap();
    // The worker's summarizing write must not overwrite user-set names.
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        true,
        changes("speakers", json!({"spk_0": "Speaker 1", "spk_1": "Speaker 2"})),
    )
    .unwrap();
    let meeting = store::get_meeting(&app.ctx.paths, &app.ctx.locks, mid).unwrap();
    assert_eq!(meeting["speakers"], json!({"spk_0": "Alice", "spk_1": "Speaker 2"}));
}

#[test]
fn rename_updates_generated_notes() {
    let app = TestApp::fake();
    let record = store::create_meeting(&app.ctx.paths, &app.ctx.locks, "notes.mp4", -1).unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[
            ("speakers", json!({"spk_0": "Speaker 1", "spk_1": "Speaker 10"})),
            ("summary", json!("Speaker 1 briefs Speaker 10 on the launch.")),
            ("takeaways", json!(["Speaker 1 owns the rollout."])),
            ("action_items", json!(["Speaker 1 sends the recap to Speaker 10."])),
        ]),
    )
    .unwrap();
    let mut names = serde_json::Map::new();
    names.insert("spk_0".into(), json!("Anna"));
    store::merge_speaker_names(&app.ctx.paths, &app.ctx.locks, mid, &names).unwrap();
    let meeting = store::get_meeting(&app.ctx.paths, &app.ctx.locks, mid).unwrap();
    // "Speaker 10" must survive the "Speaker 1" rename untouched.
    assert_eq!(meeting["summary"], "Anna briefs Speaker 10 on the launch.");
    assert_eq!(meeting["takeaways"], json!(["Anna owns the rollout."]));
    assert_eq!(meeting["action_items"], json!(["Anna sends the recap to Speaker 10."]));
}

#[test]
fn concurrent_renames_both_survive() {
    let app = TestApp::fake();
    let record = store::create_meeting(&app.ctx.paths, &app.ctx.locks, "race.mp4", -1).unwrap();
    let mid = record["id"].as_str().unwrap().to_string();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        &mid,
        false,
        changes("speakers", json!({"a": "a", "b": "b"})),
    )
    .unwrap();

    let mut handles = Vec::new();
    for (id, name) in [("a", "Alice"), ("b", "Bob")] {
        let ctx: std::sync::Arc<Ctx> = app.ctx.clone();
        let mid = mid.clone();
        handles.push(std::thread::spawn(move || {
            let mut names = serde_json::Map::new();
            names.insert(id.to_string(), json!(name));
            store::merge_speaker_names(&ctx.paths, &ctx.locks, &mid, &names).unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let meeting = store::get_meeting(&app.ctx.paths, &app.ctx.locks, &mid).unwrap();
    assert_eq!(meeting["speakers"], json!({"a": "Alice", "b": "Bob"}));
}

#[test]
fn corrupt_meeting_json_is_tolerated() {
    let app = TestApp::fake();
    let bad = app.ctx.paths.meetings_dir().join("corrupt.json");
    std::fs::write(&bad, "{not json").unwrap();
    // list skips it, direct fetch 404s
    assert!(api::list_meetings(&app.ctx).as_array().unwrap().is_empty());
    assert_eq!(api::get_meeting(&app.ctx, "corrupt").unwrap_err().code, 404);
}

#[test]
fn orphan_sweep_marks_dead_worker_error() {
    let app = TestApp::fake();
    let record = store::create_meeting(&app.ctx.paths, &app.ctx.locks, "orphan.mp4", -1).unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[("status", json!("transcribing")), ("worker_pid", json!(99999999))]),
    )
    .unwrap();

    // A fresh app instance over the same data dir — i.e. an app restart.
    let events = std::sync::Arc::new(lazymeet_core::events::CollectorEvents::default());
    let ctx2 = Ctx::new(
        lazymeet_core::config::Paths::at(app.ctx.paths.data_dir.clone()),
        true,
        events.clone(),
    );
    jobs::sweep_orphans(&ctx2);

    let meeting = api::get_meeting(&ctx2, mid).unwrap();
    assert_eq!(meeting["status"], "error");
    assert!(meeting["error"].as_str().unwrap().contains("interrupted"));
    // The sweep announced the terminal transition as an event too.
    assert_eq!(status_events(&events, mid), vec!["error"]);
}
