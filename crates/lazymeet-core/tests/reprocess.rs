//! Regeneration (notes / transcription re-runs) and the per-meeting version
//! history that records which models produced each version.

mod common;

use common::{obj, TestApp};
use lazymeet_core::{api, jobs, store};
use serde_json::json;

#[test]
fn initial_run_records_v1_with_models() {
    let app = TestApp::fake();
    let mid = app.upload("m.mp4", b"fake video bytes");
    let m = app.wait_terminal(&mid);
    assert_eq!(m["status"], "done", "{:?}", m["error"]);
    let versions = m["versions"].as_array().unwrap();
    assert_eq!(versions.len(), 1);
    assert_eq!(versions[0]["v"], 1);
    assert_eq!(versions[0]["action"], "processed");
    assert_eq!(versions[0]["asr_model"], "parakeet-tdt-0.6b-v3");
    assert_eq!(versions[0]["summary_model"], "qwen3.5-4b");
}

#[test]
fn notes_regen_uses_new_model_and_keeps_transcript() {
    let app = TestApp::fake();
    let mid = app.upload("m.mp4", b"fake video bytes");
    let m = app.wait_terminal(&mid);
    let segments_before = m["segments"].clone();

    // A manual edit made while v1 is current must survive in v1's snapshot.
    api::patch_meeting(&app.ctx, &mid, &obj(&[("summary", json!("Edited by hand."))])).unwrap();

    api::put_config(&app.ctx, &obj(&[("SUMMARY_MODEL", json!("qwen3.5-9b"))])).unwrap();
    api::reprocess_meeting(&app.ctx, &mid, "notes").unwrap();
    let m = app.wait_terminal(&mid);
    assert_eq!(m["status"], "done", "{:?}", m["error"]);
    assert_eq!(m["segments"], segments_before); // transcript untouched
    let versions = m["versions"].as_array().unwrap();
    assert_eq!(versions.len(), 2);
    assert_eq!(versions[1]["v"], 2);
    assert_eq!(versions[1]["action"], "notes");
    // the transcript didn't change, so v2 keeps v1's ASR model
    assert_eq!(versions[1]["asr_model"], versions[0]["asr_model"]);
    assert_eq!(versions[1]["summary_model"], "qwen3.5-9b");
    // snapshots: v1 holds the content as last seen (incl. the manual edit),
    // v2 holds what this run produced (= the live content)
    assert_eq!(versions[0]["snapshot"]["summary"], "Edited by hand.");
    assert_eq!(versions[0]["snapshot"]["segments"], segments_before);
    assert_eq!(versions[1]["snapshot"]["summary"], m["summary"]);
}

#[test]
fn pre_history_meeting_gets_backfilled_v1_with_snapshot() {
    let app = TestApp::fake();
    let mid = app.upload("m.mp4", b"fake video bytes");
    let m = app.wait_terminal(&mid);
    let old_summary = m["summary"].clone();

    // Simulate a meeting processed before version history existed.
    store::update_meeting(&app.ctx.paths, &app.ctx.locks, &mid, false, obj(&[("versions", json!([]))]))
        .unwrap();

    api::reprocess_meeting(&app.ctx, &mid, "notes").unwrap();
    let m = app.wait_terminal(&mid);
    let versions = m["versions"].as_array().unwrap();
    assert_eq!(versions.len(), 2);
    // backfilled original: unknown models, but the old content stays viewable
    assert_eq!(versions[0]["action"], "processed");
    assert_eq!(versions[0]["asr_model"], json!(null));
    assert_eq!(versions[0]["summary_model"], json!(null));
    assert_eq!(versions[0]["snapshot"]["summary"], old_summary);
    assert_eq!(versions[1]["action"], "notes");
}

#[test]
fn retranscribe_reruns_pipeline_with_new_models() {
    let app = TestApp::fake();
    let mid = app.upload("m.mp4", b"fake video bytes");
    let m = app.wait_terminal(&mid);
    assert_eq!(m["has_audio"], true);

    api::put_config(&app.ctx, &obj(&[("ASR_MODEL", json!("whisper-base"))])).unwrap();
    api::reprocess_meeting(&app.ctx, &mid, "transcription").unwrap();
    let m = app.wait_terminal(&mid);
    assert_eq!(m["status"], "done", "{:?}", m["error"]);
    assert!(!m["segments"].as_array().unwrap().is_empty());
    assert_eq!(m["confirmed"], json!([])); // old cluster confirmations dropped
    let versions = m["versions"].as_array().unwrap();
    assert_eq!(versions.len(), 2);
    assert_eq!(versions[1]["action"], "transcription");
    assert_eq!(versions[1]["asr_model"], "whisper-base");
}

#[test]
fn retranscribe_without_audio_is_404_but_notes_still_work() {
    let app = TestApp::fake();
    let mid = app.upload("m.mp4", b"fake video bytes");
    app.wait_terminal(&mid);
    std::fs::remove_dir_all(app.uploads_dir().join(&mid)).unwrap();

    let err = api::reprocess_meeting(&app.ctx, &mid, "transcription").unwrap_err();
    assert_eq!(err.code, 404);

    api::reprocess_meeting(&app.ctx, &mid, "notes").unwrap();
    let m = app.wait_terminal(&mid);
    assert_eq!(m["status"], "done", "{:?}", m["error"]);
}

#[test]
fn reprocess_guards() {
    let app = TestApp::fake();
    let err = api::reprocess_meeting(&app.ctx, "nope", "notes").unwrap_err();
    assert_eq!(err.code, 404);

    let mid = app.upload("m.mp4", b"fake video bytes");
    app.wait_terminal(&mid);
    let err = api::reprocess_meeting(&app.ctx, &mid, "everything").unwrap_err();
    assert_eq!(err.code, 422);
}

#[test]
fn reprocess_while_processing_is_409() {
    // fake=false keeps the models "missing", so the job waits in the queue.
    let app = TestApp::with_fake(false);
    let mid = app.upload("m.mp4", b"fake video bytes");
    let err = api::reprocess_meeting(&app.ctx, &mid, "notes").unwrap_err();
    assert_eq!(err.code, 409);
}

#[test]
fn interrupted_rejob_restores_instead_of_failing() {
    let app = TestApp::fake();
    let mid = app.upload("m.mp4", b"fake video bytes");
    app.wait_terminal(&mid);

    // Crash mid notes-regen: worker died at "summarizing", marker still set.
    // The meeting's content is intact, so the sweep restores "done" (a fresh
    // upload in that state would be failed instead).
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        &mid,
        false,
        obj(&[("status", json!("summarizing")), ("rejob", json!("notes"))]),
    )
    .unwrap();
    jobs::sweep_orphans(&app.ctx);
    let m = api::get_meeting(&app.ctx, &mid).unwrap();
    assert_eq!(m["status"], "done");

    // Crash while still queued: the sweep re-dispatches the right job kind.
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        &mid,
        false,
        obj(&[("status", json!("queued")), ("rejob", json!("notes"))]),
    )
    .unwrap();
    jobs::sweep_orphans(&app.ctx);
    let m = app.wait_terminal(&mid);
    assert_eq!(m["status"], "done", "{:?}", m["error"]);
    assert_eq!(m["versions"].as_array().unwrap().last().unwrap()["action"], "notes");
}
