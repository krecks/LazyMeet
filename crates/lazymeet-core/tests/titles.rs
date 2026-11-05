//! Generated meeting titles: the summarizer names the meeting unless the
//! user already did, and meetings processed before titles get one from
//! their notes on launch.

mod common;

use common::{obj, TestApp};
use lazymeet_core::{api, jobs, pipeline::clean_title, store};
use serde_json::json;

const FAKE_TITLE: &str = "Transcription pipeline sync";

fn wait_idle(app: &TestApp) {
    for _ in 0..500 {
        if !jobs::is_busy(&app.ctx) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("jobs never went idle");
}

/// Make a processed meeting look like one from before generated titles.
fn make_legacy(app: &TestApp, mid: &str, title: &str) {
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[("title", json!(title)), ("title_by_user", json!(null))]),
    )
    .unwrap();
}

#[test]
fn processing_names_the_meeting() {
    let app = TestApp::fake();
    let mid = app.upload("2026-04-24 12-29-55.mkv", b"fake video bytes");
    let m = app.wait_terminal(&mid);
    assert_eq!(m["status"], "done", "{:?}", m["error"]);
    assert_eq!(m["title"], FAKE_TITLE);
    assert_eq!(m["filename"], "2026-04-24 12-29-55.mkv");
    let listed = store::list_meetings(&app.ctx.paths, &app.ctx.locks);
    assert_eq!(listed[0]["title"], FAKE_TITLE);
}

#[test]
fn a_user_title_survives_regeneration() {
    let app = TestApp::fake();
    let mid = app.upload("call.mp4", b"fake video bytes");
    app.wait_terminal(&mid);
    api::patch_meeting(&app.ctx, &mid, &obj(&[("title", json!("Kevin intro call"))])).unwrap();

    api::reprocess_meeting(&app.ctx, &mid, "notes").unwrap();
    let m = app.wait_terminal(&mid);
    assert_eq!(m["title"], "Kevin intro call");
    api::reprocess_meeting(&app.ctx, &mid, "transcription").unwrap();
    assert_eq!(app.wait_terminal(&mid)["title"], "Kevin intro call");
}

#[test]
fn a_late_rename_beats_a_generated_title() {
    let app = TestApp::fake();
    let mid = app.upload("late.mp4", b"fake video bytes");
    app.wait_terminal(&mid);
    api::patch_meeting(&app.ctx, &mid, &obj(&[("title", json!("Mine"))])).unwrap();
    // What a job that read the record before the rename would write.
    let r = store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        &mid,
        false,
        obj(&[("title", json!(FAKE_TITLE)), ("title_by_user", json!(false))]),
    )
    .unwrap();
    assert_eq!(r["title"], "Mine");
    assert_eq!(r["title_by_user"], true);
}

#[test]
fn backfill_titles_only_untouched_legacy_meetings() {
    let app = TestApp::fake();
    let named_by_file = app.upload("Call.mp4", b"fake video bytes");
    app.wait_terminal(&named_by_file);
    let renamed = app.upload("other.mp4", b"fake video bytes");
    app.wait_terminal(&renamed);
    make_legacy(&app, &named_by_file, "Call.mp4");
    make_legacy(&app, &renamed, "Board prep"); // renamed before the flag existed

    jobs::queue_title_backfill(&app.ctx);
    wait_idle(&app);

    let a = api::get_meeting(&app.ctx, &named_by_file).unwrap();
    assert_eq!(a["title"], FAKE_TITLE);
    assert_eq!(a["status"], "done"); // a title pass never changes the status
    assert_eq!(api::get_meeting(&app.ctx, &renamed).unwrap()["title"], "Board prep");

    // Once titled, a later launch does not queue it again.
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        &named_by_file,
        false,
        obj(&[("title", json!("Changed by hand"))]),
    )
    .unwrap();
    jobs::queue_title_backfill(&app.ctx);
    wait_idle(&app);
    assert_eq!(api::get_meeting(&app.ctx, &named_by_file).unwrap()["title"], "Changed by hand");
}

#[test]
fn model_titles_are_cleaned() {
    assert_eq!(clean_title("\"Q3 pricing review with Acme.\""), "Q3 pricing review with Acme");
    assert_eq!(clean_title("Title: **Hiring sync**\nextra line"), "Hiring sync");
    assert_eq!(clean_title("  „Kickoff  Projekt Atlas“ "), "Kickoff Projekt Atlas");
    assert_eq!(clean_title(""), "");
    assert_eq!(clean_title("## Intro to C#"), "Intro to C#");
    // Multi-byte text where a "Title:" prefix would be checked must not panic.
    assert_eq!(clean_title("Präsentation Phase zwei"), "Präsentation Phase zwei");
    assert_eq!(clean_title("Ärger über Lieferzeiten"), "Ärger über Lieferzeiten");
    assert_eq!(clean_title("Kickoä Termin"), "Kickoä Termin"); // ä spans bytes 5..7
    assert_eq!(clean_title(&"ä".repeat(100)).chars().count(), 80);
    let long = "word ".repeat(40);
    let cleaned = clean_title(&long);
    assert!(cleaned.chars().count() <= 80 && !cleaned.ends_with(' '), "{cleaned}");
}
