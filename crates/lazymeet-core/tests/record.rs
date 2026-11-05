//! Recording in the app (FAKE capture: no devices, silent WAVs), and how a
//! recording becomes a meeting.

mod common;

use std::path::{Path, PathBuf};

use common::TestApp;
use lazymeet_core::{api, record};

#[test]
fn record_stop_and_transcribe_moves_the_file() {
    let app = TestApp::fake();
    let started = record::start(&app.ctx, None, true).unwrap();
    assert_eq!(started["sources"], serde_json::json!(["mic", "system"]));
    assert_eq!(record::status(&app.ctx)["recording"], true);
    assert_eq!(record::start(&app.ctx, None, false).unwrap_err().code, 409, "one at a time");
    std::thread::sleep(std::time::Duration::from_millis(250));
    assert!(!app.events.payloads("record-level").is_empty(), "level meters tick");

    let stopped = record::stop(&app.ctx).unwrap();
    let path = PathBuf::from(stopped["path"].as_str().unwrap());
    assert!(path.is_file());
    assert_eq!(record::status(&app.ctx)["recording"], false);
    assert_eq!(record::stop(&app.ctx).unwrap_err().code, 409);

    // Not imported yet: offered as a leftover.
    let left = record::leftovers(&app.ctx);
    assert_eq!(left.len(), 1);
    assert_eq!(left[0]["path"], stopped["path"]);

    let created = api::create_meeting(&app.ctx, &path, -1, None, None).unwrap();
    let id = created["id"].as_str().unwrap();
    assert!(!path.exists(), "our own recording is moved, not copied");
    assert!(!path.parent().unwrap().exists(), "its folder goes too");
    assert_eq!(app.wait_terminal(id)["status"], "done");
    assert!(record::leftovers(&app.ctx).is_empty());
}

#[test]
fn cancel_discards_and_user_files_are_never_moved() {
    let app = TestApp::fake();
    record::start(&app.ctx, Some(""), true).unwrap();
    record::cancel(&app.ctx).unwrap();
    assert!(record::leftovers(&app.ctx).is_empty());
    let dirs = std::fs::read_dir(app.ctx.paths.recordings_dir()).unwrap().count();
    assert_eq!(dirs, 0, "a cancelled recording leaves nothing behind");
    assert_eq!(
        record::start(&app.ctx, Some(""), false).unwrap_err().code,
        422,
        "nothing to record"
    );

    // A file of the user's own is copied, never moved — even one named like ours.
    let own = app.tmp.path().join("recording.m4a");
    std::fs::write(&own, b"fake audio").unwrap();
    assert!(record::owned_folder(&app.ctx, &own).is_none());
    let created = api::create_meeting(&app.ctx, &own, -1, None, None).unwrap();
    app.wait_terminal(created["id"].as_str().unwrap());
    assert!(own.exists());
    assert_eq!(record::discard(&app.ctx, &own).unwrap_err().code, 422);
}

#[test]
fn leftovers_can_be_discarded() {
    let app = TestApp::fake();
    record::start(&app.ctx, None, false).unwrap();
    let path = record::stop(&app.ctx).unwrap()["path"].as_str().unwrap().to_string();
    record::discard(&app.ctx, Path::new(&path)).unwrap();
    assert!(record::leftovers(&app.ctx).is_empty());
}

#[test]
fn mixing_keeps_both_sources_at_full_level() {
    let args = record::mix_args(
        &[PathBuf::from("mic.wav"), PathBuf::from("system.wav")],
        Path::new("out.m4a"),
    );
    let joined = args.join(" ");
    assert!(joined.contains("-i mic.wav -i system.wav"));
    assert!(joined.contains("amix=inputs=2:duration=longest:normalize=0"));
    assert!(joined.ends_with("-ac 1 -c:a aac -b:a 96k out.m4a"));
    let one = record::mix_args(&[PathBuf::from("mic.wav")], Path::new("out.m4a")).join(" ");
    assert!(!one.contains("amix"));
}
