//! Watch folder: what gets imported, when, and only once.

mod common;

use std::path::Path;
use std::time::{Duration, SystemTime};

use common::TestApp;
use lazymeet_core::{api, watch};
use serde_json::{json, Map};

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(dir.join(name), bytes).unwrap();
}

fn names(paths: &[std::path::PathBuf]) -> Vec<String> {
    paths.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect()
}

/// `now` for a scan: real file mtimes are "now", so move the clock forward.
fn later(secs: u64) -> SystemTime {
    SystemTime::now() + Duration::from_secs(secs)
}

#[test]
fn imports_new_recordings_once_they_stop_changing() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write(dir, "old.mkv", b"there before watching");
    let mut state = watch::WatchState::fresh(dir, later(1));
    std::thread::sleep(Duration::from_millis(1100)); // mtimes have 1 s granularity on some FSs

    write(dir, "meeting.mkv", b"recording...");
    write(dir, "notes.txt", b"not media");
    // First sight: remembered as pending, not imported yet.
    assert!(watch::scan(dir, &mut state, later(2)).is_empty());
    // Unchanged but not for long enough yet.
    assert!(watch::scan(dir, &mut state, later(5)).is_empty());
    // Still growing: the clock restarts.
    write(dir, "meeting.mkv", b"recording... more bytes");
    assert!(watch::scan(dir, &mut state, later(20)).is_empty());
    assert!(watch::scan(dir, &mut state, later(25)).is_empty());
    // Quiet for 10 s: ready, exactly once. The old file never imports.
    assert_eq!(names(&watch::scan(dir, &mut state, later(31))), vec!["meeting.mkv"]);
    assert!(watch::scan(dir, &mut state, later(60)).is_empty());
}

#[test]
fn an_obs_remux_twin_is_not_imported_again() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let mut state = watch::WatchState::fresh(dir, SystemTime::now() - Duration::from_secs(5));
    write(dir, "2026-11-05 10-00-00.mkv", b"recording");
    watch::scan(dir, &mut state, later(1));
    assert_eq!(watch::scan(dir, &mut state, later(12)).len(), 1);
    // OBS remuxes to .mp4 after stopping.
    write(dir, "2026-11-05 10-00-00.mp4", b"remuxed recording");
    watch::scan(dir, &mut state, later(13));
    assert!(watch::scan(dir, &mut state, later(30)).is_empty(), "same recording, other container");
    // A different recording a moment later is still imported.
    write(dir, "2026-11-05 11-00-00.mp4", b"next meeting");
    watch::scan(dir, &mut state, later(31));
    assert_eq!(names(&watch::scan(dir, &mut state, later(45))), vec!["2026-11-05 11-00-00.mp4"]);
}

#[test]
fn importing_creates_meetings_and_tells_the_ui() {
    let app = TestApp::fake();
    let dir = app.tmp.path().join("obs");
    std::fs::create_dir(&dir).unwrap();
    write(&dir, "call.mp4", b"fake video bytes");
    let ids = watch::import(&app.ctx, &[dir.join("call.mp4")]);
    assert_eq!(ids.len(), 1);
    let m = app.wait_terminal(&ids[0]);
    assert_eq!(m["status"], "done");
    assert!(dir.join("call.mp4").exists(), "the user's file stays where it was");
    let events = app.events.payloads("meetings-changed");
    assert_eq!(events[0]["imported"]["filename"], "call.mp4");
}

#[test]
fn the_watch_folder_setting_is_validated() {
    let app = TestApp::fake();
    let put = |v: serde_json::Value| {
        let mut fields = Map::new();
        fields.insert("WATCH_DIR".into(), v);
        api::put_config(&app.ctx, &fields)
    };
    assert_eq!(put(json!("relative/path")).unwrap_err().code, 422);
    assert_eq!(put(json!("/definitely/not/here")).unwrap_err().code, 422);
    let data = app.ctx.paths.data_dir.join("uploads");
    std::fs::create_dir_all(&data).unwrap();
    assert_eq!(put(json!(data.to_string_lossy())).unwrap_err().code, 422, "own data folder");
    let obs = app.tmp.path().join("obs");
    std::fs::create_dir(&obs).unwrap();
    put(json!(obs.to_string_lossy())).unwrap();
    assert_eq!(watch::watched_dir(&app.ctx).as_deref(), Some(obs.as_path()));
    put(json!(null)).unwrap(); // null clears an override, like every setting
    assert_eq!(watch::watched_dir(&app.ctx), None);
}
