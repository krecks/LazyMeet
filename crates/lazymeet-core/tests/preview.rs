//! extract_preview_audio: the prep screen's audio fallback for containers
//! the webview can't play (mkv, avi, …).

mod common;

use common::TestApp;
use lazymeet_core::api;

#[test]
fn preview_audio_lands_under_uploads() {
    let app = TestApp::fake();
    let src = app.tmp.path().join("meeting.mkv");
    std::fs::write(&src, b"fake mkv bytes").unwrap();
    let res = api::extract_preview_audio(&app.ctx, &src).expect("preview");
    let path = std::path::PathBuf::from(res["path"].as_str().unwrap());
    // Inside uploads/ (the asset-protocol scope) so the webview can stream it.
    assert!(path.starts_with(app.uploads_dir()), "outside uploads: {}", path.display());
    // FAKE mode copies the source through the same dispatch the native
    // pipeline routes to its AAC extraction.
    assert_eq!(std::fs::read(&path).unwrap(), b"fake mkv bytes");
    // A second call overwrites in place (one transient preview at a time).
    let res2 = api::extract_preview_audio(&app.ctx, &src).expect("preview again");
    assert_eq!(res2["path"], res["path"]);
}

#[test]
fn preview_of_missing_file_is_422() {
    let app = TestApp::fake();
    let err = api::extract_preview_audio(&app.ctx, std::path::Path::new("/nope.mkv")).unwrap_err();
    assert_eq!(err.code, 422);
}
