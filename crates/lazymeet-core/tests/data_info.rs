//! Storage-info payload (port of test_data.py; the reveal-in-file-manager
//! tests moved to the desktop shell, which owns that OS call now).

mod common;

use common::TestApp;
use lazymeet_core::{api, store};

#[test]
fn data_info_shape() {
    let app = TestApp::fake();
    let data = api::get_data_info(&app.ctx);
    assert_eq!(
        data["data_dir"].as_str().unwrap(),
        app.ctx.paths.data_dir.to_str().unwrap()
    );
    for key in ["meetings_bytes", "uploads_bytes", "models_bytes", "total_bytes"] {
        assert!(data[key].as_u64().is_some(), "{key} missing or negative");
    }
    assert_eq!(
        data["total_bytes"].as_u64().unwrap(),
        data["meetings_bytes"].as_u64().unwrap()
            + data["uploads_bytes"].as_u64().unwrap()
            + data["models_bytes"].as_u64().unwrap()
    );
    assert!(data["meeting_count"].as_u64().is_some());
}

#[test]
fn data_info_counts_meeting_files() {
    let app = TestApp::fake();
    store::create_meeting(&app.ctx.paths, &app.ctx.locks, "sized.mp4", -1).unwrap();
    let data = api::get_data_info(&app.ctx);
    assert!(data["meetings_bytes"].as_u64().unwrap() > 0);
    assert!(data["meeting_count"].as_u64().unwrap() >= 1);
}
