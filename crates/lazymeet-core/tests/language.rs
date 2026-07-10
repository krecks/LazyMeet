//! Per-meeting spoken-language hint (chosen on the prep screen): stored on
//! the record, validated at the API, and overriding ASR_LANGUAGE per job.

mod common;

use common::TestApp;
use lazymeet_core::api;

fn upload_lang(app: &TestApp, name: &str, language: Option<&str>) -> Result<String, u16> {
    let src = app.tmp.path().join("incoming");
    std::fs::create_dir_all(&src).unwrap();
    let path = src.join(name);
    std::fs::write(&path, b"fake").unwrap();
    match api::create_meeting(&app.ctx, &path, -1, None, language) {
        Ok(res) => Ok(res["id"].as_str().unwrap().to_string()),
        Err(e) => Err(e.code),
    }
}

#[test]
fn language_round_trips_on_the_record() {
    let app = TestApp::fake();
    let id = upload_lang(&app, "de.mp4", Some("de")).unwrap();
    let meeting = app.wait_terminal(&id);
    assert_eq!(meeting["status"], "done");
    assert_eq!(meeting["language"], "de");
}

#[test]
fn omitted_language_stays_null() {
    let app = TestApp::fake();
    let id = upload_lang(&app, "none.mp4", None).unwrap();
    let meeting = app.wait_terminal(&id);
    assert_eq!(meeting["language"], serde_json::Value::Null);
}

#[test]
fn auto_is_a_valid_choice() {
    let app = TestApp::fake();
    let id = upload_lang(&app, "auto.mp4", Some("auto")).unwrap();
    let meeting = app.wait_terminal(&id);
    assert_eq!(meeting["language"], "auto");
}

#[test]
fn unknown_language_is_a_422() {
    let app = TestApp::fake();
    assert_eq!(upload_lang(&app, "bad.mp4", Some("klingon")), Err(422));
}
