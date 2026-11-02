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

// Cohere Transcribe can't detect the language: the model + language pair is
// checked when it is set and when a recording would run on it.

#[test]
fn cohere_needs_a_language_it_decodes() {
    let app = TestApp::fake();
    let put = |fields: &[(&str, serde_json::Value)]| {
        api::put_config(&app.ctx, &common::obj(fields)).map(|_| ()).map_err(|e| e.code)
    };
    // The default language is "auto".
    assert_eq!(put(&[("ASR_MODEL", "cohere-transcribe".into())]), Err(422));
    assert_eq!(
        put(&[("ASR_MODEL", "cohere-transcribe".into()), ("ASR_LANGUAGE", "ru".into())]),
        Err(422)
    );
    put(&[("ASR_MODEL", "cohere-transcribe".into()), ("ASR_LANGUAGE", "de".into())]).unwrap();
    // Back to auto while Cohere stays selected: refused.
    assert_eq!(put(&[("ASR_LANGUAGE", "auto".into())]), Err(422));
    assert_eq!(put(&[("ASR_LANGUAGE", serde_json::Value::Null)]), Err(422));
    // Unrelated saves are not affected by the pair.
    put(&[("SPEAKER_THRESHOLD", 0.7.into())]).unwrap();
    // Switching to a model that detects makes auto valid again.
    put(&[("ASR_MODEL", "parakeet-tdt-0.6b-v3".into()), ("ASR_LANGUAGE", "auto".into())]).unwrap();
}

#[test]
fn recording_language_is_checked_against_the_model() {
    let app = TestApp::fake();
    api::put_config(
        &app.ctx,
        &common::obj(&[("ASR_MODEL", "cohere-transcribe".into()), ("ASR_LANGUAGE", "en".into())]),
    )
    .unwrap();
    assert_eq!(upload_lang(&app, "auto.mp4", Some("auto")), Err(422));
    assert_eq!(upload_lang(&app, "uk.mp4", Some("uk")), Err(422));
    // The setting (en) applies when the recording names no language.
    upload_lang(&app, "none.mp4", None).unwrap();
    let id = upload_lang(&app, "vi.mp4", Some("vi")).unwrap();
    assert_eq!(app.wait_terminal(&id)["status"], "done");
}
