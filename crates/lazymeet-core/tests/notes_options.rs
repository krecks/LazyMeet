//! Notes templates, the per-meeting notes language and the vocabulary reach
//! the summarizer (the fake one echoes them) and are kept on the meeting.

mod common;

use common::TestApp;
use lazymeet_core::api::{self, NotesOptions};
use lazymeet_core::config::{self, Config, Paths};
use serde_json::{json, Map, Value};

fn put(app: &TestApp, key: &str, value: Value) -> Result<Value, lazymeet_core::error::ApiError> {
    let mut fields = Map::new();
    fields.insert(key.into(), value);
    api::put_config(&app.ctx, &fields)
}

#[test]
fn guidance_combines_template_and_vocabulary() {
    let mut cfg = Config::resolve(&Paths::at("/nonexistent"));
    assert_eq!(config::notes_guidance(&cfg), "", "general + no vocabulary adds nothing");
    cfg.override_value("NOTES_TEMPLATE", json!("standup"));
    cfg.override_value("VOCABULARY", json!("Krecks\n\n  LazyMeet  \nNemotron"));
    let g = config::notes_guidance(&cfg);
    assert!(g.starts_with("This is a stand-up."), "{g}");
    assert!(g.ends_with("Krecks, LazyMeet, Nemotron."), "{g}");
    assert_eq!(cfg.vocabulary(), vec!["Krecks", "LazyMeet", "Nemotron"]);
}

#[test]
fn per_meeting_choices_reach_the_notes_and_are_kept() {
    let app = TestApp::fake();
    let src = app.tmp.path().join("standup.mp4");
    std::fs::write(&src, b"fake video bytes").unwrap();
    let notes = NotesOptions { template: Some("standup".into()), language: Some("de".into()) };
    let created = api::create_meeting_with(&app.ctx, &src, -1, None, None, &notes).unwrap();
    let mid = created["id"].as_str().unwrap().to_string();
    let m = app.wait_terminal(&mid);
    assert_eq!(m["notes_template"], "standup");
    assert_eq!(m["notes_language"], "de");
    assert!(m["summary_long"].as_str().unwrap().contains("template=standup, language=de"));
    assert_eq!(m["versions"][0]["notes_template"], "standup");

    // Regenerating with another template stores it and uses it.
    let lecture = NotesOptions { template: Some("lecture".into()), language: None };
    api::reprocess_meeting_with(&app.ctx, &mid, "notes", &lecture).unwrap();
    let m = app.wait_terminal(&mid);
    assert_eq!(m["notes_template"], "lecture");
    assert!(m["summary_long"].as_str().unwrap().contains("template=lecture, language=de"));
    assert_eq!(m["versions"].as_array().unwrap().last().unwrap()["notes_template"], "lecture");
}

#[test]
fn bad_choices_are_refused_up_front() {
    let app = TestApp::fake();
    let src = app.tmp.path().join("a.mp4");
    std::fs::write(&src, b"fake video bytes").unwrap();
    let bad = NotesOptions { template: Some("haiku".into()), language: None };
    assert_eq!(
        api::create_meeting_with(&app.ctx, &src, -1, None, None, &bad).unwrap_err().code,
        422
    );
    let bad = NotesOptions { template: None, language: Some("xx".into()) };
    assert_eq!(
        api::create_meeting_with(&app.ctx, &src, -1, None, None, &bad).unwrap_err().code,
        422
    );
    assert_eq!(put(&app, "NOTES_TEMPLATE", json!("haiku")).unwrap_err().code, 422);
    put(&app, "NOTES_TEMPLATE", json!("interview")).unwrap();
    put(&app, "VOCABULARY", json!("Acme\nZephyrine")).unwrap();

    // The saved settings apply to a meeting that didn't choose.
    let mid = app.upload("b.mp4", b"fake video bytes");
    let m = app.wait_terminal(&mid);
    assert!(m.get("notes_template").is_none() || m["notes_template"].is_null());
    let long = m["summary_long"].as_str().unwrap();
    assert!(
        long.contains("template=interview") && long.contains("vocabulary=Acme|Zephyrine"),
        "{long}"
    );

    let payload = api::config_payload(&app.ctx);
    assert_eq!(payload["notes_templates"][0], json!({"id": "general", "label": "General meeting"}));
}
