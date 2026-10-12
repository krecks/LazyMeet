//! Speaker library: renames, enrolment learning, and match verification
//! (port of test_speakers.py).

mod common;

use common::{obj, TestApp};
use lazymeet_core::{api, store};
use serde_json::json;

fn upsert(app: &TestApp, name: &str, emb: &[f64]) {
    store::upsert_speaker(&app.ctx.paths, &app.ctx.locks, name, emb);
}

fn library(app: &TestApp) -> serde_json::Map<String, serde_json::Value> {
    store::load_speaker_library(&app.ctx.paths, &app.ctx.locks)
}

fn merge_names(app: &TestApp, mid: &str, id: &str, name: &str) {
    let mut names = serde_json::Map::new();
    names.insert(id.to_string(), json!(name));
    store::merge_speaker_names(&app.ctx.paths, &app.ctx.locks, mid, &names).unwrap();
}

#[test]
fn rename_keeps_voiceprint() {
    let app = TestApp::fake();
    upsert(&app, "Alice", &[0.1, 0.2, 0.3]);
    let names = api::rename_known_speaker(&app.ctx, "Alice", "  Alice Weber ").unwrap();
    let names: Vec<&str> = names.as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(names.contains(&"Alice Weber") && !names.contains(&"Alice"));

    assert_eq!(library(&app)["Alice Weber"]["embedding"], json!([0.1, 0.2, 0.3]));
}

#[test]
fn rename_missing_is_404() {
    let app = TestApp::fake();
    assert_eq!(api::rename_known_speaker(&app.ctx, "Nobody", "X").unwrap_err().code, 404);
}

#[test]
fn rename_conflict_is_409() {
    let app = TestApp::fake();
    upsert(&app, "Alice", &[0.1]);
    upsert(&app, "Bob", &[0.2]);
    assert_eq!(api::rename_known_speaker(&app.ctx, "Alice", "Bob").unwrap_err().code, 409);
    // both untouched
    let names = store::get_speaker_names(&app.ctx.paths, &app.ctx.locks);
    assert!(names.contains(&"Alice".to_string()) && names.contains(&"Bob".to_string()));
}

#[test]
fn rename_invalid_names_are_422() {
    let app = TestApp::fake();
    upsert(&app, "Alice", &[0.1]);
    assert_eq!(api::rename_known_speaker(&app.ctx, "Alice", "   ").unwrap_err().code, 422);
    let long = "x".repeat(101);
    assert_eq!(api::rename_known_speaker(&app.ctx, "Alice", &long).unwrap_err().code, 422);
}

#[test]
fn meeting_rename_moves_library_entry() {
    // Renaming an enrolled speaker in a meeting must not leave a duplicate.
    let app = TestApp::fake();
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "lib.mp4", -1, None, None).unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[
            ("speakers", json!({"spk_0": "Speaker 1"})),
            ("voiceprints", json!({"spk_0": [0.1, 0.2]})),
        ]),
    )
    .unwrap();
    merge_names(&app, mid, "spk_0", "Christine Schneider");
    assert!(store::get_speaker_names(&app.ctx.paths, &app.ctx.locks)
        .contains(&"Christine Schneider".to_string()));

    merge_names(&app, mid, "spk_0", "Christine Weber");
    let names = store::get_speaker_names(&app.ctx.paths, &app.ctx.locks);
    assert!(names.contains(&"Christine Weber".to_string()));
    assert!(!names.contains(&"Christine Schneider".to_string())); // renamed, not duplicated
}

#[test]
fn meeting_rename_merges_into_existing_entry() {
    // Renaming a voice to a name already in the library keeps one entry.
    let app = TestApp::fake();
    upsert(&app, "Anna", &[0.9, 0.9]);
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "merge.mp4", -1, None, None).unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[
            ("speakers", json!({"spk_0": "Speaker 1"})),
            ("voiceprints", json!({"spk_0": [0.1, 0.2]})),
        ]),
    )
    .unwrap();
    merge_names(&app, mid, "spk_0", "Nana");
    merge_names(&app, mid, "spk_0", "Anna");
    let names = store::get_speaker_names(&app.ctx.paths, &app.ctx.locks);
    assert!(names.contains(&"Anna".to_string()) && !names.contains(&"Nana".to_string()));
    // the fresh voice refines the entry: matching uses the sample mean
    let entry = &library(&app)["Anna"];
    let emb: Vec<f64> =
        entry["embedding"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!((emb[0] - 0.5).abs() < 1e-9 && (emb[1] - 0.55).abs() < 1e-9);
    assert_eq!(entry["samples"].as_array().unwrap().len(), 2);
}

#[test]
fn upsert_learns_running_mean() {
    // Each enrolment adds a sample; matching uses their mean, so voiceprints
    // get more robust with every file instead of the last voice overwriting.
    let app = TestApp::fake();
    upsert(&app, "Olga", &[1.0, 0.0]);
    upsert(&app, "Olga", &[0.0, 1.0]);
    let entry = &library(&app)["Olga"];
    assert_eq!(entry["samples"], json!([[1.0, 0.0], [0.0, 1.0]]));
    assert_eq!(entry["embedding"], json!([0.5, 0.5]));
}

#[test]
fn correcting_wrong_match_keeps_victim_entry() {
    // Renaming a wrongly auto-recognized speaker must not move or delete the
    // matched person's library entry — only enrol the voice under the new name.
    let app = TestApp::fake();
    upsert(&app, "Christine Schneider", &[0.9, 0.9]);
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "wrong.mp4", -1, None, None).unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[
            ("speakers", json!({"spk_0": "Christine Schneider"})),
            ("recognized", json!(["spk_0"])),
            ("voiceprints", json!({"spk_0": [0.1, 0.2]})),
        ]),
    )
    .unwrap();
    merge_names(&app, mid, "spk_0", "Miriam");

    let lib = library(&app);
    assert_eq!(lib["Christine Schneider"]["embedding"], json!([0.9, 0.9])); // untouched
    assert_eq!(lib["Miriam"]["embedding"], json!([0.1, 0.2])); // enrolled correctly
    let meeting = store::get_meeting(&app.ctx.paths, &app.ctx.locks, mid).unwrap();
    assert_eq!(meeting["recognized"], json!([]));
}

#[test]
fn confirm_recognized_speaker_learns() {
    let app = TestApp::fake();
    upsert(&app, "Dimitri", &[1.0, 0.0]);
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "confirm.mp4", -1, None, None)
            .unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[
            ("speakers", json!({"spk_0": "Dimitri"})),
            ("recognized", json!(["spk_0"])),
            ("voiceprints", json!({"spk_0": [0.0, 1.0]})),
        ]),
    )
    .unwrap();
    let meeting = api::verify_speaker(&app.ctx, mid, "spk_0", true).unwrap();
    assert_eq!(meeting["confirmed"], json!(["spk_0"]));
    // this recording became another sample for Dimitri
    assert_eq!(library(&app)["Dimitri"]["embedding"], json!([0.5, 0.5]));
}

#[test]
fn reject_recognized_speaker_unlinks_without_poison() {
    let app = TestApp::fake();
    upsert(&app, "Christine Schneider", &[0.9, 0.9]);
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "reject.mp4", -1, None, None)
            .unwrap();
    let mid = record["id"].as_str().unwrap();
    store::update_meeting(
        &app.ctx.paths,
        &app.ctx.locks,
        mid,
        false,
        obj(&[
            ("speakers", json!({"spk_0": "Christine Schneider", "spk_1": "Speaker 1"})),
            ("recognized", json!(["spk_0"])),
            ("voiceprints", json!({"spk_0": [0.1, 0.2]})),
            ("summary", json!("Christine Schneider owns the rollout.")),
        ]),
    )
    .unwrap();
    let meeting = api::verify_speaker(&app.ctx, mid, "spk_0", false).unwrap();
    assert_eq!(meeting["speakers"]["spk_0"], "Speaker 2"); // "Speaker 1" was taken
    assert_eq!(meeting["recognized"], json!([]));
    assert_eq!(meeting["summary"], "Speaker 2 owns the rollout.");
    // the wrongly matched person's voiceprint is untouched
    assert_eq!(library(&app)["Christine Schneider"]["embedding"], json!([0.9, 0.9]));

    // nothing left to confirm — the id no longer counts as recognized
    assert_eq!(api::verify_speaker(&app.ctx, mid, "spk_0", true).unwrap_err().code, 409);
}
