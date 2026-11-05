//! Tags: cleaning, bulk tagging, search, and bulk delete.

mod common;

use common::{changes, TestApp};
use lazymeet_core::api;
use serde_json::json;

#[test]
fn tags_are_cleaned_and_deduplicated_ignoring_case() {
    let app = TestApp::fake();
    let mid = app.upload("a.mp4", b"fake video bytes");
    app.wait_terminal(&mid);

    let tags = json!(["  Acme  ", "acme", "1:1", "", "   ", "Q3   planning", "x".repeat(40)]);
    let m = api::patch_meeting(&app.ctx, &mid, &changes("tags", tags)).unwrap();
    assert_eq!(m["tags"], json!(["Acme", "1:1", "Q3 planning", "x".repeat(32)]));
    // the sidebar entry carries them too
    let listed = api::list_meetings(&app.ctx);
    let entry = listed.as_array().unwrap().iter().find(|e| e["id"] == mid.as_str()).unwrap();
    assert_eq!(entry["tags"], m["tags"]);

    for bad in [json!("Acme"), json!([1, 2]), json!(vec!["t"; 21])] {
        assert_eq!(
            api::patch_meeting(&app.ctx, &mid, &changes("tags", bad)).unwrap_err().code,
            422
        );
    }
    // a meeting without the field reads as no tags
    let other = app.upload("b.mp4", b"fake video bytes");
    assert_eq!(api::get_meeting(&app.ctx, &other).unwrap()["tags"], json!([]));
}

#[test]
fn tags_can_be_set_while_processing_and_survive_reprocessing() {
    let app = TestApp::fake();
    let mid = app.upload("a.mp4", b"fake video bytes");
    // allowed in any status, like a rename
    api::patch_meeting(&app.ctx, &mid, &changes("tags", json!(["Acme"]))).unwrap();
    app.wait_terminal(&mid);
    api::reprocess_meeting(&app.ctx, &mid, "notes").unwrap();
    let m = app.wait_terminal(&mid);
    assert_eq!(m["tags"], json!(["Acme"]));
}

#[test]
fn bulk_tagging_search_and_bulk_delete() {
    let app = TestApp::fake();
    let ids: Vec<String> = (0..3)
        .map(|i| {
            let id = app.upload(&format!("m{i}.mp4"), b"fake video bytes");
            app.wait_terminal(&id);
            id
        })
        .collect();
    let all = ids.clone();
    let r = api::tag_meetings(&app.ctx, &all, &["Acme".into()], &[]).unwrap();
    assert_eq!(r["updated"], 3);
    let two = vec![ids[0].clone(), ids[1].clone(), "gone".into()];
    let r = api::tag_meetings(&app.ctx, &two, &["Sales".into()], &["ACME".into()]).unwrap();
    assert_eq!(r["updated"], 2, "missing meetings are skipped");
    assert_eq!(api::get_meeting(&app.ctx, &ids[0]).unwrap()["tags"], json!(["Sales"]));
    assert_eq!(api::get_meeting(&app.ctx, &ids[2]).unwrap()["tags"], json!(["Acme"]));
    assert_eq!(api::tag_meetings(&app.ctx, &all, &[" ".into()], &[]).unwrap_err().code, 422);

    // search finds a tag like any other field
    let groups = api::search_meetings(&app.ctx, "sales");
    let groups = groups.as_array().unwrap();
    assert_eq!(groups.len(), 2);
    assert!(groups.iter().all(|g| g["hits"]
        .as_array()
        .unwrap()
        .iter()
        .any(|h| h["kind"] == "tag")));

    // bulk delete skips the ones already gone
    let r =
        api::delete_meetings(&app.ctx, &[ids[0].clone(), "gone".into(), ids[1].clone()]).unwrap();
    assert_eq!(r["deleted"], 2);
    assert_eq!(api::list_meetings(&app.ctx).as_array().unwrap().len(), 1);
}
