//! Global search across every meeting on disk: literal, case-insensitive,
//! grouped per meeting. The user question it answers is "in which meeting did
//! we talk about Arbeitspaket?".

mod common;

use common::{changes, obj, TestApp};
use lazymeet_core::{api, store};
use serde_json::{json, Value};

fn process_meeting(app: &TestApp, name: &str) -> Value {
    let mid = app.upload(name, b"fake video bytes");
    let meeting = app.wait_terminal(&mid);
    assert_eq!(meeting["status"], "done", "{:?}", meeting["error"]);
    meeting
}

fn search(app: &TestApp, query: &str) -> Vec<Value> {
    api::search_meetings(&app.ctx, query).as_array().unwrap().clone()
}

/// The one group for `mid`, or panic — most assertions are about a single
/// meeting's hits.
fn group_for<'a>(results: &'a [Value], mid: &str) -> &'a Value {
    results
        .iter()
        .find(|g| g["id"].as_str() == Some(mid))
        .unwrap_or_else(|| panic!("no result group for {mid} in {results:#?}"))
}

fn kinds(group: &Value) -> Vec<String> {
    group["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["kind"].as_str().unwrap().to_string())
        .collect()
}

// --------------------------------------------------------------------------
// Matching
// --------------------------------------------------------------------------

#[test]
fn substring_match_is_case_insensitive() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "sync.mp4");
    let mid = meeting["id"].as_str().unwrap();
    api::patch_segments(
        &app.ctx,
        mid,
        &[
            (0, "Das Arbeitspaket ist fertig.".into()),
            (1, "PAKET drei verschieben wir.".into()),
        ],
    )
    .unwrap();

    // "Paket" must reach "Arbeitspaket" — the compound is the whole point.
    for query in ["paket", "Paket", "PAKET"] {
        let results = search(&app, query);
        assert_eq!(results.len(), 1, "{query}");
        assert_eq!(group_for(&results, mid)["count"], 2, "{query}");
    }

    // The longer term is strictly narrower.
    let results = search(&app, "arbeitspaket");
    let group = group_for(&results, mid);
    assert_eq!(group["count"], 1);
    assert_eq!(group["hits"][0]["index"], 0);
}

#[test]
fn count_totals_occurrences_not_matching_fields() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "sync.mp4");
    let mid = meeting["id"].as_str().unwrap();
    api::patch_segments(
        &app.ctx,
        mid,
        &[(0, "Das Arbeitspaket blockiert das naechste Arbeitspaket.".into())],
    )
    .unwrap();

    let results = search(&app, "arbeitspaket");
    let group = group_for(&results, mid);
    // one row to click, but two occurrences — the same tally the transcript
    // find bar shows for this text
    assert_eq!(group["hits"].as_array().unwrap().len(), 1);
    assert_eq!(group["count"], 2);
    assert_eq!(group["truncated"], false);
}

#[test]
fn empty_and_absent_queries_return_nothing() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "sync.mp4");
    let mid = meeting["id"].as_str().unwrap();
    api::patch_segments(&app.ctx, mid, &[(0, "Das Arbeitspaket ist fertig.".into())]).unwrap();

    assert!(search(&app, "").is_empty());
    assert!(search(&app, "   ").is_empty());
    assert!(search(&app, "Zebrastreifen").is_empty());
}

#[test]
fn regex_metacharacters_are_literal() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "sync.mp4");
    let mid = meeting["id"].as_str().unwrap();
    api::patch_segments(&app.ctx, mid, &[(0, "Kosten (netto) sind offen.".into())]).unwrap();

    assert_eq!(search(&app, "(netto)").len(), 1);
    assert!(search(&app, "n.tto").is_empty()); // '.' is not a wildcard
}

// --------------------------------------------------------------------------
// What gets searched
// --------------------------------------------------------------------------

#[test]
fn every_field_kind_is_searched_and_labelled() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "sync.mp4");
    let mid = meeting["id"].as_str().unwrap();

    api::patch_meeting(&app.ctx, mid, &changes("title", json!("Paket Kickoff"))).unwrap();
    api::rename_speakers(&app.ctx, mid, &changes("SPEAKER_00", json!("Petra Paketier"))).unwrap();
    api::patch_segments(&app.ctx, mid, &[(2, "Das Arbeitspaket ist fertig.".into())]).unwrap();
    api::patch_meeting(
        &app.ctx,
        mid,
        &obj(&[
            ("summary", json!("Ein Paket wurde besprochen.")),
            ("takeaways", json!(["Paket drei verschieben."])),
            ("action_items", json!(["Arbeitspaket zuweisen."])),
        ]),
    )
    .unwrap();

    let results = search(&app, "paket");
    let group = group_for(&results, mid);
    assert_eq!(group["count"], 6);
    assert_eq!(group["truncated"], false);
    assert_eq!(
        kinds(group),
        ["title", "speaker", "segment", "summary", "takeaway", "action_item"]
    );

    // Group header carries what the result list renders.
    assert_eq!(group["title"], "Paket Kickoff");
    assert_eq!(group["status"], "done");
    assert!(group["created_at"].is_string());

    let hits = group["hits"].as_array().unwrap();
    assert_eq!(hits[0]["text"], "Paket Kickoff");
    assert_eq!(hits[1]["speaker"], "SPEAKER_00");
    assert_eq!(hits[1]["text"], "Petra Paketier");

    // A segment hit must be enough to jump to it in the transcript.
    let seg = &hits[2];
    assert_eq!(seg["index"], 2);
    assert_eq!(seg["speaker"], "SPEAKER_01");
    assert_eq!(seg["text"], "Das Arbeitspaket ist fertig.");
    assert!(seg["start"].as_f64().is_some());
    // resolves through the meeting's speakers map (default label until renamed)
    assert_eq!(seg["speaker_name"], "Speaker 2");
    assert_eq!(hits[4]["index"], 0); // takeaway position
    assert_eq!(hits[5]["text"], "Arbeitspaket zuweisen.");
}

#[test]
fn segment_hits_use_the_display_speaker_name() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "sync.mp4");
    let mid = meeting["id"].as_str().unwrap();
    api::patch_segments(&app.ctx, mid, &[(0, "Das Arbeitspaket ist fertig.".into())]).unwrap();
    api::rename_speakers(&app.ctx, mid, &changes("SPEAKER_00", json!("Anna"))).unwrap();

    let results = search(&app, "arbeitspaket");
    let hit = &group_for(&results, mid)["hits"][0];
    assert_eq!(hit["speaker"], "SPEAKER_00");
    assert_eq!(hit["speaker_name"], "Anna");
}

#[test]
fn version_snapshots_are_not_searched() {
    let app = TestApp::fake();
    let meeting = process_meeting(&app, "sync.mp4");
    let mid = meeting["id"].as_str().unwrap();
    api::patch_segments(&app.ctx, mid, &[(0, "Das Arbeitspaket ist fertig.".into())]).unwrap();
    assert_eq!(group_for(&search(&app, "paket"), mid)["count"], 1);

    // A notes re-run snapshots the whole transcript into versions[]; the same
    // sentence must still be reported once, not once per version.
    api::reprocess_meeting(&app.ctx, mid, "notes").unwrap();
    let m = app.wait_terminal(mid);
    assert_eq!(m["status"], "done", "{:?}", m["error"]);
    assert_eq!(m["versions"].as_array().unwrap().len(), 2);

    let results = search(&app, "paket");
    assert_eq!(group_for(&results, mid)["count"], 1);
}

// --------------------------------------------------------------------------
// Grouping across the library
// --------------------------------------------------------------------------

#[test]
fn hits_are_grouped_per_meeting_newest_first() {
    let app = TestApp::fake();
    let older = process_meeting(&app, "older.mp4");
    let older_id = older["id"].as_str().unwrap().to_string();
    let newer = process_meeting(&app, "newer.mp4");
    let newer_id = newer["id"].as_str().unwrap().to_string();
    let quiet = process_meeting(&app, "quiet.mp4");
    let quiet_id = quiet["id"].as_str().unwrap().to_string();

    api::patch_segments(&app.ctx, &older_id, &[(0, "Arbeitspaket eins.".into())]).unwrap();
    api::patch_segments(
        &app.ctx,
        &newer_id,
        &[(0, "Paket zwei.".into()), (1, "Und Paket drei.".into())],
    )
    .unwrap();

    let results = search(&app, "paket");
    let ids: Vec<&str> = results.iter().map(|g| g["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&quiet_id.as_str())); // no hits, no group
    assert_eq!(group_for(&results, &older_id)["count"], 1);
    assert_eq!(group_for(&results, &newer_id)["count"], 2);

    // same ordering contract as list_meetings
    let dates: Vec<&str> = results.iter().map(|g| g["created_at"].as_str().unwrap()).collect();
    assert!(dates.windows(2).all(|w| w[0] >= w[1]), "not newest-first: {dates:?}");
}

#[test]
fn hits_per_meeting_are_capped_but_the_count_is_honest() {
    let app = TestApp::fake();
    // Hand-written record: 250 matching segments is more than any fake run
    // produces, and the cap is what keeps the IPC payload bounded.
    let segments: Vec<Value> = (0..250)
        .map(|i| {
            json!({
                "start": i as f64,
                "end": i as f64 + 1.0,
                "speaker": "SPEAKER_00",
                "text": format!("Arbeitspaket {i} besprochen."),
            })
        })
        .collect();
    let mid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let record = json!({
        "id": mid,
        "filename": "big.mp4",
        "title": "Grosse Runde",
        "status": "done",
        "created_at": store::now_iso(),
        "speakers": {"SPEAKER_00": "Anna"},
        "segments": segments,
    });
    store::atomic_write_json(
        &app.ctx.paths.meetings_dir().join(format!("{mid}.json")),
        &record,
    )
    .unwrap();

    let results = search(&app, "arbeitspaket");
    let group = group_for(&results, mid);
    assert_eq!(group["count"], 250);
    assert_eq!(group["truncated"], true);
    assert_eq!(group["hits"].as_array().unwrap().len(), 200);
}

#[test]
fn meetings_still_processing_match_on_their_title() {
    let app = TestApp::fake();
    let mid = app.upload("Arbeitspaket Review.mp4", b"fake video bytes");
    let results = search(&app, "arbeitspaket");
    let group = group_for(&results, &mid);
    assert_eq!(group["hits"][0]["kind"], "title");
    // title defaults to the upload filename before the run finishes
    assert_eq!(group["hits"][0]["text"], "Arbeitspaket Review.mp4");
}
