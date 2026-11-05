//! Asking questions about meetings (fake summarizer: it quotes the question
//! and cites the first transcript time it was given).

mod common;

use std::sync::atomic::AtomicBool;

use common::TestApp;
use lazymeet_core::{ask, pipeline, store};
use serde_json::json;

#[test]
fn a_question_streams_its_answer_and_is_kept_on_the_meeting() {
    let app = TestApp::fake();
    let mid = app.upload("sync.mp4", b"fake video bytes");
    app.wait_terminal(&mid);

    let entry = ask::ask_meeting(&app.ctx, &mid, "  When is the demo?  ", "r1").unwrap();
    assert_eq!(entry["q"], "When is the demo?");
    let answer = entry["a"].as_str().unwrap();
    assert!(answer.starts_with("About \"When is the demo?\""), "{answer}");
    assert!(answer.contains("[00:0"), "cites a transcript time: {answer}");

    // Deltas add up to the answer, then one "done".
    let events = app.events.payloads("ask-progress");
    let streamed: String = events.iter().filter_map(|e| e["delta"].as_str()).collect();
    assert_eq!(streamed.trim(), answer);
    let last = events.last().unwrap();
    assert_eq!(last["done"], true);
    assert_eq!(last["cancelled"], false);
    assert!(events.iter().all(|e| e["request_id"] == "r1"));

    let m = lazymeet_core::api::get_meeting(&app.ctx, &mid).unwrap();
    assert_eq!(m["qa"].as_array().unwrap().len(), 1);
    assert_eq!(m["qa"][0]["a"], answer);
    // Asking is not an edit: no new version.
    assert_eq!(m["versions"].as_array().unwrap().len(), 1);
}

#[test]
fn the_history_keeps_the_newest_answers() {
    let app = TestApp::fake();
    let mid = app.upload("sync.mp4", b"fake video bytes");
    app.wait_terminal(&mid);
    for i in 0..(ask::MAX_QA + 5) {
        store::push_qa(&app.ctx.paths, &app.ctx.locks, &mid, json!({"q": i}), ask::MAX_QA);
    }
    let m = lazymeet_core::api::get_meeting(&app.ctx, &mid).unwrap();
    let qa = m["qa"].as_array().unwrap();
    assert_eq!(qa.len(), ask::MAX_QA);
    assert_eq!(qa[0]["q"], 5);
}

#[test]
fn bad_questions_and_unready_meetings_are_refused() {
    let app = TestApp::fake();
    let mid = app.upload("sync.mp4", b"fake video bytes");
    app.wait_terminal(&mid);
    assert_eq!(ask::ask_meeting(&app.ctx, &mid, "   ", "r").unwrap_err().code, 422);
    let long = "x".repeat(ask::MAX_QUESTION_CHARS + 1);
    assert_eq!(ask::ask_meeting(&app.ctx, &mid, &long, "r").unwrap_err().code, 422);
    assert_eq!(ask::ask_meeting(&app.ctx, "nope", "Why?", "r").unwrap_err().code, 404);
    let queued =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "x.wav", -1, None, None).unwrap();
    let qid = queued["id"].as_str().unwrap();
    assert_eq!(
        ask::ask_meeting(&app.ctx, qid, "Why?", "r").unwrap_err().code,
        409,
        "no transcript yet"
    );
}

#[test]
fn a_cancelled_answer_stops() {
    let flag = AtomicBool::new(true);
    let mut got = String::new();
    let out =
        pipeline::fake::ask("[00:03] A: hi\n\nQuestion: why?", &mut |d| got.push_str(d), &flag);
    assert!(out.is_empty() && got.is_empty());
}

#[test]
fn asking_across_meetings_labels_the_sources_by_date() {
    let app = TestApp::fake();
    assert_eq!(
        ask::ask_all(&app.ctx, "Anything?", "r").unwrap_err().code,
        409,
        "nothing to ask yet"
    );
    let a = app.upload("first.mp4", b"fake video bytes");
    app.wait_terminal(&a);
    let b = app.upload("second.mp4", b"fake video bytes");
    app.wait_terminal(&b);
    let out = ask::ask_all(&app.ctx, "What was decided about the speaker library?", "r2").unwrap();
    let sources = out["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0]["label"], "M1");
    assert_eq!(sources[0]["id"], a.as_str(), "oldest first");
    assert_eq!(sources[1]["id"], b.as_str());
    assert!(out["a"].as_str().unwrap().contains("speaker library"));
}

#[test]
fn all_meetings_prompt_keeps_only_matching_windows() {
    let src = |label: &str, lines: &[&str]| ask::Source {
        label: label.into(),
        id: label.to_lowercase(),
        title: format!("Meeting {label}"),
        created_at: "2026-11-01T10:00:00Z".into(),
        lines: lines.iter().map(|s| s.to_string()).collect(),
    };
    let sources = vec![
        src("M1", &["[00:01] A: lunch plans", "[00:05] B: weather talk"]),
        src(
            "M2",
            &["[00:01] A: the Acme contract renews in May", "[00:09] B: Acme wants a discount"],
        ),
    ];
    let prompt =
        ask::all_meetings_prompt(&sources, "When does the Acme contract renew?", &|_| true);
    assert!(prompt.contains("M2: Meeting M2 (2026-11-01)"), "{prompt}");
    assert!(!prompt.contains("M1:"), "a meeting with no match stays out: {prompt}");
    assert!(prompt.ends_with("Question: When does the Acme contract renew?"));
}
