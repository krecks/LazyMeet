//! Model catalog API, resolvers, and the processing queue (port of
//! test_models.py). The Python gating tests monkeypatched is_ready/run_worker;
//! here the same behavior is driven through the real gate: a fake=false app
//! whose models dir has no marker files.

mod common;

use common::{create_model_markers, obj, status_events, TestApp};
use lazymeet_core::{api, jobs, models};
use serde_json::json;
use std::time::{Duration, Instant};

#[test]
fn models_catalog_shape() {
    let app = TestApp::fake();
    let data = api::get_models(&app.ctx);
    for (step, default) in [("asr", "parakeet-tdt-0.6b-v3"), ("summary", "qwen3.5-4b")] {
        assert_eq!(data[step]["active"], default);
        let options = data[step]["options"].as_array().unwrap();
        let ids: Vec<&str> = options.iter().map(|o| o["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&default));
        for opt in options {
            for key in ["id", "label", "ready", "downloading", "error", "pros", "cons"] {
                assert!(opt.get(key).is_some(), "missing {key}");
            }
            // The settings UI renders a pros/cons overview for every model.
            assert!(!opt["pros"].as_array().unwrap().is_empty());
            assert!(!opt["cons"].as_array().unwrap().is_empty());
        }
    }
    let asr_ids: Vec<&str> = data["asr"]["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["id"].as_str().unwrap())
        .collect();
    assert!(asr_ids.contains(&"whisper-small") && asr_ids.contains(&"whisper-base"));
    let llm_ids: Vec<&str> = data["summary"]["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["id"].as_str().unwrap())
        .collect();
    assert_eq!(llm_ids, ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]);
}

#[test]
fn put_model_and_language_settings() {
    let app = TestApp::fake();
    api::put_config(
        &app.ctx,
        &obj(&[
            ("ASR_MODEL", json!("whisper-small")),
            ("ASR_LANGUAGE", json!("de")),
            ("SUMMARY_MODEL", json!("qwen3.5-9b")),
            ("SUMMARY_LANGUAGE", json!("de")),
        ]),
    )
    .unwrap();
    let snapshot = app.ctx.config.read().unwrap();
    assert_eq!(snapshot.asr_model(), "whisper-small");
    assert_eq!(snapshot.summary_language(), "de");
    drop(snapshot);
    assert_eq!(api::get_models(&app.ctx)["asr"]["active"], "whisper-small");
}

#[test]
fn put_rejects_unknown_models_and_languages() {
    let app = TestApp::fake();
    for body in [
        obj(&[("ASR_MODEL", json!("whisper-nonexistent"))]),
        obj(&[("SUMMARY_MODEL", json!("gpt-4"))]),
        obj(&[("ASR_LANGUAGE", json!("klingon"))]),
        obj(&[("SUMMARY_LANGUAGE", json!("xx"))]),
    ] {
        assert_eq!(api::put_config(&app.ctx, &body).unwrap_err().code, 422, "{body:?}");
    }
}

#[test]
fn download_endpoint_validates() {
    let app = TestApp::fake();
    assert_eq!(api::download_model(&app.ctx, "asr", "nope").unwrap_err().code, 404);
    assert_eq!(api::download_model(&app.ctx, "bogus", "whisper-small").unwrap_err().code, 404);
    // FAKE mode: everything counts as ready, request is a no-op success.
    assert_eq!(api::download_model(&app.ctx, "asr", "whisper-small").unwrap(), json!({"ok": true}));
}

#[test]
fn resolvers_fall_back_to_defaults() {
    let app = TestApp::fake();
    // Unknown ids resolve to the default model rather than erroring.
    let asr = models::resolve_asr(&app.ctx, "does-not-exist");
    assert_eq!(asr.kind, "nemo_transducer");
    assert_eq!(asr.encoder, app.ctx.paths.models_dir.join("parakeet").join("encoder.onnx"));
    assert_eq!(
        models::resolve_llm(&app.ctx, "does-not-exist"),
        app.ctx.paths.models_dir.join("qwen3.5-4b-q4_k_m.gguf")
    );
}

#[test]
fn uploads_process_sequentially() {
    // The FAKE worker is near-instant, so strict one-at-a-time ordering is
    // asserted from the event stream: each meeting's first stage event must
    // come after the previous meeting's terminal event.
    let app = TestApp::fake();
    let first = app.upload("first.mp4", b"fake video bytes");
    let second = app.upload("second.mp4", b"fake video bytes");
    app.wait_terminal(&first);
    app.wait_terminal(&second);

    let all: Vec<(String, String)> = app
        .events
        .payloads("meeting-updated")
        .into_iter()
        .map(|p| {
            (p["id"].as_str().unwrap().to_string(), p["status"].as_str().unwrap().to_string())
        })
        .collect();
    let first_done = all
        .iter()
        .position(|(id, s)| *id == first && (s == "done" || s == "error"))
        .expect("first finished");
    let second_started = all
        .iter()
        .position(|(id, _)| *id == second)
        .expect("second processed");
    assert!(
        second_started > first_done,
        "second meeting started before the first finished: {all:?}"
    );
}

#[test]
fn uploads_wait_for_model_downloads() {
    // While first-run models are missing, uploads queue; once the markers
    // exist a pump processes them — exactly what the download-complete path
    // does. (fake=false so is_ready() consults the real marker files; the
    // native pipeline stubs make the meeting terminal as "error", which still
    // proves the queue released it.)
    let app = TestApp::with_fake(false);
    let mid = app.upload("early-bird.mp4", b"fake video bytes");
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(api::get_meeting(&app.ctx, &mid).unwrap()["status"], "queued");

    create_model_markers(&app.ctx);
    jobs::pump_queue(&app.ctx); // what the download-complete callback does

    let deadline = Instant::now() + Duration::from_secs(15);
    let mut status = String::new();
    while Instant::now() < deadline {
        status = api::get_meeting(&app.ctx, &mid).unwrap()["status"]
            .as_str()
            .unwrap()
            .to_string();
        if status == "done" || status == "error" {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(status == "done" || status == "error", "meeting still {status}");
    // It definitely left the queue and went through the pipeline stages.
    assert!(status_events(&app.events, &mid).contains(&"extracting".to_string()));
}

#[test]
fn deleted_queued_meeting_is_skipped() {
    // Hold three uploads in the queue (models missing), delete the middle one,
    // then release the queue: the deleted meeting must never start.
    let app = TestApp::with_fake(false);
    let a = app.upload("a.mp4", b"fake video bytes");
    let b = app.upload("b.mp4", b"fake video bytes");
    let c = app.upload("c.mp4", b"fake video bytes");
    std::thread::sleep(Duration::from_millis(100));
    for id in [&a, &b, &c] {
        assert_eq!(api::get_meeting(&app.ctx, id).unwrap()["status"], "queued");
    }
    api::delete_meeting(&app.ctx, &b).unwrap();

    create_model_markers(&app.ctx);
    jobs::pump_queue(&app.ctx);

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let sa = api::get_meeting(&app.ctx, &a).unwrap()["status"].as_str().unwrap().to_string();
        let sc = api::get_meeting(&app.ctx, &c).unwrap()["status"].as_str().unwrap().to_string();
        let terminal = |s: &str| s == "done" || s == "error";
        if terminal(&sa) && terminal(&sc) {
            break;
        }
        assert!(Instant::now() < deadline, "queue never drained: {sa} {sc}");
        std::thread::sleep(Duration::from_millis(20));
    }
    // The deleted meeting produced no processing events at all.
    assert!(status_events(&app.events, &b).is_empty());
}
