//! Settings API + config.json handling (port of test_config.py; the
//! env-precedence tests live in config_env.rs — they mutate process env).

mod common;

use common::{obj, TestApp};
use lazymeet_core::{api, config, store};
use serde_json::json;

#[test]
fn get_config_shape() {
    let app = TestApp::fake();
    let data = api::config_payload(&app.ctx);
    for key in ["values", "overrides", "defaults", "env_locked", "limits"] {
        assert!(data.get(key).is_some(), "missing {key}");
    }
    let names: Vec<String> = config::tunables().iter().map(|s| s.name.to_string()).collect();
    let values: Vec<String> = data["values"].as_object().unwrap().keys().cloned().collect();
    assert_eq!(values.len(), names.len());
    for name in &names {
        assert!(values.contains(name), "missing tunable {name}");
    }
    assert_eq!(data["defaults"]["SPEAKER_THRESHOLD"], 0.6);
    assert_eq!(data["limits"]["DIAR_THRESHOLD"], json!({"min": 0.0, "max": 1.0}));
    assert_eq!(data["limits"]["KEEP_AUDIO"], json!({"min": null, "max": null}));
}

#[test]
fn put_config_persists_and_applies() {
    let app = TestApp::fake();
    let data = api::put_config(
        &app.ctx,
        &obj(&[
            ("SPEAKER_THRESHOLD", json!(0.8)),
            ("KEEP_AUDIO", json!(true)),
            ("NUM_THREADS", json!(1)), // valid on any host (bound = core count)
        ]),
    )
    .unwrap();
    assert_eq!(data["values"]["SPEAKER_THRESHOLD"], 0.8);
    assert_eq!(
        data["overrides"],
        json!({"SPEAKER_THRESHOLD": 0.8, "KEEP_AUDIO": true, "NUM_THREADS": 1})
    );
    assert!(app.ctx.paths.config_file().exists());
    let snapshot = app.ctx.config.read().unwrap();
    assert_eq!(snapshot.speaker_threshold(), 0.8);
    assert!(snapshot.keep_audio());
    drop(snapshot);

    // A fresh read agrees, and untouched values stay at their defaults.
    let again = api::config_payload(&app.ctx);
    assert_eq!(again["values"]["SPEAKER_THRESHOLD"], 0.8);
    assert_eq!(again["values"]["DIAR_THRESHOLD"], 0.7);
}

#[test]
fn put_null_removes_override() {
    let app = TestApp::fake();
    api::put_config(&app.ctx, &obj(&[("DIAR_THRESHOLD", json!(0.9))])).unwrap();
    assert_eq!(app.ctx.config.read().unwrap().diar_threshold(), 0.9);
    let data = api::put_config(&app.ctx, &obj(&[("DIAR_THRESHOLD", json!(null))])).unwrap();
    assert!(data["overrides"].get("DIAR_THRESHOLD").is_none());
    assert_eq!(app.ctx.config.read().unwrap().diar_threshold(), 0.7);
}

#[test]
fn put_config_rejects_invalid() {
    let app = TestApp::fake();
    for body in [
        obj(&[("SPEAKER_THRESHOLD", json!(1.5))]),
        obj(&[("SPEAKER_THRESHOLD", json!(-0.1))]),
        obj(&[("NUM_THREADS", json!(-1))]),
        obj(&[("NUM_THREADS", json!(config::max_threads() + 1))]),
        obj(&[("MIN_SPEAKER_SECONDS", json!(9999))]),
        obj(&[("KEEP_AUDIO", json!("sometimes"))]),
    ] {
        let err = api::put_config(&app.ctx, &body).unwrap_err();
        assert_eq!(err.code, 422, "{body:?}");
    }
}

#[test]
fn summary_prompt_round_trip() {
    let app = TestApp::fake();
    // The default is exposed so the settings UI can show (and restore) it.
    let payload = api::config_payload(&app.ctx);
    assert_eq!(payload["defaults"]["SUMMARY_PROMPT"], config::DEFAULT_SUMMARY_PROMPT);
    assert_eq!(payload["values"]["SUMMARY_PROMPT"], config::DEFAULT_SUMMARY_PROMPT);

    // A custom prompt persists and resolves.
    let data = api::put_config(
        &app.ctx,
        &obj(&[("SUMMARY_PROMPT", json!("Focus on decisions and open questions."))]),
    )
    .unwrap();
    assert_eq!(data["values"]["SUMMARY_PROMPT"], "Focus on decisions and open questions.");
    assert_eq!(
        app.ctx.config.read().unwrap().summary_prompt(),
        "Focus on decisions and open questions."
    );

    // Beyond the 2000-char cap and blank values are rejected.
    let long = "x".repeat(2001);
    let err = api::put_config(&app.ctx, &obj(&[("SUMMARY_PROMPT", json!(long))])).unwrap_err();
    assert_eq!(err.code, 422);
    let err = api::put_config(&app.ctx, &obj(&[("SUMMARY_PROMPT", json!("   "))])).unwrap_err();
    assert_eq!(err.code, 422);

    // null (the UI sends it for an emptied field) restores the default.
    let data = api::put_config(&app.ctx, &obj(&[("SUMMARY_PROMPT", json!(null))])).unwrap();
    assert_eq!(data["values"]["SUMMARY_PROMPT"], config::DEFAULT_SUMMARY_PROMPT);
}

#[test]
fn num_threads_defaults_to_auto_and_tracks_cores() {
    let cores = std::thread::available_parallelism().map(|n| n.get() as i64).unwrap_or(8);
    let maximum = config::max_threads();
    assert_eq!(maximum, cores.min(16));
    // The saved default is "auto"; the resolved count is 75 % of the cores,
    // capped, so the same config.json works on a 4-core laptop and a 32-core
    // workstation without pinning either to a thread count.
    assert_eq!(config::default_value("NUM_THREADS"), json!(config::AUTO_THREADS));
    assert_eq!(config::auto_threads(), (cores * 3 / 4).clamp(1, maximum));
    let auto = config::Config::from_overrides(&Default::default());
    assert!(auto.auto_threads());
    assert_eq!(auto.num_threads(), config::auto_threads());
    // An explicit value is honored (and still clamped to the machine).
    let mut overrides = serde_json::Map::new();
    overrides.insert("NUM_THREADS".into(), json!(1));
    let pinned = config::Config::from_overrides(&overrides);
    assert!(!pinned.auto_threads());
    assert_eq!(pinned.num_threads(), 1);
}

#[test]
fn config_payload_reports_auto_threads() {
    let app = TestApp::fake();
    let data = api::config_payload(&app.ctx);
    assert_eq!(data["cores"], json!(config::cpu_count()));
    assert_eq!(data["auto_threads"], json!(config::auto_threads()));
    assert_eq!(data["limits"]["NUM_THREADS"]["min"], json!(0.0));
}

#[test]
fn corrupt_config_json_is_tolerated() {
    let app = TestApp::fake();
    std::fs::write(app.ctx.paths.config_file(), "{not json").unwrap();
    assert!(config::load_overrides(&app.ctx.paths).is_empty());
    app.ctx.refresh_config();
    assert_eq!(app.ctx.config.read().unwrap().speaker_threshold(), 0.6);
}

#[test]
fn unknown_and_invalid_file_values_ignored() {
    // NUM_THREADS: 1 is valid on any host — the upper bound tracks the
    // machine's core count (a 2-core CI runner rejects 3, correctly).
    let app = TestApp::fake();
    std::fs::write(
        app.ctx.paths.config_file(),
        r#"{"BOGUS": 1, "DIAR_THRESHOLD": 5.0, "NUM_THREADS": 1}"#,
    )
    .unwrap();
    let overrides = config::load_overrides(&app.ctx.paths);
    assert_eq!(serde_json::Value::Object(overrides), json!({"NUM_THREADS": 1}));
}

#[test]
fn worker_picks_up_config_changes() {
    // Workers take a fresh Config snapshot per job (Python's refresh()).
    let app = TestApp::fake();
    let mut overrides = serde_json::Map::new();
    overrides.insert("MIN_SPEAKER_SECONDS".into(), json!(42.0));
    config::save_overrides(&app.ctx.paths, &overrides).unwrap();
    // The app-level snapshot has NOT been refreshed:
    assert_ne!(app.ctx.config.read().unwrap().min_speaker_seconds(), 42.0);

    // The worker still resolves the new value at job start:
    let fresh = config::Config::resolve(&app.ctx.paths);
    assert_eq!(fresh.min_speaker_seconds(), 42.0);

    // And a full FAKE run completes with the override in place.
    let mid = app.upload("cfg.mp4", b"fake video bytes");
    let meeting = app.wait_terminal(&mid);
    assert_eq!(meeting["status"], "done", "{:?}", meeting["error"]);
}

#[test]
fn onnx_providers_default_to_cpu_and_validate() {
    // CPU measures fastest for the ONNX models on every machine tested; the
    // per-stage providers default there and only accept known values.
    let app = TestApp::fake();
    let snapshot = app.ctx.config.read().unwrap();
    assert_eq!(snapshot.asr_provider(), "cpu");
    assert_eq!(snapshot.diar_provider(), "cpu");
    assert_eq!(snapshot.speaker_provider(), "cpu");
    drop(snapshot);

    let err = api::put_config(&app.ctx, &obj(&[("DIAR_PROVIDER", json!("cuda"))])).unwrap_err();
    assert_eq!(err.code, 422);
    let err = api::put_config(&app.ctx, &obj(&[("SUMMARY_PROVIDER", json!("npu"))])).unwrap_err();
    assert_eq!(err.code, 422);
    // CoreML is no longer compiled in: rejected on every platform, and a
    // value saved by an older version reads as the CPU.
    let err = api::put_config(&app.ctx, &obj(&[("ASR_PROVIDER", json!("coreml"))])).unwrap_err();
    assert_eq!(err.code, 422);
    std::fs::write(app.ctx.paths.config_file(), r#"{"DIAR_PROVIDER": "coreml"}"#).unwrap();
    app.ctx.refresh_config();
    assert_eq!(app.ctx.config.read().unwrap().diar_provider(), "cpu");
    // The UI learns what this platform offers from the config payload.
    let payload = api::config_payload(&app.ctx);
    assert!(payload["compute_options"]["SUMMARY_PROVIDER"][0].is_string());
}

#[test]
fn atomic_writes_leave_no_temp_files() {
    let app = TestApp::fake();
    let record =
        store::create_meeting(&app.ctx.paths, &app.ctx.locks, "tmp.mp4", -1, None, None).unwrap();
    store::delete_meeting(&app.ctx.paths, &app.ctx.locks, record["id"].as_str().unwrap());
    let leftovers: Vec<_> = std::fs::read_dir(app.ctx.paths.meetings_dir())
        .unwrap()
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("tmp"))
        .collect();
    assert!(leftovers.is_empty());
}

#[test]
fn legacy_use_gpu_carries_over_to_summary_provider() {
    let app = TestApp::fake();
    std::fs::write(app.ctx.paths.config_file(), r#"{"USE_GPU": false, "NUM_THREADS": 1}"#).unwrap();
    let overrides = config::load_overrides(&app.ctx.paths);
    assert_eq!(overrides.get("SUMMARY_PROVIDER"), Some(&json!("cpu")));
    assert!(overrides.get("USE_GPU").is_none());
    // An explicit SUMMARY_PROVIDER wins over the legacy key.
    let options = config::summary_provider_options();
    let explicit = options.last().unwrap();
    std::fs::write(
        app.ctx.paths.config_file(),
        format!(r#"{{"USE_GPU": false, "SUMMARY_PROVIDER": "{explicit}"}}"#),
    )
    .unwrap();
    let overrides = config::load_overrides(&app.ctx.paths);
    assert_eq!(overrides.get("SUMMARY_PROVIDER"), Some(&json!(explicit)));
}

#[test]
fn config_payload_lists_the_languages() {
    let app = TestApp::fake();
    let data = api::config_payload(&app.ctx);
    let codes: Vec<&str> =
        data["languages"].as_array().unwrap().iter().map(|l| l[0].as_str().unwrap()).collect();
    for code in ["en", "de", "ar", "el", "vi"] {
        assert!(codes.contains(&code), "missing {code}");
    }
}
