//! Env-precedence tests (env > config.json > default). These mutate process
//! environment variables, so they serialize on a lock — and live in their own
//! test binary, which cargo runs after the others finish.

mod common;

use std::sync::Mutex;

use common::TestApp;
use lazymeet_core::config;
use serde_json::json;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard(&'static str);
impl EnvGuard {
    fn set(name: &'static str, value: &str) -> Self {
        std::env::set_var(name, value);
        EnvGuard(name)
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        std::env::remove_var(self.0);
    }
}

#[test]
fn precedence_env_beats_file_beats_default() {
    let _lock = ENV_LOCK.lock().unwrap();
    let app = TestApp::fake();
    let mut overrides = serde_json::Map::new();
    overrides.insert("SPEAKER_THRESHOLD".into(), json!(0.9));
    config::save_overrides(&app.ctx.paths, &overrides).unwrap();

    {
        let _env = EnvGuard::set("LAZYMEET_SPEAKER_THRESHOLD", "0.3");
        app.ctx.refresh_config();
        assert_eq!(app.ctx.config.read().unwrap().speaker_threshold(), 0.3);
        assert!(config::env_locked().contains(&"SPEAKER_THRESHOLD".to_string()));
    }

    app.ctx.refresh_config();
    assert_eq!(app.ctx.config.read().unwrap().speaker_threshold(), 0.9);

    config::save_overrides(&app.ctx.paths, &serde_json::Map::new()).unwrap();
    app.ctx.refresh_config();
    assert_eq!(app.ctx.config.read().unwrap().speaker_threshold(), 0.6);
}

#[test]
fn summary_provider_env_toggle() {
    let _lock = ENV_LOCK.lock().unwrap();
    let app = TestApp::fake();
    // The summarizer accelerates by default where a GPU backend exists.
    let default_gpu = cfg!(target_os = "macos");
    assert_eq!(app.ctx.config.read().unwrap().use_gpu(), default_gpu);
    {
        let _env = EnvGuard::set("LAZYMEET_SUMMARY_PROVIDER", "cpu");
        app.ctx.refresh_config();
        assert!(!app.ctx.config.read().unwrap().use_gpu());
    }
    app.ctx.refresh_config();
    assert_eq!(app.ctx.config.read().unwrap().use_gpu(), default_gpu);
}

#[test]
fn env_locked_is_empty_without_env() {
    let _lock = ENV_LOCK.lock().unwrap();
    assert_eq!(config::env_locked(), Vec::<String>::new());
}
