//! Shared test harness: an isolated app instance per test (own temp data dir,
//! FAKE pipeline, event collector) — the Rust equivalent of conftest.py's
//! TestClient fixture, minus the process-global env vars.

// Each integration-test binary compiles this module separately, so every helper
// is unused in some binaries; dead-code warnings here would only be noise.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use lazymeet_core::config::Paths;
use lazymeet_core::events::CollectorEvents;
use lazymeet_core::{api, Ctx};

pub struct TestApp {
    pub ctx: Arc<Ctx>,
    pub events: Arc<CollectorEvents>,
    // Owns the temp dir for the test's lifetime.
    pub tmp: tempfile::TempDir,
}

impl TestApp {
    /// FAKE-mode app over a fresh temp data dir, dirs created (startup ran).
    pub fn fake() -> Self {
        Self::with_fake(true)
    }

    /// fake=false leaves the models "missing" — is_ready() is false until
    /// marker files appear, which is how queue-gating tests hold jobs back.
    pub fn with_fake(fake: bool) -> Self {
        let tmp = tempfile::tempdir().expect("tempdir");
        let events = Arc::new(CollectorEvents::default());
        let ctx = Ctx::new(Paths::at(tmp.path().join("data")), fake, events.clone());
        ctx.paths.ensure_dirs().expect("ensure_dirs");
        TestApp { ctx, events, tmp }
    }

    /// Write `bytes` to a scratch file named `name` and create a meeting from
    /// it — the native replacement for the old multipart upload.
    pub fn upload(&self, name: &str, bytes: &[u8]) -> String {
        let src = self.tmp.path().join("incoming");
        std::fs::create_dir_all(&src).unwrap();
        let path = src.join(name);
        std::fs::write(&path, bytes).unwrap();
        let res = api::create_meeting(&self.ctx, &path, -1, None, None).expect("create_meeting");
        res["id"].as_str().unwrap().to_string()
    }

    /// Like `upload`, but restricted to the given time ranges.
    pub fn upload_ranges(&self, name: &str, bytes: &[u8], ranges: &[(f64, f64)]) -> String {
        let src = self.tmp.path().join("incoming");
        std::fs::create_dir_all(&src).unwrap();
        let path = src.join(name);
        std::fs::write(&path, bytes).unwrap();
        let res =
            api::create_meeting(&self.ctx, &path, -1, Some(ranges), None).expect("create_meeting");
        res["id"].as_str().unwrap().to_string()
    }

    /// Poll until the meeting reaches done/error (the FAKE worker is a thread).
    pub fn wait_terminal(&self, meeting_id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut last = Value::Null;
        while Instant::now() < deadline {
            last = api::get_meeting(&self.ctx, meeting_id).expect("meeting exists");
            let status = last["status"].as_str().unwrap_or("");
            if status == "done" || status == "error" {
                return last;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("meeting stuck in {:?}", last["status"]);
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.ctx.paths.uploads_dir()
    }
}

/// Build a serde_json object Map from key/value pairs.
pub fn obj(entries: &[(&str, Value)]) -> Map<String, Value> {
    let mut map = Map::new();
    for (k, v) in entries {
        map.insert(k.to_string(), v.clone());
    }
    map
}

/// `{"speakers": {...}}`-style single-key object helper.
pub fn changes(key: &str, value: Value) -> Map<String, Value> {
    obj(&[(key, value)])
}

/// Touch every first-run model marker so is_ready() flips true (fake=false).
pub fn create_model_markers(ctx: &Ctx) {
    for spec in lazymeet_core::models::model_specs() {
        let marker = ctx.paths.models_dir.join(&spec.marker);
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, b"").unwrap();
    }
}

/// Statuses of the `meeting-updated` events recorded for one meeting id.
pub fn status_events(events: &CollectorEvents, meeting_id: &str) -> Vec<String> {
    events
        .payloads("meeting-updated")
        .into_iter()
        .filter(|p| p["id"].as_str() == Some(meeting_id))
        .map(|p| p["status"].as_str().unwrap_or("").to_string())
        .collect()
}

#[allow(dead_code)]
pub fn json_str(v: &str) -> Value {
    json!(v)
}
