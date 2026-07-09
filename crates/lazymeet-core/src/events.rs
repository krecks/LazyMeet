//! Event sink abstraction. The desktop shell implements this over Tauri's
//! `AppHandle::emit`; tests implement a collector so event sequences are
//! assertable. Replaces the old backend's poll-only progress reporting.

use std::sync::Mutex;

pub trait Events: Send + Sync {
    fn emit(&self, event: &str, payload: serde_json::Value);
}

/// Discards all events (CLI runs, tests that don't care).
pub struct NoopEvents;

impl Events for NoopEvents {
    fn emit(&self, _event: &str, _payload: serde_json::Value) {}
}

/// Records every emitted event; used by the test suite.
#[derive(Default)]
pub struct CollectorEvents {
    pub events: Mutex<Vec<(String, serde_json::Value)>>,
}

impl Events for CollectorEvents {
    fn emit(&self, event: &str, payload: serde_json::Value) {
        self.events.lock().unwrap().push((event.to_string(), payload));
    }
}

impl CollectorEvents {
    /// Names of all recorded events, in emission order.
    pub fn names(&self) -> Vec<String> {
        self.events.lock().unwrap().iter().map(|(n, _)| n.clone()).collect()
    }

    /// Payloads of every recorded event with the given name.
    pub fn payloads(&self, event: &str) -> Vec<serde_json::Value> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(n, _)| n == event)
            .map(|(_, p)| p.clone())
            .collect()
    }
}
