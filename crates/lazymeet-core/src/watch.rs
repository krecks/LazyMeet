//! Watch folder: new recordings that appear in `WATCH_DIR` (OBS's output
//! folder, say) are imported on their own — whole file, default speakers and
//! language, no prepare screen.
//!
//! Polling rather than file-system events: a scan every few seconds is the
//! same on every OS, survives network drives and sleep, and is easy to test.
//! A file is imported once it has stopped growing; files that were already
//! there when watching started are left alone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{api, Ctx};

/// Extensions the app takes as recordings (the file picker's filter too).
pub const MEDIA_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "mov", "avi", "webm", "flv", "ts", "m4v", "wav", "mp3", "m4a", "aac", "flac",
    "ogg", "oga", "opus", "wma", "aiff", "aif",
];

/// How often the folder is scanned.
pub const POLL: Duration = Duration::from_secs(5);
/// A file must keep its size and modification time this long before it's
/// taken as finished (OBS writes a recording for the whole meeting).
pub const STABLE_FOR: Duration = Duration::from_secs(10);
/// A file with the same name but another media extension imported within
/// this window is the same recording (OBS remuxes .mkv to .mp4 on stop).
pub const SAME_RECORDING_WINDOW: Duration = Duration::from_secs(600);

pub fn is_media(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| MEDIA_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

fn secs(t: SystemTime) -> f64 {
    t.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// What the watcher remembers between scans and across restarts
/// (`watch_state.json` in the data folder).
#[derive(Default, Serialize, Deserialize, Debug)]
pub struct WatchState {
    /// The folder this state belongs to; another folder starts afresh.
    pub dir: String,
    /// Files modified before this (unix seconds) were there already.
    pub since: f64,
    /// Imported or skipped files: path → (size, mtime). A file that changes
    /// after that counts as new again.
    pub seen: HashMap<String, (u64, f64)>,
    /// Recently imported recordings: (file stem, when) — for the remux rule.
    pub recent: Vec<(String, f64)>,
    /// Files still being written: path → (size, mtime, unchanged since).
    #[serde(skip)]
    pub pending: HashMap<String, (u64, f64, f64)>,
}

impl WatchState {
    /// Start watching `dir` from `now`: what's in it already is left alone.
    pub fn fresh(dir: &Path, now: SystemTime) -> Self {
        WatchState {
            dir: dir.to_string_lossy().into_owned(),
            since: secs(now),
            ..Default::default()
        }
    }

    fn load(path: &Path) -> Option<Self> {
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }

    fn save(&self, path: &Path) {
        if let Ok(bytes) = serde_json::to_vec(self) {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
    }
}

/// One pass over `dir`: the recordings that are ready to import now. They
/// are marked seen here, so a failed import is not retried in a loop.
pub fn scan(dir: &Path, state: &mut WatchState, now: SystemTime) -> Vec<PathBuf> {
    let now_s = secs(now);
    let mut ready = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return ready };
    let mut present = std::collections::HashSet::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !is_media(&path) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let key = path.to_string_lossy().into_owned();
        present.insert(key.clone());
        let size = meta.len();
        let mtime = meta.modified().map(secs).unwrap_or(0.0);
        if state.seen.get(&key) == Some(&(size, mtime)) {
            continue;
        }
        if mtime < state.since {
            // There before watching started (or before the app saw it
            // change): never import, and don't look at it again.
            state.seen.insert(key, (size, mtime));
            continue;
        }
        // Still being written? Wait until size and mtime hold still.
        let unchanged_since = match state.pending.get(&key) {
            Some(&(s, m, t)) if s == size && m == mtime => t,
            _ => {
                state.pending.insert(key, (size, mtime, now_s));
                continue;
            }
        };
        if now_s - unchanged_since < STABLE_FOR.as_secs_f64() || size == 0 {
            continue;
        }
        // Windows: OBS keeps the file locked while it records.
        if std::fs::File::open(&path).is_err() {
            continue;
        }
        state.pending.remove(&key);
        state.seen.insert(key, (size, mtime));
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let window = SAME_RECORDING_WINDOW.as_secs_f64();
        if state.recent.iter().any(|(s, at)| *s == stem && now_s - at < window) {
            continue; // the remuxed twin of a recording already imported
        }
        state.recent.push((stem, now_s));
        ready.push(path);
    }
    // Forget what's gone, so the state file can't grow forever.
    state.pending.retain(|k, _| present.contains(k));
    state.seen.retain(|k, _| present.contains(k));
    state.recent.retain(|(_, at)| now_s - at < SAME_RECORDING_WINDOW.as_secs_f64());
    ready.sort();
    ready
}

/// The folder to watch, if the setting names an existing directory.
pub fn watched_dir(ctx: &Ctx) -> Option<PathBuf> {
    let dir = ctx.config.read().unwrap().watch_dir();
    let dir = PathBuf::from(dir.trim());
    (!dir.as_os_str().is_empty() && dir.is_dir()).then_some(dir)
}

/// The watcher loop (its own thread, for the app's lifetime).
pub fn start(ctx: &Arc<Ctx>) {
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let state_path = ctx.paths.data_dir.join("watch_state.json");
        let mut state: Option<WatchState> = None;
        loop {
            std::thread::sleep(POLL);
            let Some(dir) = watched_dir(&ctx) else {
                state = None;
                continue;
            };
            let dir_key = dir.to_string_lossy().into_owned();
            let st = state.get_or_insert_with(|| {
                WatchState::load(&state_path)
                    .filter(|s| s.dir == dir_key)
                    .unwrap_or_else(|| WatchState::fresh(&dir, SystemTime::now()))
            });
            if st.dir != dir_key {
                *st = WatchState::fresh(&dir, SystemTime::now());
            }
            let ready = scan(&dir, st, SystemTime::now());
            st.save(&state_path);
            import(&ctx, &ready);
        }
    });
}

/// Hand finished recordings to the pipeline and tell the UI.
pub fn import(ctx: &Arc<Ctx>, paths: &[PathBuf]) -> Vec<String> {
    let mut imported = Vec::new();
    for path in paths {
        match api::create_meeting(ctx, path, -1, None, None) {
            Ok(meeting) => {
                let filename =
                    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                ctx.events.emit(
                    "meetings-changed",
                    json!({"imported": {"id": meeting["id"], "filename": filename}}),
                );
                if let Some(id) = meeting["id"].as_str() {
                    imported.push(id.to_string());
                }
            }
            Err(e) => {
                eprintln!("watch folder: could not import {}: {}", path.display(), e.message);
                ctx.events.emit(
                    "meetings-changed",
                    json!({"error": format!("Couldn't import {}: {}", path.display(), e.message)}),
                );
            }
        }
    }
    imported
}
