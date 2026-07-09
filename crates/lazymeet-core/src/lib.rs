//! LazyMeet backend core: configuration, meeting/speaker storage, model
//! management, the job queue, and the audio processing pipeline.
//!
//! This crate is deliberately free of any Tauri dependency so the full
//! behavioral test suite runs with
//! `cargo test -p lazymeet-core --no-default-features` in seconds. The
//! desktop shell (src-tauri) wraps the `api` module in Tauri commands and
//! implements [`events::Events`] over its event system.

pub mod api;
pub mod config;
pub mod error;
pub mod events;
pub mod jobs;
pub mod models;
pub mod pipeline;
pub mod store;

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use config::{Config, Paths};

/// Shared application state. The Python backend used module globals; this is
/// the explicit equivalent, so tests construct isolated instances instead of
/// fighting process-global environment variables.
pub struct Ctx {
    pub paths: Paths,
    /// `LAZYMEET_FAKE=1`: deterministic stub pipeline, models always "ready".
    pub fake: bool,
    /// Current tunables snapshot (env > config.json > default). Re-resolved on
    /// settings writes; the worker takes its own snapshot per job.
    pub config: RwLock<Config>,
    pub events: Arc<dyn events::Events>,
    pub models: models::ModelsState,
    pub jobs: jobs::JobsState,
    pub locks: store::StoreLocks,
    /// Bundled ffmpeg binary (native pipeline only; None in FAKE mode/tests).
    pub ffmpeg: Option<PathBuf>,
}

impl Ctx {
    pub fn new(paths: Paths, fake: bool, events: Arc<dyn events::Events>) -> Arc<Self> {
        let config = RwLock::new(Config::resolve(&paths));
        Arc::new(Self {
            paths,
            fake,
            config,
            events,
            models: models::ModelsState::default(),
            jobs: jobs::JobsState::default(),
            locks: store::StoreLocks::default(),
            ffmpeg: None,
        })
    }

    /// Environment-driven construction (the desktop shell and CLI).
    pub fn from_env(events: Arc<dyn events::Events>) -> Arc<Self> {
        Self::new(Paths::from_env(), config::fake_mode_from_env(), events)
    }

    /// Re-resolve the tunables snapshot (after a settings write).
    pub fn refresh_config(&self) {
        let fresh = Config::resolve(&self.paths);
        *self.config.write().unwrap() = fresh;
    }
}

/// Everything the app does on launch (was FastAPI's lifespan hook): ensure
/// dirs, kick off first-run model downloads, sweep orphaned meetings.
pub fn startup(ctx: &Arc<Ctx>) -> std::io::Result<()> {
    ctx.paths.ensure_dirs()?;
    sweep_stale_files(ctx);
    models::ensure_models_async(ctx);
    jobs::sweep_orphans(ctx);
    Ok(())
}

/// Remove leftovers the old Python backend could leave behind on a crash:
/// cross-process `.lock` files (locking is in-process now) and orphaned
/// atomic-write `.tmp` staging files.
fn sweep_stale_files(ctx: &Ctx) {
    for dir in [ctx.paths.meetings_dir(), ctx.paths.data_dir.clone()] {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".lock") || name.ends_with(".tmp") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}
