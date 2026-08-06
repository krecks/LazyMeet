// Prevents an extra console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use std::path::PathBuf;
use std::sync::Arc;

use tauri::{AppHandle, Emitter, Manager};

use lazymeet_core::events::Events;
use lazymeet_core::Ctx;

/// Core events go straight to the webview: `meeting-updated`,
/// `setup-progress`, `models-progress` (replaces the old HTTP polling).
struct TauriEvents(AppHandle);

impl Events for TauriEvents {
    fn emit(&self, event: &str, payload: serde_json::Value) {
        let _ = self.0.emit(event, payload);
    }
}

/// The bundled ffmpeg (Tauri externalBin) sits next to the app executable —
/// in dev builds Tauri copies it beside the debug binary too.
fn find_ffmpeg() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("LAZYMEET_FFMPEG") {
        return Some(PathBuf::from(p));
    }
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let bundled = exe_dir.join(format!("ffmpeg{}", std::env::consts::EXE_SUFFIX));
    if bundled.is_file() {
        return Some(bundled);
    }
    // Plain `cargo run` (no tauri-cli): use the fetched binary directly.
    let triple = current_triple();
    let dev = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("binaries")
        .join(format!("ffmpeg-{triple}{}", std::env::consts::EXE_SUFFIX));
    dev.is_file().then_some(dev)
}

fn current_triple() -> String {
    format!(
        "{}-{}",
        std::env::consts::ARCH,
        match std::env::consts::OS {
            "macos" => "apple-darwin",
            "windows" => "pc-windows-msvc",
            _ => "unknown-linux-gnu",
        }
    )
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::get_setup,
            commands::retry_setup,
            commands::get_config,
            commands::put_config,
            commands::get_models,
            commands::download_model,
            commands::create_meeting,
            commands::probe_media,
            commands::prepare_media_preview,
            commands::extract_preview_audio,
            commands::pick_recording,
            commands::list_meetings,
            commands::search_meetings,
            commands::get_meeting,
            commands::get_meeting_audio_path,
            commands::patch_meeting,
            commands::patch_segment,
            commands::patch_segments,
            commands::rename_speakers,
            commands::verify_speaker,
            commands::delete_meeting,
            commands::reprocess_meeting,
            commands::export_meeting,
            commands::get_speakers,
            commands::rename_known_speaker,
            commands::delete_speaker,
            commands::get_data_info,
            commands::reveal_data_dir,
            commands::run_benchmark,
        ])
        .setup(|app| {
            let events = Arc::new(TauriEvents(app.handle().clone()));
            let mut ctx = Ctx::from_env(events);
            Arc::get_mut(&mut ctx).expect("fresh ctx").ffmpeg = find_ffmpeg();

            // Segment playback streams the kept audio through the asset
            // protocol (Range/seek support); scope it to the uploads dir.
            ctx.paths.ensure_dirs()?;
            app.asset_protocol_scope().allow_directory(ctx.paths.uploads_dir(), true)?;

            // Dirs, first-run model downloads, orphaned-meeting sweep.
            lazymeet_core::startup(&ctx)?;
            app.manage(ctx);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running LazyMeet");
}
