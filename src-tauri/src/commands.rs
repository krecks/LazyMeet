//! Tauri commands: thin async wrappers around `lazymeet_core::api`.
//!
//! Rules of the layer: every command is `async` (sync commands run on the main
//! thread and would freeze the UI), filesystem-heavy work goes through
//! `spawn_blocking`, and all validation/error semantics live in core — the
//! serialized `ApiError { code, kind, message }` is what the frontend's
//! `err.status` checks read.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{Map, Value};
use tauri::{AppHandle, State};
use tauri_plugin_dialog::DialogExt;

use lazymeet_core::error::{ApiError, ApiResult};
use lazymeet_core::{api, Ctx};

type App = Arc<Ctx>;

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> ApiResult<T> + Send + 'static,
) -> ApiResult<T> {
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| ApiError::internal(format!("task failed: {e}")))?
}

// --- Setup / settings ------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn get_setup(ctx: State<'_, App>) -> ApiResult<Value> {
    Ok(api::get_setup(&ctx))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn retry_setup(ctx: State<'_, App>) -> ApiResult<Value> {
    Ok(api::retry_setup(&ctx))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_config(ctx: State<'_, App>) -> ApiResult<Value> {
    Ok(api::config_payload(&ctx))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn put_config(ctx: State<'_, App>, fields: Map<String, Value>) -> ApiResult<Value> {
    api::put_config(&ctx, &fields)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_models(ctx: State<'_, App>) -> ApiResult<Value> {
    Ok(api::get_models(&ctx))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn download_model(ctx: State<'_, App>, step: String, id: String) -> ApiResult<Value> {
    api::download_model(&ctx, &step, &id)
}

// --- Meetings ---------------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn create_meeting(
    ctx: State<'_, App>,
    path: String,
    num_speakers: i64,
) -> ApiResult<Value> {
    let ctx = ctx.inner().clone();
    blocking(move || api::create_meeting(&ctx, &PathBuf::from(path), num_speakers)).await
}

/// Native replacement for the `<input type="file">` picker (webview File
/// objects have no filesystem path).
#[tauri::command(rename_all = "snake_case")]
pub async fn pick_recording(app: AppHandle) -> ApiResult<Option<String>> {
    blocking(move || {
        let picked = app
            .dialog()
            .file()
            .add_filter(
                "Recordings",
                &[
                    "mp4", "mkv", "mov", "avi", "webm", "flv", "ts", "m4v", "wav", "mp3", "m4a",
                    "aac", "flac", "ogg", "oga", "opus", "wma", "aiff", "aif",
                ],
            )
            .blocking_pick_file();
        Ok(picked.and_then(|p| p.into_path().ok()).map(|p| p.to_string_lossy().into_owned()))
    })
    .await
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_meetings(ctx: State<'_, App>) -> ApiResult<Value> {
    Ok(api::list_meetings(&ctx))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_meeting(ctx: State<'_, App>, meeting_id: String) -> ApiResult<Value> {
    api::get_meeting(&ctx, &meeting_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_meeting_audio_path(ctx: State<'_, App>, meeting_id: String) -> ApiResult<Value> {
    api::get_meeting_audio_path(&ctx, &meeting_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn patch_meeting(
    ctx: State<'_, App>,
    meeting_id: String,
    fields: Map<String, Value>,
) -> ApiResult<Value> {
    api::patch_meeting(&ctx, &meeting_id, &fields)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn patch_segment(
    ctx: State<'_, App>,
    meeting_id: String,
    index: i64,
    fields: Map<String, Value>,
) -> ApiResult<Value> {
    api::patch_segment(&ctx, &meeting_id, index, &fields)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn rename_speakers(
    ctx: State<'_, App>,
    meeting_id: String,
    names: Map<String, Value>,
) -> ApiResult<Value> {
    api::rename_speakers(&ctx, &meeting_id, &names)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn verify_speaker(
    ctx: State<'_, App>,
    meeting_id: String,
    diarized_id: String,
    correct: bool,
) -> ApiResult<Value> {
    api::verify_speaker(&ctx, &meeting_id, &diarized_id, correct)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn delete_meeting(ctx: State<'_, App>, meeting_id: String) -> ApiResult<Value> {
    api::delete_meeting(&ctx, &meeting_id)
}

/// Export = build the markdown in core, then a native save dialog. Returns the
/// written path, or null when the user cancelled.
#[tauri::command(rename_all = "snake_case")]
pub async fn export_meeting(
    app: AppHandle,
    ctx: State<'_, App>,
    meeting_id: String,
) -> ApiResult<Option<String>> {
    let (markdown, filename) = api::export_meeting(&ctx, &meeting_id)?;
    blocking(move || {
        let Some(target) = app
            .dialog()
            .file()
            .set_file_name(&filename)
            .add_filter("Markdown", &["md"])
            .blocking_save_file()
        else {
            return Ok(None); // cancelled
        };
        let path = target
            .into_path()
            .map_err(|e| ApiError::internal(format!("invalid save location: {e}")))?;
        std::fs::write(&path, markdown)
            .map_err(|e| ApiError::internal(format!("could not write the export: {e}")))?;
        Ok(Some(path.to_string_lossy().into_owned()))
    })
    .await
}

// --- Speaker library ---------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn get_speakers(ctx: State<'_, App>) -> ApiResult<Value> {
    Ok(api::get_speakers(&ctx))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn rename_known_speaker(
    ctx: State<'_, App>,
    name: String,
    new_name: String,
) -> ApiResult<Value> {
    api::rename_known_speaker(&ctx, &name, &new_name)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn delete_speaker(ctx: State<'_, App>, name: String) -> ApiResult<Value> {
    api::delete_speaker(&ctx, &name)
}

// --- Data management ----------------------------------------------------------

#[tauri::command(rename_all = "snake_case")]
pub async fn get_data_info(ctx: State<'_, App>) -> ApiResult<Value> {
    let ctx = ctx.inner().clone();
    blocking(move || Ok(api::get_data_info(&ctx))).await
}

/// Open the app-data folder in the OS file manager. The path is app-fixed —
/// never taken from the request.
#[tauri::command(rename_all = "snake_case")]
pub async fn reveal_data_dir(ctx: State<'_, App>) -> ApiResult<Value> {
    let target = ctx.paths.data_dir.clone();
    blocking(move || {
        #[cfg(target_os = "macos")]
        let result = std::process::Command::new("open").arg(&target).spawn();
        #[cfg(target_os = "windows")]
        let result = std::process::Command::new("explorer").arg(&target).spawn();
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let result = std::process::Command::new("xdg-open").arg(&target).spawn();
        result.map_err(|e| ApiError::internal(format!("could not open folder: {e}")))?;
        Ok(serde_json::json!({"ok": true}))
    })
    .await
}
