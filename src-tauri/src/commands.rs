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
use tauri::{AppHandle, Manager, State};
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

#[tauri::command(rename_all = "snake_case")]
pub async fn remove_model(ctx: State<'_, App>, step: String, id: String) -> ApiResult<Value> {
    // May wait a few seconds for a background title pass to end.
    let ctx = ctx.inner().clone();
    blocking(move || api::remove_model(&ctx, &step, &id)).await
}

// --- Meetings ---------------------------------------------------------------

/// `ranges` (`[[start, end], ...]` in seconds, or null/omitted for the whole
/// file) restricts transcription to the user-selected spans. `language`
/// ("auto" or a code) is a per-meeting spoken-language hint that outranks the
/// ASR_LANGUAGE setting; null/omitted keeps the setting.
#[tauri::command(rename_all = "snake_case")]
pub async fn create_meeting(
    ctx: State<'_, App>,
    path: String,
    num_speakers: i64,
    ranges: Option<Vec<(f64, f64)>>,
    language: Option<String>,
    notes_template: Option<String>,
    notes_language: Option<String>,
) -> ApiResult<Value> {
    let ctx = ctx.inner().clone();
    let notes = api::NotesOptions { template: notes_template, language: notes_language };
    blocking(move || {
        api::create_meeting_with(
            &ctx,
            &PathBuf::from(path),
            num_speakers,
            ranges.as_deref(),
            language.as_deref(),
            &notes,
        )
    })
    .await
}

/// Duration + waveform peaks for the prep screen. Decodes the file with the
/// bundled ffmpeg — takes a few seconds on long recordings.
#[tauri::command(rename_all = "snake_case")]
pub async fn probe_media(ctx: State<'_, App>, path: String) -> ApiResult<Value> {
    let ctx = ctx.inner().clone();
    blocking(move || api::probe_media(&ctx, &PathBuf::from(path))).await
}

/// Audio-only AAC preview for formats the webview can't play (mkv, avi, …).
/// Decodes + encodes with the bundled ffmpeg — seconds to ~a minute on long
/// recordings; the frontend shows a "preparing" note meanwhile.
#[tauri::command(rename_all = "snake_case")]
pub async fn extract_preview_audio(ctx: State<'_, App>, path: String) -> ApiResult<Value> {
    let ctx = ctx.inner().clone();
    blocking(move || api::extract_preview_audio(&ctx, &PathBuf::from(path))).await
}

/// Allow the picked file through the asset protocol so the prep screen's
/// <video> can stream it before it is copied into the uploads dir (whose
/// directory scope is set at startup).
#[tauri::command(rename_all = "snake_case")]
pub async fn prepare_media_preview(app: AppHandle, path: String) -> ApiResult<Value> {
    let p = PathBuf::from(&path);
    if !p.is_file() {
        return Err(ApiError::invalid("file not found"));
    }
    app.asset_protocol_scope()
        .allow_file(&p)
        .map_err(|e| ApiError::internal(format!("could not enable preview: {e}")))?;
    Ok(serde_json::json!({"ok": true}))
}

/// Native replacement for the `<input type="file">` picker (webview File
/// objects have no filesystem path). Several files can be picked at once;
/// returns an empty list when cancelled.
#[tauri::command(rename_all = "snake_case")]
pub async fn pick_recording(app: AppHandle) -> ApiResult<Vec<String>> {
    blocking(move || {
        let picked = app
            .dialog()
            .file()
            .add_filter("Recordings", lazymeet_core::watch::MEDIA_EXTENSIONS)
            .blocking_pick_files()
            .unwrap_or_default();
        Ok(picked
            .into_iter()
            .filter_map(|p| p.into_path().ok())
            .map(|p| p.to_string_lossy().into_owned())
            .collect())
    })
    .await
}

/// Pick a folder (the watch folder setting). None when cancelled.
#[tauri::command(rename_all = "snake_case")]
pub async fn pick_folder(app: AppHandle) -> ApiResult<Option<String>> {
    blocking(move || {
        let picked = app.dialog().file().blocking_pick_folder();
        Ok(picked.and_then(|p| p.into_path().ok()).map(|p| p.to_string_lossy().into_owned()))
    })
    .await
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_meetings(ctx: State<'_, App>) -> ApiResult<Value> {
    Ok(api::list_meetings(&ctx))
}

/// Unlike `list_meetings` this reads every byte of every meeting file, so it
/// goes to the blocking pool rather than the async runtime.
#[tauri::command(rename_all = "snake_case")]
pub async fn search_meetings(ctx: State<'_, App>, query: String) -> ApiResult<Value> {
    let ctx = Arc::clone(&ctx);
    blocking(move || Ok(api::search_meetings(&ctx, &query))).await
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
pub async fn patch_segments(
    ctx: State<'_, App>,
    meeting_id: String,
    edits: Vec<(usize, String)>,
) -> ApiResult<Value> {
    api::patch_segments(&ctx, &meeting_id, &edits)
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

/// Regenerate with the current settings: mode "notes" re-summarizes the
/// existing transcript, mode "transcription" re-runs the whole pipeline over
/// the kept audio. Progress streams like the original processing.
#[tauri::command(rename_all = "snake_case")]
pub async fn reprocess_meeting(
    ctx: State<'_, App>,
    meeting_id: String,
    mode: String,
    notes_template: Option<String>,
    notes_language: Option<String>,
) -> ApiResult<Value> {
    let notes = api::NotesOptions { template: notes_template, language: notes_language };
    api::reprocess_meeting_with(&ctx, &meeting_id, &mode, &notes)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn delete_meetings(ctx: State<'_, App>, meeting_ids: Vec<String>) -> ApiResult<Value> {
    api::delete_meetings(&ctx, &meeting_ids)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn tag_meetings(
    ctx: State<'_, App>,
    meeting_ids: Vec<String>,
    add: Vec<String>,
    remove: Vec<String>,
) -> ApiResult<Value> {
    api::tag_meetings(&ctx, &meeting_ids, &add, &remove)
}

/// Export = build the file in core (`format`: md, txt, srt, vtt, json, docx,
/// pdf), then a native save dialog. Returns the written path, or null when
/// the user cancelled.
#[tauri::command(rename_all = "snake_case")]
pub async fn export_meeting(
    app: AppHandle,
    ctx: State<'_, App>,
    meeting_id: String,
    format: Option<String>,
) -> ApiResult<Option<String>> {
    let format = format.unwrap_or_else(|| "md".into());
    let (bytes, filename) = api::export_meeting(&ctx, &meeting_id, &format)?;
    let kind = lazymeet_core::export::Format::parse(&format).expect("validated by export_meeting");
    blocking(move || {
        let Some(target) = app
            .dialog()
            .file()
            .set_file_name(&filename)
            .add_filter(kind.label(), &[kind.ext()])
            .blocking_save_file()
        else {
            return Ok(None); // cancelled
        };
        let path = target
            .into_path()
            .map_err(|e| ApiError::internal(format!("invalid save location: {e}")))?;
        std::fs::write(&path, bytes)
            .map_err(|e| ApiError::internal(format!("could not write the export: {e}")))?;
        Ok(Some(path.to_string_lossy().into_owned()))
    })
    .await
}

/// Export several meetings: pick a folder, then one file per meeting in it.
/// Returns the written paths, or null when the user cancelled.
#[tauri::command(rename_all = "snake_case")]
pub async fn export_meetings(
    app: AppHandle,
    ctx: State<'_, App>,
    meeting_ids: Vec<String>,
    format: String,
) -> ApiResult<Option<Vec<String>>> {
    if lazymeet_core::export::Format::parse(&format).is_none() {
        return Err(ApiError::invalid(format!("unknown export format: {format}")));
    }
    let ctx = ctx.inner().clone();
    blocking(move || {
        let Some(dir) = app.dialog().file().set_title("Export to folder").blocking_pick_folder()
        else {
            return Ok(None);
        };
        let dir =
            dir.into_path().map_err(|e| ApiError::internal(format!("invalid folder: {e}")))?;
        api::export_meetings(&ctx, &meeting_ids, &format, &dir).map(Some)
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

// --- Hardware benchmark --------------------------------------------------------

/// Start the built-in backend benchmark; progress streams as
/// `benchmark-progress` events. 409s while models download, a meeting is
/// processing, or a benchmark is already running.
#[tauri::command(rename_all = "snake_case")]
pub async fn run_benchmark(ctx: State<'_, App>, path: Option<String>) -> ApiResult<Value> {
    // May wait a few seconds for a background title pass to end.
    let ctx = ctx.inner().clone();
    blocking(move || lazymeet_core::bench::start(&ctx, path.map(PathBuf::from))).await
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
