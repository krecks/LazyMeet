//! Endpoint-level operations: the exact request validation, error codes, and
//! response payloads of the old FastAPI routes (`backend/main.py`), minus the
//! HTTP. The desktop shell's Tauri commands are thin wrappers around these;
//! the test suite calls them directly.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::config;
use crate::error::{ApiError, ApiResult};
use crate::pipeline::{self, LANGUAGE_NAMES};
use crate::store::{self, StoreError};
use crate::{jobs, models, Ctx};

// --------------------------------------------------------------------------
// Setup / health
// --------------------------------------------------------------------------

pub fn get_setup(ctx: &Ctx) -> Value {
    models::status(ctx)
}

/// Restart failed model downloads (no-op if a download is already running).
pub fn retry_setup(ctx: &Arc<Ctx>) -> Value {
    models::ensure_models_async(ctx);
    models::status(ctx)
}

// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------

/// "auto" + everything the summarizer knows a name for; the same codes are
/// valid Whisper language hints.
pub fn supported_languages() -> Vec<&'static str> {
    let mut v = vec!["auto"];
    v.extend(LANGUAGE_NAMES.iter().map(|(code, _)| *code));
    v
}

pub fn config_payload(ctx: &Ctx) -> Value {
    let snapshot = ctx.config.read().unwrap();
    let overrides = config::load_overrides(&ctx.paths);
    let mut defaults = Map::new();
    let mut limits = Map::new();
    for spec in config::tunables() {
        defaults.insert(spec.name.to_string(), config::default_value(spec.name));
        limits.insert(spec.name.to_string(), json!({"min": spec.min, "max": spec.max}));
    }
    json!({
        "values": snapshot.values(),
        "overrides": overrides,
        "defaults": defaults,
        // Set by env var, which outranks config.json — the UI locks these.
        "env_locked": config::env_locked(),
        "limits": limits,
        // Backends available in this build on this platform — the UI builds
        // the compute selects from these (CPU everywhere; NPU for the
        // recognition models on Apple Silicon; Metal GPU for the summarizer).
        "compute_options": {
            "ASR_PROVIDER": config::onnx_provider_options(),
            "DIAR_PROVIDER": config::onnx_provider_options(),
            "SPEAKER_PROVIDER": config::onnx_provider_options(),
            "SUMMARY_PROVIDER": config::summary_provider_options(),
        },
    })
}

/// PUT /api/config: apply tunable changes. All fields optional; an explicit
/// null removes that override. Unknown keys are ignored (pydantic did too).
pub fn put_config(ctx: &Arc<Ctx>, fields: &Map<String, Value>) -> ApiResult<Value> {
    let specs = config::tunables();

    if let Some(v) = fields.get("ASR_MODEL").filter(|v| !v.is_null()) {
        let id = v.as_str().unwrap_or_default();
        if !models::asr_catalog().iter().any(|e| e.id == id) {
            return Err(ApiError::invalid("unknown transcription model"));
        }
    }
    if let Some(v) = fields.get("SUMMARY_MODEL").filter(|v| !v.is_null()) {
        let id = v.as_str().unwrap_or_default();
        if !models::llm_catalog().iter().any(|e| e.id == id) {
            return Err(ApiError::invalid("unknown summary model"));
        }
    }
    for key in ["ASR_LANGUAGE", "SUMMARY_LANGUAGE"] {
        if let Some(v) = fields.get(key).filter(|v| !v.is_null()) {
            let lang = v.as_str().unwrap_or_default();
            if !supported_languages().contains(&lang) {
                return Err(ApiError::invalid(format!("unsupported language for {key}")));
            }
        }
    }
    for key in ["ASR_PROVIDER", "DIAR_PROVIDER", "SPEAKER_PROVIDER"] {
        if let Some(v) = fields.get(key).filter(|v| !v.is_null()) {
            let provider = v.as_str().unwrap_or_default();
            if !config::onnx_provider_options().contains(&provider) {
                return Err(ApiError::invalid(format!("unsupported provider for {key}")));
            }
        }
    }
    if let Some(v) = fields.get("SUMMARY_PROVIDER").filter(|v| !v.is_null()) {
        let provider = v.as_str().unwrap_or_default();
        if !config::summary_provider_options().contains(&provider) {
            return Err(ApiError::invalid("unsupported provider for SUMMARY_PROVIDER"));
        }
    }

    let mut overrides = config::load_overrides(&ctx.paths);
    for (name, value) in fields {
        let Some(spec) = specs.iter().find(|s| s.name == name) else {
            continue; // unknown keys are ignored
        };
        if value.is_null() {
            overrides.remove(name);
        } else {
            // Type/bounds validation (was pydantic Field(ge=..., le=...)).
            // Booleans are strict here (pydantic was): "sometimes" is a 422,
            // while config.json loading stays lenient for compatibility.
            if spec.kind == config::Kind::Bool && !value.is_boolean() {
                return Err(ApiError::invalid(format!("invalid value for {name}")));
            }
            let coerced = config::coerce(spec, value)
                .ok_or_else(|| ApiError::invalid(format!("invalid value for {name}")))?;
            overrides.insert(name.clone(), coerced);
        }
    }
    config::save_overrides(&ctx.paths, &overrides)?;
    ctx.refresh_config();
    // A newly selected model that isn't on disk yet starts downloading now;
    // the pipeline uses the default until it lands.
    models::ensure_selected_async(ctx);
    Ok(config_payload(ctx))
}

pub fn get_models(ctx: &Ctx) -> Value {
    models::models_status(ctx)
}

pub fn download_model(ctx: &Arc<Ctx>, step: &str, id: &str) -> ApiResult<Value> {
    if !models::request_download(ctx, step, id) {
        return Err(ApiError::not_found("unknown model"));
    }
    Ok(json!({"ok": true}))
}

// --------------------------------------------------------------------------
// Meetings
// --------------------------------------------------------------------------

/// POST /api/meetings — native replacement for the multipart upload: the
/// recording is referenced by filesystem path (drag-drop / file dialog give
/// real paths) and copied into the uploads dir. `ranges` restricts
/// transcription to the selected spans (seconds, original-media time);
/// None = the whole recording.
pub fn create_meeting(
    ctx: &Arc<Ctx>,
    src: &Path,
    num_speakers: i64,
    ranges: Option<&[pipeline::TimeRange]>,
    language: Option<&str>,
) -> ApiResult<Value> {
    if !src.is_file() {
        return Err(ApiError::invalid("file not found"));
    }
    let ranges = match ranges {
        Some(raw) => Some(pipeline::normalize_ranges(raw).map_err(ApiError::invalid)?),
        None => None,
    };
    if let Some(lang) = language {
        if !supported_languages().contains(&lang) {
            return Err(ApiError::invalid("unsupported language"));
        }
    }
    let filename = src
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .unwrap_or("recording")
        .to_string();
    let record = store::create_meeting(
        &ctx.paths,
        &ctx.locks,
        &filename,
        num_speakers,
        ranges.as_deref(),
        language,
    )?;
    let id = record["id"].as_str().unwrap().to_string();
    let upload_dir = ctx.paths.uploads_dir().join(&id);
    std::fs::create_dir_all(&upload_dir)?;
    let dest = upload_dir.join(&filename);
    std::fs::copy(src, &dest).map_err(|e| {
        ApiError::internal(format!("could not copy the recording: {e}"))
    })?;
    jobs::dispatch(ctx, id.clone(), jobs::Job::Process { upload: dest });
    Ok(json!({"id": id}))
}

/// Extract an audio-only AAC preview of a recording for the prep screen —
/// the fallback when the webview can't play the original container (mkv,
/// avi, wma, …). Written under uploads/.preview/ (already asset-protocol
/// scoped), one file at a time, overwritten per call, swept on startup.
pub fn extract_preview_audio(ctx: &Ctx, src: &Path) -> ApiResult<Value> {
    if !src.is_file() {
        return Err(ApiError::invalid("file not found"));
    }
    let config = config::Config::resolve(&ctx.paths);
    let pctx = pipeline::PipelineCtx {
        fake: ctx.fake,
        config: &config,
        models_dir: &ctx.paths.models_dir,
        ffmpeg: ctx.ffmpeg.clone(),
    };
    let out = ctx.paths.uploads_dir().join(config::PREVIEW_DIR).join("preview.m4a");
    pipeline::extract_playback_audio(&pctx, src, &out).map_err(ApiError::invalid)?;
    Ok(json!({"path": out.to_string_lossy()}))
}

/// Duration + waveform peaks for the pre-transcription prep screen. Decodes
/// the file once with the bundled ffmpeg — call it from a blocking context.
pub fn probe_media(ctx: &Ctx, src: &Path) -> ApiResult<Value> {
    if !src.is_file() {
        return Err(ApiError::invalid("file not found"));
    }
    let config = config::Config::resolve(&ctx.paths);
    let pctx = pipeline::PipelineCtx {
        fake: ctx.fake,
        config: &config,
        models_dir: &ctx.paths.models_dir,
        ffmpeg: ctx.ffmpeg.clone(),
    };
    let probe = pipeline::probe_media(&pctx, src).map_err(ApiError::invalid)?;
    Ok(serde_json::to_value(probe).unwrap_or_else(|_| json!({})))
}

/// POST /api/meetings/{id}/reprocess — regenerate content with the *current*
/// settings. mode "notes": summarize the existing transcript again; mode
/// "transcription": run the whole pipeline again over the kept audio.
pub fn reprocess_meeting(ctx: &Arc<Ctx>, meeting_id: &str, mode: &str) -> ApiResult<Value> {
    let record = store::get_meeting(&ctx.paths, &ctx.locks, meeting_id)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    let status = record.get("status").and_then(|v| v.as_str()).unwrap_or("");
    if status != "done" && status != "error" {
        return Err(ApiError::conflict("the meeting is still processing"));
    }
    let job = match mode {
        "notes" => {
            let has_transcript = record
                .get("segments")
                .and_then(|v| v.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            if !has_transcript {
                return Err(ApiError::conflict("there is no transcript to summarize yet"));
            }
            jobs::Job::Resummarize
        }
        "transcription" => {
            if store::audio_path(&ctx.paths, &record).is_none() {
                return Err(ApiError::not_found("the recording is no longer on disk"));
            }
            jobs::Job::Retranscribe
        }
        _ => return Err(ApiError::invalid("mode must be \"notes\" or \"transcription\"")),
    };
    let record = jobs::queue_rejob(ctx, meeting_id, mode, job)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &record)))
}

pub fn list_meetings(ctx: &Ctx) -> Value {
    Value::Array(store::list_meetings(&ctx.paths, &ctx.locks).into_iter().map(Value::Object).collect())
}

/// Literal, case-insensitive search over every meeting on disk. An empty query
/// is a legitimate empty result, not a validation error.
pub fn search_meetings(ctx: &Ctx, query: &str) -> Value {
    Value::Array(
        store::search_meetings(&ctx.paths, &ctx.locks, query).into_iter().map(Value::Object).collect(),
    )
}

pub fn get_meeting(ctx: &Ctx, meeting_id: &str) -> ApiResult<Value> {
    let record = store::get_meeting(&ctx.paths, &ctx.locks, meeting_id)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &record)))
}

/// GET /api/meetings/{id}/audio — returns the on-disk path + download name;
/// the shell serves it through the asset protocol (Range/seek included).
pub fn get_meeting_audio_path(ctx: &Ctx, meeting_id: &str) -> ApiResult<Value> {
    let record = store::get_meeting(&ctx.paths, &ctx.locks, meeting_id)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    let path = store::audio_path(&ctx.paths, &record)
        .ok_or_else(|| ApiError::not_found("recording no longer on disk"))?;
    let mut name = record
        .get("filename")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("recording")
        .to_string();
    if path.file_name().and_then(|n| n.to_str()) == Some(config::PLAYBACK_FILENAME) {
        // video uploads are reduced to their audio track after processing
        name = PathBuf::from(&name).with_extension("m4a").to_string_lossy().into_owned();
    }
    Ok(json!({"path": path, "filename": name}))
}

/// PATCH /api/meetings/{id}: title in any status; summary/takeaways/action
/// items only once "done".
pub fn patch_meeting(ctx: &Ctx, meeting_id: &str, fields: &Map<String, Value>) -> ApiResult<Value> {
    let mut changes = Map::new();
    if let Some(v) = fields.get("title") {
        let title = v.as_str().unwrap_or_default().trim().to_string();
        if title.is_empty() || title.chars().count() > 200 {
            return Err(ApiError::invalid("title must be 1-200 characters"));
        }
        changes.insert("title".into(), json!(title));
    }
    let mut content_edit = false;
    if let Some(v) = fields.get("summary") {
        changes.insert("summary".into(), json!(v.as_str().unwrap_or_default().trim()));
        content_edit = true;
    }
    for key in ["takeaways", "action_items"] {
        if let Some(v) = fields.get(key) {
            let items: Vec<String> = v
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str())
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .take(100)
                        .collect()
                })
                .unwrap_or_default();
            changes.insert(key.into(), json!(items));
            content_edit = true;
        }
    }
    if changes.is_empty() {
        return Err(ApiError::invalid("no fields to update"));
    }
    let allowed: Option<&[&str]> = if content_edit { Some(&["done"]) } else { None };
    let updated = store::patch_meeting(&ctx.paths, &ctx.locks, meeting_id, allowed, changes)
        .map_err(status_conflict)?
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &updated)))
}

pub fn patch_segment(
    ctx: &Ctx,
    meeting_id: &str,
    index: i64,
    fields: &Map<String, Value>,
) -> ApiResult<Value> {
    let text = match fields.get("text") {
        Some(v) => {
            let t = v.as_str().unwrap_or_default().trim().to_string();
            if t.is_empty() {
                return Err(ApiError::invalid("segment text must not be empty"));
            }
            Some(t)
        }
        None => None,
    };
    let speaker = fields.get("speaker").map(|v| v.as_str().unwrap_or_default().to_string());
    if text.is_none() && speaker.is_none() {
        return Err(ApiError::invalid("no fields to update"));
    }
    if index < 0 {
        return Err(ApiError::not_found("segment not found"));
    }
    let updated = store::update_segment(
        &ctx.paths,
        &ctx.locks,
        meeting_id,
        index as usize,
        text.as_deref(),
        speaker.as_deref(),
    )
    .map_err(|e| match e {
        StoreError::UnknownSpeaker => ApiError::invalid("unknown speaker id"),
        StoreError::SegmentOutOfRange => ApiError::not_found("segment not found"),
        other => status_conflict(other),
    })?
    .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &updated)))
}

/// Bulk transcript text edit — one locked write for all listed segments
/// (the frontend's search & replace). Unlike `patch_segment`, an edit may
/// leave a segment empty: replacing a phrase with nothing is legitimate, and
/// the segment keeps its speaker and timestamps.
pub fn patch_segments(
    ctx: &Ctx,
    meeting_id: &str,
    edits: &[(usize, String)],
) -> ApiResult<Value> {
    if edits.is_empty() {
        return Err(ApiError::invalid("no segments to update"));
    }
    let updated = store::update_segment_texts(&ctx.paths, &ctx.locks, meeting_id, edits)
        .map_err(|e| match e {
            StoreError::SegmentOutOfRange => ApiError::not_found("segment not found"),
            other => status_conflict(other),
        })?
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &updated)))
}

pub fn rename_speakers(ctx: &Ctx, meeting_id: &str, names: &Map<String, Value>) -> ApiResult<Value> {
    let updated = store::merge_speaker_names(&ctx.paths, &ctx.locks, meeting_id, names)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &updated)))
}

/// Confirm or reject an auto-recognized speaker.
pub fn verify_speaker(
    ctx: &Ctx,
    meeting_id: &str,
    diarized_id: &str,
    correct: bool,
) -> ApiResult<Value> {
    let result = if correct {
        store::confirm_speaker(&ctx.paths, &ctx.locks, meeting_id, diarized_id)
    } else {
        store::reject_speaker(&ctx.paths, &ctx.locks, meeting_id, diarized_id)
    };
    let updated = result
        .map_err(|_| ApiError::conflict("speaker was not auto-recognized"))?
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &updated)))
}

pub fn delete_meeting(ctx: &Ctx, meeting_id: &str) -> ApiResult<Value> {
    if !store::delete_meeting(&ctx.paths, &ctx.locks, meeting_id) {
        return Err(ApiError::not_found("meeting not found"));
    }
    Ok(json!({"deleted": true}))
}

/// GET /api/meetings/{id}/export — returns (markdown, suggested filename);
/// the shell shows a save dialog and writes the file.
pub fn export_meeting(ctx: &Ctx, meeting_id: &str) -> ApiResult<(String, String)> {
    let record = store::get_meeting(&ctx.paths, &ctx.locks, meeting_id)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    let md = to_markdown(&record);
    let title = record
        .get("title")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| record.get("filename").and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
        .unwrap_or(meeting_id);
    // Python: re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "-")
    let safe: String = title
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == ' ')
        .collect::<String>()
        .trim()
        .replace(' ', "-");
    let safe = if safe.is_empty() { meeting_id.to_string() } else { safe };
    Ok((md, format!("{safe}.md")))
}

fn status_conflict(e: StoreError) -> ApiError {
    match e {
        StoreError::StatusConflict(status) => ApiError::conflict(format!(
            "meeting is not editable yet (status: {})",
            status.as_deref().unwrap_or("None")
        )),
        StoreError::UnknownSpeaker => ApiError::invalid("unknown speaker id"),
        StoreError::SegmentOutOfRange => ApiError::not_found("segment not found"),
        StoreError::NotRecognized => ApiError::conflict("speaker was not auto-recognized"),
    }
}

// --------------------------------------------------------------------------
// Speaker library
// --------------------------------------------------------------------------

pub fn get_speakers(ctx: &Ctx) -> Value {
    json!(store::get_speaker_names(&ctx.paths, &ctx.locks))
}

/// Rename a library entry, keeping its voiceprint (future recordings only).
pub fn rename_known_speaker(ctx: &Ctx, name: &str, new_name: &str) -> ApiResult<Value> {
    let new = new_name.trim();
    if new.is_empty() || new.chars().count() > 100 {
        return Err(ApiError::invalid("name must be 1-100 characters"));
    }
    match store::rename_speaker(&ctx.paths, &ctx.locks, name, new) {
        "missing" => Err(ApiError::not_found("speaker not found")),
        "exists" => Err(ApiError::conflict("a speaker with that name already exists")),
        _ => Ok(json!(store::get_speaker_names(&ctx.paths, &ctx.locks))),
    }
}

pub fn delete_speaker(ctx: &Ctx, name: &str) -> ApiResult<Value> {
    if !store::delete_speaker(&ctx.paths, &ctx.locks, name) {
        return Err(ApiError::not_found("speaker not found"));
    }
    Ok(json!({"deleted": true}))
}

// --------------------------------------------------------------------------
// Data management
// --------------------------------------------------------------------------

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(path) else { return 0 };
    for entry in entries.flatten() {
        // file_type() does not follow symlinks — same as Python's
        // is_file(follow_symlinks=False); links are skipped entirely.
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_file() {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
        } else if ft.is_dir() {
            total += dir_size(&entry.path());
        }
    }
    total
}

pub fn get_data_info(ctx: &Ctx) -> Value {
    let meetings = dir_size(&ctx.paths.meetings_dir());
    let uploads = dir_size(&ctx.paths.uploads_dir());
    let models = dir_size(&ctx.paths.models_dir);
    json!({
        "data_dir": ctx.paths.data_dir,
        "meetings_bytes": meetings,
        "uploads_bytes": uploads,
        "models_bytes": models,
        "total_bytes": meetings + uploads + models,
        "meeting_count": store::list_meetings(&ctx.paths, &ctx.locks).len(),
    })
}

// --------------------------------------------------------------------------
// Markdown export
// --------------------------------------------------------------------------

fn to_markdown(record: &store::Record) -> String {
    let speakers = record.get("speakers").and_then(|v| v.as_object());
    let mut out = String::new();
    let title = record
        .get("title")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| record.get("filename").and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
        .unwrap_or("Meeting");
    out.push_str(&format!("# {title}\n\n"));
    if let Some(summary) = record.get("summary").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
        out.push_str(&format!("## Summary\n\n{summary}\n\n"));
    }
    if let Some(takeaways) = record.get("takeaways").and_then(|v| v.as_array()).filter(|a| !a.is_empty()) {
        out.push_str("## Key takeaways\n\n");
        for t in takeaways {
            out.push_str(&format!("- {}\n", t.as_str().unwrap_or_default()));
        }
        out.push('\n');
    }
    if let Some(items) = record.get("action_items").and_then(|v| v.as_array()).filter(|a| !a.is_empty()) {
        out.push_str("## Action items\n\n");
        for a in items {
            out.push_str(&format!("- [ ] {}\n", a.as_str().unwrap_or_default()));
        }
        out.push('\n');
    }
    out.push_str("## Transcript\n\n");
    if let Some(segments) = record.get("segments").and_then(|v| v.as_array()) {
        for seg in segments {
            let spk_id = seg.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
            let name = speakers
                .and_then(|m| m.get(spk_id))
                .and_then(|v| v.as_str())
                .unwrap_or(spk_id);
            let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!("**{name}** [{}]: {text}\n\n", fmt_ts(start)));
        }
    }
    out
}

fn fmt_ts(seconds: f64) -> String {
    let total = seconds as i64;
    let (m, s) = (total / 60, total % 60);
    let (h, m) = (m / 60, m % 60);
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}
