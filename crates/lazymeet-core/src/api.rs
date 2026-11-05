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
        // What "Auto" threads resolve to on this machine, so the settings UI
        // can label the option "Auto — 7 of 10 threads" instead of guessing.
        "cores": config::cpu_count(),
        "auto_threads": config::auto_threads(),
        // Spoken/notes languages as (code, name); "auto" is added by the UI.
        "languages": LANGUAGE_NAMES,
        // Backends this build offers for the summarizer (Metal GPU and CPU
        // on macOS, CPU elsewhere); the UI shows the choice only when there
        // is one. The ONNX stages are CPU-only (CoreML is compiled out).
        "compute_options": {
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
    if let Some(v) = fields.get("DIAR_MODEL").filter(|v| !v.is_null()) {
        let id = v.as_str().unwrap_or_default();
        if !models::diar_catalog().iter().any(|e| e.id == id) {
            return Err(ApiError::invalid("unknown speaker diarization model"));
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
    // The model + spoken-language pair as it will be after this change:
    // Cohere Transcribe, e.g., cannot run on "auto". Checked only when the
    // change touches the pair, so an unrelated save never trips over it.
    if fields.contains_key("ASR_MODEL") || fields.contains_key("ASR_LANGUAGE") {
        let next = config::Config::from_overrides(&overrides);
        let model = next.asr_model();
        if let Some(entry) = models::asr_catalog().into_iter().find(|e| e.id == model) {
            if let Some(problem) = entry.language_problem(&next.asr_language()) {
                return Err(ApiError::invalid(problem));
            }
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

/// Delete a downloaded model (not the active or a first-run one); the
/// `models-progress` event carries the new status.
pub fn remove_model(ctx: &Arc<Ctx>, step: &str, id: &str) -> ApiResult<Value> {
    models::remove_model(ctx, step, id)?;
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
    check_asr_language(ctx, language)?;
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
    std::fs::copy(src, &dest)
        .map_err(|e| ApiError::internal(format!("could not copy the recording: {e}")))?;
    jobs::dispatch(ctx, id.clone(), jobs::Job::Process { upload: dest });
    Ok(json!({"id": id}))
}

/// Fail fast when the transcription model a job would run can't take the
/// recording's language (`language` = the meeting's choice; None = the
/// setting), instead of erroring once processing starts.
fn check_asr_language(ctx: &Ctx, language: Option<&str>) -> ApiResult<()> {
    let config = ctx.config.read().unwrap();
    let entry = models::effective_asr_entry(&ctx.paths.models_dir, ctx.fake, &config.asr_model());
    let language = language.map(str::to_string).unwrap_or_else(|| config.asr_language());
    match entry.language_problem(&language) {
        Some(problem) => Err(ApiError::invalid(problem)),
        None => Ok(()),
    }
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
            check_asr_language(ctx, record.get("language").and_then(|v| v.as_str()))?;
            jobs::Job::Retranscribe
        }
        _ => return Err(ApiError::invalid("mode must be \"notes\" or \"transcription\"")),
    };
    let record = jobs::queue_rejob(ctx, meeting_id, mode, job)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    Ok(Value::Object(store::public_meeting(&ctx.paths, &record)))
}

pub fn list_meetings(ctx: &Ctx) -> Value {
    Value::Array(
        store::list_meetings(&ctx.paths, &ctx.locks).into_iter().map(Value::Object).collect(),
    )
}

/// Literal, case-insensitive search over every meeting on disk. An empty query
/// is a legitimate empty result, not a validation error.
pub fn search_meetings(ctx: &Ctx, query: &str) -> Value {
    Value::Array(
        store::search_meetings(&ctx.paths, &ctx.locks, query)
            .into_iter()
            .map(Value::Object)
            .collect(),
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

/// Longest notes text a PATCH may store. Generous for a hand-written summary,
/// but a runaway paste would otherwise land in the record, in every version
/// snapshot, in every `meeting-updated` event and in the search index.
const MAX_NOTE_CHARS: usize = 20_000;

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
    for key in ["summary", "summary_long"] {
        if let Some(v) = fields.get(key) {
            // Reject instead of coercing: `as_str().unwrap_or_default()` would
            // turn a malformed value into an empty string and silently erase
            // the note the user asked to keep.
            let text = v
                .as_str()
                .ok_or_else(|| ApiError::invalid(format!("{key} must be a string")))?
                .trim();
            if text.chars().count() > MAX_NOTE_CHARS {
                return Err(ApiError::invalid(format!(
                    "{key} must be at most {MAX_NOTE_CHARS} characters"
                )));
            }
            changes.insert(key.into(), json!(text));
            content_edit = true;
        }
    }
    for key in ["takeaways", "action_items"] {
        if let Some(v) = fields.get(key) {
            // Plain strings (the old shape) and {short, detail} objects both
            // arrive here; store::note_items normalizes and drops empties.
            let items: Vec<Value> = store::note_items(Some(v)).into_iter().take(100).collect();
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
pub fn patch_segments(ctx: &Ctx, meeting_id: &str, edits: &[(usize, String)]) -> ApiResult<Value> {
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

pub fn rename_speakers(
    ctx: &Ctx,
    meeting_id: &str,
    names: &Map<String, Value>,
) -> ApiResult<Value> {
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
    if let Some(summary) = record.get("summary").and_then(|v| v.as_str()).filter(|s| !s.is_empty())
    {
        out.push_str(&format!("## Summary\n\n{summary}\n\n"));
    }
    if let Some(detail) =
        record.get("summary_long").and_then(|v| v.as_str()).filter(|s| !s.is_empty())
    {
        out.push_str(&format!("## In detail\n\n{detail}\n\n"));
    }
    let list = |out: &mut String, key: &str, heading: &str, checkbox: bool| {
        let items = store::note_items(record.get(key));
        if items.is_empty() {
            return;
        }
        out.push_str(&format!("## {heading}\n\n"));
        let mark = if checkbox { "- [ ] " } else { "- " };
        for item in &items {
            let field = |k: &str| item.get(k).and_then(|v| v.as_str()).unwrap_or("");
            let (short, detail) = (field("short"), field("detail"));
            out.push_str(&format!("{mark}{short}\n"));
            if !detail.is_empty() {
                // Indented continuation keeps the list item intact.
                out.push_str(&format!("  {detail}\n"));
            }
        }
        out.push('\n');
    };
    list(&mut out, "takeaways", "Key takeaways", false);
    list(&mut out, "action_items", "Action items", true);
    out.push_str("## Transcript\n\n");
    if let Some(segments) = record.get("segments").and_then(|v| v.as_array()) {
        for seg in segments {
            let spk_id = seg.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
            let name =
                speakers.and_then(|m| m.get(spk_id)).and_then(|v| v.as_str()).unwrap_or(spk_id);
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
