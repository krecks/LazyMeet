//! Processing queue + worker. Port of the worker half of `backend/main.py`.
//!
//! Uploads are processed strictly one at a time: the models are heavy enough
//! that two concurrent workers would thrash CPU/GPU and memory. New meetings
//! wait in the queue with status "queued" until the active one finishes.
//!
//! Unlike the Python backend (separate worker *process*, needed because native
//! calls held the GIL), the worker is an in-process thread — Rust has no GIL.
//! Panics are contained with `catch_unwind`; a hard native crash is covered by
//! `sweep_orphans` on next launch.

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use crate::config::Config;
use crate::pipeline::{self, PipelineCtx};
use crate::{store, Ctx};

/// What the worker should do for a queued meeting.
pub enum Job {
    /// Full pipeline over a fresh upload (the normal path).
    Process { upload: PathBuf },
    /// Full pipeline again, over the kept audio — new transcript, new notes.
    Retranscribe,
    /// Summarization only, over the existing transcript.
    Resummarize,
}

#[derive(Default)]
pub struct JobsState {
    queue: Mutex<VecDeque<(String, Job)>>,
    active: Mutex<Option<String>>,
}

/// Queue a meeting for processing; starts immediately if nothing is running.
pub fn dispatch(ctx: &Arc<Ctx>, meeting_id: String, job: Job) {
    ctx.jobs.queue.lock().unwrap().push_back((meeting_id, job));
    pump_queue(ctx);
}

/// Queue a notes/transcription re-run on an existing meeting: mark it queued
/// (with the `rejob` marker sweep_orphans uses after a crash), emit, dispatch.
pub fn queue_rejob(ctx: &Arc<Ctx>, meeting_id: &str, mode: &str, job: Job) -> Option<store::Record> {
    let record = update_and_emit(
        ctx,
        meeting_id,
        false,
        obj(&[
            ("status", json!("queued")),
            ("error", Value::Null),
            ("rejob", json!(mode)),
        ]),
    )?;
    dispatch(ctx, meeting_id.to_string(), job);
    Some(record)
}

pub fn pump_queue(ctx: &Arc<Ctx>) {
    // Nothing can process until the first-run models exist; the download
    // completion re-pumps.
    if !crate::models::is_ready(ctx) {
        return;
    }
    let job = {
        let mut active = ctx.jobs.active.lock().unwrap();
        if active.is_some() {
            return;
        }
        let mut queue = ctx.jobs.queue.lock().unwrap();
        let mut picked = None;
        while let Some((meeting_id, job)) = queue.pop_front() {
            // Skip entries deleted while they waited.
            let record = store::get_meeting(&ctx.paths, &ctx.locks, &meeting_id);
            let queued = record
                .as_ref()
                .and_then(|r| r.get("status"))
                .and_then(|v| v.as_str())
                == Some("queued");
            if queued {
                *active = Some(meeting_id.clone());
                picked = Some((meeting_id, job));
                break;
            }
        }
        picked
    };
    if let Some((meeting_id, job)) = job {
        start_worker(ctx, meeting_id, job);
    }
}

/// True while a meeting is being processed (the benchmark refuses to run then).
pub fn is_busy(ctx: &Ctx) -> bool {
    ctx.jobs.active.lock().unwrap().is_some()
}

fn finish_active(ctx: &Arc<Ctx>, meeting_id: &str) {
    {
        let mut active = ctx.jobs.active.lock().unwrap();
        if active.as_deref() == Some(meeting_id) {
            *active = None;
        }
    }
    pump_queue(ctx);
}

fn start_worker(ctx: &Arc<Ctx>, meeting_id: String, job: Job) {
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            run_worker(&ctx, &meeting_id, &job);
        }));
        if result.is_err() {
            // A panic in the pipeline must not leave the meeting "processing"
            // forever (the Python equivalent was the worker-process monitor).
            let record = store::get_meeting(&ctx.paths, &ctx.locks, &meeting_id);
            let status = record
                .as_ref()
                .and_then(|r| r.get("status"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if status != "done" && status != "error" {
                update_and_emit(
                    &ctx,
                    &meeting_id,
                    false,
                    obj(&[
                        ("status", json!("error")),
                        ("error", json!("processing crashed")),
                    ]),
                );
            }
        }
        finish_active(&ctx, &meeting_id);
    });
}

/// On startup, fail meetings whose worker died while the app wasn't running.
/// Meetings still in "queued" whose upload survived never started — resume
/// those instead. (The worker is in-process now, so at startup nothing can
/// still be running; the old PID probe is gone.)
pub fn sweep_orphans(ctx: &Arc<Ctx>) {
    for item in store::list_meetings(&ctx.paths, &ctx.locks) {
        let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if status == "done" || status == "error" {
            continue;
        }
        let Some(id) = item.get("id").and_then(|v| v.as_str()).map(String::from) else {
            continue;
        };
        let record = store::get_meeting(&ctx.paths, &ctx.locks, &id).unwrap_or_default();
        let rec_status = record.get("status").and_then(|v| v.as_str()).unwrap_or("");
        let filename = record.get("filename").and_then(|v| v.as_str()).unwrap_or("");
        let rejob = record.get("rejob").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if rec_status == "queued" {
            match rejob.as_str() {
                "notes" => {
                    dispatch(ctx, id, Job::Resummarize);
                    continue;
                }
                "transcription" => {
                    dispatch(ctx, id, Job::Retranscribe);
                    continue;
                }
                _ if !filename.is_empty() => {
                    let upload = ctx.paths.uploads_dir().join(&id).join(filename);
                    if upload.exists() {
                        dispatch(ctx, id, Job::Process { upload });
                        continue;
                    }
                }
                _ => {}
            }
        }
        // A re-run that died mid-flight left a complete meeting behind —
        // restore it instead of failing it.
        let has_segments = record
            .get("segments")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);
        if !rejob.is_empty() && has_segments {
            update_and_emit(
                ctx,
                &id,
                false,
                obj(&[("status", json!("done")), ("rejob", Value::Null)]),
            );
            continue;
        }
        update_and_emit(
            ctx,
            &id,
            false,
            obj(&[
                ("status", json!("error")),
                ("error", json!("processing was interrupted")),
            ]),
        );
    }
}

fn obj(entries: &[(&str, Value)]) -> Map<String, Value> {
    let mut map = Map::new();
    for (k, v) in entries {
        map.insert(k.to_string(), v.clone());
    }
    map
}

/// Write changes and push a `meeting-updated` event. Terminal transitions
/// (done/error) carry the full public meeting so the UI needs no follow-up
/// fetch; intermediate ones carry only `{id, status}`.
fn update_and_emit(
    ctx: &Arc<Ctx>,
    meeting_id: &str,
    preserve_speakers: bool,
    changes: Map<String, Value>,
) -> Option<store::Record> {
    let record =
        store::update_meeting(&ctx.paths, &ctx.locks, meeting_id, preserve_speakers, changes)?;
    let status = record.get("status").and_then(|v| v.as_str()).unwrap_or("");
    let mut payload = Map::new();
    payload.insert("id".into(), json!(meeting_id));
    payload.insert("status".into(), json!(status));
    if status == "done" || status == "error" {
        payload.insert(
            "meeting".into(),
            Value::Object(store::public_meeting(&ctx.paths, &record)),
        );
    }
    ctx.events.emit("meeting-updated", Value::Object(payload));
    Some(record)
}

/// The record's current content, as stored on a version entry so the version
/// stays viewable after a regeneration replaces the live fields.
fn record_snapshot(record: &store::Record) -> Value {
    let f = |k: &str| record.get(k).cloned().unwrap_or(Value::Null);
    json!({
        "summary": f("summary"),
        "takeaways": f("takeaways"),
        "action_items": f("action_items"),
        "segments": f("segments"),
        "speakers": f("speakers"),
    })
}

/// One entry in a meeting's version history, appended to `record.versions`:
/// which models produced the content each time it was (re)generated, plus a
/// content snapshot for viewing that version later. Pre-history meetings get
/// a backfilled v1 — unknown models (unknowable retroactively) but with the
/// content the record held before this re-run, so the original stays viewable.
fn versions_with(
    record: &store::Record,
    action: &str,
    asr_model: Value,
    summary_model: Value,
    snapshot: Value,
) -> Value {
    let mut versions = record
        .get("versions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if versions.is_empty() && action != "processed" {
        versions.push(json!({
            "v": 1,
            "created_at": record.get("created_at").cloned().unwrap_or(Value::Null),
            "action": "processed",
            "asr_model": Value::Null,
            "summary_model": Value::Null,
            "snapshot": record_snapshot(record),
        }));
    } else if let Some(Value::Object(last)) = versions.last_mut() {
        // Refresh the outgoing version's snapshot from the live record: it
        // picks up manual edits made since, and fills in entries written
        // before snapshots existed.
        last.insert("snapshot".into(), record_snapshot(record));
    }
    versions.push(json!({
        "v": versions.len() + 1,
        "created_at": store::now_iso(),
        "action": action,
        "asr_model": asr_model,
        "summary_model": summary_model,
        "snapshot": snapshot,
    }));
    Value::Array(versions)
}

pub fn run_worker(ctx: &Arc<Ctx>, meeting_id: &str, job: &Job) {
    // Pick up the latest settings — a fresh snapshot per job, like the fresh
    // worker process (or Python's config.refresh()) used to guarantee.
    let mut config = Config::resolve(&ctx.paths);
    let record = store::get_meeting(&ctx.paths, &ctx.locks, meeting_id).unwrap_or_default();
    // A language chosen on the prep screen outranks the saved setting for
    // this meeting (it's stored on the record, so re-transcriptions keep it).
    if let Some(lang) = record.get("language").and_then(|v| v.as_str()) {
        config.override_value("ASR_LANGUAGE", json!(lang));
    }

    let pctx = PipelineCtx {
        fake: ctx.fake,
        config: &config,
        models_dir: &ctx.paths.models_dir,
        ffmpeg: ctx.ffmpeg.clone(),
    };

    let result = match job {
        Job::Process { upload } => run_full(ctx, meeting_id, &pctx, &config, &record, upload, "processed"),
        Job::Retranscribe => match store::audio_path(&ctx.paths, &record) {
            Some(audio) => run_full(ctx, meeting_id, &pctx, &config, &record, &audio, "transcription"),
            None => Err("the recording is no longer on disk".into()),
        },
        Job::Resummarize => resummarize(ctx, meeting_id, &pctx, &config, &record),
    };

    if let Err(exc) = result {
        update_and_emit(
            ctx,
            meeting_id,
            false,
            obj(&[
                ("status", json!("error")),
                ("error", json!(exc)),
                ("rejob", Value::Null),
            ]),
        );
    }
}

/// The full pipeline: extract → transcribe → diarize → identify → summarize.
/// `action` is the version-history label — "processed" for a fresh upload,
/// "transcription" for a re-run over the kept audio (which replaces speaker
/// assignments instead of preserving them: the new diarization's cluster ids
/// need not line up with the old names).
fn run_full(
    ctx: &Arc<Ctx>,
    meeting_id: &str,
    pctx: &PipelineCtx,
    config: &Config,
    record: &store::Record,
    upload_path: &Path,
    action: &str,
) -> Result<(), String> {
    let num_speakers = record.get("num_speakers").and_then(|v| v.as_i64()).unwrap_or(-1);
    // User-selected spans to transcribe (null/absent = the whole file). Kept
    // on the record so queued resumes and re-transcriptions honor them too.
    let ranges: Option<Vec<pipeline::TimeRange>> = record
        .get("ranges")
        .filter(|v| !v.is_null())
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let retranscribe = action == "transcription";

    update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("extracting"))]));
    let wav = ctx.paths.uploads_dir().join(meeting_id).join("audio.wav");
    // Video uploads get their AAC playback track in the same decode pass.
    // The video in uploads/ is only a *copy* of the user's file (made at
    // drag-drop), so once the WAV — and, with KEEP_AUDIO, the playback
    // track — are extracted, the bulky copy is deleted right away instead
    // of surviving the whole run. The one exception: if the combined pass
    // fell back to WAV-only, the copy stays for the end-of-run playback
    // retry so playback is never lost. (A re-transcription's source is the
    // kept audio itself — never a video, so none of this triggers.)
    let is_video = !crate::config::is_audio_file(upload_path);
    let playback = ctx
        .paths
        .uploads_dir()
        .join(meeting_id)
        .join(crate::config::PLAYBACK_FILENAME);
    let want_playback = config.keep_audio() && is_video;
    pipeline::extract_audio(pctx, upload_path, &wav, want_playback.then_some(&*playback))?;
    if is_video && (!config.keep_audio() || playback.is_file()) {
        let _ = std::fs::remove_file(upload_path);
    }

    update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("transcribing"))]));
    let (segments, turns) = pipeline::transcribe_and_diarize(
        pctx,
        &wav,
        num_speakers,
        ranges.as_deref(),
        || {
            update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("diarizing"))]));
        },
    )?;

    update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("identifying"))]));
    let voiceprints = pipeline::compute_voiceprints(pctx, &wav, &turns)?;
    // Folding short speakers into the closest main voice cleans up the
    // classic diarizer's over-clustering. Nemotron tracks speakers itself and
    // keeps someone who only says a sentence or two as a speaker of their
    // own, so its output only merges voices that match outright.
    let min_speaker_seconds = match pipeline::diar_engine(pctx, num_speakers) {
        "nemotron" => 0.0,
        _ => config.min_speaker_seconds(),
    };
    let (turns, voiceprints) = pipeline::consolidate_speakers(
        &turns,
        &voiceprints,
        config.speaker_merge_threshold(),
        min_speaker_seconds,
    );
    let library = store::load_speaker_library(&ctx.paths, &ctx.locks);
    let (speakers, recognized) =
        pipeline::match_speakers(&voiceprints, &library, config.speaker_threshold());

    let merged = pipeline::assign_speakers(&segments, &turns);

    let vp_value: Map<String, Value> =
        voiceprints.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    let seg_value = serde_json::to_value(&merged).unwrap_or(json!([]));
    let mut stage = obj(&[
        ("status", json!("summarizing")),
        ("segments", seg_value.clone()),
        ("speakers", Value::Object(speakers.clone())),
        ("recognized", json!(recognized)),
        ("voiceprints", Value::Object(vp_value)),
    ]);
    if retranscribe {
        // The old confirmations described clusters that no longer exist.
        stage.insert("confirmed".into(), json!([]));
    }
    // A fresh upload must never overwrite names the user set in the meantime;
    // a re-transcription replaces them (its cluster ids are new).
    update_and_emit(ctx, meeting_id, !retranscribe, stage);

    // A summary failure (e.g. an unusually long meeting) must not throw
    // away the transcript — mark the meeting done with a note instead.
    let (summary, takeaways, action_items) = match pipeline::summarize(pctx, &merged, &speakers) {
        Ok(result) => (result.summary, result.takeaways, result.action_items),
        Err(exc) => (format!("(Automatic summary unavailable: {exc})"), vec![], vec![]),
    };
    let snapshot = json!({
        "summary": summary,
        "takeaways": takeaways,
        "action_items": action_items,
        "segments": seg_value,
        "speakers": Value::Object(speakers.clone()),
    });
    let versions = versions_with(
        record,
        action,
        json!(config.asr_model()),
        json!(config.summary_model()),
        snapshot,
    );
    update_and_emit(
        ctx,
        meeting_id,
        false,
        obj(&[
            ("status", json!("done")),
            ("summary", json!(summary)),
            ("takeaways", json!(takeaways)),
            ("action_items", json!(action_items)),
            ("error", Value::Null),
            ("rejob", Value::Null),
            ("versions", versions),
        ]),
    );

    // The extracted WAV is a processing intermediate either way. With
    // KEEP_AUDIO (default) only the audio stays for segment playback:
    // audio-only uploads are kept as-is, video uploads were reduced to
    // their playback track back in the extracting stage. With it off
    // everything goes.
    let _ = std::fs::remove_file(&wav);
    if !config.keep_audio() {
        let _ = std::fs::remove_dir_all(ctx.paths.uploads_dir().join(meeting_id));
    } else if upload_path.is_file() && is_video && !playback.is_file() {
        // The combined extraction fell back to WAV-only — one retry, and
        // only then does the video copy go.
        let _ = pipeline::extract_playback_audio(pctx, upload_path, &playback);
        if playback.is_file() {
            let _ = std::fs::remove_file(upload_path);
        }
        // still no playback track: keep the video rather than lose playback
    }
    Ok(())
}

/// Notes-only re-run over the existing transcript. Unlike the full run, a
/// summarizer failure here must not overwrite the notes the user already has —
/// the meeting goes to "error" (via run_worker) with its content intact.
fn resummarize(
    ctx: &Arc<Ctx>,
    meeting_id: &str,
    pctx: &PipelineCtx,
    config: &Config,
    record: &store::Record,
) -> Result<(), String> {
    let merged: Vec<pipeline::MergedSegment> = record
        .get("segments")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    if merged.is_empty() {
        return Err("there is no transcript to summarize".into());
    }
    let speakers = record
        .get("speakers")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("summarizing"))]));
    let result = pipeline::summarize(pctx, &merged, &speakers)?;

    // The transcript is untouched, so the new version keeps its ASR model.
    let asr = record
        .get("versions")
        .and_then(|v| v.as_array())
        .and_then(|a| a.last())
        .and_then(|e| e.get("asr_model"))
        .cloned()
        .unwrap_or(Value::Null);
    let snapshot = json!({
        "summary": result.summary,
        "takeaways": result.takeaways,
        "action_items": result.action_items,
        "segments": record.get("segments").cloned().unwrap_or(json!([])),
        "speakers": Value::Object(speakers.clone()),
    });
    let versions = versions_with(record, "notes", asr, json!(config.summary_model()), snapshot);
    update_and_emit(
        ctx,
        meeting_id,
        false,
        obj(&[
            ("status", json!("done")),
            ("summary", json!(result.summary)),
            ("takeaways", json!(result.takeaways)),
            ("action_items", json!(result.action_items)),
            ("error", Value::Null),
            ("rejob", Value::Null),
            ("versions", versions),
        ]),
    );
    Ok(())
}
