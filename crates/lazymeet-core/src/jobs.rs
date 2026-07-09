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
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use crate::config::Config;
use crate::pipeline::{self, PipelineCtx};
use crate::{store, Ctx};

#[derive(Default)]
pub struct JobsState {
    queue: Mutex<VecDeque<(String, PathBuf)>>,
    active: Mutex<Option<String>>,
}

/// Queue a meeting for processing; starts immediately if nothing is running.
pub fn dispatch(ctx: &Arc<Ctx>, meeting_id: String, upload_path: PathBuf) {
    ctx.jobs.queue.lock().unwrap().push_back((meeting_id, upload_path));
    pump_queue(ctx);
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
        while let Some((meeting_id, upload_path)) = queue.pop_front() {
            // Skip entries deleted while they waited.
            let record = store::get_meeting(&ctx.paths, &ctx.locks, &meeting_id);
            let queued = record
                .as_ref()
                .and_then(|r| r.get("status"))
                .and_then(|v| v.as_str())
                == Some("queued");
            if queued {
                *active = Some(meeting_id.clone());
                picked = Some((meeting_id, upload_path));
                break;
            }
        }
        picked
    };
    if let Some((meeting_id, upload_path)) = job {
        start_worker(ctx, meeting_id, upload_path);
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

fn start_worker(ctx: &Arc<Ctx>, meeting_id: String, upload_path: PathBuf) {
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            run_worker(&ctx, &meeting_id, &upload_path);
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
        if rec_status == "queued" && !filename.is_empty() {
            let upload = ctx.paths.uploads_dir().join(&id).join(filename);
            if upload.exists() {
                dispatch(ctx, id, upload);
                continue;
            }
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

pub fn run_worker(ctx: &Arc<Ctx>, meeting_id: &str, upload_path: &PathBuf) {
    // Pick up the latest settings — a fresh snapshot per job, like the fresh
    // worker process (or Python's config.refresh()) used to guarantee.
    let config = Config::resolve(&ctx.paths);
    let record = store::get_meeting(&ctx.paths, &ctx.locks, meeting_id).unwrap_or_default();
    let num_speakers = record.get("num_speakers").and_then(|v| v.as_i64()).unwrap_or(-1);

    let pctx = PipelineCtx {
        fake: ctx.fake,
        config: &config,
        models_dir: &ctx.paths.models_dir,
        ffmpeg: ctx.ffmpeg.clone(),
    };

    let result = (|| -> Result<(), String> {
        update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("extracting"))]));
        let wav = ctx.paths.uploads_dir().join(meeting_id).join("audio.wav");
        // Video uploads get their AAC playback track in the same decode pass.
        // The video in uploads/ is only a *copy* of the user's file (made at
        // drag-drop), so once the WAV — and, with KEEP_AUDIO, the playback
        // track — are extracted, the bulky copy is deleted right away instead
        // of surviving the whole run. The one exception: if the combined pass
        // fell back to WAV-only, the copy stays for the end-of-run playback
        // retry so playback is never lost.
        let is_video = !crate::config::is_audio_file(upload_path);
        let playback = ctx
            .paths
            .uploads_dir()
            .join(meeting_id)
            .join(crate::config::PLAYBACK_FILENAME);
        let want_playback = config.keep_audio() && is_video;
        pipeline::extract_audio(&pctx, upload_path, &wav, want_playback.then_some(&*playback))?;
        if is_video && (!config.keep_audio() || playback.is_file()) {
            let _ = std::fs::remove_file(upload_path);
        }

        update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("transcribing"))]));
        let segments = pipeline::transcribe(&pctx, &wav)?;

        update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("diarizing"))]));
        let turns = pipeline::diarize(&pctx, &wav, num_speakers)?;

        update_and_emit(ctx, meeting_id, false, obj(&[("status", json!("identifying"))]));
        let voiceprints = pipeline::compute_voiceprints(&pctx, &wav, &turns)?;
        let (turns, voiceprints) = pipeline::consolidate_speakers(
            &turns,
            &voiceprints,
            config.speaker_merge_threshold(),
            config.min_speaker_seconds(),
        );
        let library = store::load_speaker_library(&ctx.paths, &ctx.locks);
        let (speakers, recognized) =
            pipeline::match_speakers(&voiceprints, &library, config.speaker_threshold());

        let merged = pipeline::assign_speakers(&segments, &turns);

        let vp_value: Map<String, Value> =
            voiceprints.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        update_and_emit(
            ctx,
            meeting_id,
            true, // never overwrite names the user set in the meantime
            obj(&[
                ("status", json!("summarizing")),
                ("segments", serde_json::to_value(&merged).unwrap_or(json!([]))),
                ("speakers", Value::Object(speakers.clone())),
                ("recognized", json!(recognized)),
                ("voiceprints", Value::Object(vp_value)),
            ]),
        );

        // A summary failure (e.g. an unusually long meeting) must not throw
        // away the transcript — mark the meeting done with a note instead.
        match pipeline::summarize(&pctx, &merged, &speakers) {
            Ok(result) => {
                update_and_emit(
                    ctx,
                    meeting_id,
                    false,
                    obj(&[
                        ("status", json!("done")),
                        ("summary", json!(result.summary)),
                        ("takeaways", json!(result.takeaways)),
                        ("action_items", json!(result.action_items)),
                    ]),
                );
            }
            Err(exc) => {
                update_and_emit(
                    ctx,
                    meeting_id,
                    false,
                    obj(&[
                        ("status", json!("done")),
                        ("summary", json!(format!("(Automatic summary unavailable: {exc})"))),
                        ("takeaways", json!([])),
                        ("action_items", json!([])),
                    ]),
                );
            }
        }

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
            let _ = pipeline::extract_playback_audio(&pctx, upload_path, &playback);
            if playback.is_file() {
                let _ = std::fs::remove_file(upload_path);
            }
            // still no playback track: keep the video rather than lose playback
        }
        Ok(())
    })();

    if let Err(exc) = result {
        update_and_emit(
            ctx,
            meeting_id,
            false,
            obj(&[("status", json!("error")), ("error", json!(exc))]),
        );
    }
}
