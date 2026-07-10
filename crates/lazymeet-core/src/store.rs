//! JSON-file persistence for meetings and the cross-file speaker library.
//!
//! Port of `backend/store.py`. There is no database: each meeting is a single
//! JSON file under `meetings/` and the speaker library is one `speakers.json`.
//! Writes are atomic (temp file + rename) so readers never see a half-written
//! file. Records are kept as ordered JSON maps rather than structs so fields
//! written by the old Python backend (e.g. `worker_pid`) survive
//! read-modify-write untouched.
//!
//! The old cross-process `.lock` files are gone: the pipeline worker now runs
//! in-process, so plain mutexes serialize all read-modify-writes.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{json, Map, Value};

use crate::config::{Paths, PLAYBACK_FILENAME};

/// In-process read-modify-write serialization. Lock ordering: `meetings`
/// before `speakers` (merge_speaker_names enrols voices while holding the
/// meeting lock — same nesting the Python RLock allowed).
#[derive(Default)]
pub struct StoreLocks {
    meetings: Mutex<()>,
    speakers: Mutex<()>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StoreError {
    /// A mutation was rejected because the meeting is in the wrong status.
    StatusConflict(Option<String>),
    /// A segment was assigned a speaker id the meeting doesn't know.
    UnknownSpeaker,
    /// Segment index out of range.
    SegmentOutOfRange,
    /// Confirm/reject on a speaker that was not auto-recognized.
    NotRecognized,
}

pub type Record = Map<String, Value>;

pub fn now_iso() -> String {
    // Matches Python's datetime.now(timezone.utc).isoformat():
    // microseconds + "+00:00" — lexicographically sortable next to old records.
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}

pub fn atomic_write_json(path: &Path, data: &Value) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".{}.tmp", uuid::Uuid::new_v4().simple()));
    let body = serde_json::to_string_pretty(data)?;
    let result = std::fs::write(&tmp, body).and_then(|_| std::fs::rename(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn meeting_path(paths: &Paths, meeting_id: &str) -> PathBuf {
    paths.meetings_dir().join(format!("{meeting_id}.json"))
}

fn read_json_map(path: &Path) -> Option<Record> {
    let raw = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str::<Value>(&raw) {
        Ok(Value::Object(map)) => Some(map),
        _ => None,
    }
}

// --------------------------------------------------------------------------
// Meetings
// --------------------------------------------------------------------------

pub fn create_meeting(
    paths: &Paths,
    locks: &StoreLocks,
    filename: &str,
    num_speakers: i64,
) -> std::io::Result<Record> {
    let meeting_id = uuid::Uuid::new_v4().simple().to_string();
    let mut record = Map::new();
    record.insert("id".into(), json!(meeting_id));
    record.insert("filename".into(), json!(filename));
    // user-editable; filename stays the immutable original
    record.insert("title".into(), json!(filename));
    record.insert("created_at".into(), json!(now_iso()));
    record.insert("status".into(), json!("queued"));
    record.insert("error".into(), Value::Null);
    record.insert("num_speakers".into(), json!(num_speakers)); // -1 = auto-detect
    record.insert("speakers".into(), json!({})); // diarized id -> display name
    record.insert("recognized".into(), json!([])); // auto-matched from the library
    record.insert("confirmed".into(), json!([])); // recognized ids the user verified
    record.insert("segments".into(), json!([])); // [{start, end, speaker, text}]
    record.insert("summary".into(), json!(""));
    record.insert("takeaways".into(), json!([]));
    record.insert("action_items".into(), json!([]));
    // transient voiceprints captured during identification, used when the user
    // later renames a speaker so we can enrol that voice.
    record.insert("voiceprints".into(), json!({}));
    // one entry per (re)generation: which models produced the content
    record.insert("versions".into(), json!([]));

    let _g = locks.meetings.lock().unwrap();
    paths.ensure_dirs()?;
    atomic_write_json(&meeting_path(paths, &meeting_id), &Value::Object(record.clone()))?;
    Ok(record)
}

pub fn get_meeting(paths: &Paths, locks: &StoreLocks, meeting_id: &str) -> Option<Record> {
    let _g = locks.meetings.lock().unwrap();
    get_meeting_unlocked(paths, meeting_id)
}

fn get_meeting_unlocked(paths: &Paths, meeting_id: &str) -> Option<Record> {
    read_json_map(&meeting_path(paths, meeting_id))
}

/// Re-read, apply `changes`, and write back under the lock.
///
/// `preserve_speakers=true` makes existing `speakers` entries win over the
/// passed ones — used by the worker so its diarization defaults never
/// overwrite names the user set in the meantime.
pub fn update_meeting(
    paths: &Paths,
    locks: &StoreLocks,
    meeting_id: &str,
    preserve_speakers: bool,
    changes: Record,
) -> Option<Record> {
    let _g = locks.meetings.lock().unwrap();
    let mut record = get_meeting_unlocked(paths, meeting_id)?;
    let mut changes = changes;
    if preserve_speakers {
        if let Some(Value::Object(new_speakers)) = changes.get("speakers") {
            let mut merged = new_speakers.clone();
            if let Some(Value::Object(existing)) = record.get("speakers") {
                for (k, v) in existing {
                    merged.insert(k.clone(), v.clone());
                }
            }
            changes.insert("speakers".into(), Value::Object(merged));
        }
    }
    for (k, v) in changes {
        record.insert(k, v);
    }
    atomic_write_json(&meeting_path(paths, meeting_id), &Value::Object(record.clone())).ok()?;
    Some(record)
}

/// User-initiated edit: locked read-modify-write with a status guard.
///
/// The guard must run *inside* the lock — checking status before taking it
/// would race the worker's terminal write.
pub fn patch_meeting(
    paths: &Paths,
    locks: &StoreLocks,
    meeting_id: &str,
    allowed_statuses: Option<&[&str]>,
    changes: Record,
) -> Result<Option<Record>, StoreError> {
    let _g = locks.meetings.lock().unwrap();
    let Some(mut record) = get_meeting_unlocked(paths, meeting_id) else {
        return Ok(None);
    };
    if let Some(allowed) = allowed_statuses {
        let status = record.get("status").and_then(|v| v.as_str());
        if !status.map(|s| allowed.contains(&s)).unwrap_or(false) {
            return Err(StoreError::StatusConflict(status.map(String::from)));
        }
    }
    for (k, v) in changes {
        record.insert(k, v);
    }
    let _ = atomic_write_json(&meeting_path(paths, meeting_id), &Value::Object(record.clone()));
    Ok(Some(record))
}

/// Edit one transcript segment in place, under the lock.
///
/// Reassigning a speaker deliberately leaves `speakers` and `voiceprints`
/// untouched — an id whose last segment moves away simply stops appearing in
/// the legend; deleting it from the map would break `recognized` and rename
/// history.
pub fn update_segment(
    paths: &Paths,
    locks: &StoreLocks,
    meeting_id: &str,
    index: usize,
    text: Option<&str>,
    speaker: Option<&str>,
) -> Result<Option<Record>, StoreError> {
    let _g = locks.meetings.lock().unwrap();
    let Some(mut record) = get_meeting_unlocked(paths, meeting_id) else {
        return Ok(None);
    };
    if record.get("status").and_then(|v| v.as_str()) != Some("done") {
        let status = record.get("status").and_then(|v| v.as_str()).map(String::from);
        return Err(StoreError::StatusConflict(status));
    }
    if let Some(speaker) = speaker {
        let known = record
            .get("speakers")
            .and_then(|v| v.as_object())
            .map(|m| m.contains_key(speaker))
            .unwrap_or(false);
        if !known {
            return Err(StoreError::UnknownSpeaker);
        }
    }
    let segments = record
        .get_mut("segments")
        .and_then(|v| v.as_array_mut())
        .ok_or(StoreError::SegmentOutOfRange)?;
    let seg = segments.get_mut(index).ok_or(StoreError::SegmentOutOfRange)?;
    if let Some(obj) = seg.as_object_mut() {
        if let Some(speaker) = speaker {
            obj.insert("speaker".into(), json!(speaker));
        }
        if let Some(text) = text {
            obj.insert("text".into(), json!(text));
        }
    }
    let _ = atomic_write_json(&meeting_path(paths, meeting_id), &Value::Object(record.clone()));
    Ok(Some(record))
}

/// Is `c` a word character in the sense of Python's `\w` (unicode-aware)?
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Replace whole-word occurrences of `old` with `new` — the equivalent of
/// Python's `(?<!\w)OLD(?!\w)` so "Speaker 1" never matches inside "Speaker 10".
fn replace_name(text: &str, old: &str, new: &str) -> String {
    if old.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut consumed = 0usize; // bytes of `text` already pushed to `out`
    while let Some(pos) = rest.find(old) {
        let abs = consumed + pos;
        let before_ok = text[..abs].chars().next_back().map(|c| !is_word_char(c)).unwrap_or(true);
        let after = abs + old.len();
        let after_ok = text[after..].chars().next().map(|c| !is_word_char(c)).unwrap_or(true);
        if before_ok && after_ok {
            out.push_str(&text[consumed..abs]);
            out.push_str(new);
            consumed = after;
            rest = &text[after..];
        } else {
            // skip past this occurrence
            let next = abs + old.len();
            out.push_str(&text[consumed..next]);
            consumed = next;
            rest = &text[next..];
        }
    }
    out.push_str(&text[consumed..]);
    out
}

/// Rewrite one speaker's display name inside the generated notes.
fn rename_in_notes(record: &mut Record, old: &str, new: &str) {
    if let Some(Value::String(s)) = record.get("summary") {
        let replaced = replace_name(s, old, new);
        record.insert("summary".into(), json!(replaced));
    }
    for key in ["takeaways", "action_items"] {
        if let Some(Value::Array(items)) = record.get(key) {
            let replaced: Vec<Value> = items
                .iter()
                .map(|v| match v.as_str() {
                    Some(s) => json!(replace_name(s, old, new)),
                    None => v.clone(),
                })
                .collect();
            record.insert(key.into(), json!(replaced));
        }
    }
}

/// Apply user renames and enrol those voices in the speaker library.
pub fn merge_speaker_names(
    paths: &Paths,
    locks: &StoreLocks,
    meeting_id: &str,
    names: &Map<String, Value>,
) -> Option<Record> {
    let _g = locks.meetings.lock().unwrap();
    let mut record = get_meeting_unlocked(paths, meeting_id)?;
    let mut speakers = record
        .get("speakers")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let mut recognized: Vec<String> = as_string_vec(record.get("recognized"));
    let voiceprints = record
        .get("voiceprints")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    for (diarized_id, new_name) in names {
        let new_name = new_name.as_str().unwrap_or_default().trim().to_string();
        if new_name.is_empty() {
            continue;
        }
        let old_name = speakers.get(diarized_id).and_then(|v| v.as_str()).map(String::from);
        speakers.insert(diarized_id.clone(), json!(new_name));
        if let Some(old_name) = old_name.filter(|o| *o != new_name) {
            rename_in_notes(&mut record, &old_name, &new_name);
            if recognized.contains(diarized_id) {
                // The auto-match was wrong: the old name is a different person,
                // whose library entry must survive untouched.
                recognized.retain(|r| r != diarized_id);
                let confirmed: Vec<String> = as_string_vec(record.get("confirmed"))
                    .into_iter()
                    .filter(|c| c != diarized_id)
                    .collect();
                record.insert("confirmed".into(), json!(confirmed));
            } else {
                // Manually named earlier: move the library entry rather than
                // leaving a duplicate person behind under the old name. If the
                // new name already exists, the voice merges into it.
                if rename_speaker(paths, locks, &old_name, &new_name) == "exists" {
                    delete_speaker(paths, locks, &old_name);
                }
            }
        }
        // Enrol this voice so the name follows it into future recordings.
        if let Some(emb) = voiceprints.get(diarized_id).and_then(|v| v.as_array()) {
            if !emb.is_empty() {
                let emb: Vec<f64> = emb.iter().filter_map(|v| v.as_f64()).collect();
                upsert_speaker(paths, locks, &new_name, &emb);
            }
        }
    }
    record.insert("speakers".into(), Value::Object(speakers));
    record.insert("recognized".into(), json!(recognized));
    let _ = atomic_write_json(&meeting_path(paths, meeting_id), &Value::Object(record.clone()));
    Some(record)
}

/// User confirmed an auto-match: learn this recording's voice.
pub fn confirm_speaker(
    paths: &Paths,
    locks: &StoreLocks,
    meeting_id: &str,
    diarized_id: &str,
) -> Result<Option<Record>, StoreError> {
    let _g = locks.meetings.lock().unwrap();
    let Some(mut record) = get_meeting_unlocked(paths, meeting_id) else {
        return Ok(None);
    };
    let recognized = as_string_vec(record.get("recognized"));
    if !recognized.iter().any(|r| r == diarized_id) {
        return Err(StoreError::NotRecognized);
    }
    let mut confirmed = as_string_vec(record.get("confirmed"));
    if !confirmed.iter().any(|c| c == diarized_id) {
        confirmed.push(diarized_id.to_string());
    }
    let name = record
        .get("speakers")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get(diarized_id))
        .and_then(|v| v.as_str())
        .map(String::from);
    let emb: Vec<f64> = record
        .get("voiceprints")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get(diarized_id))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    if let Some(name) = name.filter(|_| !emb.is_empty()) {
        upsert_speaker(paths, locks, &name, &emb);
    }
    record.insert("confirmed".into(), json!(confirmed));
    let _ = atomic_write_json(&meeting_path(paths, meeting_id), &Value::Object(record.clone()));
    Ok(Some(record))
}

/// User said the auto-match is wrong: unlink without touching the library.
/// The display name reverts to a fresh "Speaker N".
pub fn reject_speaker(
    paths: &Paths,
    locks: &StoreLocks,
    meeting_id: &str,
    diarized_id: &str,
) -> Result<Option<Record>, StoreError> {
    let _g = locks.meetings.lock().unwrap();
    let Some(mut record) = get_meeting_unlocked(paths, meeting_id) else {
        return Ok(None);
    };
    let mut recognized = as_string_vec(record.get("recognized"));
    if !recognized.iter().any(|r| r == diarized_id) {
        return Err(StoreError::NotRecognized);
    }
    let mut speakers = record
        .get("speakers")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let taken: Vec<String> =
        speakers.values().filter_map(|v| v.as_str().map(String::from)).collect();
    let mut n = 1;
    while taken.contains(&format!("Speaker {n}")) {
        n += 1;
    }
    let new_name = format!("Speaker {n}");
    let old_name = speakers.get(diarized_id).and_then(|v| v.as_str()).map(String::from);
    speakers.insert(diarized_id.to_string(), json!(new_name));
    if let Some(old_name) = old_name {
        rename_in_notes(&mut record, &old_name, &new_name);
    }
    recognized.retain(|r| r != diarized_id);
    let confirmed: Vec<String> = as_string_vec(record.get("confirmed"))
        .into_iter()
        .filter(|c| c != diarized_id)
        .collect();
    record.insert("speakers".into(), Value::Object(speakers));
    record.insert("recognized".into(), json!(recognized));
    record.insert("confirmed".into(), json!(confirmed));
    let _ = atomic_write_json(&meeting_path(paths, meeting_id), &Value::Object(record.clone()));
    Ok(Some(record))
}

pub fn list_meetings(paths: &Paths, locks: &StoreLocks) -> Vec<Record> {
    let _g = locks.meetings.lock().unwrap();
    let mut items: Vec<Record> = Vec::new();
    let entries = match std::fs::read_dir(paths.meetings_dir()) {
        Ok(e) => e,
        Err(_) => return items,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(rec) = read_json_map(&path) else { continue };
        let mut item = Map::new();
        item.insert("id".into(), rec.get("id").cloned().unwrap_or(Value::Null));
        item.insert("filename".into(), rec.get("filename").cloned().unwrap_or(Value::Null));
        // fallback covers records created before titles existed
        let title = rec
            .get("title")
            .filter(|v| v.as_str().map(|s| !s.is_empty()).unwrap_or(false))
            .or(rec.get("filename"))
            .cloned()
            .unwrap_or(Value::Null);
        item.insert("title".into(), title);
        item.insert("status".into(), rec.get("status").cloned().unwrap_or(Value::Null));
        item.insert("created_at".into(), rec.get("created_at").cloned().unwrap_or(Value::Null));
        items.push(item);
    }
    items.sort_by(|a, b| {
        let ka = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        let kb = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        kb.cmp(ka) // newest first
    });
    items
}

pub fn delete_meeting(paths: &Paths, locks: &StoreLocks, meeting_id: &str) -> bool {
    let _g = locks.meetings.lock().unwrap();
    let path = meeting_path(paths, meeting_id);
    let existed = path.is_file();
    if existed {
        let _ = std::fs::remove_file(&path);
    }
    let upload_dir = paths.uploads_dir().join(meeting_id);
    if upload_dir.exists() {
        let _ = std::fs::remove_dir_all(&upload_dir);
    }
    existed
}

/// The kept audio on disk (extracted track, or the original upload when it was
/// audio-only), or None if it was cleaned up.
pub fn audio_path(paths: &Paths, record: &Record) -> Option<PathBuf> {
    let id = record.get("id")?.as_str()?;
    let upload_dir = paths.uploads_dir().join(id);
    let playback = upload_dir.join(PLAYBACK_FILENAME);
    if playback.is_file() {
        return Some(playback);
    }
    let filename = record.get("filename")?.as_str()?;
    if filename.is_empty() {
        return None;
    }
    let path = upload_dir.join(filename);
    path.is_file().then_some(path)
}

/// Strip internal fields (voiceprints, worker pid, rejob marker) before
/// returning to the client.
pub fn public_meeting(paths: &Paths, record: &Record) -> Record {
    let mut out = Map::new();
    for (k, v) in record {
        if k != "voiceprints" && k != "worker_pid" && k != "rejob" {
            out.insert(k.clone(), v.clone());
        }
    }
    // fallback covers records created before titles existed (no disk migration)
    let title = record
        .get("title")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| record.get("filename").and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
        .unwrap_or("Meeting");
    out.insert("title".into(), json!(title));
    // whether segment playback is possible (recording still on disk)
    out.insert("has_audio".into(), json!(audio_path(paths, record).is_some()));
    out
}

fn as_string_vec(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

// --------------------------------------------------------------------------
// Speaker library (persistent voiceprints for cross-file naming)
// --------------------------------------------------------------------------

pub fn load_speaker_library(paths: &Paths, locks: &StoreLocks) -> Map<String, Value> {
    let _g = locks.speakers.lock().unwrap();
    load_library_unlocked(paths)
}

fn load_library_unlocked(paths: &Paths) -> Map<String, Value> {
    read_json_map(&paths.speakers_file()).unwrap_or_default()
}

pub fn get_speaker_names(paths: &Paths, locks: &StoreLocks) -> Vec<String> {
    let mut names: Vec<String> = load_speaker_library(paths, locks).keys().cloned().collect();
    names.sort();
    names
}

/// Per-person cap on stored voice samples. The matching embedding is their
/// mean, so each confirmed recording refines the voiceprint; the cap keeps one
/// person from growing unbounded while still spanning different mics and rooms.
const MAX_SAMPLES: usize = 10;

/// Enrol one heard voice under `name`, refining any existing voiceprint.
pub fn upsert_speaker(paths: &Paths, locks: &StoreLocks, name: &str, embedding: &[f64]) {
    let _g = locks.speakers.lock().unwrap();
    let mut lib = load_library_unlocked(paths);
    let entry = lib.get(name).and_then(|v| v.as_object()).cloned().unwrap_or_default();
    let mut samples: Vec<Vec<f64>> = entry
        .get("samples")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| {
                    s.as_array().map(|xs| xs.iter().filter_map(|x| x.as_f64()).collect())
                })
                .collect()
        })
        .unwrap_or_default();
    if samples.is_empty() {
        // Legacy entry with only an embedding upgrades to samples=[embedding].
        if let Some(emb) = entry.get("embedding").and_then(|v| v.as_array()) {
            if !emb.is_empty() {
                samples.push(emb.iter().filter_map(|x| x.as_f64()).collect());
            }
        }
    }
    // A changed embedding model means old samples can't be averaged in.
    samples.retain(|s| s.len() == embedding.len());
    samples.push(embedding.to_vec());
    if samples.len() > MAX_SAMPLES {
        samples = samples.split_off(samples.len() - MAX_SAMPLES);
    }
    let dim = embedding.len();
    let mut centroid = vec![0.0f64; dim];
    for s in &samples {
        for (c, x) in centroid.iter_mut().zip(s) {
            *c += x;
        }
    }
    for c in &mut centroid {
        *c /= samples.len() as f64;
    }
    lib.insert(
        name.to_string(),
        json!({"embedding": centroid, "samples": samples, "updated_at": now_iso()}),
    );
    let _ = atomic_write_json(&paths.speakers_file(), &Value::Object(lib));
}

/// Move a voiceprint to a new name, keeping the embedding.
/// Returns "ok" | "missing" | "exists". Only affects future recognition.
pub fn rename_speaker(paths: &Paths, locks: &StoreLocks, old: &str, new: &str) -> &'static str {
    let _g = locks.speakers.lock().unwrap();
    let mut lib = load_library_unlocked(paths);
    if !lib.contains_key(old) {
        return "missing";
    }
    if lib.contains_key(new) && new != old {
        return "exists";
    }
    let entry = lib.remove(old).unwrap();
    lib.insert(new.to_string(), entry);
    let _ = atomic_write_json(&paths.speakers_file(), &Value::Object(lib));
    "ok"
}

pub fn delete_speaker(paths: &Paths, locks: &StoreLocks, name: &str) -> bool {
    let _g = locks.speakers.lock().unwrap();
    let mut lib = load_library_unlocked(paths);
    if lib.remove(name).is_some() {
        let _ = atomic_write_json(&paths.speakers_file(), &Value::Object(lib));
        true
    } else {
        false
    }
}
