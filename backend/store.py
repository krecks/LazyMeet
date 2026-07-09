"""JSON-file persistence for meetings and the cross-file speaker library.

There is no database: each meeting is a single JSON file under ``meetings/`` and
the speaker library is one ``speakers.json``. Writes are atomic (temp file +
``os.replace``) so readers never see a half-written file, and read-modify-writes
are guarded by an in-process lock *and* a per-file lock file — the pipeline
worker runs in a separate OS process, where a ``threading`` lock alone would not
prevent lost updates (e.g. a speaker rename landing while the worker writes).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config

_lock = threading.RLock()


class StatusConflict(Exception):
    """A mutation was rejected because the meeting is in the wrong status."""

    def __init__(self, status: Optional[str]):
        super().__init__(f"meeting status is {status!r}")
        self.status = status


class UnknownSpeaker(ValueError):
    """A segment was assigned a speaker id the meeting doesn't know."""


@contextmanager
def _os_lock(path: Path, timeout: float = 10.0, stale: float = 30.0):
    """Cross-process mutual exclusion for one file's read-modify-write.

    Uses an ``O_CREAT | O_EXCL`` lock file, which is portable (``fcntl`` is
    POSIX-only). Lock files older than ``stale`` seconds are treated as left
    behind by a crashed holder and broken. Best-effort by design: on timeout we
    proceed unlocked rather than deadlock the UI.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            time.sleep(0.05)
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        acquired = True
        break
    try:
        yield
    finally:
        if acquired:
            lock.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _meeting_path(meeting_id: str) -> Path:
    return config.MEETINGS_DIR / f"{meeting_id}.json"


# --------------------------------------------------------------------------- #
# Meetings
# --------------------------------------------------------------------------- #
def create_meeting(filename: str, num_speakers: int = -1) -> dict:
    meeting_id = uuid.uuid4().hex
    record = {
        "id": meeting_id,
        "filename": filename,
        "title": filename,  # user-editable; filename stays the immutable original
        "created_at": _now(),
        "status": "queued",
        "error": None,
        "num_speakers": num_speakers,  # -1 = auto-detect
        "speakers": {},      # diarized id -> display name
        "recognized": [],    # diarized ids auto-matched from the speaker library
        "segments": [],      # [{start, end, speaker, text}]
        "summary": "",
        "takeaways": [],
        "action_items": [],
        # transient voiceprints captured during identification, used when the
        # user later renames a speaker so we can enrol that voice.
        "voiceprints": {},   # diarized id -> [floats]
    }
    with _lock:
        config.ensure_dirs()
        _atomic_write(_meeting_path(meeting_id), record)
    return record


def get_meeting(meeting_id: str) -> Optional[dict]:
    path = _meeting_path(meeting_id)
    if not path.exists():
        return None
    with _lock:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None


def update_meeting(
    meeting_id: str, *, preserve_speakers: bool = False, **changes: Any
) -> Optional[dict]:
    """Re-read, apply ``changes``, and write back under both locks.

    ``preserve_speakers=True`` makes existing ``speakers`` entries win over the
    passed ones — used by the worker so its diarization defaults never overwrite
    names the user set in the meantime.
    """
    with _lock, _os_lock(_meeting_path(meeting_id)):
        record = get_meeting(meeting_id)
        if record is None:
            return None
        if preserve_speakers and "speakers" in changes:
            changes = dict(changes)
            changes["speakers"] = {**changes["speakers"], **record.get("speakers", {})}
        record.update(changes)
        _atomic_write(_meeting_path(meeting_id), record)
        return record


def patch_meeting(
    meeting_id: str,
    *,
    allowed_statuses: Optional[tuple[str, ...]] = None,
    **changes: Any,
) -> Optional[dict]:
    """User-initiated edit: locked read-modify-write with a status guard.

    The guard must run *inside* the locks — a route-level status check would
    race the worker's terminal write (e.g. a summary edit landing just as the
    worker writes its own summary would survive it and then be clobbered, or
    worse, clobber it).
    """
    with _lock, _os_lock(_meeting_path(meeting_id)):
        record = get_meeting(meeting_id)
        if record is None:
            return None
        if allowed_statuses is not None and record.get("status") not in allowed_statuses:
            raise StatusConflict(record.get("status"))
        record.update(changes)
        _atomic_write(_meeting_path(meeting_id), record)
        return record


def update_segment(
    meeting_id: str,
    index: int,
    *,
    text: Optional[str] = None,
    speaker: Optional[str] = None,
) -> Optional[dict]:
    """Edit one transcript segment in place, under both locks.

    Must own the whole read-modify-write: composing this from
    ``update_meeting(segments=...)`` at the route would drop concurrent edits
    to other segments. Reassigning a speaker deliberately leaves ``speakers``
    and ``voiceprints`` untouched — an id whose last segment moves away simply
    stops appearing in the legend (which derives from segments); deleting it
    from the map would break ``recognized`` and rename history.
    """
    with _lock, _os_lock(_meeting_path(meeting_id)):
        record = get_meeting(meeting_id)
        if record is None:
            return None
        if record.get("status") != "done":
            raise StatusConflict(record.get("status"))
        segments = record.get("segments", [])
        if not 0 <= index < len(segments):
            raise IndexError(index)
        if speaker is not None:
            if speaker not in record.get("speakers", {}):
                raise UnknownSpeaker(speaker)
            segments[index]["speaker"] = speaker
        if text is not None:
            segments[index]["text"] = text
        _atomic_write(_meeting_path(meeting_id), record)
        return record


def _rename_in_notes(record: dict, old: str, new: str) -> None:
    """Rewrite one speaker's display name inside the generated notes.

    The summary text bakes in the names it was generated with, so a rename
    must follow it there too. Guarded lookarounds keep "Speaker 1" from also
    matching inside "Speaker 10".
    """
    pattern = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)")
    record["summary"] = pattern.sub(new, record.get("summary", ""))
    record["takeaways"] = [pattern.sub(new, t) for t in record.get("takeaways", [])]
    record["action_items"] = [pattern.sub(new, a) for a in record.get("action_items", [])]


def merge_speaker_names(meeting_id: str, names: dict) -> Optional[dict]:
    """Apply user renames and enrol those voices in the speaker library.

    The whole read-modify-write happens under the locks so concurrent renames
    (or a worker write) can't drop each other's names.
    """
    with _lock, _os_lock(_meeting_path(meeting_id)):
        record = get_meeting(meeting_id)
        if record is None:
            return None
        speakers = dict(record.get("speakers", {}))
        voiceprints = record.get("voiceprints", {})
        for diarized_id, new_name in names.items():
            new_name = new_name.strip()
            if not new_name:
                continue
            old_name = speakers.get(diarized_id)
            speakers[diarized_id] = new_name
            if old_name and old_name != new_name:
                _rename_in_notes(record, old_name, new_name)
                # Move the library entry rather than leaving a duplicate
                # person behind under the old name. If the new name already
                # exists, the voice merges into it and the old entry goes.
                if rename_speaker(old_name, new_name) == "exists":
                    delete_speaker(old_name)
            # Enrol this voice so the name follows it into future recordings.
            emb = voiceprints.get(diarized_id)
            if emb:
                upsert_speaker(new_name, emb)
        record["speakers"] = speakers
        _atomic_write(_meeting_path(meeting_id), record)
        return record


def list_meetings() -> list[dict]:
    with _lock:
        items = []
        for path in config.MEETINGS_DIR.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            items.append(
                {
                    "id": rec.get("id"),
                    "filename": rec.get("filename"),
                    # fallback covers records created before titles existed
                    "title": rec.get("title") or rec.get("filename"),
                    "status": rec.get("status"),
                    "created_at": rec.get("created_at"),
                }
            )
    items.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return items


def delete_meeting(meeting_id: str) -> bool:
    with _lock, _os_lock(_meeting_path(meeting_id)):
        path = _meeting_path(meeting_id)
        existed = path.exists()
        if existed:
            path.unlink()
        upload_dir = config.UPLOADS_DIR / meeting_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        return existed


# Audio track extracted from a video upload once processing succeeds; the
# original video is deleted then, so playback prefers this file.
PLAYBACK_FILENAME = "playback.m4a"


def audio_path(record: dict) -> Optional[Path]:
    """The kept audio on disk (extracted track, or the original upload when it
    was audio-only), or None if it was cleaned up."""
    if not record.get("id"):
        return None
    upload_dir = config.UPLOADS_DIR / record["id"]
    playback = upload_dir / PLAYBACK_FILENAME
    if playback.is_file():
        return playback
    filename = record.get("filename")
    if not filename:
        return None
    path = upload_dir / filename
    return path if path.is_file() else None


def public_meeting(record: dict) -> dict:
    """Strip internal fields (voiceprints, worker pid) before returning to the client."""
    out = {k: v for k, v in record.items() if k not in ("voiceprints", "worker_pid")}
    # fallback covers records created before titles existed (no disk migration)
    out["title"] = record.get("title") or record.get("filename") or "Meeting"
    # whether segment playback is possible (recording still on disk)
    out["has_audio"] = audio_path(record) is not None
    return out


# --------------------------------------------------------------------------- #
# Speaker library (persistent voiceprints for cross-file naming)
# --------------------------------------------------------------------------- #
def load_speaker_library() -> dict:
    with _lock:
        if not config.SPEAKERS_FILE.exists():
            return {}
        try:
            with open(config.SPEAKERS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}


def get_speaker_names() -> list[str]:
    return sorted(load_speaker_library().keys())


def upsert_speaker(name: str, embedding: list[float]) -> None:
    with _lock:
        lib = load_speaker_library()
        lib[name] = {"embedding": list(embedding), "updated_at": _now()}
        _atomic_write(config.SPEAKERS_FILE, lib)


def rename_speaker(old: str, new: str) -> str:
    """Move a voiceprint to a new name, keeping the embedding.

    Only affects future recognition — meetings keep the names they were saved
    with. Returns "ok" | "missing" | "exists".
    """
    with _lock:
        lib = load_speaker_library()
        if old not in lib:
            return "missing"
        if new in lib and new != old:
            return "exists"
        lib[new] = lib.pop(old)
        _atomic_write(config.SPEAKERS_FILE, lib)
        return "ok"


def delete_speaker(name: str) -> bool:
    with _lock:
        lib = load_speaker_library()
        if name in lib:
            del lib[name]
            _atomic_write(config.SPEAKERS_FILE, lib)
            return True
        return False
