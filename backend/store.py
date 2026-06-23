"""JSON-file persistence for meetings and the cross-file speaker library.

There is no database: each meeting is a single JSON file under ``meetings/`` and
the speaker library is one ``speakers.json``. Writes are atomic (temp file +
``os.replace``) and guarded by a re-entrant lock so the request handlers and the
background worker never see a half-written file.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config

_lock = threading.RLock()


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
def create_meeting(filename: str) -> dict:
    meeting_id = uuid.uuid4().hex
    record = {
        "id": meeting_id,
        "filename": filename,
        "created_at": _now(),
        "status": "queued",
        "error": None,
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
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def update_meeting(meeting_id: str, **changes: Any) -> Optional[dict]:
    with _lock:
        record = get_meeting(meeting_id)
        if record is None:
            return None
        record.update(changes)
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
                    "status": rec.get("status"),
                    "created_at": rec.get("created_at"),
                }
            )
    items.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return items


def delete_meeting(meeting_id: str) -> bool:
    with _lock:
        path = _meeting_path(meeting_id)
        existed = path.exists()
        if existed:
            path.unlink()
        upload_dir = config.UPLOADS_DIR / meeting_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        return existed


def public_meeting(record: dict) -> dict:
    """Strip internal fields (voiceprints) before returning to the client."""
    return {k: v for k, v in record.items() if k != "voiceprints"}


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


def delete_speaker(name: str) -> bool:
    with _lock:
        lib = load_speaker_library()
        if name in lib:
            del lib[name]
            _atomic_write(config.SPEAKERS_FILE, lib)
            return True
        return False
