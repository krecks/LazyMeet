"""FastAPI sidecar: HTTP API + background transcription pipeline.

Started by the Tauri shell. On launch it binds a localhost port, prints
``READY <port>`` to stdout (the Rust side waits for this line), then serves the
API and the static frontend. All heavy work runs in a background worker so the
UI can poll for progress.
"""

from __future__ import annotations

import io
import os
import re
import shutil
from collections import deque
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, models_setup, store
from .pipeline import audio, diarize, identify, merge, summarize, transcribe

@asynccontextmanager
async def _lifespan(app: FastAPI):
    config.ensure_dirs()
    # Uploads made while the first-run download is still going sit in the
    # processing queue; pump it the moment the models land.
    models_setup.set_ready_callback(_pump_queue)
    models_setup.ensure_models_async()
    _sweep_orphans()
    yield


app = FastAPI(title="LazyMeet", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # localhost desktop app
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Setup / health
# --------------------------------------------------------------------------- #
@app.get("/api/setup")
def get_setup():
    return models_setup.status()


@app.post("/api/setup/retry")
def retry_setup():
    """Restart failed model downloads (no-op if a download is already running)."""
    models_setup.ensure_models_async()
    return models_setup.status()


@app.get("/api/health")
def health():
    return {"ok": True, "fake": config.FAKE_MODE}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
# "auto" + everything the summarizer knows a name for; the same codes are valid
# Whisper language hints.
SUPPORTED_LANGUAGES = ("auto",) + tuple(summarize.LANGUAGE_NAMES)


class ConfigUpdate(BaseModel):
    """All fields optional; an explicit null removes that override."""

    ASR_MODEL: str | None = None
    ASR_LANGUAGE: str | None = None
    SUMMARY_MODEL: str | None = None
    SUMMARY_LANGUAGE: str | None = None
    SPEAKER_THRESHOLD: float | None = Field(None, ge=0.0, le=1.0)
    DIAR_THRESHOLD: float | None = Field(None, ge=0.0, le=1.0)
    SPEAKER_MERGE_THRESHOLD: float | None = Field(None, ge=0.0, le=1.0)
    MIN_SPEAKER_SECONDS: float | None = Field(None, ge=0.0, le=600.0)
    NUM_THREADS: int | None = Field(None, ge=1, le=config.TUNABLES["NUM_THREADS"][4])
    USE_GPU: bool | None = None
    KEEP_AUDIO: bool | None = None


def _config_payload() -> dict:
    return {
        "values": {name: getattr(config, name) for name in config.TUNABLES},
        "overrides": config.load_overrides(),
        "defaults": {name: spec[2] for name, spec in config.TUNABLES.items()},
        # Set by env var, which outranks config.json — the UI locks these.
        "env_locked": config.env_locked(),
        "limits": {
            name: {"min": spec[3], "max": spec[4]}
            for name, spec in config.TUNABLES.items()
        },
    }


@app.get("/api/config")
def get_config():
    return _config_payload()


@app.put("/api/config")
def put_config(body: ConfigUpdate):
    fields = body.model_dump(exclude_unset=True)
    if fields.get("ASR_MODEL") is not None and fields["ASR_MODEL"] not in models_setup.ASR_CATALOG:
        raise HTTPException(422, "unknown transcription model")
    if (
        fields.get("SUMMARY_MODEL") is not None
        and fields["SUMMARY_MODEL"] not in models_setup.LLM_CATALOG
    ):
        raise HTTPException(422, "unknown summary model")
    for key in ("ASR_LANGUAGE", "SUMMARY_LANGUAGE"):
        if fields.get(key) is not None and fields[key] not in SUPPORTED_LANGUAGES:
            raise HTTPException(422, f"unsupported language for {key}")

    overrides = config.load_overrides()
    for name, value in fields.items():
        if value is None:
            overrides.pop(name, None)
        else:
            overrides[name] = value
    config.save_overrides(overrides)
    config.refresh()
    # A newly selected model that isn't on disk yet starts downloading now; the
    # pipeline uses the default until it lands.
    models_setup.ensure_selected_async()
    return _config_payload()


@app.get("/api/models")
def get_models():
    return models_setup.models_status()


class ModelDownload(BaseModel):
    step: str  # "asr" | "summary"
    id: str


@app.post("/api/models/download")
def download_model(body: ModelDownload):
    if not models_setup.request_download(body.step, body.id):
        raise HTTPException(404, "unknown model")
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Meetings
# --------------------------------------------------------------------------- #
@app.post("/api/meetings")
async def create_meeting(file: UploadFile, num_speakers: int = Form(-1)):
    record = store.create_meeting(file.filename or "recording", num_speakers=num_speakers)
    upload_dir = config.UPLOADS_DIR / record["id"]
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (file.filename or "recording")
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    _dispatch(record["id"], dest)
    return {"id": record["id"]}


@app.get("/api/meetings")
def list_meetings():
    return store.list_meetings()


@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    record = store.get_meeting(meeting_id)
    if record is None:
        raise HTTPException(404, "meeting not found")
    return store.public_meeting(record)


@app.get("/api/meetings/{meeting_id}/audio")
def get_meeting_audio(meeting_id: str):
    """The kept audio, for in-transcript segment playback.

    404s when it is gone (pre-existing meetings, or KEEP_AUDIO turned off);
    the UI only shows play buttons when ``has_audio`` says it is here.
    FileResponse handles Range requests, which the webview needs to seek.
    """
    record = store.get_meeting(meeting_id)
    if record is None:
        raise HTTPException(404, "meeting not found")
    path = store.audio_path(record)
    if path is None:
        raise HTTPException(404, "recording no longer on disk")
    name = record.get("filename") or "recording"
    if path.name == store.PLAYBACK_FILENAME:
        # video uploads are reduced to their audio track after processing
        name = str(Path(name).with_suffix(".m4a"))
    return FileResponse(path, filename=name)


class MeetingPatch(BaseModel):
    title: str | None = None
    summary: str | None = None
    takeaways: list[str] | None = None
    action_items: list[str] | None = None


@app.patch("/api/meetings/{meeting_id}")
def patch_meeting(meeting_id: str, body: MeetingPatch):
    """Edit user-facing fields.

    The title may change in any status (the worker never writes it), but
    summary / takeaways / action items only once the meeting is ``done`` —
    before that the worker's terminal write would clobber the edit, and an
    ``error`` meeting shows its error message in the summary slot.
    """
    fields = body.model_dump(exclude_unset=True)
    changes: dict = {}
    if "title" in fields:
        title = (fields["title"] or "").strip()
        if not 1 <= len(title) <= 200:
            raise HTTPException(422, "title must be 1-200 characters")
        changes["title"] = title
    content_edit = False
    if "summary" in fields:
        changes["summary"] = (fields["summary"] or "").strip()
        content_edit = True
    for key in ("takeaways", "action_items"):
        if key in fields:
            items = [s.strip() for s in (fields[key] or []) if s and s.strip()]
            changes[key] = items[:100]
            content_edit = True
    if not changes:
        raise HTTPException(422, "no fields to update")
    try:
        updated = store.patch_meeting(
            meeting_id,
            allowed_statuses=("done",) if content_edit else None,
            **changes,
        )
    except store.StatusConflict as exc:
        raise HTTPException(409, f"meeting is not editable yet (status: {exc.status})")
    if updated is None:
        raise HTTPException(404, "meeting not found")
    return store.public_meeting(updated)


class SegmentPatch(BaseModel):
    text: str | None = None
    speaker: str | None = None  # must be an existing diarized id


@app.patch("/api/meetings/{meeting_id}/segments/{index}")
def patch_segment(meeting_id: str, index: int, body: SegmentPatch):
    fields = body.model_dump(exclude_unset=True)
    if "text" in fields:
        fields["text"] = (fields["text"] or "").strip()
        if not fields["text"]:
            raise HTTPException(422, "segment text must not be empty")
    if not fields:
        raise HTTPException(422, "no fields to update")
    try:
        updated = store.update_segment(meeting_id, index, **fields)
    except store.StatusConflict as exc:
        raise HTTPException(409, f"meeting is not editable yet (status: {exc.status})")
    except store.UnknownSpeaker:
        raise HTTPException(422, "unknown speaker id")
    except IndexError:
        raise HTTPException(404, "segment not found")
    if updated is None:
        raise HTTPException(404, "meeting not found")
    return store.public_meeting(updated)


class SpeakerRename(BaseModel):
    names: dict[str, str]  # diarized id -> new display name


@app.patch("/api/meetings/{meeting_id}/speakers")
def rename_speakers(meeting_id: str, body: SpeakerRename):
    updated = store.merge_speaker_names(meeting_id, body.names)
    if updated is None:
        raise HTTPException(404, "meeting not found")
    return store.public_meeting(updated)


class SpeakerVerify(BaseModel):
    correct: bool


@app.post("/api/meetings/{meeting_id}/speakers/{diarized_id}/verify")
def verify_speaker(meeting_id: str, diarized_id: str, body: SpeakerVerify):
    """Confirm or reject an auto-recognized speaker.

    Confirming enrols this recording's voice as another sample for the person,
    so recognition learns from every confirmed file. Rejecting unlinks the
    name without touching the library — a bad match can't poison the saved
    voiceprint — and reverts the display name to a fresh "Speaker N".
    """
    try:
        action = store.confirm_speaker if body.correct else store.reject_speaker
        updated = action(meeting_id, diarized_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    if updated is None:
        raise HTTPException(404, "meeting not found")
    return store.public_meeting(updated)


@app.delete("/api/meetings/{meeting_id}")
def delete_meeting(meeting_id: str):
    if not store.delete_meeting(meeting_id):
        raise HTTPException(404, "meeting not found")
    return {"deleted": True}


@app.get("/api/meetings/{meeting_id}/export")
def export_meeting(meeting_id: str):
    record = store.get_meeting(meeting_id)
    if record is None:
        raise HTTPException(404, "meeting not found")
    md = _to_markdown(record)
    title = record.get("title") or record.get("filename") or meeting_id
    safe = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "-") or meeting_id
    return PlainTextResponse(
        md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe}.md"'},
    )


# --------------------------------------------------------------------------- #
# Speaker library
# --------------------------------------------------------------------------- #
@app.get("/api/speakers")
def get_speakers():
    return store.get_speaker_names()


class SpeakerPatch(BaseModel):
    name: str


@app.patch("/api/speakers/{name}")
def rename_known_speaker(name: str, body: SpeakerPatch):
    """Rename a library entry, keeping its voiceprint (future recordings only)."""
    new = body.name.strip()
    if not 1 <= len(new) <= 100:
        raise HTTPException(422, "name must be 1-100 characters")
    result = store.rename_speaker(name, new)
    if result == "missing":
        raise HTTPException(404, "speaker not found")
    if result == "exists":
        raise HTTPException(409, "a speaker with that name already exists")
    return store.get_speaker_names()


@app.delete("/api/speakers/{name}")
def delete_speaker(name: str):
    if not store.delete_speaker(name):
        raise HTTPException(404, "speaker not found")
    return {"deleted": True}


# --------------------------------------------------------------------------- #
# Data management
# --------------------------------------------------------------------------- #
def _dir_size(path: Path) -> int:
    total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += _dir_size(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        pass
    return total


@app.get("/api/data")
def get_data_info():
    sizes = {
        "meetings_bytes": _dir_size(config.MEETINGS_DIR),
        "uploads_bytes": _dir_size(config.UPLOADS_DIR),
        "models_bytes": _dir_size(config.MODELS_DIR),
    }
    return {
        "data_dir": str(config.DATA_DIR),
        **sizes,
        "total_bytes": sum(sizes.values()),
        "meeting_count": len(store.list_meetings()),
    }


@app.post("/api/data/reveal")
def reveal_data_dir():
    """Open the app-data folder in the OS file manager.

    Done here rather than in the webview because no Tauri shell/fs plugin is
    enabled. The path is server-fixed — never taken from the request.
    """
    target = str(config.DATA_DIR)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", target])
        elif sys.platform == "win32":
            os.startfile(target)  # noqa: S606 - fixed local path
        else:
            subprocess.Popen(["xdg-open", target])
    except OSError as exc:
        raise HTTPException(500, f"could not open folder: {exc}")
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Pipeline worker
# --------------------------------------------------------------------------- #
# Uploads are processed strictly one at a time: the models are heavy enough
# that two concurrent workers would thrash CPU/GPU and memory. New meetings
# wait in this queue with status "queued" until the active one finishes.
_queue: deque = deque()
_queue_lock = threading.Lock()
_active_meeting: str | None = None


def _dispatch(meeting_id: str, upload_path: Path) -> None:
    """Queue a meeting for processing; starts immediately if nothing is running."""
    with _queue_lock:
        _queue.append((meeting_id, Path(upload_path)))
    _pump_queue()


def _pump_queue() -> None:
    global _active_meeting
    # Nothing can process until the first-run models exist; the ready callback
    # re-pumps once the download completes.
    if not models_setup.is_ready():
        return
    with _queue_lock:
        if _active_meeting is not None:
            return
        while _queue:
            meeting_id, upload_path = _queue.popleft()
            record = store.get_meeting(meeting_id)
            # Skip entries deleted while they waited.
            if record is not None and record.get("status") == "queued":
                _active_meeting = meeting_id
                break
        else:
            return
    _start_worker(meeting_id, upload_path)


def _finish_active(meeting_id: str) -> None:
    global _active_meeting
    with _queue_lock:
        if _active_meeting == meeting_id:
            _active_meeting = None
    _pump_queue()


def _start_worker(meeting_id: str, upload_path: Path) -> None:
    """Start processing one meeting without blocking the server.

    The real pipeline runs in a separate OS process: sherpa-onnx and llama.cpp
    make long native calls that hold the Python GIL, which would otherwise
    freeze this server's event loop and stall the UI's progress polling. The
    file-based store lets the worker report progress that the server just reads.
    """
    if config.FAKE_MODE:
        def _run():
            try:
                run_worker(meeting_id, str(upload_path))
            finally:
                _finish_active(meeting_id)

        threading.Thread(target=_run, daemon=True).start()
        return

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--worker", meeting_id, str(upload_path)]
    else:
        cmd = [sys.executable, "-m", "backend.main", "--worker", meeting_id, str(upload_path)]
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    store.update_meeting(meeting_id, worker_pid=proc.pid)
    threading.Thread(target=_monitor_worker, args=(meeting_id, proc), daemon=True).start()


def _monitor_worker(meeting_id: str, proc: subprocess.Popen) -> None:
    """Reap the worker; fail the meeting if it died before finishing.

    A segfault in native code (sherpa-onnx, llama.cpp) kills the worker before
    it can write a terminal status, which would otherwise leave the meeting
    "processing" forever — and the child a zombie.
    """
    rc = proc.wait()
    record = store.get_meeting(meeting_id)
    if record and record.get("status") not in ("done", "error"):
        store.update_meeting(
            meeting_id, status="error", error=f"processing crashed (exit code {rc})"
        )
    _finish_active(meeting_id)


def _sweep_orphans() -> None:
    """On startup, fail meetings whose worker died while the app wasn't running.

    Meetings still in "queued" whose upload survived never started (they were
    waiting in the processing queue when the app quit) — resume those instead.
    """
    for item in store.list_meetings():
        if item.get("status") in ("done", "error"):
            continue
        record = store.get_meeting(item["id"]) or {}
        if record.get("status") == "queued" and record.get("filename"):
            upload = config.UPLOADS_DIR / item["id"] / record["filename"]
            if upload.exists():
                _dispatch(item["id"], upload)
                continue
        pid = record.get("worker_pid")
        if pid:
            try:
                os.kill(pid, 0)  # probe — does not actually signal
                continue
            except OSError:
                pass
        store.update_meeting(
            item["id"], status="error", error="processing was interrupted"
        )


def run_worker(meeting_id: str, upload_path: str) -> None:
    # Pick up the latest settings. Real workers are fresh processes and get
    # them at import, but FAKE_MODE workers are threads in the server process,
    # whose config would otherwise be frozen at server start.
    config.refresh()
    path = Path(upload_path)
    record = store.get_meeting(meeting_id) or {}
    num_speakers = int(record.get("num_speakers", -1))
    try:
        store.update_meeting(meeting_id, status="extracting")
        wav = config.UPLOADS_DIR / meeting_id / "audio.wav"
        audio.extract_audio(path, wav)

        store.update_meeting(meeting_id, status="transcribing")
        segments = transcribe.transcribe(wav)

        store.update_meeting(meeting_id, status="diarizing")
        turns = diarize.diarize(wav, num_speakers=num_speakers)

        store.update_meeting(meeting_id, status="identifying")
        voiceprints = identify.compute_voiceprints(wav, turns)
        turns, voiceprints = identify.consolidate_speakers(turns, voiceprints)
        speakers, recognized = identify.match_speakers(voiceprints)

        merged = merge.assign_speakers(segments, turns)

        store.update_meeting(meeting_id, preserve_speakers=True, status="summarizing",
                             segments=merged, speakers=speakers, recognized=recognized,
                             voiceprints=voiceprints)

        # A summary failure (e.g. an unusually long meeting) must not throw away
        # the transcript — mark the meeting done with a note instead.
        try:
            result = summarize.summarize(merged, speakers)
            store.update_meeting(
                meeting_id,
                status="done",
                summary=result["summary"],
                takeaways=result["takeaways"],
                action_items=result["action_items"],
            )
        except Exception as exc:
            store.update_meeting(
                meeting_id,
                status="done",
                summary=f"(Automatic summary unavailable: {exc})",
                takeaways=[],
                action_items=[],
            )

        # The extracted WAV is a processing intermediate either way. With
        # KEEP_AUDIO (default) only the audio stays for segment playback: an
        # audio-only upload is kept as-is, a video upload is reduced to its
        # audio track and the much larger video deleted. With it off
        # everything goes — export works off the JSON record, re-enrolment
        # off stored voiceprints. Everything is kept on error so a failed
        # run can be debugged or reprocessed.
        wav.unlink(missing_ok=True)
        if not config.KEEP_AUDIO:
            shutil.rmtree(config.UPLOADS_DIR / meeting_id, ignore_errors=True)
        elif path.is_file() and not audio.is_audio_file(path):
            try:
                audio.extract_playback_audio(path, path.parent / store.PLAYBACK_FILENAME)
                path.unlink(missing_ok=True)
            except Exception:
                pass  # keep the video rather than lose playback entirely
    except Exception as exc:
        store.update_meeting(meeting_id, status="error", error=str(exc))


def _to_markdown(record: dict) -> str:
    speakers = record.get("speakers", {})
    out = io.StringIO()
    out.write(f"# {record.get('title') or record.get('filename') or 'Meeting'}\n\n")
    if record.get("summary"):
        out.write("## Summary\n\n" + record["summary"] + "\n\n")
    if record.get("takeaways"):
        out.write("## Key takeaways\n\n")
        for t in record["takeaways"]:
            out.write(f"- {t}\n")
        out.write("\n")
    if record.get("action_items"):
        out.write("## Action items\n\n")
        for a in record["action_items"]:
            out.write(f"- [ ] {a}\n")
        out.write("\n")
    out.write("## Transcript\n\n")
    for seg in record.get("segments", []):
        name = speakers.get(seg["speaker"], seg["speaker"])
        ts = _fmt_ts(seg["start"])
        out.write(f"**{name}** [{ts}]: {seg['text']}\n\n")
    return out.getvalue()


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# Serve the static frontend (also used for browser-based dev/testing).
_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def _watch_parent(on_dead) -> None:
    """Run ``on_dead`` once the Tauri app (``LAZYMEET_PARENT_PID``) disappears."""
    raw = os.environ.get("LAZYMEET_PARENT_PID", "")
    parent_pid = int(raw) if raw.isdigit() else None

    def _loop():
        while True:
            time.sleep(1.0)
            gone = False
            if parent_pid is not None:
                try:
                    os.kill(parent_pid, 0)  # probe — does not actually signal
                except OSError:
                    gone = True
            elif os.getppid() == 1:
                gone = True
            if gone:
                on_dead()

    threading.Thread(target=_loop, daemon=True).start()


def _install_exit_guard() -> None:
    """Tear the whole sidecar process tree down when the Tauri app exits.

    The frozen sidecar is a PyInstaller bootloader + child, and it spawns worker
    subprocesses of its own, so simply killing the process Tauri launched leaves
    orphans. Instead we put the server (and therefore every worker it spawns,
    which inherit the group) into a fresh process group, then watch the Tauri
    app's PID; when it disappears we signal the entire group at once. This runs
    only for the managed sidecar (``LAZYMEET_PARENT_PID`` is set by the Rust
    shell), so plain ``python -m backend.main`` dev runs are unaffected.
    """
    try:
        os.setpgrp()  # new process group, pgid == our pid; workers inherit it
    except OSError:
        pass

    def _shutdown(*_):
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    _watch_parent(_shutdown)


def _pick_port() -> int:
    env_port = os.environ.get("LAZYMEET_PORT")
    if env_port:
        return int(env_port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    # Worker mode: re-invoked as a subprocess to process one meeting, then exit.
    args = sys.argv[1:]
    if args and args[0] == "--worker":
        config.ensure_dirs()
        # Belt-and-suspenders: if this worker ever escapes the server's process
        # group, it still self-terminates within ~1s of the app exiting.
        if os.environ.get("LAZYMEET_PARENT_PID"):
            _watch_parent(lambda: os._exit(0))
        run_worker(args[1], args[2])
        return

    # Managed by the Tauri shell: clean up the whole tree when the app exits.
    if os.environ.get("LAZYMEET_PARENT_PID"):
        _install_exit_guard()

    port = _pick_port()
    # The Tauri shell waits for this exact line before loading the UI.
    print(f"READY {port}", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
