"""FastAPI sidecar: HTTP API + background transcription pipeline.

Started by the Tauri shell. On launch it binds a localhost port, prints
``READY <port>`` to stdout (the Rust side waits for this line), then serves the
API and the static frontend. All heavy work runs in a background worker so the
UI can poll for progress.
"""

from __future__ import annotations

import io
import os
import shutil
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
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, models_setup, store
from .pipeline import audio, diarize, identify, merge, summarize, transcribe

@asynccontextmanager
async def _lifespan(app: FastAPI):
    config.ensure_dirs()
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


class SpeakerRename(BaseModel):
    names: dict[str, str]  # diarized id -> new display name


@app.patch("/api/meetings/{meeting_id}/speakers")
def rename_speakers(meeting_id: str, body: SpeakerRename):
    updated = store.merge_speaker_names(meeting_id, body.names)
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
    return PlainTextResponse(
        md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{meeting_id}.md"'},
    )


# --------------------------------------------------------------------------- #
# Speaker library
# --------------------------------------------------------------------------- #
@app.get("/api/speakers")
def get_speakers():
    return store.get_speaker_names()


@app.delete("/api/speakers/{name}")
def delete_speaker(name: str):
    if not store.delete_speaker(name):
        raise HTTPException(404, "speaker not found")
    return {"deleted": True}


# --------------------------------------------------------------------------- #
# Pipeline worker
# --------------------------------------------------------------------------- #
def _dispatch(meeting_id: str, upload_path: Path) -> None:
    """Start processing a meeting without blocking the server.

    The real pipeline runs in a separate OS process: sherpa-onnx and llama.cpp
    make long native calls that hold the Python GIL, which would otherwise
    freeze this server's event loop and stall the UI's progress polling. The
    file-based store lets the worker report progress that the server just reads.
    """
    if config.FAKE_MODE:
        threading.Thread(
            target=run_worker, args=(meeting_id, str(upload_path)), daemon=True
        ).start()
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


def _sweep_orphans() -> None:
    """On startup, fail meetings whose worker died while the app wasn't running."""
    for item in store.list_meetings():
        if item.get("status") in ("done", "error"):
            continue
        record = store.get_meeting(item["id"]) or {}
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

        # The recording is no longer needed (export works off the JSON record,
        # re-enrolment off stored voiceprints) — drop it for privacy and disk.
        # Kept on error so a failed run can be debugged or reprocessed.
        if not config.KEEP_AUDIO:
            shutil.rmtree(config.UPLOADS_DIR / meeting_id, ignore_errors=True)
    except Exception as exc:
        store.update_meeting(meeting_id, status="error", error=str(exc))


def _to_markdown(record: dict) -> str:
    speakers = record.get("speakers", {})
    out = io.StringIO()
    out.write(f"# {record.get('filename', 'Meeting')}\n\n")
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
