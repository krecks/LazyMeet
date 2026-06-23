"""FastAPI sidecar: HTTP API + background transcription pipeline.

Started by the Tauri shell. On launch it binds a localhost port, prints
``READY <port>`` to stdout (the Rust side waits for this line), then serves the
API and the static frontend. All heavy work runs in a background worker so the
UI can poll for progress.
"""

from __future__ import annotations

import io
import os
import socket
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, models_setup, store
from .pipeline import audio, diarize, identify, merge, summarize, transcribe

app = FastAPI(title="LazyMeet")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # localhost desktop app
    allow_methods=["*"],
    allow_headers=["*"],
)

# One job at a time is plenty for a personal desktop tool.
_worker_lock = threading.Lock()


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    models_setup.ensure_models_async()


# --------------------------------------------------------------------------- #
# Setup / health
# --------------------------------------------------------------------------- #
@app.get("/api/setup")
def get_setup():
    return models_setup.status()


@app.get("/api/health")
def health():
    return {"ok": True, "fake": config.FAKE_MODE}


# --------------------------------------------------------------------------- #
# Meetings
# --------------------------------------------------------------------------- #
@app.post("/api/meetings")
async def create_meeting(background: BackgroundTasks, file: UploadFile):
    record = store.create_meeting(file.filename or "recording")
    upload_dir = config.UPLOADS_DIR / record["id"]
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (file.filename or "recording")
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    background.add_task(_process_meeting, record["id"], dest)
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
    record = store.get_meeting(meeting_id)
    if record is None:
        raise HTTPException(404, "meeting not found")

    speakers = dict(record.get("speakers", {}))
    voiceprints = record.get("voiceprints", {})
    for diarized_id, new_name in body.names.items():
        new_name = new_name.strip()
        if not new_name:
            continue
        speakers[diarized_id] = new_name
        # Enrol this voice so the name follows it into future recordings.
        emb = voiceprints.get(diarized_id)
        if emb:
            store.upsert_speaker(new_name, emb)
    updated = store.update_meeting(meeting_id, speakers=speakers)
    return store.public_meeting(updated)


@app.delete("/api/meetings/{meeting_id}")
def delete_meeting(meeting_id: str):
    if not store.delete_meeting(meeting_id):
        raise HTTPException(404, "meeting not found")
    return {"deleted": True}


@app.get("/api/meetings/{meeting_id}/export")
def export_meeting(meeting_id: str, format: str = "md"):
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
def _process_meeting(meeting_id: str, upload_path: Path) -> None:
    with _worker_lock:
        try:
            store.update_meeting(meeting_id, status="extracting")
            wav = config.UPLOADS_DIR / meeting_id / "audio.wav"
            audio.extract_audio(upload_path, wav)

            store.update_meeting(meeting_id, status="transcribing")
            segments = transcribe.transcribe(wav)

            store.update_meeting(meeting_id, status="diarizing")
            turns = diarize.diarize(wav)

            store.update_meeting(meeting_id, status="identifying")
            voiceprints = identify.compute_voiceprints(wav, turns)
            speakers, recognized = identify.match_speakers(voiceprints)

            merged = merge.assign_speakers(segments, turns)

            store.update_meeting(meeting_id, status="summarizing", segments=merged,
                                 speakers=speakers, recognized=recognized,
                                 voiceprints=voiceprints)
            result = summarize.summarize(merged, speakers)

            store.update_meeting(
                meeting_id,
                status="done",
                summary=result["summary"],
                takeaways=result["takeaways"],
                action_items=result["action_items"],
            )
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
    port = _pick_port()
    # The Tauri shell waits for this exact line before loading the UI.
    print(f"READY {port}", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
