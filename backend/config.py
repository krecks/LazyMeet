"""Central configuration: app-data paths, model locations, tunables.

Everything is derived from a single per-user app-data directory so the bundled
desktop app keeps its models and transcripts in the conventional OS location
(e.g. ``~/Library/Application Support/LazyMeet`` on macOS, ``%APPDATA%\\LazyMeet``
on Windows, ``~/.local/share/LazyMeet`` on Linux). Paths can be overridden with
environment variables, which is what the test harness and the Tauri shell use.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "LazyMeet"

# ``LAZYMEET_FAKE=1`` swaps the heavy ASR / diarization / LLM stages for cheap
# deterministic stubs so the whole app can be exercised without any models or
# native libraries installed.
FAKE_MODE = os.environ.get("LAZYMEET_FAKE") == "1"


def _default_data_dir() -> Path:
    override = os.environ.get("LAZYMEET_DATA_DIR")
    if override:
        return Path(override)
    try:  # platformdirs is a runtime dep but optional for the stub paths
        from platformdirs import user_data_dir

        return Path(user_data_dir(APP_NAME, appauthor=False))
    except Exception:
        return Path.home() / ".local" / "share" / APP_NAME


DATA_DIR = _default_data_dir()
MODELS_DIR = Path(os.environ.get("LAZYMEET_MODELS_DIR", DATA_DIR / "models"))
MEETINGS_DIR = DATA_DIR / "meetings"
UPLOADS_DIR = DATA_DIR / "uploads"
SPEAKERS_FILE = DATA_DIR / "speakers.json"

# --- Model locations (populated on first run by models_setup.py) -------------
ASR_MODEL_DIR = MODELS_DIR / "parakeet"            # sherpa-onnx Parakeet model dir
VAD_MODEL = MODELS_DIR / "silero_vad.onnx"          # sherpa-onnx Silero VAD
DIAR_SEG_MODEL = MODELS_DIR / "diarization_segmentation.onnx"
DIAR_EMB_MODEL = MODELS_DIR / "speaker_embedding.onnx"  # also used for identification
LLM_MODEL = MODELS_DIR / "summarizer.gguf"

# --- Tunables ----------------------------------------------------------------
# Cosine-similarity threshold for matching a diarized speaker to a saved voice.
SPEAKER_THRESHOLD = float(os.environ.get("LAZYMEET_SPEAKER_THRESHOLD", "0.5"))
NUM_THREADS = int(os.environ.get("LAZYMEET_NUM_THREADS", "4"))
# Whisper-equivalent embedding dimension used by the deterministic fake embedder.
FAKE_EMBED_DIM = 192


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, MEETINGS_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)
