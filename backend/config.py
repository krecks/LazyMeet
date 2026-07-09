"""Central configuration: app-data paths, model locations, tunables.

Everything is derived from a single per-user app-data directory so the bundled
desktop app keeps its models and transcripts in the conventional OS location
(e.g. ``~/Library/Application Support/LazyMeet`` on macOS, ``%APPDATA%\\LazyMeet``
on Windows, ``~/.local/share/LazyMeet`` on Linux). Paths can be overridden with
environment variables, which is what the test harness and the Tauri shell use.

Tunables listed in ``TUNABLES`` can additionally be persisted to
``DATA_DIR/config.json`` (the Settings UI writes it via ``/api/config``).
Precedence: environment variable > config.json > default. Paths are
deliberately env-only — config.json lives inside DATA_DIR, so DATA_DIR cannot
come from it. Every consumer accesses these via ``config.X`` attribute lookup,
which lets ``refresh()`` rebind the module globals at runtime.
"""

from __future__ import annotations

import json
import os
import tempfile
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
# Summarizer GGUF paths live in models_setup.LLM_CATALOG (selectable per model).

# --- Tunables ----------------------------------------------------------------
# name -> (env var, type, default, min, max). min/max are None for bools.
# What each one means:
#   SPEAKER_THRESHOLD       cosine similarity to match a diarized speaker to a
#                           saved voice
#   DIAR_THRESHOLD          auto-clustering threshold (higher => fewer speakers;
#                           0.7 avoids the over-segmentation 0.5 produced)
#   SPEAKER_MERGE_THRESHOLD auto clustering still over-splits real meetings (a
#                           2-person recording came back as 63 labels in
#                           testing), so post-diarization we merge labels whose
#                           long-window voiceprints match at this similarity
#   MIN_SPEAKER_SECONDS     speakers with less speech than this are folded into
#                           the closest main speaker
#   NUM_THREADS             ONNX / llama.cpp CPU threads (default ~75% of cores)
#   USE_GPU                 offload the summarizer LLM to the GPU (Metal) when
#                           llama.cpp supports it, falling back to the CPU
#                           automatically; the ONNX models always run on the
#                           CPU (see onnx_provider)
#   KEEP_AUDIO              keep raw recordings after processing (default: they
#                           are deleted for privacy + disk once a meeting is
#                           fully processed)
#   ASR_MODEL / SUMMARY_MODEL      ids from models_setup.ASR_CATALOG/LLM_CATALOG
#                                  (validated at the API layer; pipeline falls
#                                  back to the default while not downloaded)
#   ASR_LANGUAGE                   spoken-language hint ("auto" or ISO code);
#                                  used by Whisper models, Parakeet auto-detects
#   SUMMARY_LANGUAGE               language of the generated notes ("auto" =
#                                  match the meeting)
# The CPU-bound pipeline stages scale well with threads; default to ~75% of
# the cores so processing is fast but the machine stays responsive.
_CPU_COUNT = os.cpu_count() or 8
_MAX_THREADS = min(16, _CPU_COUNT)
_DEFAULT_THREADS = max(1, min(_MAX_THREADS, _CPU_COUNT * 3 // 4))

TUNABLES: dict[str, tuple[str, type, object, object, object]] = {
    "ASR_MODEL": ("LAZYMEET_ASR_MODEL", str, "parakeet-tdt-0.6b-v3", None, None),
    "ASR_LANGUAGE": ("LAZYMEET_ASR_LANGUAGE", str, "auto", None, None),
    "SUMMARY_MODEL": ("LAZYMEET_SUMMARY_MODEL", str, "qwen3.5-4b", None, None),
    "SUMMARY_LANGUAGE": ("LAZYMEET_SUMMARY_LANGUAGE", str, "auto", None, None),
    "SPEAKER_THRESHOLD": ("LAZYMEET_SPEAKER_THRESHOLD", float, 0.5, 0.0, 1.0),
    "DIAR_THRESHOLD": ("LAZYMEET_DIAR_THRESHOLD", float, 0.7, 0.0, 1.0),
    "SPEAKER_MERGE_THRESHOLD": ("LAZYMEET_MERGE_THRESHOLD", float, 0.6, 0.0, 1.0),
    "MIN_SPEAKER_SECONDS": ("LAZYMEET_MIN_SPEAKER_SECONDS", float, 15.0, 0.0, 600.0),
    "NUM_THREADS": ("LAZYMEET_NUM_THREADS", int, _DEFAULT_THREADS, 1, _MAX_THREADS),
    "USE_GPU": ("LAZYMEET_USE_GPU", bool, True, None, None),
    "KEEP_AUDIO": ("LAZYMEET_KEEP_AUDIO", bool, False, None, None),
}

CONFIG_FILE = DATA_DIR / "config.json"

# Diarization speaker count: > 0 forces an exact count; -1 means "auto", in
# which case DIAR_THRESHOLD decides. Env-only — the UI sets it per upload.
DIAR_NUM_SPEAKERS = int(os.environ.get("LAZYMEET_NUM_SPEAKERS", "-1"))

# Summarization LLM context window. The transcript is chunked to fit, but a
# larger window means fewer chunks (and faster, more coherent summaries).
LLM_N_CTX = int(os.environ.get("LAZYMEET_LLM_N_CTX", "16384"))

# Whisper-equivalent embedding dimension used by the deterministic fake embedder.
FAKE_EMBED_DIM = 192


def _coerce(name: str, raw: object) -> object | None:
    """Parse+validate one tunable value; None if unusable (ignored, not clamped)."""
    _, typ, _, lo, hi = TUNABLES[name]
    try:
        if typ is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return None
        if typ is str:
            if not isinstance(raw, str):
                return None
            value = raw.strip()
            # Catalog/choice membership is enforced at the API layer; consumers
            # fall back to defaults on unknown ids, so a stale value is inert.
            return value if 0 < len(value) <= 200 else None
        value = typ(raw)
    except (TypeError, ValueError):
        return None
    if lo is not None and (value < lo or value > hi):
        return None
    return value


def load_overrides() -> dict:
    """Read config.json; unknown keys and invalid values are dropped."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for name, raw in data.items():
        if name not in TUNABLES:
            continue
        value = _coerce(name, raw)
        if value is not None:
            out[name] = value
    return out


def save_overrides(values: dict) -> None:
    # Atomic write (same tmp + os.replace pattern as store._atomic_write, which
    # can't be imported here: store imports config).
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_FILE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_FILE)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def env_locked() -> list[str]:
    """Tunables pinned by an environment variable (the Settings UI locks them)."""
    return [n for n, (env, *_rest) in TUNABLES.items() if os.environ.get(env)]


def refresh() -> None:
    """Rebind tunable globals: env var > config.json > default."""
    overrides = load_overrides()
    for name, (env, _typ, default, _lo, _hi) in TUNABLES.items():
        raw_env = os.environ.get(env)
        value = _coerce(name, raw_env) if raw_env is not None else None
        if value is None:
            value = overrides.get(name)
        if value is None:
            value = default
        globals()[name] = value


refresh()


def onnx_provider() -> str:
    """Execution provider for the sherpa-onnx models.

    Always "cpu": CoreML measured ~2x slower than the CPU provider for these
    models (Parakeet 300ms vs 152ms, speaker embedding 61ms vs 35ms per call
    on an M1 Pro) — onnxruntime can only place a fraction of their ops on the
    ANE, and the CPU<->CoreML hand-offs between the resulting graph islands
    cost more than the acceleration saves. USE_GPU still controls Metal
    offload for the summarizer LLM, which does benefit.
    """
    return "cpu"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, MEETINGS_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)
