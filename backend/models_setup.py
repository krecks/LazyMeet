"""First-run model downloader with progress reporting.

All weights are token-free: the ASR / VAD / diarization models come from
sherpa-onnx GitHub releases and the summarization GGUF from an ungated Hugging
Face repo. Downloads run on a background thread; the UI polls ``status()`` and
shows a progress screen until everything is ready, after which the app is fully
offline.

URLs are best-known defaults and can be overridden via environment variables
(handy if a newer model release is preferred).
"""

from __future__ import annotations

import os
import tarfile
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class ModelSpec:
    name: str
    url: str
    # File (relative to MODELS_DIR) whose existence means this item is ready.
    marker: Path
    # If the URL is a .tar.bz2 archive, extract it and move the right files out.
    archive: bool = False
    # For archives: (path-inside-archive-suffix -> destination relative to MODELS_DIR)
    extract_map: dict = field(default_factory=dict)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="Parakeet ASR (multilingual)",
        url=_env(
            "LAZYMEET_ASR_URL",
            f"{REL}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        ),
        archive=True,
        marker=Path("parakeet") / "encoder.onnx",
        extract_map={
            "encoder": Path("parakeet") / "encoder.onnx",
            "decoder": Path("parakeet") / "decoder.onnx",
            "joiner": Path("parakeet") / "joiner.onnx",
            "tokens.txt": Path("parakeet") / "tokens.txt",
        },
    ),
    ModelSpec(
        name="Silero VAD",
        url=_env("LAZYMEET_VAD_URL", f"{REL}/asr-models/silero_vad.onnx"),
        marker=Path("silero_vad.onnx"),
    ),
    ModelSpec(
        name="Diarization segmentation",
        url=_env(
            "LAZYMEET_DIAR_SEG_URL",
            f"{REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        ),
        archive=True,
        marker=Path("diarization_segmentation.onnx"),
        extract_map={"model.onnx": Path("diarization_segmentation.onnx")},
    ),
    ModelSpec(
        name="Speaker embedding",
        # English TitaNet — discriminates English/German voices far better than
        # the zh-cn model (which over-segmented real meetings into dozens of
        # spurious speakers).
        url=_env(
            "LAZYMEET_DIAR_EMB_URL",
            f"{REL}/speaker-recongition-models/nemo_en_titanet_small.onnx",
        ),
        marker=Path("speaker_embedding.onnx"),
    ),
    ModelSpec(
        name="Summarization LLM",
        url=_env(
            "LAZYMEET_LLM_URL",
            "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/"
            "Qwen3.5-4B-Q4_K_M.gguf",
        ),
        marker=Path("qwen3.5-4b-q4_k_m.gguf"),
    ),
]

# --------------------------------------------------------------------------- #
# Selectable model catalogs (Settings → Transcription / Summary).
#
# The first entry of each catalog is the default that MODEL_SPECS installs on
# first run; the others download on demand when selected. Pipeline code
# resolves the active id through resolve_asr()/resolve_llm(), which fall back
# to the default while a download is still in flight (or failed) so a meeting
# never errors because of a settings choice.
# --------------------------------------------------------------------------- #
ASR_CATALOG: dict[str, dict] = {
    "parakeet-tdt-0.6b-v3": {
        "label": "Parakeet TDT 0.6B v3 — 25 European languages (default)",
        "size": "≈700 MB",
        "kind": "nemo_transducer",
        "dir": "parakeet",
        "pros": [
            "Best accuracy for European languages, incl. German and English",
            "Fastest option — transcribes far faster than real time on the CPU",
            "Detects the language by itself, with punctuation and capitalization",
        ],
        "cons": [
            "Only 25 (European) languages — no Chinese, Japanese or Korean",
            "Less robust than Whisper on heavy accents or noisy recordings",
        ],
        "spec": MODEL_SPECS[0],  # installed on first run
    },
    "whisper-small": {
        "label": "Whisper Small — 90+ languages",
        "size": "≈600 MB download",
        "kind": "whisper",
        "dir": "whisper-small",
        "pros": [
            "Widest coverage — 90+ languages incl. Chinese, Japanese, Korean",
            "Robust with accents, background noise and far-from-mic speech",
            "Accepts a spoken-language hint (setting below)",
        ],
        "cons": [
            "Several times slower than Parakeet",
            "Can invent text during long silences or music",
        ],
        "spec": ModelSpec(
            name="Whisper Small ASR",
            url=_env(
                "LAZYMEET_WHISPER_SMALL_URL",
                f"{REL}/asr-models/sherpa-onnx-whisper-small.tar.bz2",
            ),
            archive=True,
            marker=Path("whisper-small") / "encoder.onnx",
            extract_map={
                "small-encoder.int8.onnx": Path("whisper-small") / "encoder.onnx",
                "small-decoder.int8.onnx": Path("whisper-small") / "decoder.onnx",
                "small-tokens.txt": Path("whisper-small") / "tokens.txt",
            },
        ),
    },
    "whisper-base": {
        "label": "Whisper Base — smallest, less accurate",
        "size": "≈150 MB download",
        "kind": "whisper",
        "dir": "whisper-base",
        "pros": [
            "Smallest download and lowest memory use",
            "Fastest of the Whisper family",
            "Same 90+ language coverage",
        ],
        "cons": [
            "Least accurate — noticeably more transcription errors",
            "Struggles with accents, jargon and people talking over each other",
        ],
        "spec": ModelSpec(
            name="Whisper Base ASR",
            url=_env(
                "LAZYMEET_WHISPER_BASE_URL",
                f"{REL}/asr-models/sherpa-onnx-whisper-base.tar.bz2",
            ),
            archive=True,
            marker=Path("whisper-base") / "encoder.onnx",
            extract_map={
                "base-encoder.int8.onnx": Path("whisper-base") / "encoder.onnx",
                "base-decoder.int8.onnx": Path("whisper-base") / "decoder.onnx",
                "base-tokens.txt": Path("whisper-base") / "tokens.txt",
            },
        ),
    },
}

# Qwen 3.5 small series (Q4_K_M GGUF from unsloth; llama.cpp ≥ b-March-2026,
# verified with the bundled llama-cpp-python 0.3.33).
def _qwen35(
    size: str,
    env: str,
    gb: str,
    note: str,
    pros: list[str],
    cons: list[str],
    default: bool = False,
) -> dict:
    fname = f"qwen3.5-{size.lower()}-q4_k_m.gguf"
    return {
        "label": f"Qwen 3.5 {size} — {note}" + (" (default)" if default else ""),
        "size": gb if default else f"{gb} download",
        "file": Path(fname),
        "pros": pros,
        "cons": cons,
        "spec": ModelSpec(
            name=f"Qwen 3.5 {size} LLM",
            url=_env(
                env,
                f"https://huggingface.co/unsloth/Qwen3.5-{size}-GGUF/resolve/main/"
                f"Qwen3.5-{size}-Q4_K_M.gguf",
            ),
            marker=Path(fname),
        ),
    }


LLM_CATALOG: dict[str, dict] = {
    "qwen3.5-0.8b": _qwen35(
        "0.8B", "LAZYMEET_LLM_08B_URL", "≈0.5 GB", "tiny, fastest",
        pros=[
            "Notes in seconds, even on modest hardware",
            "Tiny download, runs in about 1 GB of memory",
        ],
        cons=[
            "Short, plain notes that miss nuance",
            "Action items often incomplete; weakest outside English",
        ],
    ),
    "qwen3.5-2b": _qwen35(
        "2B", "LAZYMEET_LLM_2B_URL", "≈1.3 GB", "fast, light",
        pros=[
            "Quick summaries with about 2 GB of memory",
            "Clearly better notes than the 0.8B",
        ],
        cons=[
            "Takeaways stay brief — subtle decisions can be missed",
            "Less reliable than the 4B in languages other than English",
        ],
    ),
    "qwen3.5-4b": {
        **_qwen35(
            "4B", "LAZYMEET_LLM_URL", "≈2.7 GB", "balanced",
            pros=[
                "Good balance of quality and speed",
                "Reliable takeaways and action items",
                "Solid notes in German, English and other major languages",
            ],
            cons=[
                "Needs roughly 3–4 GB of free memory",
                "A long meeting takes a few minutes on CPU-only machines",
            ],
            default=True,
        ),
        "spec": MODEL_SPECS[4],  # installed on first run
    },
    "qwen3.5-9b": _qwen35(
        "9B", "LAZYMEET_LLM_9B_URL", "≈5.7 GB", "strongest, slowest",
        pros=[
            "Highest-quality notes — best for long, complex or multilingual meetings",
            "Most faithful decisions and action items",
        ],
        cons=[
            "Large download and needs roughly 7 GB of free memory",
            "Slowest — can take many minutes without GPU acceleration",
        ],
    ),
}

ASR_DEFAULT = "parakeet-tdt-0.6b-v3"
LLM_DEFAULT = "qwen3.5-4b"

_progress: dict = {"state": "idle", "items": {}, "error": None}
_lock = threading.Lock()
_thread: threading.Thread | None = None
_inflight: set[str] = set()  # spec names with an on-demand download running
_ready_callback = None


def set_ready_callback(cb) -> None:
    """Called once all first-run models are downloaded (used to pump the
    processing queue: the UI is usable during the download, and uploads made
    in the meantime wait as "queued")."""
    global _ready_callback
    _ready_callback = cb


def _marker_path(spec: ModelSpec) -> Path:
    return config.MODELS_DIR / spec.marker


def is_ready() -> bool:
    if config.FAKE_MODE:
        return True
    return all(_marker_path(s).exists() for s in MODEL_SPECS)


def status() -> dict:
    with _lock:
        items = []
        for s in MODEL_SPECS:
            prog = _progress["items"].get(s.name, {})
            items.append(
                {
                    "name": s.name,
                    "ready": _marker_path(s).exists() or config.FAKE_MODE,
                    "downloaded": prog.get("downloaded", 0),
                    "total": prog.get("total", 0),
                }
            )
        return {
            "ready": is_ready(),
            "state": "ready" if is_ready() else _progress["state"],
            "error": _progress["error"],
            "items": items,
        }


def ensure_models_async() -> None:
    """Kick off background downloads of any missing models (idempotent)."""
    global _thread
    if config.FAKE_MODE or is_ready():
        return
    with _lock:
        if _thread and _thread.is_alive():
            return
        _progress["state"] = "downloading"
        _progress["error"] = None
        _thread = threading.Thread(target=_download_all, daemon=True)
        _thread.start()


def _download_all() -> None:
    try:
        config.ensure_dirs()
        for spec in MODEL_SPECS:
            if _marker_path(spec).exists():
                continue
            _download_one(spec)
        with _lock:
            _progress["state"] = "ready"
        if _ready_callback is not None:
            try:
                _ready_callback()
            except Exception:
                pass
    except Exception as exc:  # surfaced to the UI
        with _lock:
            _progress["state"] = "error"
            _progress["error"] = str(exc)


# --------------------------------------------------------------------------- #
# Selectable models: resolution + on-demand downloads
# --------------------------------------------------------------------------- #
def _entry_ready(entry: dict) -> bool:
    return config.FAKE_MODE or _marker_path(entry["spec"]).exists()


def resolve_asr(model_id: str) -> dict:
    """Paths + loader kind for the active ASR model, defaulting when missing."""
    entry = ASR_CATALOG.get(model_id)
    if entry is None or not _entry_ready(entry):
        entry = ASR_CATALOG[ASR_DEFAULT]
    d = config.MODELS_DIR / entry["dir"]
    return {
        "kind": entry["kind"],
        "encoder": d / "encoder.onnx",
        "decoder": d / "decoder.onnx",
        "joiner": d / "joiner.onnx",
        "tokens": d / "tokens.txt",
    }


def resolve_llm(model_id: str) -> Path:
    """GGUF path for the active summarizer, defaulting when missing."""
    entry = LLM_CATALOG.get(model_id)
    if entry is None or not _entry_ready(entry):
        entry = LLM_CATALOG[LLM_DEFAULT]
    return config.MODELS_DIR / entry["file"]


def _catalog_options(catalog: dict) -> list[dict]:
    with _lock:
        options = []
        for model_id, entry in catalog.items():
            name = entry["spec"].name
            prog = _progress["items"].get(name, {})
            options.append(
                {
                    "id": model_id,
                    "label": entry["label"],
                    "size": entry.get("size", ""),
                    "pros": entry.get("pros", []),
                    "cons": entry.get("cons", []),
                    "ready": _entry_ready(entry),
                    "downloading": name in _inflight,
                    "downloaded": prog.get("downloaded", 0),
                    "total": prog.get("total", 0),
                    "error": prog.get("error"),
                }
            )
        return options


def models_status() -> dict:
    return {
        "asr": {"active": config.ASR_MODEL, "options": _catalog_options(ASR_CATALOG)},
        "summary": {"active": config.SUMMARY_MODEL, "options": _catalog_options(LLM_CATALOG)},
    }


def request_download(step: str, model_id: str) -> bool:
    """Start a background download of one selectable model (idempotent).

    Returns False for an unknown step/id; already-ready and already-running
    downloads return True without doing anything.
    """
    catalog = {"asr": ASR_CATALOG, "summary": LLM_CATALOG}.get(step)
    if catalog is None or model_id not in catalog:
        return False
    entry = catalog[model_id]
    if config.FAKE_MODE or _entry_ready(entry):
        return True
    spec = entry["spec"]
    with _lock:
        if spec.name in _inflight:
            return True
        _inflight.add(spec.name)
        _progress["items"][spec.name] = {"downloaded": 0, "total": 0}

    def _run():
        try:
            config.ensure_dirs()
            _download_one(spec)
        except Exception as exc:
            with _lock:
                item = _progress["items"].setdefault(spec.name, {})
                item["error"] = str(exc)
        finally:
            with _lock:
                _inflight.discard(spec.name)

    threading.Thread(target=_run, daemon=True).start()
    return True


def ensure_selected_async() -> None:
    """Kick off downloads for the currently selected models, if missing."""
    request_download("asr", config.ASR_MODEL)
    request_download("summary", config.SUMMARY_MODEL)


def _download_one(spec: ModelSpec) -> None:
    tmp = config.MODELS_DIR / (spec.name.replace(" ", "_") + ".download")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "LazyMeet"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with _lock:
            _progress["items"][spec.name] = {"downloaded": 0, "total": total}
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                with _lock:
                    _progress["items"][spec.name] = {"downloaded": downloaded, "total": total}

    if spec.archive:
        _extract_archive(tmp, spec)
        tmp.unlink(missing_ok=True)
    else:
        dest = _marker_path(spec)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dest)


def _extract_archive(archive_path: Path, spec: ModelSpec) -> None:
    """Extract the mapped files, committing them atomically.

    Everything is staged as ``*.part`` first and only renamed into place once
    the whole archive extracted, with the marker file renamed last — a crash
    mid-extract must never leave a truncated model that satisfies ``is_ready()``.
    """
    staged: dict = {}  # dest path relative to MODELS_DIR -> staged .part path
    try:
        with tarfile.open(archive_path, "r:bz2") as tar:
            for member in tar.getmembers():
                base = os.path.basename(member.name)
                for needle, dest_rel in spec.extract_map.items():
                    if dest_rel in staged:
                        continue
                    if base == needle or (needle in base and base.endswith(".onnx")):
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            continue
                        dest = config.MODELS_DIR / dest_rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        part = dest.with_suffix(dest.suffix + ".part")
                        with open(part, "wb") as out:
                            out.write(extracted.read())
                        staged[dest_rel] = part
        missing = [str(d) for d in spec.extract_map.values() if d not in staged]
        if missing:
            raise RuntimeError(f"{spec.name}: archive is missing {', '.join(missing)}")
        for dest_rel, part in staged.items():
            if dest_rel != spec.marker:
                os.replace(part, config.MODELS_DIR / dest_rel)
        os.replace(staged[spec.marker], config.MODELS_DIR / spec.marker)
    except BaseException:
        for part in staged.values():
            part.unlink(missing_ok=True)
        raise
