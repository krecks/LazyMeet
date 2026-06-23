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
        url=_env(
            "LAZYMEET_DIAR_EMB_URL",
            f"{REL}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        ),
        marker=Path("speaker_embedding.onnx"),
    ),
    ModelSpec(
        name="Summarization LLM",
        url=_env(
            "LAZYMEET_LLM_URL",
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
            "qwen2.5-3b-instruct-q4_k_m.gguf",
        ),
        marker=Path("summarizer.gguf"),
    ),
]

_progress: dict = {"state": "idle", "items": {}, "error": None}
_lock = threading.Lock()
_thread: threading.Thread | None = None


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
    except Exception as exc:  # surfaced to the UI
        with _lock:
            _progress["state"] = "error"
            _progress["error"] = str(exc)


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
    with tarfile.open(archive_path, "r:bz2") as tar:
        members = tar.getmembers()
        for member in members:
            base = os.path.basename(member.name)
            for needle, dest_rel in spec.extract_map.items():
                if base == needle or (needle in base and base.endswith(".onnx")):
                    dest = config.MODELS_DIR / dest_rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        with open(dest, "wb") as out:
                            out.write(extracted.read())
