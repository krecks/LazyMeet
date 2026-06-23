"""Speaker identification: turn anonymous diarization labels into saved names.

For each diarized speaker we compute one averaged voiceprint (sherpa-onnx
``SpeakerEmbeddingExtractor`` — the same embedding model used for diarization)
and compare it to the persistent speaker library by cosine similarity. A match
above ``SPEAKER_THRESHOLD`` relabels the speaker with the saved name; the raw
voiceprints are returned so the API can enrol a voice when the user renames it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .. import config, store

_extractor = None


def _load_extractor():
    global _extractor
    if _extractor is not None:
        return _extractor

    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(config.DIAR_EMB_MODEL),
        num_threads=config.NUM_THREADS,
    )
    _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    return _extractor


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def compute_voiceprints(wav_path: Path, turns: list[dict]) -> dict[str, list[float]]:
    """Return ``{speaker_id: embedding}`` averaged over each speaker's turns."""
    if config.FAKE_MODE:
        speakers = {t["speaker"] for t in turns}
        return {spk: _fake_embedding(spk) for spk in speakers}
    return _real_voiceprints(wav_path, turns)


def _real_voiceprints(wav_path: Path, turns: list[dict]) -> dict[str, list[float]]:
    import numpy as np
    import soundfile as sf

    extractor = _load_extractor()
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Use each speaker's longest turns (more reliable embedding).
    by_speaker: dict[str, list[dict]] = {}
    for t in turns:
        by_speaker.setdefault(t["speaker"], []).append(t)

    voiceprints: dict[str, list[float]] = {}
    for spk, spk_turns in by_speaker.items():
        spk_turns = sorted(spk_turns, key=lambda t: t["end"] - t["start"], reverse=True)[:5]
        embs = []
        for t in spk_turns:
            a = audio[int(t["start"] * sr) : int(t["end"] * sr)]
            if len(a) < sr * 0.3:  # skip < 0.3s
                continue
            stream = extractor.create_stream()
            stream.accept_waveform(sr, a)
            stream.input_finished()
            if extractor.is_ready(stream):
                embs.append(np.asarray(extractor.compute(stream), dtype="float32"))
        if embs:
            voiceprints[spk] = np.mean(embs, axis=0).tolist()
    return voiceprints


def match_speakers(voiceprints: dict[str, list[float]]) -> tuple[dict[str, str], list[str]]:
    """Map diarized ids to display names using the saved speaker library.

    Returns ``(speakers, recognized)`` where ``speakers`` maps every diarized id
    to a display name (saved name if matched, otherwise ``Speaker N``) and
    ``recognized`` lists the ids that were auto-matched.
    """
    library = store.load_speaker_library()
    speakers: dict[str, str] = {}
    recognized: list[str] = []

    # Stable "Speaker N" numbering from the sorted diarized ids.
    ordered = sorted(voiceprints.keys())
    for idx, spk in enumerate(ordered, start=1):
        emb = voiceprints[spk]
        best_name, best_score = None, -1.0
        for name, entry in library.items():
            score = cosine(emb, entry.get("embedding", []))
            if score > best_score:
                best_name, best_score = name, score
        if best_name is not None and best_score >= config.SPEAKER_THRESHOLD:
            speakers[spk] = best_name
            recognized.append(spk)
        else:
            speakers[spk] = f"Speaker {idx}"
    return speakers, recognized


def _fake_embedding(speaker_label: str) -> list[float]:
    """Deterministic unit-ish vector seeded by the speaker label.

    Same label → same vector, so enrolling a fake speaker and re-processing a
    stub with the same label exercises the real match/enrol path.
    """
    seed = hashlib.sha256(speaker_label.encode()).digest()
    vals = []
    for i in range(config.FAKE_EMBED_DIM):
        b = seed[i % len(seed)]
        vals.append((b / 255.0) * 2 - 1)
    return vals
