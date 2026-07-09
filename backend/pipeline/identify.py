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
        provider=config.onnx_provider(),
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


_TINY_MERGE_FLOOR = 0.4   # a short speaker folds into a main speaker above this
_NOISE_MAX_SECONDS = 5.0  # below the floor and shorter than this => dropped as noise


def consolidate_speakers(
    turns: list[dict], voiceprints: dict[str, list[float]]
) -> tuple[list[dict], dict[str, list[float]]]:
    """Collapse over-split diarization labels into real speakers.

    Auto clustering works on short, noisy per-segment embeddings and routinely
    splits one voice into many labels. The averaged long-window voiceprints are
    far more reliable, so: (1) greedily merge labels whose voiceprints match at
    ``SPEAKER_MERGE_THRESHOLD``, folding the shorter speaker into the longer one
    (the survivor keeps its own voiceprint, which prevents chain-merging two
    real speakers); (2) fold remaining sub-``MIN_SPEAKER_SECONDS`` speakers into
    the closest main speaker; (3) drop tiny fragments that resemble no one
    (laughs, music, cross-talk) — their transcript segments reattach to the
    nearest surviving turn in ``merge.assign_speakers``.
    """
    dur: dict[str, float] = {}
    for t in turns:
        dur[t["speaker"]] = dur.get(t["speaker"], 0.0) + (t["end"] - t["start"])

    vp = dict(voiceprints)
    alias: dict[str, str] = {}  # folded label -> surviving label

    def fold(loser: str, winner: str) -> None:
        for k, v in alias.items():
            if v == loser:
                alias[k] = winner
        alias[loser] = winner
        dur[winner] = dur.get(winner, 0.0) + dur.pop(loser, 0.0)
        vp.pop(loser, None)

    # Stage 1: merge matching voiceprints, highest similarity first. Survivors
    # keep their voiceprints, so pair scores never change and one sorted pass
    # over the precomputed pairs is exact greedy merging.
    spks = sorted(vp)
    pairs = sorted(
        ((cosine(vp[a], vp[b]), a, b) for i, a in enumerate(spks) for b in spks[i + 1 :]),
        reverse=True,
    )
    for score, a, b in pairs:
        if score < config.SPEAKER_MERGE_THRESHOLD:
            break
        if a in alias or b in alias:
            continue
        loser, winner = (a, b) if dur.get(a, 0.0) < dur.get(b, 0.0) else (b, a)
        fold(loser, winner)

    # Stages 2+3: deal with what's left below MIN_SPEAKER_SECONDS. Without any
    # main speaker (e.g. a very short clip) everything is kept as-is.
    cores = [s for s in vp if dur.get(s, 0.0) >= config.MIN_SPEAKER_SECONDS]
    dropped: set[str] = set()
    if cores:
        for spk in [s for s in vp if s not in cores]:
            best = max(cores, key=lambda c: cosine(vp[spk], vp[c]))
            if cosine(vp[spk], vp[best]) >= _TINY_MERGE_FLOOR:
                fold(spk, best)
            elif dur.get(spk, 0.0) < _NOISE_MAX_SECONDS:
                dropped.add(spk)
                del vp[spk]
        # Labels with no usable voiceprint at all can't be compared to anyone.
        for spk in {t["speaker"] for t in turns} - set(vp) - set(alias):
            if dur.get(spk, 0.0) < _NOISE_MAX_SECONDS:
                dropped.add(spk)

    out = []
    for t in turns:
        spk = alias.get(t["speaker"], t["speaker"])
        if spk in dropped:
            continue
        out.append({**t, "speaker": spk})
    return out, vp


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
