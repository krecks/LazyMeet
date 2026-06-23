"""Attach a speaker to each transcript segment by maximum temporal overlap.

This mirrors what WhisperX does internally: for every ASR segment we pick the
diarization turn it overlaps with the most.
"""

from __future__ import annotations


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    merged = []
    for seg in segments:
        best_speaker, best_ov = None, 0.0
        for turn in turns:
            ov = _overlap(seg["start"], seg["end"], turn["start"], turn["end"])
            if ov > best_ov:
                best_speaker, best_ov = turn["speaker"], ov
        if best_speaker is None and turns:
            # No overlap (e.g. boundary segment): fall back to nearest turn.
            mid = (seg["start"] + seg["end"]) / 2
            best_speaker = min(
                turns, key=lambda t: abs(((t["start"] + t["end"]) / 2) - mid)
            )["speaker"]
        merged.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": best_speaker or "SPEAKER_00",
                "text": seg["text"],
            }
        )
    return merged
