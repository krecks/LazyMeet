"""Speaker diarization with sherpa-onnx (segmentation + speaker embedding).

Returns anonymous, per-file speaker turns ``[{start, end, speaker}]`` where
``speaker`` is ``SPEAKER_00``, ``SPEAKER_01``, ... The mapping of these labels to
real names happens later in ``identify.py`` against the speaker library.
"""

from __future__ import annotations

from pathlib import Path

from .. import config

_diarizer = None


def _load_diarizer():
    global _diarizer
    if _diarizer is not None:
        return _diarizer

    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(config.DIAR_SEG_MODEL)
            ),
            num_threads=config.NUM_THREADS,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(config.DIAR_EMB_MODEL),
            num_threads=config.NUM_THREADS,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
    )
    _diarizer = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return _diarizer


def diarize(wav_path: Path) -> list[dict]:
    if config.FAKE_MODE:
        return _fake_diarize()
    return _real_diarize(wav_path)


def _real_diarize(wav_path: Path) -> list[dict]:
    import soundfile as sf

    diarizer = _load_diarizer()
    audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    result = diarizer.process(audio).sort_by_start_time()
    turns = []
    for seg in result:
        turns.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": f"SPEAKER_{seg.speaker:02d}",
            }
        )
    return turns


def _fake_diarize() -> list[dict]:
    # Two alternating speakers matching the fake transcript timeline.
    return [
        {"start": 0.0, "end": 7.2, "speaker": "SPEAKER_00"},
        {"start": 7.2, "end": 11.1, "speaker": "SPEAKER_01"},
        {"start": 11.1, "end": 15.5, "speaker": "SPEAKER_00"},
        {"start": 15.5, "end": 19.2, "speaker": "SPEAKER_01"},
    ]
