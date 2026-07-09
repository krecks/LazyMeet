"""Speaker diarization with sherpa-onnx (segmentation + speaker embedding).

Returns anonymous, per-file speaker turns ``[{start, end, speaker}]`` where
``speaker`` is ``SPEAKER_00``, ``SPEAKER_01``, ... The mapping of these labels to
real names happens later in ``identify.py`` against the speaker library.
"""

from __future__ import annotations

from pathlib import Path

from .. import config


def _build_diarizer(num_speakers: int):
    import sherpa_onnx

    # A meeting-supplied count (>0) wins; otherwise fall back to the auto default.
    n = num_speakers if num_speakers and num_speakers > 0 else config.DIAR_NUM_SPEAKERS

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(config.DIAR_SEG_MODEL)
            ),
            num_threads=config.NUM_THREADS,
            provider=config.onnx_provider(),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(config.DIAR_EMB_MODEL),
            num_threads=config.NUM_THREADS,
            provider=config.onnx_provider(),
        ),
        # num_clusters>0 forces an exact speaker count; -1 lets `threshold`
        # decide (higher threshold => fewer speakers). min_duration_* smooth out
        # the sub-second fragments that otherwise explode into hundreds of
        # spurious speakers.
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=n,
            threshold=config.DIAR_THRESHOLD,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def diarize(wav_path: Path, num_speakers: int = -1) -> list[dict]:
    if config.FAKE_MODE:
        return _fake_diarize()
    return _real_diarize(wav_path, num_speakers)


def _real_diarize(wav_path: Path, num_speakers: int) -> list[dict]:
    import soundfile as sf

    diarizer = _build_diarizer(num_speakers)
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
