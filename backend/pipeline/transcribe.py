"""Speech-to-text with NVIDIA Parakeet via sherpa-onnx.

Parakeet is utterance-level, so we run sherpa-onnx's Silero VAD first to cut the
audio into speech segments and decode each one; the timing comes from the VAD.
Models are loaded lazily and cached for the life of the process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config, models_setup

_recognizer = None
_vad = None
_loaded_key = None  # (model paths kind, language) the cached recognizer was built for


def _load_models():
    global _recognizer, _vad, _loaded_key

    spec = models_setup.resolve_asr(config.ASR_MODEL)
    language = "" if config.ASR_LANGUAGE == "auto" else config.ASR_LANGUAGE
    key = (str(spec["encoder"]), language)
    if _recognizer is not None and _loaded_key == key:
        return _recognizer, _vad

    import sherpa_onnx

    if spec["kind"] == "whisper":
        _recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(spec["encoder"]),
            decoder=str(spec["decoder"]),
            tokens=str(spec["tokens"]),
            language=language,  # "" lets Whisper detect the language itself
            task="transcribe",
            num_threads=config.NUM_THREADS,
            provider=config.onnx_provider(),
        )
    else:
        # Parakeet (nemo_transducer) — multilingual, always auto-detects.
        _recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(spec["encoder"]),
            decoder=str(spec["decoder"]),
            joiner=str(spec["joiner"]),
            tokens=str(spec["tokens"]),
            num_threads=config.NUM_THREADS,
            provider=config.onnx_provider(),
            model_type="nemo_transducer",
        )
    _loaded_key = key

    if _vad is None:
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(config.VAD_MODEL)
        vad_config.sample_rate = 16000
        vad_config.provider = config.onnx_provider()
        _vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)
    return _recognizer, _vad


def transcribe(wav_path: Path) -> list[dict]:
    """Return ``[{start, end, text}]`` ordered by time."""
    if config.FAKE_MODE:
        return _fake_transcribe()
    return _real_transcribe(wav_path)


def _real_transcribe(wav_path: Path) -> list[dict]:
    import numpy as np
    import soundfile as sf

    recognizer, vad = _load_models()
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]

    segments: list[dict] = []
    window = 512
    i = 0
    vad.reset()
    while i < len(audio):
        vad.accept_waveform(audio[i : i + window])
        i += window
        while not vad.empty():
            seg = vad.front
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, seg.samples)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                start = seg.start / sample_rate
                end = start + len(seg.samples) / sample_rate
                segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, seg.samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if text:
            start = seg.start / sample_rate
            end = start + len(seg.samples) / sample_rate
            segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
        vad.pop()
    return segments


def _fake_transcribe() -> list[dict]:
    return [
        {"start": 0.0, "end": 3.2, "text": "Hi everyone, thanks for joining the weekly sync."},
        {"start": 3.4, "end": 7.1, "text": "Let's start with the transcription pipeline status."},
        {"start": 7.3, "end": 11.0, "text": "I finished wiring up the diarization step yesterday."},
        {"start": 11.2, "end": 15.4, "text": "Great. Can you also add the speaker library before Friday?"},
        {"start": 15.6, "end": 19.0, "text": "Yes, I'll have it ready and send the demo recording."},
    ]
