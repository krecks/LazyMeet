"""Extract a 16 kHz mono WAV from any OBS recording using bundled ffmpeg.

The ffmpeg binary ships inside the app via the ``imageio-ffmpeg`` wheel, so the
user never installs ffmpeg themselves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .. import config


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: Path, out_wav: Path) -> Path:
    """Decode ``video_path`` to a 16 kHz mono PCM WAV at ``out_wav``."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if config.FAKE_MODE:
        # Write a tiny valid (silent) WAV so downstream stubs have a real file.
        _write_silent_wav(out_wav)
        return out_wav

    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


def _write_silent_wav(out_wav: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    import struct
    import wave

    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<" + "h" * int(rate * seconds), *([0] * int(rate * seconds))))
