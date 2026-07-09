"""Extract a 16 kHz mono WAV from any OBS recording using bundled ffmpeg.

The ffmpeg binary ships inside the app via the ``imageio-ffmpeg`` wheel, so the
user never installs ffmpeg themselves.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .. import config

# Uploads with these extensions are audio-only already; anything else (OBS
# .mkv/.mp4/.mov, unknown containers) is treated as video and gets its audio
# track extracted for playback so the large video file can be deleted.
AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif"}
)


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


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


def extract_playback_audio(src: Path, out_m4a: Path) -> Path:
    """Extract the audio track of ``src`` to an AAC ``.m4a`` for playback.

    96 kbps AAC is transparent for speech at a fraction of the video's size.
    ``+faststart`` moves the index to the front so the webview can seek while
    streaming over Range requests.
    """
    out_m4a.parent.mkdir(parents=True, exist_ok=True)

    if config.FAKE_MODE:
        shutil.copyfile(src, out_m4a)
        return out_m4a

    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i", str(src),
        "-vn",
        "-c:a", "aac",
        "-b:a", "96k",
        "-movflags", "+faststart",
        str(out_m4a),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_m4a


def _write_silent_wav(out_wav: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    import struct
    import wave

    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<" + "h" * int(rate * seconds), *([0] * int(rate * seconds))))
