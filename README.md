# LazyMeet

**Local, private meeting transcription & summaries.** Drag and drop a recording
(e.g. an OBS capture) and get a **per-speaker transcript**, a **summary**, and
**action items** — everything runs on your machine. No cloud, no accounts, **no
API keys**.

Speaker identification is built in: name a speaker once and that name
**auto-applies to the same voice in future recordings**, backed by a local
voiceprint library that learns from every confirmed match.

## How it works

```
Tauri desktop app
 └─ Python sidecar (FastAPI)
     1. ffmpeg (bundled)        → 16 kHz mono audio
     2. sherpa-onnx VAD         → speech utterances
     3. NVIDIA Parakeet (ONNX)  → transcript with timestamps
     4. sherpa-onnx diarization → who-spoke-when
     5. speaker identification  → match/name voices via a saved speaker library
     6. llama.cpp (in-process)  → summary, takeaways, action items
```

All AI models are **token-free** and download once on first run, then run
fully offline:

| Component | Engine | Source |
|---|---|---|
| Transcription | NVIDIA parakeet-tdt-0.6b-v3 (25 European languages; Whisper Small/Base selectable) | sherpa-onnx GitHub releases |
| Voice activity | Silero VAD | sherpa-onnx GitHub releases |
| Diarization + speaker ID | pyannote-seg + speaker-embedding (ONNX) | sherpa-onnx GitHub releases |
| Summarization | Qwen 3.5 (0.8B–9B selectable, GGUF) via llama-cpp-python | Hugging Face (ungated) |
| ffmpeg | static binary | `imageio-ffmpeg` wheel |

## Install (end users)

1. Download the installer for your OS from the latest build artifacts
   (`.dmg`/`.app`, `.msi`/`.exe`, or `.AppImage`/`.deb`).
2. Launch **LazyMeet**. On first run it downloads the AI models (a few GB) with a
   progress screen — this happens once.
3. Drag a recording onto the window. Watch the stages, then read the transcript
   and summary. Rename `Speaker 1/2/3` to real names; those names are remembered
   and auto-applied to the same voices next time.

You install nothing else — ffmpeg, the models, and the LLM are all bundled or
auto-downloaded.

## Speaker naming across files

Diarization numbers speakers per file (Speaker 1 here ≠ Speaker 1 next time).
When you rename a speaker, LazyMeet saves that person's **voiceprint** to a local
**speaker library** (`speakers.json` in the app-data folder). On every future
upload, each detected speaker is matched against the library by voice similarity;
known voices are auto-labelled (shown with a *recognized* badge), unknown ones
stay `Speaker N` until you name them. Manage saved voices in the **Known
speakers** panel.

## Privacy

Audio, transcripts, summaries, and voiceprints never leave your machine. They
live in the OS app-data directory (`~/Library/Application Support/LazyMeet`,
`%APPDATA%\LazyMeet`, or `~/.local/share/LazyMeet`). After the one-time model
download the app needs no network at all. The uploaded recording itself is
deleted as soon as processing finishes — only the transcript and summary are
kept (set `LAZYMEET_KEEP_AUDIO=1` to keep recordings).

## Development / building from source

Requirements: Python 3.11+, Rust (stable) + Tauri CLI (`cargo install tauri-cli
--version "^2"`), and the platform webview libs (Linux: `libwebkit2gtk-4.1-dev`).

```bash
# 1. Backend deps
pip install -r backend/requirements.txt

# 2. App icons (or: cargo tauri icon src-tauri/icons/icon.png)
python scripts/make_icons.py

# 3. Freeze the backend (PyInstaller onedir, bundled as an app resource)
python scripts/build_sidecar.py

# 4. Run the desktop app (dev)
cd src-tauri && cargo tauri dev
# ...or build installers:
cd src-tauri && cargo tauri build
```

### Backend-only / browser dev

You can run just the Python backend and open the UI in a browser — handy for
iterating without rebuilding the sidecar:

```bash
# Fake mode skips the heavy models and returns canned data:
LAZYMEET_FAKE=1 python -m backend.main
# It prints "READY <port>"; open http://127.0.0.1:<port>/
```

Without `LAZYMEET_FAKE`, the backend downloads real models on first run and does
real transcription (all on the CPU; the summarizer LLM offloads to the GPU
where llama.cpp supports it).

### Tests

Smoke tests exercise the whole API over the stubbed (`LAZYMEET_FAKE`) pipeline:

```bash
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
python -m pytest tests/
```

### Cross-platform builds

`.github/workflows/build.yml` builds installers for macOS, Windows, and Linux via
a CI matrix (Tauri can't cross-compile). Each runner builds the PyInstaller
sidecar, then bundles the app, and uploads the installer as an artifact.

## Configuration

Environment variables (all optional):

| Var | Default | Purpose |
|---|---|---|
| `LAZYMEET_DATA_DIR` | OS app-data dir | Where models + transcripts live |
| `LAZYMEET_SPEAKER_THRESHOLD` | `0.5` | Voice-match strictness (cosine) |
| `LAZYMEET_MERGE_THRESHOLD` | `0.6` | Voiceprint similarity at which over-split diarized labels merge |
| `LAZYMEET_MIN_SPEAKER_SECONDS` | `15` | Speakers with less speech fold into the closest main voice |
| `LAZYMEET_NUM_THREADS` | ~75% of cores | ONNX/LLM CPU threads |
| `LAZYMEET_ASR_URL` / `LAZYMEET_LLM_URL` / … | release URLs | Override model downloads |
| `LAZYMEET_KEEP_AUDIO` | unset | `1` = keep uploaded recordings after processing |
| `LAZYMEET_FAKE` | unset | `1` = stubbed pipeline for dev/tests |
