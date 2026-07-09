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
Tauri desktop app — one binary, backend in Rust (no Python, no sidecar)
     1. ffmpeg (bundled)        → 16 kHz mono audio
     2. sherpa-onnx VAD         → speech utterances
     3. NVIDIA Parakeet (ONNX)  → transcript with timestamps
     4. sherpa-onnx diarization → who-spoke-when
     5. speaker identification  → match/name voices via a saved speaker library
     6. llama.cpp (in-process)  → summary, takeaways, action items
```

sherpa-onnx and llama.cpp are linked statically into the app; ffmpeg ships as
a bundled static binary. Progress streams to the UI as native events.

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
known voices are auto-labelled with a *known?* badge asking you to verify the
match, unknown ones stay `Speaker N` until you name them.

Auto-matches are treated as unverified until you act on the badge: confirming
(✓) adds that recording's voice as another sample for the person — up to ten
samples are averaged, so recognition **learns from every file you confirm**
and gets better across microphones and rooms. Rejecting (✗) unlinks the name
without touching the saved voiceprint, so a wrong match can't poison the
library; the speaker reverts to `Speaker N`, ready to be renamed to the right
(or a brand-new) person. Correcting a wrong match by typing over the name has
the same safe behavior. If wrong names still appear, raise **Voice match
strictness** in Settings; lower it if familiar voices go unrecognized. Manage
saved voices in the **Known speakers** panel.

## Privacy

Audio, transcripts, summaries, and voiceprints never leave your machine. They
live in the OS app-data directory (`~/Library/Application Support/LazyMeet`,
`%LOCALAPPDATA%\LazyMeet`, or `~/.local/share/LazyMeet`). After the one-time model
download the app needs no network at all. The recording's audio is kept on
disk (locally, next to the transcript) so you can replay transcript sections —
video uploads are reduced to their audio track during processing, so the bulky
video itself never sticks around. Turn **Keep recording audio** off in Settings
(or set `LAZYMEET_KEEP_AUDIO=0`) to delete each recording as soon as
processing finishes instead.

## Development / building from source

Requirements: Rust (stable) + Tauri CLI (`cargo install tauri-cli --version
"^2"`), cmake (sherpa-onnx and llama.cpp build from source), and the platform
webview libs (Linux: `libwebkit2gtk-4.1-dev`). No Python.

```bash
# 1. Fetch the bundled ffmpeg for your platform (once)
./scripts/fetch_ffmpeg.sh

# 2. Run the desktop app (dev) — LAZYMEET_FAKE=1 skips the heavy models
cd src-tauri && cargo tauri dev
# ...or build installers:
cd src-tauri && cargo tauri build
```

The workspace has two crates: `crates/lazymeet-core` (config, storage, model
downloads, job queue, the ML pipeline — no Tauri dependency) and `src-tauri`
(the desktop shell: thin Tauri commands + events over core).

On Linux, static sherpa-onnx needs
`RUSTFLAGS="-C relocation-model=dynamic-no-pic"` (CI sets it).

### CLI (process a file without the app)

`lazymeet-cli` runs the full pipeline from the terminal against the same
models and prints the meeting JSON — useful for debugging and benchmarks:

```bash
cargo run --release -p lazymeet-core --bin lazymeet-cli -- recording.mp4
```

### Tests

The behavioral test suite (a 1:1 port of the old pytest suite, plus event and
Python-data-compat coverage) runs the stubbed (`LAZYMEET_FAKE`) pipeline and
needs no models or cmake:

```bash
cargo test -p lazymeet-core --no-default-features
```

### Cross-platform builds

`.github/workflows/build.yml` runs the core test suite, then builds installers
for macOS, Windows, and Linux via a CI matrix (Tauri can't cross-compile) and
uploads them as artifacts.

## Configuration

Environment variables (all optional):

| Var | Default | Purpose |
|---|---|---|
| `LAZYMEET_DATA_DIR` | OS app-data dir | Where models + transcripts live |
| `LAZYMEET_SPEAKER_THRESHOLD` | `0.6` | Voice-match strictness (cosine) |
| `LAZYMEET_MERGE_THRESHOLD` | `0.6` | Voiceprint similarity at which over-split diarized labels merge |
| `LAZYMEET_MIN_SPEAKER_SECONDS` | `15` | Speakers with less speech fold into the closest main voice |
| `LAZYMEET_NUM_THREADS` | ~75% of cores | ONNX/LLM CPU threads |
| `LAZYMEET_ASR_URL` / `LAZYMEET_LLM_URL` / … | release URLs | Override model downloads |
| `LAZYMEET_KEEP_AUDIO` | `1` | `1` = keep only the audio track for playback (video files are deleted after processing); `0` = delete recordings entirely (disables segment playback) |
| `LAZYMEET_FAKE` | unset | `1` = stubbed pipeline for dev/tests |
