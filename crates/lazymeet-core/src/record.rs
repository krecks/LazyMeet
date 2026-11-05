//! Recording in the app: the microphone and the computer's own audio (the
//! other people on a call), each to its own WAV, mixed into one file when
//! the recording stops. That file then goes through the normal prepare →
//! transcribe flow, and `api::create_meeting` moves it instead of copying.
//!
//! System audio: Core Audio taps on macOS 14.6+, WASAPI loopback on Windows,
//! the PulseAudio/PipeWire monitor of the default output on Linux — all
//! through cpal. Where it isn't available the microphone still records.
//!
//! Each recording lives in `recordings/<stamp>/` until it is imported. The
//! WAV headers are rewritten every second, so a crash loses at most that;
//! folders left over at launch are offered again (`leftovers`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::error::{ApiError, ApiResult};
use crate::Ctx;

/// The mixed file a finished recording leaves in its folder.
pub const MIXED: &str = "recording.m4a";
const SOURCES: [&str; 2] = ["mic", "system"];

/// How the capture thread reports that it started: the sources recording
/// and any warnings, or why nothing could record.
type Ready = mpsc::Sender<Result<(Vec<&'static str>, Vec<String>), String>>;

/// The recording in progress, if any.
#[derive(Default)]
pub struct RecorderState {
    active: Mutex<Option<Control>>,
}

struct Control {
    stop: mpsc::Sender<bool>, // true = keep, false = discard
    done: mpsc::Receiver<Result<(), String>>,
    dir: PathBuf,
    started: Instant,
    sources: Vec<&'static str>,
}

/// The audio inputs to offer, and whether system audio can be recorded here.
pub fn list_inputs(ctx: &Ctx) -> ApiResult<Value> {
    if ctx.fake {
        return Ok(
            json!({"inputs": [{"name": "Fake microphone", "default": true}], "system_audio": true}),
        );
    }
    capture::list_inputs()
}

pub fn status(ctx: &Ctx) -> Value {
    match ctx.recorder.active.lock().unwrap().as_ref() {
        Some(c) => json!({
            "recording": true,
            "elapsed": c.started.elapsed().as_secs_f64(),
            "sources": c.sources,
        }),
        None => json!({"recording": false}),
    }
}

/// Start recording. `mic`: an input's name, None for the default, or "" for
/// no microphone; `system`: also record the computer's audio.
pub fn start(ctx: &Arc<Ctx>, mic: Option<&str>, system: bool) -> ApiResult<Value> {
    let mut active = ctx.recorder.active.lock().unwrap();
    if active.is_some() {
        return Err(ApiError::conflict("already recording"));
    }
    if mic == Some("") && !system {
        return Err(ApiError::invalid("pick a microphone or system audio to record"));
    }
    let stamp = chrono::Local::now().format("%Y-%m-%d %H-%M-%S").to_string();
    let dir = ctx.paths.recordings_dir().join(&stamp);
    std::fs::create_dir_all(&dir)?;

    let (stop_tx, stop_rx) = mpsc::channel::<bool>();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();
    let thread_ctx = ctx.clone();
    let thread_dir = dir.clone();
    let mic = mic.map(str::to_string);
    std::thread::spawn(move || {
        let result = if thread_ctx.fake {
            fake_capture(&thread_ctx, &thread_dir, system, stop_rx, ready_tx)
        } else {
            capture::run(&thread_ctx, &thread_dir, mic.as_deref(), system, stop_rx, ready_tx)
        };
        let _ = done_tx.send(result);
    });
    let (sources, warnings) = match ready_rx.recv_timeout(Duration::from_secs(15)) {
        Ok(Ok(started)) => started,
        Ok(Err(e)) => {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(ApiError::invalid(e));
        }
        Err(_) => {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(ApiError::internal("the audio device did not start"));
        }
    };
    *active = Some(Control {
        stop: stop_tx,
        done: done_rx,
        dir,
        started: Instant::now(),
        sources: sources.clone(),
    });
    Ok(json!({"recording": true, "sources": sources, "warnings": warnings, "name": stamp}))
}

/// Stop and mix: returns {path} of the file to prepare and transcribe.
pub fn stop(ctx: &Ctx) -> ApiResult<Value> {
    let control = ctx
        .recorder
        .active
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| ApiError::conflict("not recording"))?;
    let _ = control.stop.send(true);
    match control.done.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(ApiError::internal(format!("the recording failed: {e}"))),
        Err(_) => return Err(ApiError::internal("the recording stopped unexpectedly")),
    }
    let path = finish(ctx, &control.dir)?;
    Ok(json!({"path": path.to_string_lossy(), "seconds": control.started.elapsed().as_secs_f64()}))
}

/// Stop and throw the recording away.
pub fn cancel(ctx: &Ctx) -> ApiResult<Value> {
    if let Some(control) = ctx.recorder.active.lock().unwrap().take() {
        let _ = control.stop.send(false);
        let _ = control.done.recv();
        let _ = std::fs::remove_dir_all(&control.dir);
    }
    Ok(json!({"recording": false}))
}

/// The folder's single file to hand on: the sources mixed (or the only one
/// converted) to `recording.m4a`. Without ffmpeg (tests) a lone WAV is used
/// as it is.
fn finish(ctx: &Ctx, dir: &Path) -> ApiResult<PathBuf> {
    let mixed = dir.join(MIXED);
    if mixed.is_file() {
        return Ok(mixed);
    }
    let inputs: Vec<PathBuf> = SOURCES
        .iter()
        .map(|s| dir.join(format!("{s}.wav")))
        .filter(|p| p.metadata().is_ok_and(|m| m.len() > 44))
        .collect();
    if inputs.is_empty() {
        return Err(ApiError::invalid("nothing was recorded"));
    }
    let Some(ffmpeg) = ctx.ffmpeg.as_ref() else {
        return Ok(inputs[0].clone());
    };
    let out = std::process::Command::new(ffmpeg)
        .args(mix_args(&inputs, &mixed))
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| ApiError::internal(format!("could not run ffmpeg: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: String = err.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
        return Err(ApiError::internal(format!("mixing the recording failed: {tail}")));
    }
    for input in &inputs {
        let _ = std::fs::remove_file(input);
    }
    Ok(mixed)
}

/// ffmpeg arguments mixing the source WAVs into one mono AAC file. Both
/// sources keep their level (`normalize=0`) — a quiet microphone shouldn't
/// be halved because system audio is there too.
pub fn mix_args(inputs: &[PathBuf], out: &Path) -> Vec<String> {
    let mut args: Vec<String> =
        vec!["-hide_banner".into(), "-loglevel".into(), "error".into(), "-y".into()];
    for input in inputs {
        args.push("-i".into());
        args.push(input.to_string_lossy().into_owned());
    }
    if inputs.len() > 1 {
        args.push("-filter_complex".into());
        args.push(format!("amix=inputs={}:duration=longest:normalize=0", inputs.len()));
    }
    for a in ["-ac", "1", "-c:a", "aac", "-b:a", "96k"] {
        args.push(a.into());
    }
    args.push(out.to_string_lossy().into_owned());
    args
}

/// Recordings never imported (the app quit or crashed, or the prepare
/// screen was cancelled): one mixed file per folder, newest first. Empty
/// folders are removed.
pub fn leftovers(ctx: &Ctx) -> Vec<Value> {
    let recording = ctx.recorder.active.lock().unwrap().as_ref().map(|c| c.dir.clone());
    let Ok(entries) = std::fs::read_dir(ctx.paths.recordings_dir()) else { return Vec::new() };
    let mut out: Vec<(String, Value)> = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() || Some(&dir) == recording.as_ref() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        match finish(ctx, &dir) {
            Ok(path) => {
                out.push((name.clone(), json!({"name": name, "path": path.to_string_lossy()})))
            }
            Err(_) => {
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.into_iter().map(|(_, v)| v).collect()
}

/// Delete a leftover recording the user doesn't want.
pub fn discard(ctx: &Ctx, path: &Path) -> ApiResult<Value> {
    let dir =
        owned_folder(ctx, path).ok_or_else(|| ApiError::invalid("not a LazyMeet recording"))?;
    let _ = std::fs::remove_dir_all(dir);
    Ok(json!({"discarded": true}))
}

/// The recording folder `path` sits in, if it is one of ours.
pub fn owned_folder(ctx: &Ctx, path: &Path) -> Option<PathBuf> {
    let root = ctx.paths.recordings_dir().canonicalize().ok()?;
    let path = path.canonicalize().ok()?;
    let dir = path.parent()?;
    (dir.parent()? == root).then(|| dir.to_path_buf())
}

/// Level meters: RMS of the latest audio, per source, as f32 bits.
#[derive(Default)]
struct Levels([AtomicU32; 2]);

impl Levels {
    fn set(&self, source: usize, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        self.0[source].store(rms.to_bits(), Ordering::Relaxed);
    }
    fn get(&self, source: usize) -> f32 {
        f32::from_bits(self.0[source].load(Ordering::Relaxed))
    }
}

fn emit_levels(ctx: &Ctx, levels: &Levels, started: Instant) {
    ctx.events.emit(
        "record-level",
        json!({"mic": levels.get(0), "system": levels.get(1), "elapsed": started.elapsed().as_secs_f64()}),
    );
}

/// FAKE mode: no devices; level events at a fixed level, and a short silent
/// WAV per source when stopped.
fn fake_capture(
    ctx: &Ctx,
    dir: &Path,
    system: bool,
    stop: mpsc::Receiver<bool>,
    ready: Ready,
) -> Result<(), String> {
    let sources: Vec<&'static str> = if system { vec!["mic", "system"] } else { vec!["mic"] };
    let _ = ready.send(Ok((sources.clone(), Vec::new())));
    let levels = Levels::default();
    levels.set(0, &[0.1]);
    let started = Instant::now();
    let keep = loop {
        match stop.recv_timeout(Duration::from_millis(100)) {
            Ok(keep) => break keep,
            Err(mpsc::RecvTimeoutError::Disconnected) => break true,
            Err(mpsc::RecvTimeoutError::Timeout) => emit_levels(ctx, &levels, started),
        }
    };
    if keep {
        for s in sources {
            crate::pipeline::fake::write_silent_wav(&dir.join(format!("{s}.wav")))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(feature = "capture")]
mod capture {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    fn device_name(device: &cpal::Device) -> String {
        device.description().map(|d| d.name().to_string()).unwrap_or_default()
    }

    /// Linux records system audio from a monitor source, which only the
    /// PulseAudio host exposes (PipeWire serves it too, via pipewire-pulse).
    fn host() -> cpal::Host {
        #[cfg(target_os = "linux")]
        if let Ok(host) = cpal::host_from_id(cpal::HostId::PulseAudio) {
            return host;
        }
        cpal::default_host()
    }

    fn is_monitor(device: &cpal::Device) -> bool {
        device_name(device).to_lowercase().contains("monitor")
    }

    /// The device whose input is the computer's own audio.
    fn system_device(host: &cpal::Host) -> Option<cpal::Device> {
        if cfg!(target_os = "linux") {
            host.input_devices().ok()?.find(is_monitor)
        } else {
            // Core Audio taps / WASAPI loopback: an input stream on the
            // default output device records what it plays.
            host.default_output_device()
        }
    }

    pub fn list_inputs() -> ApiResult<Value> {
        let host = host();
        let default = host.default_input_device().map(|d| device_name(&d));
        let inputs: Vec<Value> = host
            .input_devices()
            .map_err(|e| ApiError::internal(format!("could not list audio inputs: {e}")))?
            .filter(|d| !is_monitor(d))
            .map(|d| {
                let name = device_name(&d);
                json!({"name": name, "default": Some(&name) == default.as_ref()})
            })
            .collect();
        Ok(json!({"inputs": inputs, "system_audio": system_device(&host).is_some()}))
    }

    /// One source: its stream (kept alive), where its audio arrives, and the
    /// WAV it goes to.
    struct Source {
        _stream: cpal::Stream,
        rx: mpsc::Receiver<Vec<f32>>,
        wav: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    }

    fn open(
        device: &cpal::Device,
        loopback: bool,
        index: usize,
        dir: &Path,
        levels: &Arc<Levels>,
    ) -> Result<Source, String> {
        let config = if loopback && !cfg!(target_os = "linux") {
            device.default_output_config()
        } else {
            device.default_input_config()
        }
        .map_err(|e| e.to_string())?;
        let channels = config.channels() as usize;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: config.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let wav = hound::WavWriter::create(dir.join(format!("{}.wav", SOURCES[index])), spec)
            .map_err(|e| e.to_string())?;
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(512);
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => build::<f32>(device, &config, channels, index, tx, levels),
            cpal::SampleFormat::I16 => build::<i16>(device, &config, channels, index, tx, levels),
            cpal::SampleFormat::I32 => build::<i32>(device, &config, channels, index, tx, levels),
            cpal::SampleFormat::U16 => build::<u16>(device, &config, channels, index, tx, levels),
            other => return Err(format!("unsupported sample format {other}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Source { _stream: stream, rx, wav })
    }

    fn build<T>(
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
        channels: usize,
        index: usize,
        tx: mpsc::SyncSender<Vec<f32>>,
        levels: &Arc<Levels>,
    ) -> Result<cpal::Stream, String>
    where
        T: cpal::SizedSample,
        f32: cpal::FromSample<T>,
    {
        use cpal::Sample;
        let levels = levels.clone();
        device
            .build_input_stream(
                config.config(),
                move |data: &[T], _: &cpal::InputCallbackInfo| {
                    // Down-mix to mono here; the WAV thread only writes.
                    let mono: Vec<f32> = data
                        .chunks(channels.max(1))
                        .map(|f| {
                            f.iter().map(|s| f32::from_sample(*s)).sum::<f32>() / f.len() as f32
                        })
                        .collect();
                    levels.set(index, &mono);
                    let _ = tx.try_send(mono); // never block the audio thread
                },
                |e| eprintln!("recording: stream error: {e}"),
                None,
            )
            .map_err(|e| e.to_string())
    }

    pub fn run(
        ctx: &Ctx,
        dir: &Path,
        mic: Option<&str>,
        system: bool,
        stop: mpsc::Receiver<bool>,
        ready: Ready,
    ) -> Result<(), String> {
        let host = host();
        let levels = Arc::new(Levels::default());
        let mut sources: Vec<Source> = Vec::new();
        let mut names: Vec<&'static str> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();
        if mic != Some("") {
            let device = match mic {
                Some(name) => host
                    .input_devices()
                    .ok()
                    .and_then(|mut it| it.find(|d| device_name(d) == name))
                    .or_else(|| host.default_input_device()),
                None => host.default_input_device(),
            };
            match device
                .ok_or("no microphone found".to_string())
                .and_then(|d| open(&d, false, 0, dir, &levels))
            {
                Ok(src) => {
                    sources.push(src);
                    names.push("mic");
                }
                Err(e) => warnings.push(format!("The microphone could not be recorded: {e}")),
            }
        }
        if system {
            let opened = system_device(&host)
                .ok_or("no system audio device".to_string())
                .and_then(|d| open(&d, true, 1, dir, &levels));
            match opened {
                Ok(src) => {
                    sources.push(src);
                    names.push("system");
                }
                Err(e) => warnings.push(format!(
                    "System audio could not be recorded ({e}) — on a Mac it needs macOS 14.6 or newer and the permission to record system audio."
                )),
            }
        }
        if sources.is_empty() {
            let reason = warnings.join(" ");
            let _ = ready.send(Err(if reason.is_empty() {
                "nothing to record".into()
            } else {
                reason
            }));
            return Ok(());
        }
        let _ = ready.send(Ok((names, warnings)));

        let started = Instant::now();
        let mut last_flush = Instant::now();
        let keep = loop {
            match stop.recv_timeout(Duration::from_millis(100)) {
                Ok(keep) => break keep,
                Err(mpsc::RecvTimeoutError::Disconnected) => break true,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            for src in &mut sources {
                drain(&src.rx, &mut src.wav)?;
            }
            emit_levels(ctx, &levels, started);
            if last_flush.elapsed() >= Duration::from_secs(1) {
                for src in &mut sources {
                    src.wav.flush().map_err(|e| e.to_string())?;
                }
                last_flush = Instant::now();
            }
        };
        // Streams stop when dropped; write what they delivered until then.
        let mut writers = Vec::new();
        for Source { _stream, rx, mut wav } in sources {
            drop(_stream);
            drain(&rx, &mut wav)?;
            writers.push(wav);
        }
        if keep {
            for wav in writers {
                wav.finalize().map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn drain(
        rx: &mpsc::Receiver<Vec<f32>>,
        wav: &mut hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    ) -> Result<(), String> {
        while let Ok(chunk) = rx.try_recv() {
            for s in chunk {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                wav.write_sample(v).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "capture"))]
mod capture {
    use super::*;

    pub fn list_inputs() -> ApiResult<Value> {
        Ok(json!({"inputs": [], "system_audio": false, "unavailable": true}))
    }

    pub fn run(
        _: &Ctx,
        _: &Path,
        _: Option<&str>,
        _: bool,
        _: mpsc::Receiver<bool>,
        ready: Ready,
    ) -> Result<(), String> {
        let _ = ready.send(Err("this build can't record audio".into()));
        Ok(())
    }
}
