//! First-run model downloader with progress events + the selectable model
//! catalogs. Port of `backend/models_setup.py`.
//!
//! All weights are token-free: ASR / VAD / diarization models come from
//! sherpa-onnx GitHub releases and the summarization GGUF from an ungated
//! Hugging Face repo. Missing models download concurrently, one thread per
//! file. Progress is pushed as `setup-progress` / `models-progress` events
//! (throttled) instead of the old poll-only dict; the payloads are identical
//! to what `GET /api/setup` / `GET /api/models` returned.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::Ctx;

fn env_or(name: &str, default: String) -> String {
    std::env::var(name).ok().filter(|v| !v.is_empty()).unwrap_or(default)
}

const REL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download";

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub url: String,
    /// File (relative to models_dir) whose existence means this item is ready.
    pub marker: PathBuf,
    /// If the URL is a .tar.bz2 archive, extract it and move the right files out.
    pub archive: bool,
    /// For archives: (path-inside-archive needle -> destination relative to models_dir)
    pub extract_map: Vec<(String, PathBuf)>,
}

impl ModelSpec {
    fn plain(name: &str, url: String, marker: &str) -> Self {
        Self {
            name: name.into(),
            url,
            marker: PathBuf::from(marker),
            archive: false,
            extract_map: Vec::new(),
        }
    }
    fn archive(name: &str, url: String, marker: &str, map: &[(&str, &str)]) -> Self {
        Self {
            name: name.into(),
            url,
            marker: PathBuf::from(marker),
            archive: true,
            extract_map: map.iter().map(|(n, d)| (n.to_string(), PathBuf::from(d))).collect(),
        }
    }
}

/// The five first-run models (order matters — the setup screen lists them).
pub fn model_specs() -> Vec<ModelSpec> {
    vec![
        ModelSpec::archive(
            "Parakeet ASR (multilingual)",
            env_or(
                "LAZYMEET_ASR_URL",
                format!("{REL}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"),
            ),
            "parakeet/encoder.onnx",
            &[
                ("encoder", "parakeet/encoder.onnx"),
                ("decoder", "parakeet/decoder.onnx"),
                ("joiner", "parakeet/joiner.onnx"),
                ("tokens.txt", "parakeet/tokens.txt"),
            ],
        ),
        ModelSpec::plain(
            "Silero VAD",
            env_or("LAZYMEET_VAD_URL", format!("{REL}/asr-models/silero_vad.onnx")),
            "silero_vad.onnx",
        ),
        ModelSpec::archive(
            "Diarization segmentation",
            env_or(
                "LAZYMEET_DIAR_SEG_URL",
                format!(
                    "{REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
                ),
            ),
            "diarization_segmentation.onnx",
            &[("model.onnx", "diarization_segmentation.onnx")],
        ),
        // English TitaNet — discriminates English/German voices far better than
        // the zh-cn model (which over-segmented real meetings).
        ModelSpec::plain(
            "Speaker embedding",
            env_or(
                "LAZYMEET_DIAR_EMB_URL",
                format!("{REL}/speaker-recongition-models/nemo_en_titanet_small.onnx"),
            ),
            "speaker_embedding.onnx",
        ),
        ModelSpec::plain(
            "Summarization LLM",
            env_or(
                "LAZYMEET_LLM_URL",
                "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
                    .into(),
            ),
            "qwen3.5-4b-q4_k_m.gguf",
        ),
    ]
}

#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub id: &'static str,
    pub label: String,
    pub size: String,
    /// ASR: loader kind ("nemo_transducer" | "whisper"); empty for LLMs.
    pub kind: &'static str,
    /// ASR: model dir under models_dir; LLM: the GGUF filename.
    pub path: PathBuf,
    pub pros: Vec<&'static str>,
    pub cons: Vec<&'static str>,
    pub spec: ModelSpec,
}

pub const ASR_DEFAULT: &str = "parakeet-tdt-0.6b-v3";
pub const LLM_DEFAULT: &str = "qwen3.5-4b";

pub fn asr_catalog() -> Vec<CatalogEntry> {
    let specs = model_specs();
    vec![
        CatalogEntry {
            id: "parakeet-tdt-0.6b-v3",
            label: "Parakeet TDT 0.6B v3 — 25 European languages (default)".into(),
            size: "≈700 MB".into(),
            kind: "nemo_transducer",
            path: PathBuf::from("parakeet"),
            pros: vec![
                "Best accuracy for European languages, incl. German and English",
                "Fastest option — transcribes far faster than real time on the CPU",
                "Detects the language by itself, with punctuation and capitalization",
            ],
            cons: vec![
                "Only 25 (European) languages — no Chinese, Japanese or Korean",
                "Less robust than Whisper on heavy accents or noisy recordings",
            ],
            spec: specs[0].clone(), // installed on first run
        },
        CatalogEntry {
            id: "whisper-small",
            label: "Whisper Small — 90+ languages".into(),
            size: "≈600 MB download".into(),
            kind: "whisper",
            path: PathBuf::from("whisper-small"),
            pros: vec![
                "Widest coverage — 90+ languages incl. Chinese, Japanese, Korean",
                "Robust with accents, background noise and far-from-mic speech",
                "Accepts a spoken-language hint (setting below)",
            ],
            cons: vec![
                "Several times slower than Parakeet",
                "Can invent text during long silences or music",
            ],
            spec: ModelSpec::archive(
                "Whisper Small ASR",
                env_or(
                    "LAZYMEET_WHISPER_SMALL_URL",
                    format!("{REL}/asr-models/sherpa-onnx-whisper-small.tar.bz2"),
                ),
                "whisper-small/encoder.onnx",
                &[
                    ("small-encoder.int8.onnx", "whisper-small/encoder.onnx"),
                    ("small-decoder.int8.onnx", "whisper-small/decoder.onnx"),
                    ("small-tokens.txt", "whisper-small/tokens.txt"),
                ],
            ),
        },
        CatalogEntry {
            id: "whisper-base",
            label: "Whisper Base — smallest, less accurate".into(),
            size: "≈150 MB download".into(),
            kind: "whisper",
            path: PathBuf::from("whisper-base"),
            pros: vec![
                "Smallest download and lowest memory use",
                "Fastest of the Whisper family",
                "Same 90+ language coverage",
            ],
            cons: vec![
                "Least accurate — noticeably more transcription errors",
                "Struggles with accents, jargon and people talking over each other",
            ],
            spec: ModelSpec::archive(
                "Whisper Base ASR",
                env_or(
                    "LAZYMEET_WHISPER_BASE_URL",
                    format!("{REL}/asr-models/sherpa-onnx-whisper-base.tar.bz2"),
                ),
                "whisper-base/encoder.onnx",
                &[
                    ("base-encoder.int8.onnx", "whisper-base/encoder.onnx"),
                    ("base-decoder.int8.onnx", "whisper-base/decoder.onnx"),
                    ("base-tokens.txt", "whisper-base/tokens.txt"),
                ],
            ),
        },
    ]
}

#[allow(clippy::too_many_arguments)] // catalog-table constructor
fn qwen35(
    id: &'static str,
    size: &str,
    env: &str,
    gb: &str,
    note: &str,
    pros: Vec<&'static str>,
    cons: Vec<&'static str>,
    default: bool,
) -> CatalogEntry {
    let fname = format!("qwen3.5-{}-q4_k_m.gguf", size.to_lowercase());
    CatalogEntry {
        id,
        label: format!("Qwen 3.5 {size} — {note}{}", if default { " (default)" } else { "" }),
        size: if default { gb.to_string() } else { format!("{gb} download") },
        kind: "",
        path: PathBuf::from(&fname),
        pros,
        cons,
        spec: ModelSpec::plain(
            &format!("Qwen 3.5 {size} LLM"),
            env_or(
                env,
                format!(
                    "https://huggingface.co/unsloth/Qwen3.5-{size}-GGUF/resolve/main/Qwen3.5-{size}-Q4_K_M.gguf"
                ),
            ),
            &fname,
        ),
    }
}

pub fn llm_catalog() -> Vec<CatalogEntry> {
    vec![
        qwen35(
            "qwen3.5-0.8b",
            "0.8B",
            "LAZYMEET_LLM_08B_URL",
            "≈0.5 GB",
            "tiny, fastest",
            vec![
                "Notes in seconds, even on modest hardware",
                "Tiny download, runs in about 1 GB of memory",
            ],
            vec![
                "Short, plain notes that miss nuance",
                "Action items often incomplete; weakest outside English",
            ],
            false,
        ),
        qwen35(
            "qwen3.5-2b",
            "2B",
            "LAZYMEET_LLM_2B_URL",
            "≈1.3 GB",
            "fast, light",
            vec![
                "Quick summaries with about 2 GB of memory",
                "Clearly better notes than the 0.8B",
            ],
            vec![
                "Takeaways stay brief — subtle decisions can be missed",
                "Less reliable than the 4B in languages other than English",
            ],
            false,
        ),
        qwen35(
            "qwen3.5-4b",
            "4B",
            "LAZYMEET_LLM_URL",
            "≈2.7 GB",
            "balanced",
            vec![
                "Good balance of quality and speed",
                "Reliable takeaways and action items",
                "Solid notes in German, English and other major languages",
            ],
            vec![
                "Needs roughly 3–4 GB of free memory",
                "A long meeting takes a few minutes on CPU-only machines",
            ],
            true,
        ),
        qwen35(
            "qwen3.5-9b",
            "9B",
            "LAZYMEET_LLM_9B_URL",
            "≈5.7 GB",
            "strongest, slowest",
            vec![
                "Highest-quality notes — best for long, complex or multilingual meetings",
                "Most faithful decisions and action items",
            ],
            vec![
                "Large download and needs roughly 7 GB of free memory",
                "Slowest — can take many minutes without GPU acceleration",
            ],
            false,
        ),
    ]
}

// --------------------------------------------------------------------------
// Download state
// --------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct ItemProgress {
    downloaded: u64,
    total: u64,
    error: Option<String>,
}

#[derive(Default)]
struct Progress {
    state: String, // "idle" | "downloading" | "ready" | "error"
    items: HashMap<String, ItemProgress>,
    error: Option<String>,
}

pub struct ModelsState {
    progress: Mutex<Progress>,
    first_run_active: AtomicBool,
    inflight: Mutex<HashSet<String>>,
    /// Event throttle: last emission instant per event name.
    last_emit: Mutex<HashMap<String, Instant>>,
}

impl Default for ModelsState {
    fn default() -> Self {
        Self {
            progress: Mutex::new(Progress {
                state: "idle".into(),
                items: HashMap::new(),
                error: None,
            }),
            first_run_active: AtomicBool::new(false),
            inflight: Mutex::new(HashSet::new()),
            last_emit: Mutex::new(HashMap::new()),
        }
    }
}

const EMIT_EVERY: Duration = Duration::from_millis(250);

/// Emit `event` with `payload`, rate-limited unless `force`.
fn emit_throttled(ctx: &Ctx, event: &str, payload: Value, force: bool) {
    if !force {
        let mut last = ctx.models.last_emit.lock().unwrap();
        let now = Instant::now();
        if let Some(prev) = last.get(event) {
            if now.duration_since(*prev) < EMIT_EVERY {
                return;
            }
        }
        last.insert(event.to_string(), now);
    }
    ctx.events.emit(event, payload);
}

fn marker_path(ctx: &Ctx, spec: &ModelSpec) -> PathBuf {
    ctx.paths.models_dir.join(&spec.marker)
}

pub fn is_ready(ctx: &Ctx) -> bool {
    if ctx.fake {
        return true;
    }
    model_specs().iter().all(|s| marker_path(ctx, s).exists())
}

/// The `GET /api/setup` payload (also the `setup-progress` event payload).
pub fn status(ctx: &Ctx) -> Value {
    let prog = ctx.models.progress.lock().unwrap();
    let items: Vec<Value> = model_specs()
        .iter()
        .map(|s| {
            let p = prog.items.get(&s.name).cloned().unwrap_or_default();
            json!({
                "name": s.name,
                "ready": marker_path(ctx, s).exists() || ctx.fake,
                "downloaded": p.downloaded,
                "total": p.total,
            })
        })
        .collect();
    let ready = if ctx.fake {
        true
    } else {
        model_specs().iter().all(|s| marker_path(ctx, s).exists())
    };
    json!({
        "ready": ready,
        "state": if ready { "ready" } else { prog.state.as_str() },
        "error": prog.error,
        "items": items,
    })
}

/// Kick off background downloads of any missing first-run models (idempotent).
pub fn ensure_models_async(ctx: &Arc<Ctx>) {
    if ctx.fake || is_ready(ctx) {
        return;
    }
    if ctx.models.first_run_active.swap(true, Ordering::SeqCst) {
        return;
    }
    {
        let mut prog = ctx.models.progress.lock().unwrap();
        prog.state = "downloading".into();
        prog.error = None;
    }
    emit_throttled(ctx, "setup-progress", status(ctx), true);
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        download_all(&ctx);
        ctx.models.first_run_active.store(false, Ordering::SeqCst);
    });
}

fn download_all(ctx: &Arc<Ctx>) {
    let result = (|| -> Result<(), String> {
        ctx.paths.ensure_dirs().map_err(|e| e.to_string())?;
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for spec in model_specs() {
            if marker_path(ctx, &spec).exists() {
                continue;
            }
            let ctx = ctx.clone();
            let errors = errors.clone();
            handles.push(std::thread::spawn(move || {
                if let Err(e) = download_one(&ctx, &spec, "setup-progress") {
                    errors.lock().unwrap().push(format!("{}: {e}", spec.name));
                }
            }));
        }
        for h in handles {
            let _ = h.join();
        }
        let errors = errors.lock().unwrap();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    })();

    match result {
        Ok(()) => {
            ctx.models.progress.lock().unwrap().state = "ready".into();
            emit_throttled(ctx, "setup-progress", status(ctx), true);
            // Uploads made during the download sat in the queue as "queued".
            crate::jobs::pump_queue(ctx);
        }
        Err(e) => {
            let mut prog = ctx.models.progress.lock().unwrap();
            prog.state = "error".into();
            prog.error = Some(e);
            drop(prog);
            emit_throttled(ctx, "setup-progress", status(ctx), true);
        }
    }
}

// --------------------------------------------------------------------------
// Selectable models: resolution + on-demand downloads
// --------------------------------------------------------------------------

fn entry_ready(ctx: &Ctx, entry: &CatalogEntry) -> bool {
    ctx.fake || marker_path(ctx, &entry.spec).exists()
}

pub struct AsrPaths {
    pub kind: &'static str,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

/// Paths + loader kind for the active ASR model, defaulting when missing.
pub fn resolve_asr(ctx: &Ctx, model_id: &str) -> AsrPaths {
    resolve_asr_at(&ctx.paths.models_dir, ctx.fake, model_id)
}

/// Ctx-free variant for the pipeline (which only carries models_dir + fake).
pub fn resolve_asr_at(models_dir: &std::path::Path, fake: bool, model_id: &str) -> AsrPaths {
    let catalog = asr_catalog();
    let entry = catalog
        .iter()
        .find(|e| e.id == model_id)
        .filter(|e| fake || models_dir.join(&e.spec.marker).exists())
        .or_else(|| catalog.iter().find(|e| e.id == ASR_DEFAULT))
        .unwrap();
    let d = models_dir.join(&entry.path);
    AsrPaths {
        kind: entry.kind,
        encoder: d.join("encoder.onnx"),
        decoder: d.join("decoder.onnx"),
        joiner: d.join("joiner.onnx"),
        tokens: d.join("tokens.txt"),
    }
}

/// GGUF path for the active summarizer, defaulting when missing.
pub fn resolve_llm(ctx: &Ctx, model_id: &str) -> PathBuf {
    resolve_llm_at(&ctx.paths.models_dir, ctx.fake, model_id)
}

pub fn resolve_llm_at(models_dir: &std::path::Path, fake: bool, model_id: &str) -> PathBuf {
    let catalog = llm_catalog();
    let entry = catalog
        .iter()
        .find(|e| e.id == model_id)
        .filter(|e| fake || models_dir.join(&e.spec.marker).exists())
        .or_else(|| catalog.iter().find(|e| e.id == LLM_DEFAULT))
        .unwrap();
    models_dir.join(&entry.path)
}

fn catalog_options(ctx: &Ctx, catalog: &[CatalogEntry]) -> Vec<Value> {
    let prog = ctx.models.progress.lock().unwrap();
    let inflight = ctx.models.inflight.lock().unwrap();
    catalog
        .iter()
        .map(|entry| {
            let p = prog.items.get(&entry.spec.name).cloned().unwrap_or_default();
            json!({
                "id": entry.id,
                "label": entry.label,
                "size": entry.size,
                "pros": entry.pros,
                "cons": entry.cons,
                "ready": entry_ready(ctx, entry),
                "downloading": inflight.contains(&entry.spec.name),
                "downloaded": p.downloaded,
                "total": p.total,
                "error": p.error,
            })
        })
        .collect()
}

/// The `GET /api/models` payload (also the `models-progress` event payload).
pub fn models_status(ctx: &Ctx) -> Value {
    let config = ctx.config.read().unwrap();
    json!({
        "asr": {"active": config.asr_model(), "options": catalog_options(ctx, &asr_catalog())},
        "summary": {"active": config.summary_model(), "options": catalog_options(ctx, &llm_catalog())},
    })
}

/// Start a background download of one selectable model (idempotent).
/// Returns false for an unknown step/id; already-ready and already-running
/// downloads return true without doing anything.
pub fn request_download(ctx: &Arc<Ctx>, step: &str, model_id: &str) -> bool {
    let catalog = match step {
        "asr" => asr_catalog(),
        "summary" => llm_catalog(),
        _ => return false,
    };
    let Some(entry) = catalog.into_iter().find(|e| e.id == model_id) else {
        return false;
    };
    if ctx.fake || entry_ready(ctx, &entry) {
        return true;
    }
    let spec = entry.spec;
    {
        let mut inflight = ctx.models.inflight.lock().unwrap();
        if inflight.contains(&spec.name) {
            return true;
        }
        inflight.insert(spec.name.clone());
        ctx.models
            .progress
            .lock()
            .unwrap()
            .items
            .insert(spec.name.clone(), ItemProgress::default());
    }
    emit_throttled(ctx, "models-progress", models_status(ctx), true);
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let result = ctx.paths.ensure_dirs().map_err(|e| e.to_string()).and_then(|_| {
            download_one(&ctx, &spec, "models-progress")
        });
        if let Err(e) = result {
            let mut prog = ctx.models.progress.lock().unwrap();
            prog.items.entry(spec.name.clone()).or_default().error = Some(e);
        }
        ctx.models.inflight.lock().unwrap().remove(&spec.name);
        emit_throttled(&ctx, "models-progress", models_status(&ctx), true);
    });
    true
}

/// Kick off downloads for the currently selected models, if missing.
pub fn ensure_selected_async(ctx: &Arc<Ctx>) {
    let (asr, summary) = {
        let config = ctx.config.read().unwrap();
        (config.asr_model(), config.summary_model())
    };
    request_download(ctx, "asr", &asr);
    request_download(ctx, "summary", &summary);
}

// --------------------------------------------------------------------------
// The actual download + archive extraction
// --------------------------------------------------------------------------

fn set_item_progress(ctx: &Ctx, name: &str, downloaded: u64, total: u64) {
    let mut prog = ctx.models.progress.lock().unwrap();
    let item = prog.items.entry(name.to_string()).or_default();
    item.downloaded = downloaded;
    item.total = total;
}

fn download_one(ctx: &Arc<Ctx>, spec: &ModelSpec, event: &str) -> Result<(), String> {
    let tmp = ctx.paths.models_dir.join(format!("{}.download", spec.name.replace(' ', "_")));
    let resp = ureq::get(&spec.url)
        .set("User-Agent", "LazyMeet")
        .call()
        .map_err(|e| e.to_string())?;
    let total: u64 =
        resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
    set_item_progress(ctx, &spec.name, 0, total);

    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 1 << 20];
    let mut downloaded: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n]).map_err(|e| e.to_string())?;
        downloaded += n as u64;
        set_item_progress(ctx, &spec.name, downloaded, total);
        let payload = if event == "setup-progress" { status(ctx) } else { models_status(ctx) };
        emit_throttled(ctx, event, payload, false);
    }
    drop(file);

    if spec.archive {
        let result = extract_archive(ctx, &tmp, spec);
        let _ = std::fs::remove_file(&tmp);
        result?;
    } else {
        let dest = marker_path(ctx, spec);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Extract the mapped files, committing them atomically: everything staged as
/// `*.part` first and only renamed into place once the whole archive
/// extracted, with the marker file renamed last — a crash mid-extract must
/// never leave a truncated model that satisfies `is_ready()`.
fn extract_archive(ctx: &Ctx, archive_path: &PathBuf, spec: &ModelSpec) -> Result<(), String> {
    let mut staged: HashMap<PathBuf, PathBuf> = HashMap::new(); // dest rel -> .part path
    let result = (|| -> Result<(), String> {
        let file = std::fs::File::open(archive_path).map_err(|e| e.to_string())?;
        let mut archive = tar::Archive::new(bzip2::read::BzDecoder::new(file));
        for entry in archive.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path().map_err(|e| e.to_string())?.into_owned();
            let base = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            for (needle, dest_rel) in &spec.extract_map {
                if staged.contains_key(dest_rel) {
                    continue;
                }
                if base == *needle || (base.contains(needle.as_str()) && base.ends_with(".onnx")) {
                    let dest = ctx.paths.models_dir.join(dest_rel);
                    if let Some(parent) = dest.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                    let part = dest.with_extension(format!(
                        "{}.part",
                        dest.extension().and_then(|e| e.to_str()).unwrap_or("")
                    ));
                    let mut out = std::fs::File::create(&part).map_err(|e| e.to_string())?;
                    std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
                    staged.insert(dest_rel.clone(), part);
                    break;
                }
            }
        }
        let missing: Vec<String> = spec
            .extract_map
            .iter()
            .filter(|(_, d)| !staged.contains_key(d))
            .map(|(_, d)| d.display().to_string())
            .collect();
        if !missing.is_empty() {
            return Err(format!("{}: archive is missing {}", spec.name, missing.join(", ")));
        }
        for (dest_rel, part) in &staged {
            if *dest_rel != spec.marker {
                std::fs::rename(part, ctx.paths.models_dir.join(dest_rel))
                    .map_err(|e| e.to_string())?;
            }
        }
        std::fs::rename(&staged[&spec.marker], ctx.paths.models_dir.join(&spec.marker))
            .map_err(|e| e.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        for part in staged.values() {
            let _ = std::fs::remove_file(part);
        }
    }
    result
}
