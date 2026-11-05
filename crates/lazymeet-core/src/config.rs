//! Central configuration: app-data paths, tunables, persistence.
//!
//! Port of `backend/config.py`. Precedence per tunable: environment variable >
//! `config.json` override > default. Paths are env-only (`LAZYMEET_DATA_DIR`,
//! `LAZYMEET_MODELS_DIR`).
//!
//! ⚠ Path compatibility: the Python backend used `platformdirs.user_data_dir(
//! "LazyMeet", appauthor=False)`. Existing users' data lives there, so this
//! must resolve identically — NOT to Tauri's bundle-identifier directory:
//!   macOS   ~/Library/Application Support/LazyMeet
//!   Windows %LOCALAPPDATA%\LazyMeet
//!   Linux   $XDG_DATA_HOME/LazyMeet or ~/.local/share/LazyMeet

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::error::{ApiError, ApiResult};

pub const APP_NAME: &str = "LazyMeet";

/// Extracted-audio filename for video uploads (kept for playback).
pub const PLAYBACK_FILENAME: &str = "playback.m4a";

/// Scratch dir under uploads/ for the prep screen's audio preview (formats
/// the webview can't play directly). One file at a time, overwritten per
/// probe and swept on startup. Uploads/ is asset-protocol scoped, so the
/// preview streams without extra scope grants; the dotted name can never
/// collide with a meeting id (hex uuid).
pub const PREVIEW_DIR: &str = ".preview";

// --------------------------------------------------------------------------
// Paths
// --------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Paths {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
}

impl Paths {
    /// Resolve from the environment, exactly like the Python backend did.
    pub fn from_env() -> Self {
        let data_dir = std::env::var_os("LAZYMEET_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(default_data_dir);
        let models_dir = std::env::var_os("LAZYMEET_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("models"));
        Self { data_dir, models_dir }
    }

    /// Explicit paths (tests, CLI --data-dir).
    pub fn at(data_dir: impl Into<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        let models_dir = data_dir.join("models");
        Self { data_dir, models_dir }
    }

    pub fn meetings_dir(&self) -> PathBuf {
        self.data_dir.join("meetings")
    }
    pub fn uploads_dir(&self) -> PathBuf {
        self.data_dir.join("uploads")
    }
    pub fn speakers_file(&self) -> PathBuf {
        self.data_dir.join("speakers.json")
    }
    pub fn config_file(&self) -> PathBuf {
        self.data_dir.join("config.json")
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        for d in [&self.data_dir, &self.models_dir, &self.meetings_dir(), &self.uploads_dir()] {
            std::fs::create_dir_all(d)?;
        }
        Ok(())
    }
}

/// platformdirs-compatible per-user data dir (see module docs).
fn default_data_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    let base = dirs::data_local_dir();
    #[cfg(not(target_os = "windows"))]
    let base = dirs::data_dir();
    base.unwrap_or_else(|| {
        // Python's fallback when platformdirs was unavailable.
        dirs::home_dir().unwrap_or_default().join(".local").join("share")
    })
    .join(APP_NAME)
}

// --------------------------------------------------------------------------
// Tunables
// --------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Str,
    Float,
    Int,
    Bool,
}

#[derive(Debug, Clone)]
pub struct TunableSpec {
    pub name: &'static str,
    pub env: &'static str,
    pub kind: Kind,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

pub fn cpu_count() -> i64 {
    std::thread::available_parallelism().map(|n| n.get() as i64).unwrap_or(8)
}

/// min(16, cores) — the ceiling shown in the settings UI.
pub fn max_threads() -> i64 {
    cpu_count().min(16)
}

/// The "Auto" thread count: 75 % of the cores, at most `max_threads`. More
/// threads than that buy nothing on these models (they stop scaling around
/// the physical core count) while still occupying the whole machine.
pub fn auto_threads() -> i64 {
    (cpu_count() * 3 / 4).clamp(1, max_threads())
}

/// The `NUM_THREADS` value meaning "pick for me": resolved at use time, so a
/// saved config keeps following the machine it runs on.
pub const AUTO_THREADS: i64 = 0;

/// Same order as Python's TUNABLES dict (the settings UI relies on it).
pub fn tunables() -> Vec<TunableSpec> {
    let t = |name, env, kind, min, max| TunableSpec { name, env, kind, min, max };
    vec![
        t("ASR_MODEL", "LAZYMEET_ASR_MODEL", Kind::Str, None, None),
        t("ASR_LANGUAGE", "LAZYMEET_ASR_LANGUAGE", Kind::Str, None, None),
        t("DIAR_MODEL", "LAZYMEET_DIAR_MODEL", Kind::Str, None, None),
        t("SUMMARY_MODEL", "LAZYMEET_SUMMARY_MODEL", Kind::Str, None, None),
        t("SUMMARY_LANGUAGE", "LAZYMEET_SUMMARY_LANGUAGE", Kind::Str, None, None),
        // For Str tunables, `max` is the character cap (default 200 when None).
        t("SUMMARY_PROMPT", "LAZYMEET_SUMMARY_PROMPT", Kind::Str, None, Some(2000.0)),
        // Per-stage ONNX execution provider — only "cpu" in this build (see
        // onnx_provider_options); kept per stage so an accelerator can slot
        // in again without a settings migration.
        t("ASR_PROVIDER", "LAZYMEET_ASR_PROVIDER", Kind::Str, None, None),
        t("DIAR_PROVIDER", "LAZYMEET_DIAR_PROVIDER", Kind::Str, None, None),
        t("SPEAKER_PROVIDER", "LAZYMEET_SPEAKER_PROVIDER", Kind::Str, None, None),
        t("SPEAKER_THRESHOLD", "LAZYMEET_SPEAKER_THRESHOLD", Kind::Float, Some(0.0), Some(1.0)),
        t("DIAR_THRESHOLD", "LAZYMEET_DIAR_THRESHOLD", Kind::Float, Some(0.0), Some(1.0)),
        t("SPEAKER_MERGE_THRESHOLD", "LAZYMEET_MERGE_THRESHOLD", Kind::Float, Some(0.0), Some(1.0)),
        t(
            "MIN_SPEAKER_SECONDS",
            "LAZYMEET_MIN_SPEAKER_SECONDS",
            Kind::Float,
            Some(0.0),
            Some(600.0),
        ),
        t("SUMMARY_PROVIDER", "LAZYMEET_SUMMARY_PROVIDER", Kind::Str, None, None),
        t(
            "NUM_THREADS",
            "LAZYMEET_NUM_THREADS",
            Kind::Int,
            Some(AUTO_THREADS as f64),
            Some(max_threads() as f64),
        ),
        t("KEEP_AUDIO", "LAZYMEET_KEEP_AUDIO", Kind::Bool, None, None),
    ]
}

/// Default summarizer system prompt (user-editable via SUMMARY_PROMPT). The
/// JSON response format and the notes-language instruction are appended by
/// the pipeline and are NOT part of this prompt, so editing it can't break
/// summary parsing or the language setting.
///
/// Written for small models: one rule per line, the anti-hallucination rule
/// first, and an explicit shape for every field. The short/long split is what
/// keeps the notes scannable and still detailed — a single paragraph prompt
/// makes 4B models produce either a bare list or a wall of text.
pub const DEFAULT_SUMMARY_PROMPT: &str = "You are an expert meeting note-taker. You read a \
transcript with speaker labels and write the notes someone who missed the meeting needs.\n\n\
Rules:\n\
- Use only what the transcript says. Never invent facts, names, numbers, owners, or deadlines.\n\
- Use the speakers' names exactly as written; say \"the team\" only when nobody is named.\n\
- Keep exact figures, dates, product names and decisions instead of paraphrasing them away.\n\
- Skip small talk, filler and anything that was merely mentioned in passing.\n\
- Write plain prose in the strings: no markdown, no bullet characters, no headings.\n\n\
What to write:\n\
- summary: 2-3 sentences — what the meeting was about and what came out of it.\n\
- summary_long: 2-4 short paragraphs, in order, covering the topics discussed, the \
reasoning and trade-offs, any disagreement, and the decisions reached.\n\
- takeaways: 3-6 points. short is one line (max ~15 words) stating the outcome; detail \
adds the context, number or reason behind it (1-2 sentences).\n\
- action_items: only concrete commitments (a named person does something). short is \
\"Owner: verb + object\" (~15 words); detail carries the deadline, dependency and how it \
was agreed. Leave this list empty rather than inventing owners or dates.";

pub fn default_value(name: &str) -> Value {
    match name {
        "ASR_MODEL" => json!("parakeet-tdt-0.6b-v3"),
        "ASR_LANGUAGE" => json!("auto"),
        "DIAR_MODEL" => json!("nemotron-3-diarization"),
        "SUMMARY_MODEL" => json!("qwen3.5-4b"),
        "SUMMARY_LANGUAGE" => json!("auto"),
        "SUMMARY_PROMPT" => json!(DEFAULT_SUMMARY_PROMPT),
        "ASR_PROVIDER" | "DIAR_PROVIDER" | "SPEAKER_PROVIDER" => json!("cpu"),
        "SUMMARY_PROVIDER" => json!(if cfg!(target_os = "macos") { "gpu" } else { "cpu" }),
        "SPEAKER_THRESHOLD" => json!(0.6),
        "DIAR_THRESHOLD" => json!(0.7),
        "SPEAKER_MERGE_THRESHOLD" => json!(0.6),
        "MIN_SPEAKER_SECONDS" => json!(15.0),
        "NUM_THREADS" => json!(AUTO_THREADS),
        "KEEP_AUDIO" => json!(true),
        _ => Value::Null,
    }
}

/// Parse+validate one tunable value; None if unusable (ignored, not clamped).
/// Mirrors Python `_coerce`, including its permissive numeric coercions.
pub fn coerce(spec: &TunableSpec, raw: &Value) -> Option<Value> {
    match spec.kind {
        Kind::Bool => match raw {
            Value::Bool(b) => Some(json!(b)),
            Value::String(s) => {
                let v = s.trim().to_ascii_lowercase();
                Some(json!(matches!(v.as_str(), "1" | "true" | "yes" | "on")))
            }
            _ => None,
        },
        Kind::Str => match raw {
            Value::String(s) => {
                let v = s.trim();
                let cap = spec.max.map(|m| m as usize).unwrap_or(200);
                if !v.is_empty() && v.chars().count() <= cap {
                    Some(json!(v))
                } else {
                    None
                }
            }
            _ => None,
        },
        Kind::Float => {
            let v = match raw {
                Value::Number(n) => n.as_f64()?,
                Value::Bool(b) => *b as i64 as f64, // float(True) == 1.0 in Python
                Value::String(s) => s.trim().parse::<f64>().ok()?,
                _ => return None,
            };
            in_range(spec, v).then(|| json!(v))
        }
        Kind::Int => {
            let v = match raw {
                Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64))?,
                Value::Bool(b) => *b as i64, // int(True) == 1 in Python
                Value::String(s) => s.trim().parse::<i64>().ok()?,
                _ => return None,
            };
            in_range(spec, v as f64).then(|| json!(v))
        }
    }
}

fn in_range(spec: &TunableSpec, v: f64) -> bool {
    match (spec.min, spec.max) {
        (Some(lo), Some(hi)) => v >= lo && v <= hi,
        _ => true,
    }
}

// --------------------------------------------------------------------------
// Snapshot + persistence
// --------------------------------------------------------------------------

/// One resolved view of every tunable (env > config.json > default).
/// The worker takes a fresh snapshot at job start, like Python's `refresh()`.
#[derive(Debug, Clone)]
pub struct Config {
    values: Map<String, Value>,
}

impl Config {
    pub fn resolve(paths: &Paths) -> Self {
        Self::from_overrides(&load_overrides(paths))
    }

    /// The snapshot `overrides` (config.json's contents) resolves to.
    pub fn from_overrides(overrides: &Map<String, Value>) -> Self {
        let mut values = Map::new();
        for spec in tunables() {
            let from_env = std::env::var(spec.env)
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| coerce(&spec, &json!(s)));
            let value = from_env
                .or_else(|| overrides.get(spec.name).cloned())
                .unwrap_or_else(|| default_value(spec.name));
            values.insert(spec.name.to_string(), value);
        }
        Self { values }
    }

    pub fn values(&self) -> &Map<String, Value> {
        &self.values
    }

    /// Replace one value in this snapshot — for per-meeting overrides (e.g. a
    /// language chosen on the prep screen) that outrank the saved settings
    /// for a single job without touching config.json.
    pub fn override_value(&mut self, name: &str, value: Value) {
        self.values.insert(name.to_string(), value);
    }

    fn str_of(&self, name: &str) -> String {
        self.values[name].as_str().unwrap_or_default().to_string()
    }
    fn f64_of(&self, name: &str) -> f64 {
        self.values[name].as_f64().unwrap_or_default()
    }

    pub fn asr_model(&self) -> String {
        self.str_of("ASR_MODEL")
    }
    pub fn asr_language(&self) -> String {
        self.str_of("ASR_LANGUAGE")
    }
    pub fn diar_model(&self) -> String {
        self.str_of("DIAR_MODEL")
    }
    pub fn summary_model(&self) -> String {
        self.str_of("SUMMARY_MODEL")
    }
    pub fn summary_language(&self) -> String {
        self.str_of("SUMMARY_LANGUAGE")
    }
    /// User-editable summarizer system prompt; DEFAULT_SUMMARY_PROMPT when unset.
    pub fn summary_prompt(&self) -> String {
        self.str_of("SUMMARY_PROMPT")
    }
    pub fn speaker_threshold(&self) -> f64 {
        self.f64_of("SPEAKER_THRESHOLD")
    }
    pub fn diar_threshold(&self) -> f64 {
        self.f64_of("DIAR_THRESHOLD")
    }
    pub fn speaker_merge_threshold(&self) -> f64 {
        self.f64_of("SPEAKER_MERGE_THRESHOLD")
    }
    pub fn min_speaker_seconds(&self) -> f64 {
        self.f64_of("MIN_SPEAKER_SECONDS")
    }
    /// The thread count to actually run with: `AUTO_THREADS` resolves to 75 %
    /// of the cores here, so the saved value stays machine-independent.
    pub fn num_threads(&self) -> i64 {
        match self.values["NUM_THREADS"].as_i64().unwrap_or(AUTO_THREADS) {
            AUTO_THREADS => auto_threads(),
            n => n.clamp(1, max_threads()),
        }
    }
    /// Whether `NUM_THREADS` is left on "Auto".
    pub fn auto_threads(&self) -> bool {
        self.values["NUM_THREADS"].as_i64().unwrap_or(AUTO_THREADS) == AUTO_THREADS
    }
    /// The summarizer runs on the GPU when SUMMARY_PROVIDER is "gpu".
    pub fn use_gpu(&self) -> bool {
        self.str_of("SUMMARY_PROVIDER") == "gpu"
    }
    pub fn summary_provider(&self) -> String {
        self.str_of("SUMMARY_PROVIDER")
    }
    pub fn keep_audio(&self) -> bool {
        self.values["KEEP_AUDIO"].as_bool().unwrap_or(true)
    }

    /// Per-stage ONNX execution provider; `LAZYMEET_ONNX_PROVIDER` forces all
    /// three at once (benchmark scripts / experiments). A value this build
    /// doesn't offer (e.g. "coreml" saved by an older version) means the CPU.
    fn provider_of(&self, name: &str) -> String {
        let value = std::env::var("LAZYMEET_ONNX_PROVIDER")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| self.str_of(name));
        if onnx_provider_options().contains(&value.as_str()) {
            value
        } else {
            "cpu".into()
        }
    }
    pub fn asr_provider(&self) -> String {
        self.provider_of("ASR_PROVIDER")
    }
    pub fn diar_provider(&self) -> String {
        self.provider_of("DIAR_PROVIDER")
    }
    pub fn speaker_provider(&self) -> String {
        self.provider_of("SPEAKER_PROVIDER")
    }
}

/// ONNX execution providers available in this build on this platform. Only
/// the CPU: the prebuilt static onnxruntime that sherpa-onnx links ships
/// without the CoreML provider (sherpa-onnx compiles CoreML out for static
/// builds), and CoreML measured slower than the CPU for these models anyway.
/// CoreML/CUDA/DirectML would slot in here once compiled in.
pub fn onnx_provider_options() -> Vec<&'static str> {
    vec!["cpu"]
}

/// Summarizer backends available in this build on this platform. llama.cpp's
/// accelerator here is the Metal GPU (macOS); CUDA/Vulkan would slot in once
/// compiled in. There is no NPU path for llama.cpp.
pub fn summary_provider_options() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    if cfg!(target_os = "macos") {
        v.push("gpu");
    }
    v
}

/// Read config.json; unknown keys and invalid values are dropped.
pub fn load_overrides(paths: &Paths) -> Map<String, Value> {
    let raw = match std::fs::read_to_string(paths.config_file()) {
        Ok(s) => s,
        Err(_) => return Map::new(),
    };
    let data: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return Map::new(),
    };
    let Value::Object(data) = data else { return Map::new() };
    let specs: BTreeMap<&str, TunableSpec> = tunables().into_iter().map(|s| (s.name, s)).collect();
    let mut out = Map::new();
    // Configs from before v0.1.1 carry USE_GPU, which SUMMARY_PROVIDER replaced:
    // carry the choice over (dropped from the file on the next save).
    let legacy_gpu = data.get("USE_GPU").and_then(Value::as_bool);
    for (name, raw) in data {
        if let Some(spec) = specs.get(name.as_str()) {
            if let Some(value) = coerce(spec, &raw) {
                out.insert(name, value);
            }
        }
    }
    if let Some(gpu) = legacy_gpu {
        let provider = if gpu { "gpu" } else { "cpu" };
        if !out.contains_key("SUMMARY_PROVIDER") && summary_provider_options().contains(&provider) {
            out.insert("SUMMARY_PROVIDER".into(), json!(provider));
        }
    }
    out
}

pub fn save_overrides(paths: &Paths, values: &Map<String, Value>) -> ApiResult<()> {
    crate::store::atomic_write_json(&paths.config_file(), &Value::Object(values.clone()))
        .map_err(|e| ApiError::internal(format!("could not save config: {e}")))
}

/// Tunables pinned by a non-empty environment variable (the settings UI locks them).
pub fn env_locked() -> Vec<String> {
    tunables()
        .iter()
        .filter(|s| std::env::var(s.env).map(|v| !v.is_empty()).unwrap_or(false))
        .map(|s| s.name.to_string())
        .collect()
}

// --------------------------------------------------------------------------
// Env-only knobs
// --------------------------------------------------------------------------

/// `LAZYMEET_FAKE=1` swaps the heavy pipeline stages for deterministic stubs.
pub fn fake_mode_from_env() -> bool {
    std::env::var("LAZYMEET_FAKE").map(|v| v == "1").unwrap_or(false)
}

/// Summarization LLM context window (`LAZYMEET_LLM_N_CTX`, default 16384).
pub fn llm_n_ctx() -> u32 {
    std::env::var("LAZYMEET_LLM_N_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(16384)
}

/// Forced diarization speaker count (`LAZYMEET_NUM_SPEAKERS`, -1 = auto).
pub fn env_num_speakers() -> i64 {
    std::env::var("LAZYMEET_NUM_SPEAKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(-1)
}

/// Embedding dimension used by the deterministic fake embedder.
pub const FAKE_EMBED_DIM: usize = 192;

/// True when the upload is already audio-only (no video track to strip).
pub fn is_audio_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "wav" | "mp3" | "m4a" | "aac" | "flac" | "ogg" | "oga" | "opus" | "wma" | "aiff" | "aif"
    )
}
