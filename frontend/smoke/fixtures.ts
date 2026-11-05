// Backend payloads for the smoke stub, shaped like api::config_payload and
// models::models_status. catalog.json is the real catalog text (label, size,
// pros, cons, languages) dumped from models_status; the download states
// below are made up to exercise every card state.
import catalog from "./catalog.json";
const LANGUAGES = [
  ["en", "English"], ["de", "German"], ["fr", "French"], ["es", "Spanish"],
  ["it", "Italian"], ["pt", "Portuguese"], ["nl", "Dutch"], ["pl", "Polish"],
  ["ru", "Russian"], ["uk", "Ukrainian"], ["tr", "Turkish"], ["zh", "Chinese"],
  ["ja", "Japanese"], ["ko", "Korean"], ["ar", "Arabic"], ["el", "Greek"], ["vi", "Vietnamese"],
];

const DEFAULTS = {
  ASR_MODEL: "parakeet-tdt-0.6b-v3",
  ASR_LANGUAGE: "auto",
  DIAR_MODEL: "nemotron-3-diarization",
  SUMMARY_MODEL: "qwen3.5-4b",
  SUMMARY_LANGUAGE: "auto",
  SUMMARY_PROMPT: "You are a meeting assistant. Given a transcript with speaker labels, produce a concise summary, the key takeaways, and concrete action items.",
  ASR_PROVIDER: "cpu",
  DIAR_PROVIDER: "cpu",
  SPEAKER_PROVIDER: "cpu",
  SPEAKER_THRESHOLD: 0.6,
  DIAR_THRESHOLD: 0.7,
  SPEAKER_MERGE_THRESHOLD: 0.6,
  MIN_SPEAKER_SECONDS: 15,
  SUMMARY_PROVIDER: "gpu",
  NUM_THREADS: 0,
  KEEP_AUDIO: true,
};
export const LIMITS = {
  SPEAKER_THRESHOLD: { min: 0, max: 1 },
  DIAR_THRESHOLD: { min: 0, max: 1 },
  SPEAKER_MERGE_THRESHOLD: { min: 0, max: 1 },
  MIN_SPEAKER_SECONDS: { min: 0, max: 600 },
  NUM_THREADS: { min: 0, max: 12 },
  SUMMARY_PROMPT: { min: null, max: 2000 },
};

export function config() {
  return {
    values: { ...DEFAULTS },
    overrides: {},
    defaults: { ...DEFAULTS },
    env_locked: [],
    limits: LIMITS,
    cores: 10,
    auto_threads: 7,
    languages: LANGUAGES,
    compute_options: { SUMMARY_PROVIDER: ["cpu", "gpu"] },
  };
}

// name/tagline: the label split at " — ", as CatalogEntry::short_name does.
const option = (id, extra = {}) => {
  const { label, ...rest } = catalog[id];
  const [name, tagline = ""] = label.split(" — ");
  return {
    id,
    name,
    tagline,
    ...rest,
    ready: false,
    downloading: false,
    downloaded: 0,
    total: 0,
    error: null,
    disk_bytes: 0,
    remove_blocked: null,
    ...extra,
  };
};
const inUse = { ready: true, remove_blocked: "In use — select another model first" };

const docs = () => location.hash === "#docs";

export function models() {
  return {
    asr: {
      active: "parakeet-tdt-0.6b-v3",
      effective: "parakeet-tdt-0.6b-v3",
      options: [
        option("parakeet-tdt-0.6b-v3", { ...inUse, disk_bytes: 6.7e8 }),
        option("whisper-small", docs() ? {} : { downloading: true, downloaded: 3e8, total: 6e8 }),
        option("whisper-base", { ready: true, disk_bytes: 1.6e8 }),
        option("qwen3-asr-0.6b", docs() ? {} : { error: "connection reset" }),
        option("cohere-transcribe", { ready: true, disk_bytes: 2.9e9 }),
      ],
    },
    diar: {
      active: "nemotron-3-diarization",
      effective: "nemotron-3-diarization",
      options: [
        option("nemotron-3-diarization", { ...inUse, disk_bytes: 1.2e8 }),
        option("pyannote-3.0", { ready: true, disk_bytes: 6e6 }),
      ],
    },
    summary: {
      active: "qwen3.5-4b",
      effective: "qwen3.5-4b",
      options: [
        option("qwen3.5-0.8b"),
        option("qwen3.5-2b"),
        option("qwen3.5-4b", { ...inUse, disk_bytes: 2.7e9 }),
        option("qwen3.5-9b", { ready: true, disk_bytes: 5.7e9 }),
      ],
    },
  };
}

// Saved voices: none by default (the boot check expects the empty state);
// `#speakers` fills the Known speakers page for screenshots.
export function speakers(): string[] {
  return location.hash === "#speakers"
    ? ["Anna Berger", "Jonas Keller", "Mara Lindqvist", "Tomás Ortega", "Yuki Sato"]
    : [];
}
