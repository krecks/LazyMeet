// Backend payloads for the smoke stub, shaped like api::config_payload and
// models::models_status (abridged catalog text).
const LANGUAGES = [
  ["en", "English"], ["de", "German"], ["fr", "French"], ["es", "Spanish"],
  ["it", "Italian"], ["pt", "Portuguese"], ["nl", "Dutch"], ["pl", "Polish"],
  ["ru", "Russian"], ["uk", "Ukrainian"], ["tr", "Turkish"], ["zh", "Chinese"],
  ["ja", "Japanese"], ["ko", "Korean"], ["ar", "Arabic"], ["el", "Greek"], ["vi", "Vietnamese"],
];
const COHERE = ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"];

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
  NUM_THREADS: 7,
  KEEP_AUDIO: true,
};
const LIMITS = {
  SPEAKER_THRESHOLD: { min: 0, max: 1 },
  DIAR_THRESHOLD: { min: 0, max: 1 },
  SPEAKER_MERGE_THRESHOLD: { min: 0, max: 1 },
  MIN_SPEAKER_SECONDS: { min: 0, max: 600 },
  NUM_THREADS: { min: 1, max: 12 },
  SUMMARY_PROMPT: { min: null, max: 2000 },
};

export function config() {
  return {
    values: { ...DEFAULTS },
    overrides: {},
    defaults: { ...DEFAULTS },
    env_locked: [],
    limits: LIMITS,
    languages: LANGUAGES,
    compute_options: { SUMMARY_PROVIDER: ["cpu", "gpu"] },
  };
}

const option = (id, label, size, extra = {}) => ({
  id,
  label,
  size,
  languages: [],
  needs_language: false,
  pros: ["Fast", "Accurate"],
  cons: ["Large"],
  ready: false,
  downloading: false,
  downloaded: 0,
  total: 0,
  error: null,
  disk_bytes: 0,
  removable: false,
  ...extra,
});

export function models() {
  return {
    asr: {
      active: "parakeet-tdt-0.6b-v3",
      options: [
        option("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3 — 25 European languages (default)", "≈700 MB", { ready: true, disk_bytes: 6.7e8 }),
        option("whisper-small", "Whisper Small — 90+ languages", "≈600 MB download", { downloading: true, downloaded: 3e8, total: 6e8 }),
        option("whisper-base", "Whisper Base — smallest, less accurate", "≈150 MB download", { ready: true, disk_bytes: 1.6e8, removable: true }),
        option("qwen3-asr-0.6b", "Qwen3-ASR 0.6B — 30 languages incl. Chinese, Japanese, Korean", "≈880 MB download", { error: "connection reset" }),
        option("cohere-transcribe", "Cohere Transcribe 2B — 14 languages, most accurate", "≈1.7 GB download", { ready: true, disk_bytes: 2.9e9, removable: true, languages: COHERE, needs_language: true }),
      ],
    },
    diar: {
      active: "nemotron-3-diarization",
      options: [
        option("nemotron-3-diarization", "Nemotron 3 Diarization — up to 8 speakers (default)", "≈120 MB", { ready: true, disk_bytes: 1.2e8 }),
        option("pyannote-3.0", "Classic (pyannote 3.0) — any number of speakers", "≈6 MB download", { ready: true, disk_bytes: 6e6, removable: true }),
      ],
    },
    summary: {
      active: "qwen3.5-4b",
      options: [
        option("qwen3.5-0.8b", "Qwen 3.5 0.8B — tiny, fastest", "≈0.5 GB download"),
        option("qwen3.5-2b", "Qwen 3.5 2B — fast, light", "≈1.3 GB download"),
        option("qwen3.5-4b", "Qwen 3.5 4B — balanced (default)", "≈2.7 GB", { ready: true, disk_bytes: 2.7e9 }),
        option("qwen3.5-9b", "Qwen 3.5 9B — strongest, slowest", "≈5.7 GB download", { ready: true, disk_bytes: 5.7e9, removable: true }),
      ],
    },
  };
}
