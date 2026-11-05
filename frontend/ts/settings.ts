// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------
import { $, call } from "./core";
import { openBench } from "./bench";
import { confirmThen, disarm, el, fmtBytes, iconBtn, showToast } from "./helpers";

interface ConfigMeta {
  section: string;
  label: string;
  help: string;
  model?: string;
  /** A language list: its "auto" label. */
  languages?: string;
  /** Restrict the list to the languages of this step's selected model. */
  languagesOf?: string;
  providers?: boolean;
  textarea?: boolean;
  /**
   * A thread count: a select with "Auto" first (the backend resolves it to
   * 75 % of the cores) and every explicit count after it.
   */
  threads?: boolean;
  toggle?: boolean;
  number?: boolean;
  step?: number;
  /** Shown only while this holds for the form's current values. */
  when?: (values: Record<string, unknown>) => boolean;
}

const PYANNOTE_ONLY = (v) => v.DIAR_MODEL === "pyannote-3.0";

const CONFIG_META: Record<string, ConfigMeta> = {
  ASR_MODEL: {
    section: "config-transcription",
    label: "Transcription model",
    help: "The speech-to-text model that listens. A newly selected model downloads once, then runs offline.",
    model: "asr",
  },
  ASR_LANGUAGE: {
    section: "config-transcription",
    label: "Spoken language",
    help: "The default for new recordings — the prep screen can change it per recording. Parakeet and Whisper detect the language themselves, Qwen3-ASR takes it as a hint, Cohere Transcribe needs it.",
    languages: "Auto-detect",
    languagesOf: "asr",
  },
  DIAR_MODEL: {
    section: "config-speakers",
    label: "Speaker separation model",
    help: "The model that works out who spoke when. A newly selected model downloads once, then runs offline.",
    model: "diar",
  },
  SPEAKER_THRESHOLD: {
    section: "config-speakers",
    label: "Voice match strictness",
    help: "How closely a voice must match a saved speaker to reuse their name. Higher is stricter.",
  },
  SPEAKER_MERGE_THRESHOLD: {
    section: "config-speakers",
    label: "Merge similar speakers",
    help: "After separation, voices more similar than this are folded back into one person.",
  },
  DIAR_THRESHOLD: {
    section: "config-speakers",
    label: "Speaker separation",
    help: "How different voices must sound to count as separate people. Higher means fewer speakers.",
    when: PYANNOTE_ONLY,
  },
  MIN_SPEAKER_SECONDS: {
    section: "config-speakers",
    label: "Minimum speaking time",
    help: "Speakers with fewer seconds of speech than this are merged into the closest main voice.",
    step: 5,
    when: PYANNOTE_ONLY,
  },
  SUMMARY_MODEL: {
    section: "config-summary",
    label: "Summary model",
    help: "The on-device LLM that writes the summary, takeaways, and action items. A newly selected model downloads once, then runs offline.",
    model: "summary",
  },
  SUMMARY_LANGUAGE: {
    section: "config-summary",
    label: "Notes language",
    help: "The language the notes are written in. “Match the meeting” detects the language from the transcript; pick one explicitly if a meeting mixes languages.",
    languages: "Match the meeting",
  },
  SUMMARY_PROVIDER: {
    section: "config-summary",
    label: "Summary compute",
    help: "Where the summary model runs. The GPU is several times faster on most Macs — the benchmark measures both on this one.",
    providers: true,
  },
  KEEP_AUDIO: {
    section: "config-data",
    label: "Keep recording audio",
    help: "Keep each recording's audio so transcript sections can be played back and re-transcribed. Video files are replaced by their audio track to save space. Turn off to delete it once the notes are ready.",
    toggle: true,
  },
  NUM_THREADS: {
    section: "config-advanced",
    label: "CPU threads",
    help: "Threads for transcription and speaker separation (and the summary when it runs on the CPU). Auto takes three quarters of the cores and leaves the machine responsive; pick a number to override it.",
    threads: true,
  },
  SUMMARY_PROMPT: {
    section: "config-advanced",
    label: "System prompt",
    help: "The instruction the summary model follows when it reads the transcript — edit it to change the tone or focus of the notes. The response format and the notes-language instruction are always appended, so they can't be broken here.",
    textarea: true,
  },
};

const SECTIONS = [
  "config-transcription",
  "config-speakers",
  "config-summary",
  "config-data",
  "config-advanced",
];

export const PROVIDER_LABELS = { cpu: "CPU", gpu: "GPU" };

const STEP_LABELS = { asr: "Transcription", diar: "Speaker separation", summary: "Summary" };
const STEPS = Object.keys(STEP_LABELS);

let lastModels = null;
let lastConfig = null;
/** The values the form was rendered from: the reference for unsaved changes. */
let loadedValues: Record<string, unknown> = {};

/** Open Settings; `field` opens the Advanced card and focuses that control. */
export async function openSettings(field?: string) {
  $("app").classList.add("hidden");
  $("settings").classList.remove("hidden");
  $("settings").querySelector("h1").focus();
  try {
    const [cfg, data, models] = await Promise.all([
      call("get_config"),
      call("get_data_info"),
      call("get_models"),
    ]);
    renderConfigForm(cfg, models);
    renderDataSection(data);
    if (field) {
      const input = document.getElementById(`cfg-${field}`);
      const details = input?.closest("details");
      if (details) details.open = true;
      input?.focus();
    }
  } catch (e) {
    showToast(`Couldn't load settings: ${e.message}`);
  }
}

/** Back to the meetings; with unsaved changes the first click only asks. */
export function closeSettings() {
  const back = $("settings-back");
  if (isDirty()) confirmThen(back, "Discard unsaved changes? Click again", leaveSettings);
  else leaveSettings();
}

function leaveSettings() {
  disarm($("settings-back"));
  $("settings").classList.add("hidden");
  $("app").classList.remove("hidden");
}

export function openSpeakers() {
  $("app").classList.add("hidden");
  $("speakers").classList.remove("hidden");
  $("speakers").querySelector("h1").focus();
  refreshSpeakers();
}

export function closeSpeakers() {
  $("speakers").classList.add("hidden");
  $("app").classList.remove("hidden");
}

export async function refreshSpeakers() {
  let names;
  try {
    names = await call("get_speakers");
  } catch (e) {
    showToast(`Couldn't load speakers: ${e.message}`);
    return;
  }
  // Known names double as rename suggestions in the transcript legend, so an
  // existing person can be picked instead of mistyped into a duplicate.
  const datalist = $("speaker-names");
  datalist.innerHTML = "";
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    datalist.appendChild(opt);
  }
  const list = $("speaker-list");
  list.innerHTML = "";
  if (!names.length) {
    list.innerHTML = `<li class="list-empty muted">No saved voices yet. Rename a speaker in a transcript to save their voice.</li>`;
    return;
  }
  for (const name of names) {
    const li = document.createElement("li");
    const label = el("span", "speaker-name");
    label.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#i-user" /></svg>`;
    const nameSpan = el("span", "name-text");
    nameSpan.textContent = name;
    label.appendChild(nameSpan);
    const actions = el("span", "speaker-actions");
    actions.appendChild(
      iconBtn("i-pencil", `Rename ${name}`, () => startSpeakerRename(name, nameSpan))
    );
    const del = document.createElement("button");
    del.type = "button";
    del.className = "btn danger";
    del.innerHTML = `<span>Forget</span>`;
    del.setAttribute("aria-label", `Forget voice ${name}`);
    del.addEventListener("click", () => {
      confirmThen(del, "Sure?", async () => {
        try {
          await call("delete_speaker", { name });
          await refreshSpeakers();
        } catch (e) {
          showToast(`Couldn't forget voice: ${e.message}`);
        }
      });
    });
    actions.appendChild(del);
    li.append(label, actions);
    list.appendChild(li);
  }
}

// Rename a library entry in place — the voiceprint is kept, only the label
// changes, and it applies to future recordings (past transcripts keep the
// names they were saved with).
function startSpeakerRename(name, nameSpan) {
  if (nameSpan.querySelector("input")) return;
  const input = document.createElement("input");
  input.type = "text";
  input.value = name;
  input.maxLength = 100;
  input.setAttribute("aria-label", `Rename ${name}`);
  nameSpan.textContent = "";
  nameSpan.appendChild(input);
  input.focus();
  input.select();

  let settled = false;
  const settle = (fn) => {
    if (settled) return;
    settled = true;
    fn();
  };
  const cancel = () => settle(() => (nameSpan.textContent = name));
  const commit = () =>
    settle(async () => {
      const value = input.value.trim();
      if (!value || value === name) {
        nameSpan.textContent = name;
        return;
      }
      try {
        await call("rename_known_speaker", { name, new_name: value });
        showToast(`Renamed to “${value}” — applies to future recordings.`, 3000, true);
      } catch (e) {
        showToast(
          e.status === 409 ? "That name already exists." : `Rename failed: ${e.message}`
        );
      }
      refreshSpeakers();
    });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    if (e.key === "Escape") cancel();
  });
  input.addEventListener("blur", commit);
}

function renderConfigForm(cfg, models) {
  lastModels = models;
  lastConfig = cfg;
  for (const id of SECTIONS) $(id).innerHTML = "";
  for (const [name, meta] of Object.entries(CONFIG_META)) {
    if (!(name in cfg.values)) continue;
    // A compute choice with a single option is no choice (Windows/Linux).
    if (meta.providers && (cfg.compute_options?.[name] || []).length < 2) continue;
    $(meta.section).appendChild(configRow(cfg, models, name, meta));
  }
  loadedValues = formValues();
  syncDependentRows();
  updateSaveBar();
}

function configRow(cfg, models, name: string, meta: ConfigMeta) {
  const locked = cfg.env_locked.includes(name);
  const limits = cfg.limits[name] || {};
  const id = `cfg-${name}`;
  const row = el("div", "config-row" + (locked ? " locked" : ""));
  row.dataset.row = name;
  const label = el("label", "config-label");
  label.htmlFor = id;
  label.textContent = meta.label;
  const help = el("p", "config-help");
  help.id = `${id}-help`;
  help.textContent = meta.help;
  const control = el("div", "config-control");
  const describe = (input: HTMLElement) => {
    input.id = id;
    input.dataset.config = name;
    input.setAttribute("aria-describedby", help.id);
    (input as HTMLInputElement).disabled = locked;
    input.addEventListener("input", onFormInput);
    input.addEventListener("change", onFormInput);
  };

  // Inline rows: label, control, help; stacked rows put the help first.
  let parts: HTMLElement[] = [label, control, help];

  if (meta.model) {
    row.classList.add("stacked", "model-row");
    // The cards are the control; the hidden input carries the value.
    const value = document.createElement("input");
    value.type = "hidden";
    value.value = cfg.values[name];
    describe(value);
    const status = el("p", "model-status");
    status.dataset.modelStep = meta.model;
    status.setAttribute("aria-live", "polite");
    parts = [label, help, value, modelCards(meta.model, value, label, locked), status];
    updateModelStatus(meta.model, value.value, status);
  } else if (meta.providers) {
    const select = document.createElement("select");
    describe(select);
    for (const v of cfg.compute_options[name]) {
      select.appendChild(new Option(PROVIDER_LABELS[v] || v, v, false, v === cfg.values[name]));
    }
    control.appendChild(select);
    const bench = el("button", "link-btn");
    bench.type = "button";
    bench.textContent = "Open the benchmark";
    bench.addEventListener("click", () => openBench("settings"));
    help.append(" ", bench);
  } else if (meta.languages) {
    const select = document.createElement("select");
    describe(select);
    select.dataset.autoLabel = meta.languages;
    fillLanguages(select, cfg.values[name], null);
    control.appendChild(select);
    if (meta.languagesOf) {
      const note = el("p", "config-note");
      note.id = `${id}-note`;
      note.setAttribute("aria-live", "polite");
      parts.push(note);
    }
  } else if (meta.textarea) {
    row.classList.add("stacked");
    const ta = document.createElement("textarea");
    describe(ta);
    ta.rows = 5;
    ta.value = cfg.values[name];
    if (limits.max) ta.maxLength = limits.max;
    const restore = el("button", "link-btn");
    restore.type = "button";
    restore.textContent = "Restore the default prompt";
    restore.disabled = locked;
    restore.addEventListener("click", () => {
      ta.value = cfg.defaults[name] || "";
      onFormInput();
    });
    control.append(ta, restore);
    parts = [label, help, control];
  } else if (meta.toggle) {
    const sw = el("span", "switch");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!cfg.values[name];
    describe(input);
    sw.append(input, el("span", "track"));
    control.appendChild(sw);
  } else if (meta.threads) {
    // Auto is the default and the only sane choice for most people; the
    // explicit counts stay available for the "leave two cores for Zoom" case.
    const select = document.createElement("select");
    select.dataset.numeric = "1";
    describe(select);
    const current = Number(cfg.values[name] ?? 0);
    const auto = Number(cfg.auto_threads || 0);
    const cores = Number(cfg.cores || 0);
    const autoLabel = cores
      ? `Auto — ${auto} of ${cores} threads (75 % of the cores)`
      : `Auto — ${auto} threads (75 % of the cores)`;
    select.appendChild(new Option(autoLabel, "0", false, current === 0));
    const maxThreads = Number(limits.max) || auto || 1;
    for (let n = 1; n <= maxThreads; n += 1) {
      select.appendChild(new Option(`${n} thread${n === 1 ? "" : "s"}`, String(n), false, current === n));
    }
    control.appendChild(select);
  } else if (meta.number) {
    const input = document.createElement("input");
    input.type = "number";
    input.min = limits.min;
    input.max = limits.max;
    input.step = "1";
    input.value = cfg.values[name];
    describe(input);
    control.appendChild(input);
  } else {
    const input = document.createElement("input");
    input.type = "range";
    input.min = limits.min;
    input.max = limits.max;
    input.step = String(meta.step || 0.05);
    input.value = cfg.values[name];
    describe(input);
    const value = el("span", "range-value");
    value.setAttribute("aria-hidden", "true");
    const fmt = (v) => (meta.step ? `${Math.round(v)} s` : Number(v).toFixed(2));
    const show = () => {
      value.textContent = fmt(input.value);
      input.setAttribute("aria-valuetext", fmt(input.value));
    };
    show();
    input.addEventListener("input", show);
    control.append(input, value);
  }

  row.append(...parts);
  if (locked) row.appendChild(lockedHint());
  return row;
}

function lockedHint() {
  const hint = el("p", "locked-hint");
  hint.textContent = "Set by an environment variable — can't be changed here.";
  return hint;
}

/** The catalog option selected for `step` in the form (not the saved value).
 *  Read from the control itself: formValues() skips env-locked (disabled)
 *  fields, whose model still decides the language list. */
function selectedModel(step: string) {
  const field = Object.keys(CONFIG_META).find((n) => CONFIG_META[n].model === step);
  const id = (document.getElementById(`cfg-${field}`) as HTMLInputElement | null)?.value;
  return lastModels?.[step]?.options.find((o) => o.id === id) || null;
}

/** Language options for `select`: every app language, restricted (and
 *  without "auto") for a model that needs one it decodes. Shared with the
 *  prep screen. */
export function fillLanguageSelect(
  select: HTMLSelectElement,
  languages: [string, string][],
  value: string,
  model,
  autoLabel = "Auto-detect",
) {
  const needs = !!model?.needs_language;
  const allowed = needs ? new Set(model.languages) : null;
  select.innerHTML = "";
  if (!needs) select.appendChild(new Option(autoLabel, "auto"));
  for (const [code, name] of languages || []) {
    if (!allowed || allowed.has(code)) select.appendChild(new Option(name, code));
  }
  select.value = value;
  if (select.value !== value) select.value = pickLanguage(select);
}

function fillLanguages(select: HTMLSelectElement, value: string, model) {
  fillLanguageSelect(select, lastConfig?.languages, value, model, select.dataset.autoLabel);
}

/** A sensible language when the current one is no longer offered: the
 *  system's, else English, else the first. */
function pickLanguage(select: HTMLSelectElement) {
  const codes = [...select.options].map((o) => o.value);
  const system = (navigator.language || "").slice(0, 2).toLowerCase();
  return codes.find((c) => c === system) || codes.find((c) => c === "en") || codes[0];
}

/** Rows that depend on other values: the classic-model sliders and the
 *  language list of the selected transcription model (rebuilt only when
 *  that model changes). */
function syncDependentRows(values = formValues()) {
  for (const [name, meta] of Object.entries(CONFIG_META)) {
    const row = document.querySelector<HTMLElement>(`[data-row="${name}"]`);
    if (meta.when && row) row.hidden = !meta.when(values);
    const select = document.getElementById(`cfg-${name}`) as HTMLSelectElement;
    if (!meta.languagesOf || !select) continue;
    const model = selectedModel(meta.languagesOf);
    if (select.dataset.model === (model?.id ?? "")) continue;
    select.dataset.model = model?.id ?? "";
    fillLanguages(select, select.value, model);
    const note = document.getElementById(`cfg-${name}-note`);
    if (note) {
      note.textContent = model?.needs_language
        ? `${model.name} can't detect the language — set the one spoken in your recordings.`
        : "";
    }
  }
}

function onFormInput() {
  const values = formValues();
  syncDependentRows(values);
  updateSaveBar(values);
}

const downloadPct = (opt) => (opt.total ? Math.round((opt.downloaded / opt.total) * 100) : 0);
const bareSize = (opt) => (opt.size || "").replace(/ download$/, "");

// Model picker: one card per catalog option with its pros and cons, as a
// radio group (arrow keys move the choice, one tab stop).
function modelCards(step: string, value: HTMLInputElement, label: HTMLElement, locked: boolean) {
  const group = el("div", "model-overview");
  group.setAttribute("role", "radiogroup");
  label.id = `${value.id}-label`;
  group.setAttribute("aria-labelledby", label.id);
  label.addEventListener("click", () => group.querySelector<HTMLElement>("[aria-checked=true]")?.focus());
  const options = lastModels?.[step]?.options || [];
  const choose = (id: string, focus = false) => {
    value.value = id;
    syncModelCards(group, id);
    if (focus) group.querySelector<HTMLElement>(`[data-model-id="${id}"]`)?.focus();
    updateModelStatus(step, id, group.parentElement?.querySelector(".model-status"));
    onFormInput();
  };
  for (const opt of options) {
    const card = el("button", "model-card");
    card.type = "button";
    card.setAttribute("role", "radio");
    card.dataset.modelId = opt.id;
    card.dataset.step = step;
    card.disabled = locked;
    const head = el("div", "model-card-head");
    const name = el("span", "model-card-name");
    name.textContent = opt.name;
    const badge = el("span", "model-card-badge");
    head.append(name, badge);
    card.appendChild(head);
    if (opt.tagline) {
      const tagline = el("p", "model-card-tagline");
      tagline.textContent = opt.tagline;
      card.appendChild(tagline);
    }
    card.appendChild(prosConsList(opt.pros, "pros"));
    card.appendChild(prosConsList(opt.cons, "cons"));
    card.addEventListener("click", () => choose(opt.id));
    card.addEventListener("keydown", (e) => {
      const ids = options.map((o) => o.id);
      const i = ids.indexOf(opt.id);
      const next = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (next === undefined) return;
      e.preventDefault();
      choose(ids[(i + next + ids.length) % ids.length], true);
    });
    group.appendChild(card);
  }
  syncModelCards(group, value.value);
  return group;
}

function prosConsList(items, kind) {
  const ul = el("ul", `model-${kind}`);
  for (const text of items || []) {
    const li = document.createElement("li");
    li.textContent = text;
    ul.appendChild(li);
  }
  return ul;
}

function syncModelCards(group: Element, activeId: string) {
  for (const card of group.querySelectorAll<HTMLElement>(".model-card")) {
    const on = card.dataset.modelId === activeId;
    card.classList.toggle("selected", on);
    card.setAttribute("aria-checked", String(on));
    card.tabIndex = on ? 0 : -1;
    const opt = lastModels?.[card.dataset.step]?.options.find((o) => o.id === card.dataset.modelId);
    const badge = card.querySelector(".model-card-badge");
    if (!opt || !badge) continue;
    badge.className = "model-card-badge";
    if (opt.ready) {
      badge.textContent = "Installed";
      badge.classList.add("ready");
    } else if (opt.downloading) {
      badge.textContent = `Downloading ${downloadPct(opt)} %`;
    } else if (opt.error) {
      badge.textContent = "Download failed";
      badge.classList.add("failed");
    } else {
      badge.textContent = `${bareSize(opt)} download`.trim();
    }
  }
}

// Status line under a model picker: download progress, error + retry, or a
// "downloads when you save" note for the chosen model.
function updateModelStatus(step: string, id: string, status: Element | null) {
  if (!status) return;
  const opt = lastModels?.[step]?.options.find((o) => o.id === id);
  status.innerHTML = "";
  if (!opt || opt.ready) return;
  if (opt.downloading) {
    const pct = downloadPct(opt);
    status.textContent = `Downloading ${opt.name}… ${pct} %`;
    status.appendChild(progressBar(pct));
  } else if (opt.error) {
    status.textContent = `Download failed: ${opt.error} `;
    const retry = el("button", "btn ghost");
    retry.type = "button";
    retry.textContent = "Retry";
    retry.addEventListener("click", async () => {
      try {
        await call("download_model", { step, id: opt.id });
      } catch (e) {
        showToast(`Couldn't start download: ${e.message}`);
      }
    });
    status.appendChild(retry);
  } else {
    status.textContent = `Not downloaded yet (${opt.size || "size unknown"}) — it downloads when you save; the default model is used until it finishes.`;
  }
}

function progressBar(pct: number) {
  const bar = el("span", "bar");
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-label", "Download progress");
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  bar.setAttribute("aria-valuenow", String(pct));
  const fill = el("span");
  fill.style.width = `${pct}%`;
  bar.appendChild(fill);
  return bar;
}

function refreshModelViews() {
  for (const row of document.querySelectorAll<HTMLElement>("#settings .model-row")) {
    const value = row.querySelector<HTMLInputElement>("input[data-config]");
    const group = row.querySelector(".model-overview");
    const status = row.querySelector(".model-status");
    const step = (status as HTMLElement)?.dataset.modelStep;
    if (!value || !group || !step) continue;
    syncModelCards(group, value.value);
    updateModelStatus(step, value.value, status);
  }
  renderModelDiskList();
}

// `models-progress` events keep the cards and bars moving while any
// on-demand download runs.
export function onModelsProgress(models) {
  const before = lastModels
    ? new Set(
        STEPS.flatMap((s) =>
          lastModels[s].options.filter((o) => o.downloading).map((o) => `${s}:${o.id}`)
        )
      )
    : new Set();
  lastModels = models;
  refreshModelViews();
  for (const s of STEPS) {
    for (const o of models[s].options) {
      if (before.has(`${s}:${o.id}`) && o.ready) {
        showToast(`${o.name} downloaded — it will be used for the next recording.`, 4000, true);
      }
    }
  }
}

let diskListKey = "";

// Every model on disk with its real size; selectable ones that aren't in use
// can be removed to free space. Rebuilt only when a row changes, so download
// progress events don't reset a half-confirmed Remove button.
function renderModelDiskList() {
  const list = $("model-disk-list");
  if (!list || !lastModels) return;
  const rows = STEPS.flatMap((step) =>
    lastModels[step].options.filter((o) => o.ready).map((opt) => ({ step, opt }))
  );
  const key = JSON.stringify(rows.map(({ step, opt }) => [step, opt.id, opt.disk_bytes, opt.remove_blocked]));
  if (key === diskListKey && list.childElementCount) return;
  diskListKey = key;
  list.innerHTML = "";
  for (const { step, opt } of rows) {
    const li = el("li");
    const name = el("span", "disk-name");
    name.textContent = opt.name;
    const meta = el("span", "disk-meta muted");
    const size = opt.disk_bytes ? fmtBytes(opt.disk_bytes) : bareSize(opt);
    meta.textContent = `${STEP_LABELS[step]} · ${size}`;
    const info = el("span", "disk-info");
    info.append(name, meta);
    li.appendChild(info);
    if (!opt.remove_blocked) {
      const remove = el("button", "btn danger");
      remove.type = "button";
      remove.innerHTML = "<span>Remove</span>";
      remove.setAttribute("aria-label", `Remove ${opt.name}`);
      remove.addEventListener("click", () =>
        confirmThen(remove, "Remove?", async () => {
          try {
            // The models-progress event refreshes the cards and this list.
            await call("remove_model", { step, id: opt.id });
            renderDataSection(await call("get_data_info"));
            showToast(`${opt.name} removed.`, 3000, true);
          } catch (e) {
            showToast(`Couldn't remove: ${e.message}`);
          }
        })
      );
      li.appendChild(remove);
    } else {
      const why = el("span", "disk-why muted small");
      why.textContent = opt.remove_blocked;
      li.appendChild(why);
    }
    list.appendChild(li);
  }
}

/** The form's current values, typed (hidden rows included: they keep theirs). */
function formValues(): Record<string, unknown> {
  const out = {};
  const controls = document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(
    "#settings [data-config]",
  );
  for (const input of controls) {
    if (input.disabled) continue;
    const name = input.dataset.config;
    if (input.tagName === "SELECT" || input.type === "hidden") {
      out[name] = (input as HTMLSelectElement).dataset?.numeric ? parseInt(input.value, 10) : input.value;
    }
    else if (input.tagName === "TEXTAREA") out[name] = input.value.trim();
    else if (input.type === "checkbox") out[name] = (input as HTMLInputElement).checked;
    else if (input.type === "number") out[name] = parseInt(input.value, 10);
    else out[name] = parseFloat(input.value);
  }
  return out;
}

function isDirty(now = formValues()) {
  return Object.keys(now).some((k) => now[k] !== loadedValues[k]);
}

function updateSaveBar(values = formValues()) {
  $("settings-savebar").hidden = !isDirty(values);
}

/** The request body: a value equal to its default clears the override, so
 *  a later release's better default still reaches this machine. */
function collectConfigValues() {
  const body = {};
  const defaults = lastConfig?.defaults || {};
  for (const [name, value] of Object.entries(formValues())) {
    const fallback = defaults[name];
    const same =
      value === fallback ||
      value === "" ||
      (typeof value === "number" && typeof fallback === "number" && Math.abs(value - fallback) < 1e-9);
    body[name] = same ? null : value;
  }
  return body;
}

export async function saveConfig() {
  try {
    const cfg = await call("put_config", { fields: collectConfigValues() });
    const models = await call("get_models");
    renderConfigForm(cfg, models);
    showToast("Settings saved — they apply to the next recording.", 3500, true);
  } catch (err) {
    showToast(`Couldn't save settings: ${err.message}`);
  }
}

export function discardConfig() {
  if (lastConfig) renderConfigForm(lastConfig, lastModels);
}

export function resetConfig() {
  const btn = $("config-reset");
  confirmThen(btn, "Reset everything? Click again", async () => {
    const body = {};
    for (const name of Object.keys(lastConfig?.values || CONFIG_META)) body[name] = null;
    try {
      const cfg = await call("put_config", { fields: body });
      const models = await call("get_models");
      renderConfigForm(cfg, models);
      showToast("Settings reset to defaults.", 3000, true);
    } catch (err) {
      showToast(`Couldn't reset settings: ${err.message}`);
    }
  });
}

function renderDataSection(data) {
  renderModelDiskList();
  $("data-dir").textContent = data.data_dir;
  const parts = [
    `${fmtBytes(data.total_bytes)} on disk`,
    `models ${fmtBytes(data.models_bytes)}`,
    `transcripts ${fmtBytes(data.meetings_bytes)}`,
    `recordings ${fmtBytes(data.uploads_bytes)}`,
    `${data.meeting_count} meeting${data.meeting_count === 1 ? "" : "s"}`,
  ];
  $("data-usage").textContent = parts.join(" · ");
}

export async function revealDataDir() {
  try {
    await call("reveal_data_dir");
  } catch (e) {
    showToast(`Couldn't open folder: ${e.message}`);
  }
}

