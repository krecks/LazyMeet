// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------
const LANGUAGES = [
    ["en", "English"], ["de", "German"], ["fr", "French"], ["es", "Spanish"],
    ["it", "Italian"], ["pt", "Portuguese"], ["nl", "Dutch"], ["pl", "Polish"],
    ["ru", "Russian"], ["uk", "Ukrainian"], ["tr", "Turkish"],
    ["zh", "Chinese"], ["ja", "Japanese"], ["ko", "Korean"],
];
const CONFIG_META = {
    ASR_MODEL: {
        section: "config-transcription",
        label: "Transcription model",
        help: "The speech-to-text model that listens. A newly selected model downloads once, then runs offline.",
        model: "asr",
    },
    ASR_LANGUAGE: {
        section: "config-transcription",
        label: "Spoken language",
        help: "Hint for Whisper and Qwen3-ASR — Parakeet always detects the language itself.",
        languages: "Auto-detect",
    },
    DIAR_MODEL: {
        section: "config-transcription",
        label: "Speaker separation model",
        help: "The model that works out who spoke when. A newly selected model downloads once, then runs offline.",
        model: "diar",
    },
    SPEAKER_THRESHOLD: {
        section: "config-transcription",
        label: "Voice match strictness",
        help: "How closely a voice must match a saved speaker to reuse their name. Higher is stricter.",
    },
    DIAR_THRESHOLD: {
        section: "config-transcription",
        label: "Speaker separation",
        help: "How different voices must sound to count as separate people. Higher means fewer speakers. Classic model only.",
    },
    SPEAKER_MERGE_THRESHOLD: {
        section: "config-transcription",
        label: "Merge similar speakers",
        help: "After separation, voices more similar than this are folded back into one person.",
    },
    MIN_SPEAKER_SECONDS: {
        section: "config-transcription",
        label: "Minimum speaking time",
        help: "Speakers with fewer seconds of speech than this are merged into the closest main voice. Classic model only — Nemotron keeps short speakers as their own.",
        step: 5,
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
    SUMMARY_PROMPT: {
        section: "config-summary",
        label: "System prompt",
        help: "The instruction the summary model follows when it reads the transcript — edit it to change the tone or focus of the notes. The response format and the notes-language instruction are always appended automatically, so they can't be broken here. Clear the field and save to restore the default.",
        textarea: true,
    },
    SUMMARY_PROVIDER: {
        section: "config-processing",
        label: "Summary compute",
        help: "Where the summary model runs. Its engine (llama.cpp) accelerates on the GPU; the Neural Engine is not available for it. Options show what this build supports on this machine — the benchmark (gauge icon) measures them all and can apply the fastest.",
        providers: true,
    },
    ASR_PROVIDER: {
        section: "config-processing",
        label: "Transcription compute",
        help: "Where the speech-to-text model runs. Its engine (onnxruntime) runs on the CPU in this build — the Neural Engine measured slower for these models.",
        providers: true,
    },
    DIAR_PROVIDER: {
        section: "config-processing",
        label: "Speaker separation compute",
        help: "Where the who-spoke-when model runs (CPU in this build).",
        providers: true,
    },
    SPEAKER_PROVIDER: {
        section: "config-processing",
        label: "Voice matching compute",
        help: "Where the voiceprint model runs (CPU in this build).",
        providers: true,
    },
    NUM_THREADS: {
        section: "config-processing",
        label: "CPU threads",
        help: "Threads for the parts that run on the CPU. More finish faster but occupy more of the computer.",
        number: true,
    },
    KEEP_AUDIO: {
        section: "config-processing",
        label: "Keep recording audio",
        help: "Keep each recording's audio after processing so transcript sections can be played back. Video files are replaced by their audio track to save space. Turn off to delete everything once the notes are ready.",
        toggle: true,
    },
};
let lastModels = null;
async function openSettings() {
    $("app").classList.add("hidden");
    $("settings").classList.remove("hidden");
    try {
        const [cfg, data, models] = await Promise.all([
            call("get_config"),
            call("get_data_info"),
            call("get_models"),
        ]);
        renderConfigForm(cfg, models);
        renderDataSection(data);
    }
    catch (e) {
        showToast(`Couldn't load settings: ${e.message}`);
    }
}
function closeSettings() {
    $("settings").classList.add("hidden");
    $("app").classList.remove("hidden");
}
let benchRunning = false;
let benchRec = null; // recommendation from the last completed run
function openBench() {
    $("app").classList.add("hidden");
    $("bench").classList.remove("hidden");
}
function closeBench() {
    $("bench").classList.add("hidden");
    $("app").classList.remove("hidden");
}
async function startBenchmark(path) {
    try {
        $("bench-results").classList.add("hidden");
        $("bench-results").querySelector("tbody").innerHTML = "";
        $("bench-rec").classList.add("hidden");
        $("bench-apply").classList.add("hidden");
        await call("run_benchmark", { path: path || null });
        benchRunning = true;
        $("bench-run").disabled = true;
        $("bench-run-file").disabled = true;
        $("bench-status").textContent = "Starting…";
    }
    catch (e) {
        showToast(e.message);
    }
}
async function pickBenchmarkFile() {
    try {
        const path = await call("pick_recording");
        if (path)
            startBenchmark(path);
    }
    catch (e) {
        showToast(`Couldn't open the file picker: ${e.message}`);
    }
}
// `benchmark-progress` events: rows accumulate as each measurement lands.
function onBenchProgress(p) {
    if (p.threads) {
        const env = $("bench-env");
        const cores = p.cores
            ? `Using ${p.threads} of ${p.cores} CPU cores`
            : `Using ${p.threads} CPU threads`;
        env.textContent = `${cores} (change under Settings → CPU threads) · Source: ${p.source || "synthetic audio"}.`;
        env.classList.remove("hidden");
    }
    if (p.rows && p.rows.length) {
        const tbody = $("bench-results").querySelector("tbody");
        tbody.innerHTML = "";
        for (const row of p.rows) {
            const tr = document.createElement("tr");
            const cell = (text, cls) => {
                const td = document.createElement("td");
                td.textContent = text;
                if (cls)
                    td.className = cls;
                tr.appendChild(td);
            };
            cell(row.stage);
            cell(row.backend);
            if (row.error) {
                const td = document.createElement("td");
                td.colSpan = 2;
                td.className = "bench-error";
                td.textContent = row.error;
                tr.appendChild(td);
            }
            else {
                cell(`${row.seconds} s`);
                cell(row.speed);
            }
            tbody.appendChild(tr);
        }
        $("bench-results").classList.remove("hidden");
    }
    if (p.status === "running") {
        $("bench-status").textContent = p.step || "Measuring…";
    }
    else if (p.status === "done") {
        benchRunning = false;
        $("bench-run").disabled = false;
        $("bench-run-file").disabled = false;
        $("bench-status").textContent = "Done.";
        renderBenchRecommendation(p.recommendation, p.current);
    }
    else if (p.status === "error") {
        benchRunning = false;
        $("bench-run").disabled = false;
        $("bench-run-file").disabled = false;
        $("bench-status").textContent = `Benchmark failed: ${p.message}`;
    }
}
const PROVIDER_LABELS = {
    cpu: "CPU",
    coreml: "Neural Engine (NPU)",
    gpu: "GPU",
};
function benchSummaryText(rec) {
    return (`Fastest on this machine: transcription ${PROVIDER_LABELS[rec.ASR_PROVIDER]}` +
        ` · speaker separation ${PROVIDER_LABELS[rec.DIAR_PROVIDER]}` +
        ` · voice matching ${PROVIDER_LABELS[rec.SPEAKER_PROVIDER]}` +
        ` · summarizer ${PROVIDER_LABELS[rec.SUMMARY_PROVIDER]}.`);
}
function renderBenchRecommendation(rec, current) {
    if (!rec)
        return;
    benchRec = rec;
    const matches = current &&
        ["ASR_PROVIDER", "DIAR_PROVIDER", "SPEAKER_PROVIDER", "SUMMARY_PROVIDER"].every((k) => current[k] === rec[k]);
    const line = $("bench-rec");
    line.textContent =
        benchSummaryText(rec) +
            (matches ? " Your settings already match." : " Your settings differ.");
    line.classList.remove("hidden");
    $("bench-apply").classList.toggle("hidden", matches);
}
async function applyBenchRecommendation() {
    if (!benchRec)
        return;
    try {
        await call("put_config", { fields: benchRec });
        $("bench-apply").classList.add("hidden");
        $("bench-rec").textContent =
            benchSummaryText(benchRec) + " Applied — used from the next recording on.";
        showToast("Fastest configuration applied.", 3500, true);
    }
    catch (e) {
        showToast(`Couldn't apply: ${e.message}`);
    }
}
function openSpeakers() {
    $("app").classList.add("hidden");
    $("speakers").classList.remove("hidden");
    refreshSpeakers();
}
function closeSpeakers() {
    $("speakers").classList.add("hidden");
    $("app").classList.remove("hidden");
}
async function refreshSpeakers() {
    let names;
    try {
        names = await call("get_speakers");
    }
    catch (e) {
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
        actions.appendChild(iconBtn("i-pencil", `Rename ${name}`, () => startSpeakerRename(name, nameSpan)));
        const del = document.createElement("button");
        del.type = "button";
        del.className = "btn ghost";
        del.innerHTML = `<span>Forget</span>`;
        del.setAttribute("aria-label", `Forget voice ${name}`);
        del.addEventListener("click", () => {
            confirmThen(del, "Sure?", async () => {
                try {
                    await call("delete_speaker", { name });
                    await refreshSpeakers();
                }
                catch (e) {
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
    if (nameSpan.querySelector("input"))
        return;
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
        if (settled)
            return;
        settled = true;
        fn();
    };
    const cancel = () => settle(() => (nameSpan.textContent = name));
    const commit = () => settle(async () => {
        const value = input.value.trim();
        if (!value || value === name) {
            nameSpan.textContent = name;
            return;
        }
        try {
            await call("rename_known_speaker", { name, new_name: value });
            showToast(`Renamed to “${value}” — applies to future recordings.`, 3000, true);
        }
        catch (e) {
            showToast(e.status === 409 ? "That name already exists." : `Rename failed: ${e.message}`);
        }
        refreshSpeakers();
    });
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter")
            commit();
        if (e.key === "Escape")
            cancel();
    });
    input.addEventListener("blur", commit);
}
function renderConfigForm(cfg, models) {
    lastModels = models;
    for (const id of ["config-transcription", "config-summary", "config-processing"]) {
        $(id).innerHTML = "";
    }
    for (const [name, meta] of Object.entries(CONFIG_META)) {
        if (!(name in cfg.values))
            continue;
        const locked = cfg.env_locked.includes(name);
        const limits = cfg.limits[name] || {};
        const row = el("div", "config-row" + (locked ? " locked" : ""));
        const label = el("div", "config-label");
        label.textContent = meta.label;
        const help = el("p", "config-help");
        help.textContent = meta.help;
        const control = el("div", "config-control");
        if (meta.model) {
            row.classList.add("wide");
            const select = document.createElement("select");
            select.dataset.config = name;
            select.disabled = locked;
            const options = (models && models[meta.model] && models[meta.model].options) || [];
            for (const opt of options) {
                const o = document.createElement("option");
                o.value = opt.id;
                o.textContent = opt.label + (opt.ready ? "" : ` (${opt.size || "download"})`);
                if (opt.id === cfg.values[name])
                    o.selected = true;
                select.appendChild(o);
            }
            if (![...select.options].some((o) => o.value === cfg.values[name])) {
                const o = document.createElement("option");
                o.value = cfg.values[name];
                o.textContent = cfg.values[name];
                o.selected = true;
                select.appendChild(o);
            }
            const status = el("p", "model-status");
            status.dataset.modelStep = meta.model;
            const overview = buildModelOverview(select, options, locked);
            select.addEventListener("change", () => {
                updateModelStatus(select, status);
                syncModelOverview(overview, select.value);
            });
            control.appendChild(select);
            row.append(label, control, help, status, overview);
            updateModelStatus(select, status);
            if (locked)
                row.appendChild(lockedHint());
            $(meta.section).appendChild(row);
            continue;
        }
        if (meta.providers) {
            const select = document.createElement("select");
            select.dataset.config = name;
            const options = (cfg.compute_options && cfg.compute_options[name]) || ["cpu"];
            select.disabled = locked || options.length < 2;
            for (const value of options) {
                const o = document.createElement("option");
                o.value = value;
                o.textContent = PROVIDER_LABELS[value] || value;
                if (value === cfg.values[name])
                    o.selected = true;
                select.appendChild(o);
            }
            control.appendChild(select);
        }
        else if (meta.languages) {
            const select = document.createElement("select");
            select.dataset.config = name;
            select.disabled = locked;
            for (const [code, lang] of [["auto", meta.languages], ...LANGUAGES]) {
                const o = document.createElement("option");
                o.value = code;
                o.textContent = lang;
                if (code === cfg.values[name])
                    o.selected = true;
                select.appendChild(o);
            }
            control.appendChild(select);
        }
        else if (meta.textarea) {
            row.classList.add("stacked");
            const ta = document.createElement("textarea");
            ta.dataset.config = name;
            ta.rows = 4;
            ta.value = cfg.values[name];
            ta.placeholder = cfg.defaults[name] || "";
            if (limits.max)
                ta.maxLength = limits.max;
            ta.disabled = locked;
            ta.setAttribute("aria-label", meta.label);
            control.appendChild(ta);
            // Stacked layout reads label → help → control.
            row.append(label, help, control);
            if (locked)
                row.appendChild(lockedHint());
            $(meta.section).appendChild(row);
            continue;
        }
        else if (meta.toggle) {
            const sw = el("label", "switch");
            const input = document.createElement("input");
            input.type = "checkbox";
            input.checked = !!cfg.values[name];
            input.disabled = locked;
            input.dataset.config = name;
            sw.appendChild(input);
            sw.appendChild(el("span", "track"));
            control.appendChild(sw);
        }
        else if (meta.number) {
            const input = document.createElement("input");
            input.type = "number";
            input.min = limits.min;
            input.max = limits.max;
            input.step = "1";
            input.value = cfg.values[name];
            input.disabled = locked;
            input.dataset.config = name;
            control.appendChild(input);
        }
        else {
            const input = document.createElement("input");
            input.type = "range";
            input.min = limits.min;
            input.max = limits.max;
            input.step = String(meta.step || 0.05);
            input.value = cfg.values[name];
            input.disabled = locked;
            input.dataset.config = name;
            const value = el("span", "range-value");
            const fmt = (v) => (meta.step ? `${Math.round(v)}s` : Number(v).toFixed(2));
            value.textContent = fmt(cfg.values[name]);
            input.addEventListener("input", () => (value.textContent = fmt(input.value)));
            control.append(input, value);
        }
        row.append(label, control, help);
        if (locked)
            row.appendChild(lockedHint());
        $(meta.section).appendChild(row);
    }
}
function lockedHint() {
    const hint = el("p", "locked-hint");
    hint.textContent = "Set by an environment variable — can't be changed here.";
    return hint;
}
// Comparison cards under a model select: every catalog option with its pros
// and cons. Clicking a card drives the select, so saving still goes through
// the normal config flow.
function buildModelOverview(select, options, locked) {
    const overview = el("div", "model-overview");
    for (const opt of options) {
        const card = el("button", "model-card");
        card.type = "button";
        card.dataset.modelId = opt.id;
        card.disabled = locked;
        const dash = opt.label.indexOf(" — ");
        const head = el("div", "model-card-head");
        const name = el("span", "model-card-name");
        name.textContent = dash === -1 ? opt.label : opt.label.slice(0, dash);
        const badge = el("span", "model-card-badge" + (opt.ready ? " ready" : ""));
        badge.textContent = opt.ready ? "Installed" : opt.size || "Download";
        head.append(name, badge);
        card.appendChild(head);
        if (dash !== -1) {
            const tagline = el("p", "model-card-tagline");
            tagline.textContent = opt.label.slice(dash + 3);
            card.appendChild(tagline);
        }
        card.appendChild(prosConsList(opt.pros, "pros"));
        card.appendChild(prosConsList(opt.cons, "cons"));
        card.addEventListener("click", () => {
            select.value = opt.id;
            select.dispatchEvent(new Event("change"));
        });
        overview.appendChild(card);
    }
    syncModelOverview(overview, select.value);
    return overview;
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
function syncModelOverview(overview, activeId) {
    for (const card of overview.querySelectorAll(".model-card")) {
        card.classList.toggle("selected", card.dataset.modelId === activeId);
    }
}
// Status line under a model select: download progress, error + retry, or a
// "downloads when you save" note for models not on disk yet.
function updateModelStatus(select, status) {
    const step = status.dataset.modelStep;
    const opt = lastModels?.[step]?.options.find((o) => o.id === select.value);
    status.innerHTML = "";
    if (!opt || opt.ready)
        return;
    if (opt.downloading) {
        const pct = opt.total ? Math.round((opt.downloaded / opt.total) * 100) : 0;
        status.textContent = `Downloading… ${pct}%`;
        const bar = el("span", "bar");
        bar.innerHTML = `<span style="width:${pct}%"></span>`;
        status.appendChild(bar);
    }
    else if (opt.error) {
        status.textContent = `Download failed: ${opt.error} `;
        const retry = el("button", "btn ghost");
        retry.type = "button";
        retry.textContent = "Retry";
        retry.addEventListener("click", async () => {
            try {
                await call("download_model", { step, id: opt.id });
            }
            catch (e) {
                showToast(`Couldn't start download: ${e.message}`);
            }
        });
        status.appendChild(retry);
    }
    else {
        status.textContent = `Not downloaded yet (${opt.size || "size unknown"}) — the download starts when you save. The default model is used until it finishes.`;
    }
}
function refreshModelStatuses() {
    for (const status of document.querySelectorAll(".model-status")) {
        const select = status.parentElement.querySelector("select[data-config]");
        if (select)
            updateModelStatus(select, status);
    }
}
// `models-progress` events keep the settings progress bars moving while any
// on-demand download runs (replaces the old 1.2 s polling loop).
function onModelsProgress(models) {
    const before = lastModels
        ? new Set(["asr", "diar", "summary"].flatMap((s) => lastModels[s].options.filter((o) => o.downloading).map((o) => `${s}:${o.id}`)))
        : new Set();
    lastModels = models;
    refreshModelStatuses();
    for (const s of ["asr", "diar", "summary"]) {
        for (const o of models[s].options) {
            if (before.has(`${s}:${o.id}`) && o.ready) {
                showToast(`Model downloaded — it will be used for the next recording.`, 4000, true);
            }
        }
    }
}
function collectConfigValues() {
    const body = {};
    const controls = document.querySelectorAll("#settings [data-config]");
    for (const input of controls) {
        if (input.disabled)
            continue;
        const name = input.dataset.config;
        if (input.tagName === "SELECT")
            body[name] = input.value;
        else if (input.tagName === "TEXTAREA")
            body[name] = input.value.trim() || null; // empty resets to default
        else if (input.type === "checkbox")
            body[name] = input.checked;
        else if (input.type === "number")
            body[name] = parseInt(input.value, 10);
        else
            body[name] = parseFloat(input.value);
    }
    return body;
}
async function saveConfig() {
    try {
        const cfg = await call("put_config", { fields: collectConfigValues() });
        const models = await call("get_models");
        renderConfigForm(cfg, models);
        showToast("Settings saved — they apply to the next recording.", 3500, true);
    }
    catch (err) {
        showToast(`Couldn't save settings: ${err.message}`);
    }
}
async function resetConfig() {
    const body = {};
    for (const name of Object.keys(CONFIG_META))
        body[name] = null;
    try {
        const cfg = await call("put_config", { fields: body });
        const models = await call("get_models");
        renderConfigForm(cfg, models);
        showToast("Settings reset to defaults.", 3000, true);
    }
    catch (err) {
        showToast(`Couldn't reset settings: ${err.message}`);
    }
}
function renderDataSection(data) {
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
async function revealDataDir() {
    try {
        await call("reveal_data_dir");
    }
    catch (e) {
        showToast(`Couldn't open folder: ${e.message}`);
    }
}
