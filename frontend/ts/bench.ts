// --------------------------------------------------------------------------
// Benchmark: the selected models over five minutes of audio, run the way a
// meeting runs them, scaled to an estimate for one hour of meeting.
// --------------------------------------------------------------------------
import { $, call } from "./core";
import { el, showToast } from "./helpers";
import { openSettings, PROVIDER_LABELS } from "./settings";

/** Where the back button returns to. */
let benchFrom: "app" | "settings" = "app";
let benchRec = null; // recommendation from the last completed run

export function openBench(from: "app" | "settings" = "app") {
  benchFrom = from;
  $(from).classList.add("hidden");
  $("bench").classList.remove("hidden");
  $("bench").querySelector("h1").focus();
}

export function closeBench() {
  $("bench").classList.add("hidden");
  $(benchFrom).classList.remove("hidden");
  if (benchFrom === "settings") $("settings").querySelector("h1").focus();
}

/** "CPU threads" link: Settings → Advanced, focused on the threads field. */
function openThreadsSetting() {
  $("bench").classList.add("hidden");
  if (benchFrom === "settings") $("settings").classList.add("hidden");
  benchFrom = "app";
  openSettings("NUM_THREADS");
}

function setRunning(running: boolean) {
  $("bench-run").disabled = running;
  $("bench-run-file").disabled = running;
}

export async function startBenchmark(path) {
  try {
    $("bench-results").classList.add("hidden");
    $("bench-results").querySelector("tbody").innerHTML = "";
    $("bench-estimate").classList.add("hidden");
    $("bench-rec").classList.add("hidden");
    await call("run_benchmark", { path: path || null });
    setRunning(true);
    $("bench-status").textContent = "Starting…";
  } catch (e) {
    showToast(e.message);
  }
}

export async function pickBenchmarkFile() {
  try {
    const path = await call("pick_recording");
    if (path) startBenchmark(path);
  } catch (e) {
    showToast(`Couldn't open the file picker: ${e.message}`);
  }
}

/** 316 → "5 min", 4100 → "1 h 8 min", 45 → "45 s". */
function fmtDuration(secs: number) {
  const s = Math.round(secs);
  if (s < 90) return `${s} s`;
  const m = Math.round(s / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

// `benchmark-progress` events: rows accumulate as each measurement lands.
export function onBenchProgress(p) {
  if (p.threads) {
    const env = $("bench-env");
    const used = p.cores ? `${p.threads} of ${p.cores} CPU threads` : `${p.threads} CPU threads`;
    const models = p.models
      ? `${p.models.asr} · ${p.models.diar} · ${p.models.summary}`
      : "";
    const source = !p.source || p.source === "synthetic audio" ? "synthetic" : p.source;
    env.innerHTML = "";
    env.append([models, used, `audio: ${source}`].filter(Boolean).join(" · ") + " · ");
    const link = el("button", "link-btn");
    link.type = "button";
    link.textContent = "Change threads";
    link.addEventListener("click", openThreadsSetting);
    env.append(link);
    env.classList.remove("hidden");
  }
  if (p.rows && p.rows.length) renderRows(p.rows, p.recommendation?.SUMMARY_PROVIDER);
  if (p.status === "running") {
    $("bench-status").textContent = p.step || "Measuring…";
  } else if (p.status === "done") {
    setRunning(false);
    $("bench-status").textContent = "Done.";
    renderEstimate(p);
    renderRecommendation(p.rows, p.recommendation, p.current);
  } else if (p.status === "error") {
    setRunning(false);
    $("bench-status").textContent = `Benchmark failed: ${p.message}`;
  }
}

/** `best`: the recommended summary provider (once the run is done). */
function renderRows(rows, best?: string) {
  const tbody = $("bench-results").querySelector("tbody");
  tbody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (best && row.stage === "Summary" && row.provider === best) tr.classList.add("best");
    const cell = (text: string, cls?: string) => {
      const td = document.createElement("td");
      td.textContent = text ?? "";
      if (cls) td.className = cls;
      tr.appendChild(td);
      return td;
    };
    cell(row.stage);
    cell(row.model);
    cell(row.backend);
    if (row.error) {
      cell(row.error, "bench-error").colSpan = 2;
    } else {
      cell(`${row.seconds} s`, "num");
      cell(row.speed);
    }
    tbody.appendChild(tr);
  }
  $("bench-results").classList.remove("hidden");
}

function renderEstimate(p) {
  const box = $("bench-estimate");
  const e = p.estimate;
  if (!e) {
    const failed = (p.rows || []).find((r) => r.error);
    $("bench-estimate-head").textContent = "No estimate for a meeting";
    $("bench-estimate-parts").textContent = failed
      ? failed.error
      : "A stage could not be measured.";
  } else {
    $("bench-estimate-head").textContent =
      `About ${fmtDuration(e.per_hour_secs)} for a 1-hour meeting on this machine`;
    $("bench-estimate-parts").textContent =
      `Transcription and speaker separation ${fmtDuration(e.transcription_secs)} · ` +
      `summary ${fmtDuration(e.summary_secs)} · voice matching ${fmtDuration(e.voice_secs)}`;
  }
  box.classList.remove("hidden");
}

function renderRecommendation(rows, rec, current) {
  benchRec = rec;
  const line = $("bench-rec-text");
  const wanted = rec?.SUMMARY_PROVIDER;
  if (!wanted) {
    $("bench-rec").classList.add("hidden");
    return;
  }
  const secs = (provider) =>
    rows.find((r) => r.stage === "Summary" && r.provider === provider && !r.error)?.seconds;
  const other = wanted === "gpu" ? "cpu" : "gpu";
  const faster =
    `The summary ran faster on the ${PROVIDER_LABELS[wanted]} ` +
    `(${secs(wanted)} s vs ${secs(other)} s on the ${PROVIDER_LABELS[other]}).`;
  const matches = current?.SUMMARY_PROVIDER === wanted;
  line.textContent = matches ? `${faster} Your settings already use it.` : faster;
  $("bench-apply").classList.toggle("hidden", matches);
  $("bench-apply").textContent = `Use the ${PROVIDER_LABELS[wanted]} for summaries`;
  $("bench-rec").classList.remove("hidden");
}

export async function applyBenchRecommendation() {
  if (!benchRec?.SUMMARY_PROVIDER) return;
  try {
    await call("put_config", { fields: { SUMMARY_PROVIDER: benchRec.SUMMARY_PROVIDER } });
    $("bench-apply").classList.add("hidden");
    $("bench-rec-text").textContent +=
      " Applied — used from the next recording on.";
    showToast("Summary compute updated.", 3500, true);
  } catch (e) {
    showToast(`Couldn't apply: ${e.message}`);
  }
}
