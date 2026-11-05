// --------------------------------------------------------------------------
// Recording — the microphone and the computer's audio, from a button under
// the dropzone. Stopping opens the prepare screen with the mixed file.
// Recordings never transcribed (quit, crash, cancelled prep) are offered
// again under the dropzone.
// --------------------------------------------------------------------------
import { $, call } from "./core";
import { el, fmtTs, openPopover, closeMenu, prefs, showToast } from "./helpers";
import { openPrepare } from "./prepare";

let inputs: { inputs: { name: string; default: boolean }[]; system_audio: boolean } | null = null;
let ticking: ReturnType<typeof setInterval> | null = null;
let startedAt = 0;

/// `record-level` events: {mic, system, elapsed}.
export function onRecordLevel(p) {
  setMeter("rec-meter-mic", p.mic);
  setMeter("rec-meter-system", p.system);
  if (typeof p.elapsed === "number") startedAt = Date.now() - p.elapsed * 1000;
}

function setMeter(id: string, rms: number) {
  // RMS → a rough loudness bar: -60 dB empty, 0 dB full.
  const db = 20 * Math.log10(Math.max(rms || 0, 1e-6));
  const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  ($(id).firstElementChild as HTMLElement).style.width = `${pct}%`;
}

function showBar(sources: string[]) {
  $("dropzone").classList.add("hidden");
  $("record-btn").classList.add("hidden");
  $("record-bar").classList.remove("hidden");
  $("rec-meter-mic").parentElement!.classList.toggle("hidden", !sources.includes("mic"));
  $("rec-meter-system").parentElement!.classList.toggle("hidden", !sources.includes("system"));
  if (!ticking) {
    ticking = setInterval(() => {
      $("rec-time").textContent = fmtTs((Date.now() - startedAt) / 1000);
    }, 250);
  }
}

function hideBar() {
  if (ticking) clearInterval(ticking);
  ticking = null;
  $("record-bar").classList.add("hidden");
  $("dropzone").classList.remove("hidden");
  $("record-btn").classList.remove("hidden");
  $("rec-time").textContent = "0:00";
}

async function openRecordPanel() {
  const anchor = $("record-btn");
  const panel = el("div", "menu record-panel");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Record a meeting");
  const head = el("p", "regen-head");
  head.textContent = "Record a meeting";
  const mic = document.createElement("select");
  mic.setAttribute("aria-label", "Microphone");
  const sysWrap = el("label", "record-check");
  const sys = document.createElement("input");
  sys.type = "checkbox";
  const sysText = el("span");
  sysText.textContent = "The computer's audio (the others on a call)";
  sysWrap.append(sys, sysText);
  const note = el("p", "muted small");
  const go = el("button", "btn small");
  go.type = "button";
  go.innerHTML = '<span class="rec-dot"></span><span>Start recording</span>';
  const micField = el("label", "regen-field");
  const micLabel = el("span", "field-label");
  micLabel.textContent = "Microphone";
  micField.append(micLabel, mic);
  panel.append(head, micField, sysWrap, note, go);
  openPopover(anchor, panel);

  try {
    inputs = await call("list_audio_inputs");
  } catch (e) {
    note.textContent = `Couldn't list audio inputs: ${e.message}`;
  }
  mic.appendChild(new Option("Default microphone", "default"));
  for (const i of inputs?.inputs || []) mic.appendChild(new Option(i.name, i.name));
  mic.appendChild(new Option("No microphone", ""));
  const savedMic = prefs.get("recordMic");
  if (savedMic !== null && [...mic.options].some((o) => o.value === savedMic)) mic.value = savedMic;
  sys.checked = prefs.get("recordSystem") !== "0";
  if (inputs && !inputs.system_audio) {
    sys.checked = false;
    sys.disabled = true;
    note.textContent = "This computer's own audio can't be recorded here — only the microphone.";
  } else {
    note.textContent = "The recording stays on this computer. Stop it to choose what to transcribe.";
  }
  go.addEventListener("click", async () => {
    prefs.set("recordMic", mic.value);
    prefs.set("recordSystem", sys.checked ? "1" : "0");
    closeMenu();
    await startRecording(mic.value === "default" ? null : mic.value, sys.checked);
  });
}

async function startRecording(mic: string | null, systemAudio: boolean) {
  try {
    const r = await call("start_recording", { mic, system_audio: systemAudio });
    startedAt = Date.now();
    showBar(r.sources);
    for (const w of r.warnings || []) showToast(w, 6000);
  } catch (e) {
    showToast(`Couldn't start recording: ${e.message}`, 6000);
  }
}

async function stopRecording() {
  const btn = $("rec-stop") as HTMLButtonElement;
  btn.disabled = true;
  try {
    const r = await call("stop_recording");
    hideBar();
    await openPrepare(r.path);
  } catch (e) {
    showToast(`Couldn't finish the recording: ${e.message}`, 6000);
    hideBar();
  } finally {
    btn.disabled = false;
  }
}

async function cancelRecording() {
  await call("cancel_recording").catch(() => {});
  hideBar();
  showToast("Recording discarded", 2500, true);
}

/// Recordings left from before: one row each, "Transcribe" or "Discard".
export async function renderLeftovers() {
  const box = $("leftovers");
  box.innerHTML = "";
  let list: { name: string; path: string }[] = [];
  try {
    list = (await call("leftover_recordings")) || [];
  } catch {
    return;
  }
  box.classList.toggle("hidden", !list.length);
  for (const item of list) {
    const row = el("div", "leftover");
    const text = el("span");
    text.textContent = `Recording from ${item.name.replace(/(\d{2})-(\d{2})-(\d{2})$/, "$1:$2")}`;
    const open = el("button", "link-btn");
    open.type = "button";
    open.textContent = "Transcribe";
    open.addEventListener("click", () => openPrepare(item.path));
    const drop = el("button", "link-btn danger");
    drop.type = "button";
    drop.textContent = "Discard";
    drop.addEventListener("click", async () => {
      await call("discard_recording", { path: item.path }).catch((e) => showToast(e.message));
      renderLeftovers();
    });
    row.append(text, open, drop);
    box.appendChild(row);
  }
}

export async function setupRecording() {
  $("record-btn").addEventListener("click", openRecordPanel);
  $("rec-stop").addEventListener("click", stopRecording);
  $("rec-cancel").addEventListener("click", cancelRecording);
  // A reload in the middle of a recording picks the bar back up.
  try {
    const s = await call("recording_status");
    if (s?.recording) {
      startedAt = Date.now() - s.elapsed * 1000;
      showBar(s.sources);
    }
  } catch {
    /* no recorder — the button still shows */
  }
  renderLeftovers();
}
