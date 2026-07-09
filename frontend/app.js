// LazyMeet frontend — vanilla JS, talks to the Python sidecar over localhost.

// API base is resolved at boot: inside Tauri we ask the Rust shell for the
// sidecar port; in a plain browser we use the same origin that served the page.
let API = "";

async function resolveApiBase() {
  const tauri = window.__TAURI__;
  if (tauri && tauri.core && typeof tauri.core.invoke === "function") {
    const port = await tauri.core.invoke("get_backend_port");
    API = `http://127.0.0.1:${port}`;
  } else if (window.__LAZYMEET_PORT__) {
    API = `http://127.0.0.1:${window.__LAZYMEET_PORT__}`;
  } else {
    API = ""; // same-origin (FastAPI serves the static frontend in browser dev)
  }
}

const STAGES = [
  ["extracting", "Extracting audio"],
  ["transcribing", "Transcribing (Parakeet)"],
  ["diarizing", "Separating speakers"],
  ["identifying", "Identifying speakers"],
  ["summarizing", "Summarizing"],
];

const SPEAKER_COLORS = ["#6ea8fe", "#7ee787", "#f0883e", "#d2a8ff", "#79c0ff", "#ffa657"];

const $ = (id) => document.getElementById(id);
let activeId = null;
let pollTimer = null;
let pollFailures = 0;
let uploading = false;

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = new Error(`${res.status} ${await res.text()}`);
    err.status = res.status;
    throw err;
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

// --------------------------------------------------------------------------
// First-run setup
// --------------------------------------------------------------------------
async function waitForSetup() {
  let failures = 0;
  while (true) {
    let status;
    try {
      status = await api("/api/setup");
      failures = 0;
    } catch (e) {
      if (++failures >= 30) throw e; // backend is gone — escalate to the fatal screen
      await sleep(1000);
      continue;
    }
    if (status.ready) return;
    $("setup").classList.remove("hidden");
    renderSetup(status);
    $("setup-error").textContent = status.error || "";
    $("setup-error").classList.toggle("hidden", !status.error);
    $("setup-retry").classList.toggle("hidden", !status.error);
    await sleep(1000);
  }
}

async function retrySetup() {
  $("setup-error").classList.add("hidden");
  $("setup-retry").classList.add("hidden");
  try {
    await api("/api/setup/retry", { method: "POST" });
  } catch (e) {
    showToast(e.message);
  }
}

function renderSetup(status) {
  $("setup-items").innerHTML = "";
  for (const item of status.items) {
    const pct = item.total ? Math.round((item.downloaded / item.total) * 100) : (item.ready ? 100 : 0);
    const el = document.createElement("div");
    el.className = "setup-item";
    el.innerHTML = `<div>${item.name} ${item.ready ? "✓" : `${pct}%`}</div>
      <div class="bar"><span style="width:${pct}%"></span></div>`;
    $("setup-items").appendChild(el);
  }
}

// --------------------------------------------------------------------------
// Upload + meeting list
// --------------------------------------------------------------------------
function setupDropzone() {
  const dz = $("dropzone");
  const input = $("file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => input.files[0] && upload(input.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  });
}

async function upload(file) {
  if (uploading) return;
  const dz = $("dropzone");
  const label = dz.querySelector("strong");
  const originalLabel = label.textContent;
  uploading = true;
  dz.classList.add("uploading");
  label.textContent = "Uploading…";
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("num_speakers", $("num-speakers")?.value ?? "-1");
    const { id } = await api("/api/meetings", { method: "POST", body: form });
    await refreshMeetings();
    selectMeeting(id);
  } catch (e) {
    showToast(`Upload failed: ${e.message}`);
  } finally {
    uploading = false;
    dz.classList.remove("uploading");
    label.textContent = originalLabel;
    $("file-input").value = "";
  }
}

async function refreshMeetings() {
  let meetings;
  try {
    meetings = await api("/api/meetings");
  } catch (e) {
    showToast(`Couldn't load meetings: ${e.message}`);
    return;
  }
  const list = $("meeting-list");
  list.innerHTML = "";
  if (!meetings.length) {
    list.innerHTML = `<li class="list-empty muted">No meetings yet</li>`;
    return;
  }
  for (const m of meetings) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    if (m.id === activeId) btn.classList.add("active");
    btn.innerHTML = `<span>${escapeHtml(m.filename)}</span><span class="status">${m.status}</span>`;
    btn.addEventListener("click", () => selectMeeting(m.id));
    li.appendChild(btn);
    list.appendChild(li);
  }
}

async function refreshSpeakers() {
  let names;
  try {
    names = await api("/api/speakers");
  } catch (e) {
    showToast(`Couldn't load speakers: ${e.message}`);
    return;
  }
  const list = $("speaker-list");
  list.innerHTML = "";
  if (!names.length) {
    list.innerHTML = `<li class="list-empty muted">No saved voices yet</li>`;
    return;
  }
  for (const name of names) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escapeHtml(name)}</span>`;
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = "Forget this voice";
    del.setAttribute("aria-label", `Forget voice ${name}`);
    del.addEventListener("click", () => {
      confirmThen(del, "Sure?", async () => {
        try {
          await api(`/api/speakers/${encodeURIComponent(name)}`, { method: "DELETE" });
          await refreshSpeakers();
        } catch (e) {
          showToast(`Couldn't forget voice: ${e.message}`);
        }
      });
    });
    li.appendChild(del);
    list.appendChild(li);
  }
}

// --------------------------------------------------------------------------
// Meeting view + polling
// --------------------------------------------------------------------------
function selectMeeting(id) {
  activeId = id;
  if (pollTimer) clearInterval(pollTimer);
  $("empty").classList.add("hidden");
  refreshMeetings();
  poll();
  pollTimer = setInterval(poll, 2000);
}

async function poll() {
  const id = activeId;
  if (!id) return;
  let m;
  try {
    m = await api(`/api/meetings/${id}`);
    pollFailures = 0;
    $("conn-banner").classList.add("hidden");
  } catch (e) {
    if (id !== activeId) return; // stale response for a deselected meeting
    if (e.status === 404) {
      clearInterval(pollTimer);
      pollTimer = null;
      activeId = null;
      $("processing").classList.add("hidden");
      $("result").classList.add("hidden");
      $("empty").classList.remove("hidden");
      showToast("Meeting no longer exists");
      refreshMeetings();
    } else if (++pollFailures >= 3) {
      $("conn-banner").classList.remove("hidden");
    }
    return;
  }
  if (id !== activeId) return; // user selected another meeting mid-request
  if (m.status === "done" || m.status === "error") {
    clearInterval(pollTimer);
    pollTimer = null;
    renderResult(m);
  } else {
    renderProcessing(m);
  }
  refreshMeetings();
}

function renderProcessing(m) {
  $("result").classList.add("hidden");
  $("processing").classList.remove("hidden");
  $("proc-title").textContent = `Processing "${m.filename}"`;
  const stages = $("stages");
  stages.innerHTML = "";
  const currentIdx = STAGES.findIndex(([key]) => key === m.status);
  STAGES.forEach(([key, label], i) => {
    const div = document.createElement("div");
    div.className = "stage" + (i < currentIdx ? " done" : i === currentIdx ? " active" : "");
    div.innerHTML = `<span class="dot"></span><span>${label}</span>`;
    stages.appendChild(div);
  });
}

function renderResult(m) {
  $("processing").classList.add("hidden");
  $("result").classList.remove("hidden");
  $("result-title").textContent = m.filename;

  // Wire the toolbar first so Delete/Export always work — even for an errored
  // meeting that takes the early-return path below.
  const deleteBtn = $("delete-btn");
  disarm(deleteBtn);
  $("export-btn").onclick = () => {
    window.open(`${API}/api/meetings/${m.id}/export`, "_blank");
  };
  deleteBtn.onclick = () => {
    confirmThen(deleteBtn, "Confirm delete", async () => {
      try {
        await api(`/api/meetings/${m.id}`, { method: "DELETE" });
        activeId = null;
        $("result").classList.add("hidden");
        $("empty").classList.remove("hidden");
        refreshMeetings();
      } catch (e) {
        showToast(`Delete failed: ${e.message}`);
      }
    });
  };

  // On error, still show whatever transcript was produced before the failure.
  if (m.status === "error" && (!m.segments || m.segments.length === 0)) {
    $("summary-text").textContent = "Error: " + (m.error || "unknown");
    $("takeaways").innerHTML = "";
    $("action-items").innerHTML = "";
    $("transcript").innerHTML = "";
    $("speaker-legend").innerHTML = "";
    return;
  }

  $("summary-text").textContent =
    m.status === "error" ? "Error: " + (m.error || "unknown") : (m.summary || "");
  fillList($("takeaways"), m.takeaways);
  fillList($("action-items"), m.action_items);

  const colorOf = buildColorMap(m.segments);
  renderLegend(m, colorOf);
  renderTranscript(m, colorOf);
}

function buildColorMap(segments) {
  const ids = [...new Set(segments.map((s) => s.speaker))].sort();
  const map = {};
  ids.forEach((id, i) => (map[id] = SPEAKER_COLORS[i % SPEAKER_COLORS.length]));
  return map;
}

function renderLegend(m, colorOf) {
  const legend = $("speaker-legend");
  legend.innerHTML = "";
  const ids = Object.keys(colorOf);
  for (const id of ids) {
    const recognized = (m.recognized || []).includes(id);
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `<span class="swatch" style="background:${colorOf[id]}"></span>`;
    const input = document.createElement("input");
    input.value = m.speakers[id] || id;
    input.title = "Rename — saves this voice for future recordings";
    input.setAttribute("aria-label", `Rename ${m.speakers[id] || id}`);
    input.addEventListener("change", () => renameSpeaker(m.id, id, input.value));
    chip.appendChild(input);
    if (recognized) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "recognized";
      chip.appendChild(badge);
    }
    legend.appendChild(chip);
  }
}

function renderTranscript(m, colorOf) {
  const t = $("transcript");
  t.innerHTML = "";
  for (const seg of m.segments) {
    const div = document.createElement("div");
    div.className = "bubble";
    div.style.borderLeft = `3px solid ${colorOf[seg.speaker]}`;
    const name = m.speakers[seg.speaker] || seg.speaker;
    div.innerHTML = `<div class="meta"><span class="who" style="color:${colorOf[seg.speaker]}">${escapeHtml(name)}</span> · ${fmtTs(seg.start)}</div>
      <div>${escapeHtml(seg.text)}</div>`;
    t.appendChild(div);
  }
}

async function renameSpeaker(meetingId, diarizedId, newName) {
  try {
    const m = await api(`/api/meetings/${meetingId}/speakers`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names: { [diarizedId]: newName } }),
    });
    renderResult({ ...m, status: "done" });
    refreshSpeakers();
  } catch (e) {
    showToast(`Rename failed: ${e.message}`);
  }
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
function fillList(ul, items) {
  ul.innerHTML = "";
  if (!items || !items.length) {
    ul.innerHTML = `<li class="list-empty muted">None</li>`;
    return;
  }
  for (const it of items) {
    const li = document.createElement("li");
    li.textContent = it;
    ul.appendChild(li);
  }
}

function showToast(msg, ttl = 5000) {
  const div = document.createElement("div");
  div.className = "toast";
  div.textContent = msg;
  $("toasts").appendChild(div);
  setTimeout(() => div.remove(), ttl);
}

function showFatal(err) {
  $("setup").classList.add("hidden");
  $("app").classList.add("hidden");
  $("fatal").classList.remove("hidden");
  $("fatal-msg").textContent = err && err.message ? err.message : String(err);
}

// Two-click guard for destructive buttons: the first click arms the button for
// 3 s, the second runs the action. (Native confirm() is unreliable in webviews.)
function confirmThen(btn, armedText, action) {
  if (btn.dataset.armed !== undefined) {
    clearTimeout(+btn.dataset.timer);
    disarm(btn);
    action();
    return;
  }
  btn.dataset.armed = btn.textContent;
  btn.dataset.timer = setTimeout(() => disarm(btn), 3000);
  btn.classList.add("armed");
  btn.textContent = armedText;
}

function disarm(btn) {
  if (btn.dataset.armed !== undefined) btn.textContent = btn.dataset.armed;
  delete btn.dataset.armed;
  delete btn.dataset.timer;
  btn.classList.remove("armed");
}

function fmtTs(seconds) {
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  const m = Math.floor(seconds / 60) % 60;
  const h = Math.floor(seconds / 3600);
  return h ? `${h}:${m.toString().padStart(2, "0")}:${s}` : `${m}:${s}`;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------
async function main() {
  await resolveApiBase();
  $("setup-retry").addEventListener("click", retrySetup);
  await waitForSetup();
  $("setup").classList.add("hidden");
  $("app").classList.remove("hidden");
  setupDropzone();
  await refreshMeetings();
  await refreshSpeakers();
}

main().catch(showFatal);
