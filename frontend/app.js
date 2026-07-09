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

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// State. Renders are imperative; `editing` is the anti-clobber guard: while an
// inline edit is open, fresh meeting data is stashed in `pendingMeeting`
// instead of re-rendering (which would destroy the user's half-typed edit).
// ---------------------------------------------------------------------------
let activeId = null;
let pollTimer = null;
let pollFailures = 0;
let uploading = false;
let meetings = []; // cached list — search re-filters without refetching
let searchQuery = "";
let currentMeeting = null;
let editing = null; // "title" | "summary" | "seg:3" | "takeaways:1" | … | null
let pendingMeeting = null;

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

const jsonOpts = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

// --------------------------------------------------------------------------
// First-run setup. The download runs server-side either way; the user can
// enter the app immediately ("Use the app while it downloads") and follow
// progress in a header pill. Uploads made meanwhile wait in the processing
// queue until the models are on disk.
// --------------------------------------------------------------------------
let modelsReady = false;

async function watchSetup() {
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
    if (status.ready) {
      modelsReady = true;
      return;
    }
    if (appInitialized) {
      renderDownloadPill(status);
    } else {
      $("setup").classList.remove("hidden");
      renderSetup(status);
      $("setup-error").textContent = status.error || "";
      $("setup-error").classList.toggle("hidden", !status.error);
      $("setup-retry").classList.toggle("hidden", !status.error);
    }
    await sleep(1000);
  }
}

function renderDownloadPill(status) {
  const pill = $("dl-pill");
  pill.classList.remove("hidden");
  if (status.error) {
    pill.classList.add("failed");
    pill.textContent = "Model download failed — click to retry";
    return;
  }
  pill.classList.remove("failed");
  const done = status.items.filter((i) => i.ready).length;
  let downloaded = 0;
  let total = 0;
  for (const item of status.items) {
    if (!item.ready && item.total) {
      downloaded += item.downloaded;
      total += item.total;
    }
  }
  const pct = total ? ` · ${Math.round((downloaded / total) * 100)}%` : "";
  pill.textContent = `Downloading models ${done}/${status.items.length}${pct}`;
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
    const div = el("div", "setup-item");
    div.innerHTML = `<div class="setup-name"><span>${escapeHtml(item.name)}</span><span>${item.ready ? "✓" : `${pct}%`}</span></div>
      <div class="bar"><span style="width:${pct}%"></span></div>`;
    $("setup-items").appendChild(div);
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
  // Drops are accepted anywhere in the window, not just on the dropzone.
  // Tauri's native drag-drop interception is disabled (dragDropEnabled: false
  // in tauri.conf.json) so these standard HTML5 events fire in the webview;
  // preventDefault on document is also what stops a stray drop from making
  // the webview navigate to the dropped file.
  const isFileDrag = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  ["dragover", "dragenter"].forEach((ev) =>
    document.addEventListener(ev, (e) => {
      e.preventDefault();
      if (isFileDrag(e)) dz.classList.add("drag");
    }));
  document.addEventListener("dragleave", (e) => {
    if (!e.relatedTarget) dz.classList.remove("drag"); // left the window
  });
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
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
  try {
    meetings = await api("/api/meetings");
  } catch (e) {
    showToast(`Couldn't load meetings: ${e.message}`);
    return;
  }
  renderMeetingList();
}

function badgeFor(status) {
  if (status === "done") return ["done", "done"];
  if (status === "error") return ["error", "failed"];
  if (status === "queued") return ["queued", "queued"];
  return ["processing", status];
}

function renderMeetingList() {
  const list = $("meeting-list");
  list.innerHTML = "";
  const q = searchQuery.trim().toLowerCase();
  const shown = q
    ? meetings.filter((m) => (m.title || m.filename || "").toLowerCase().includes(q))
    : meetings;
  if (!shown.length) {
    list.innerHTML = `<li class="list-empty muted">${meetings.length ? "No matches" : "No meetings yet"}</li>`;
    return;
  }
  for (const m of shown) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    if (m.id === activeId) btn.classList.add("active");
    const [cls, label] = badgeFor(m.status);
    btn.innerHTML = `<span class="m-title">${escapeHtml(m.title || m.filename)}</span>
      <span class="m-meta"><span class="badge ${cls}">${label}</span><span>${fmtRelative(m.created_at)}</span></span>`;
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
    del.className = "btn ghost";
    del.innerHTML = `<span>Forget</span>`;
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
        await api(`/api/speakers/${encodeURIComponent(name)}`, jsonOpts("PATCH", { name: value }));
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

// --------------------------------------------------------------------------
// Meeting view + polling
// --------------------------------------------------------------------------
function selectMeeting(id) {
  activeId = id;
  editing = null; // abandon any open edit — it belonged to the previous meeting
  pendingMeeting = null;
  if (pollTimer) clearInterval(pollTimer);
  $("empty").classList.add("hidden");
  renderMeetingList();
  poll();
  pollTimer = setInterval(poll, 2000);
}

function showEmpty() {
  activeId = null;
  currentMeeting = null;
  editing = null;
  pendingMeeting = null;
  $("processing").classList.add("hidden");
  $("result").classList.add("hidden");
  $("empty").classList.remove("hidden");
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
      showEmpty();
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
  const title = m.title || m.filename;
  $("proc-title").textContent =
    m.status === "queued"
      ? modelsReady
        ? `Queued — “${title}” starts when the current recording finishes`
        : `Queued — “${title}” starts once the models finish downloading`
      : `Processing “${title}”`;
  const stages = $("stages");
  stages.innerHTML = "";
  const currentIdx = STAGES.findIndex(([key]) => key === m.status);
  STAGES.forEach(([key, label], i) => {
    const div = el("div", "stage" + (i < currentIdx ? " done" : i === currentIdx ? " active" : ""));
    div.innerHTML = `<span class="dot"></span><span>${label}</span>`;
    stages.appendChild(div);
  });
}

function renderResult(m) {
  if (editing) {
    // An inline edit is open — re-rendering now would destroy it. Apply once
    // the edit commits or cancels (finishEdit).
    pendingMeeting = m;
    return;
  }
  currentMeeting = m;
  $("processing").classList.add("hidden");
  $("result").classList.remove("hidden");

  $("result-title").textContent = m.title || m.filename;
  $("result-meta").textContent = fmtDate(m.created_at);
  $("result-title").onclick = startTitleEdit;
  $("title-edit-btn").onclick = startTitleEdit;

  // Wire the toolbar first so Delete/Export always work — even for an errored
  // meeting with nothing else to show.
  const deleteBtn = $("delete-btn");
  disarm(deleteBtn);
  $("export-btn").onclick = () => {
    window.open(`${API}/api/meetings/${m.id}/export`, "_blank");
  };
  $("copy-notes-btn").onclick = () => copyText(notesAsText(m), "Notes copied");
  $("copy-transcript-btn").onclick = () => copyText(transcriptAsText(m), "Transcript copied");
  deleteBtn.onclick = () => {
    confirmThen(deleteBtn, "Sure?", async () => {
      try {
        await api(`/api/meetings/${m.id}`, { method: "DELETE" });
        showEmpty();
        refreshMeetings();
      } catch (e) {
        showToast(`Delete failed: ${e.message}`);
      }
    });
  };

  const errNote = $("error-note");
  if (m.status === "error") {
    errNote.textContent = m.error || "Processing failed.";
    errNote.classList.remove("hidden");
  } else {
    errNote.classList.add("hidden");
  }

  // Content edits are only allowed once processing succeeded — mirrors the
  // backend, which 409s earlier (the worker would clobber the edit).
  const editable = m.status === "done";
  $("summary-text").textContent = m.summary || "";
  $("summary-edit-btn").classList.toggle("hidden", !editable);
  $("summary-edit-btn").onclick = startSummaryEdit;
  renderNoteList("takeaways", "takeaways", editable);
  renderNoteList("action-items", "action_items", editable);

  const colorOf = buildColorMap(m.segments || []);
  renderLegend(m, colorOf);
  renderTranscript(m, colorOf, editable);
}

// Called when an inline edit ends. `fresh` is the PATCH response (or null on
// cancel); any update that arrived mid-edit was stashed in pendingMeeting.
function finishEdit(fresh) {
  editing = null;
  const m = fresh || pendingMeeting || currentMeeting;
  pendingMeeting = null;
  if (m && m.id === activeId) renderResult(m);
}

function editError(e) {
  return e.status === 409
    ? "Still processing — try again when it's done."
    : `Couldn't save: ${e.message}`;
}

// --------------------------------------------------------------------------
// Inline editing: title, summary, takeaway/action lists, segments
// --------------------------------------------------------------------------
function startTitleEdit() {
  if (editing || !currentMeeting) return;
  editing = "title";
  const h2 = $("result-title");
  const btn = $("title-edit-btn");
  h2.classList.add("hidden");
  btn.classList.add("hidden");
  const input = document.createElement("input");
  input.className = "title-input";
  input.maxLength = 200;
  input.value = currentMeeting.title || currentMeeting.filename;
  input.setAttribute("aria-label", "Meeting title");
  h2.parentNode.insertBefore(input, h2);
  input.focus();
  input.select();

  let settled = false;
  const settle = (fn) => {
    if (settled) return;
    settled = true;
    input.remove();
    h2.classList.remove("hidden");
    btn.classList.remove("hidden");
    fn();
  };
  const commit = async () => {
    const value = input.value.trim();
    if (!value || value === (currentMeeting.title || currentMeeting.filename)) {
      settle(() => finishEdit(null));
      return;
    }
    try {
      const m = await api(`/api/meetings/${currentMeeting.id}`, jsonOpts("PATCH", { title: value }));
      settle(() => finishEdit(m));
      refreshMeetings();
    } catch (e) {
      showToast(editError(e));
      settle(() => finishEdit(null));
    }
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    if (e.key === "Escape") settle(() => finishEdit(null));
  });
  input.addEventListener("blur", commit);
}

function startSummaryEdit() {
  if (editing || !currentMeeting) return;
  editing = "summary";
  const p = $("summary-text");
  p.classList.add("hidden");
  $("summary-edit-btn").classList.add("hidden");
  const editor = el("div", "editor");
  const ta = document.createElement("textarea");
  ta.rows = 6;
  ta.value = currentMeeting.summary || "";
  ta.setAttribute("aria-label", "Summary");
  editor.appendChild(ta);
  const cleanup = () => {
    editor.remove();
    p.classList.remove("hidden");
  };
  const commit = async () => {
    try {
      const m = await api(
        `/api/meetings/${currentMeeting.id}`,
        jsonOpts("PATCH", { summary: ta.value })
      );
      cleanup();
      finishEdit(m);
    } catch (e) {
      showToast(editError(e));
    }
  };
  const cancel = () => {
    cleanup();
    finishEdit(null);
  };
  editor.appendChild(editorActions(commit, cancel));
  wireEditorKeys(ta, commit, cancel);
  p.parentNode.insertBefore(editor, p.nextSibling);
  ta.focus();
}

function renderNoteList(elId, field, editable) {
  const ul = $(elId);
  ul.innerHTML = "";
  const items = currentMeeting[field] || [];
  if (!items.length) {
    const li = el("li", "list-empty muted");
    li.textContent = "None";
    ul.appendChild(li);
  }
  items.forEach((text, i) => {
    const li = document.createElement("li");
    const span = el("span", "item-text");
    span.textContent = text;
    li.appendChild(span);
    if (editable) {
      const actions = el("span", "item-actions");
      actions.appendChild(iconBtn("i-pencil", "Edit", () => startItemEdit(field, i, li, span)));
      actions.appendChild(
        iconBtn("i-trash", "Remove", (e) =>
          confirmThen(e.currentTarget, "Click again to remove", () =>
            saveList(field, items.filter((_, j) => j !== i))
          )
        )
      );
      li.appendChild(actions);
    }
    ul.appendChild(li);
  });
  // the DOM is rebuilt on every render — drop the previous add-row first
  ul.parentNode.querySelectorAll(".add-row").forEach((r) => r.remove());
  if (editable) {
    const add = el("div", "add-row");
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = field === "takeaways" ? "Add a takeaway…" : "Add an action item…";
    input.setAttribute("aria-label", input.placeholder);
    const submit = () => {
      const value = input.value.trim();
      if (!value) return;
      saveList(field, [...items, value]);
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        submit();
      }
    });
    add.appendChild(input);
    add.appendChild(iconBtn("i-plus", "Add", submit, "icon-btn"));
    ul.parentNode.appendChild(add);
  }
}

function startItemEdit(field, index, li, span) {
  if (editing || !currentMeeting) return;
  editing = `${field}:${index}`;
  span.classList.add("hidden");
  const actions = li.querySelector(".item-actions");
  if (actions) actions.classList.add("hidden");
  const input = document.createElement("input");
  input.type = "text";
  input.className = "item-text";
  input.value = currentMeeting[field][index];
  li.insertBefore(input, span);
  input.focus();

  let settled = false;
  const settle = (fn) => {
    if (settled) return;
    settled = true;
    fn();
  };
  const commit = () => {
    const value = input.value.trim();
    if (!value || value === currentMeeting[field][index]) {
      settle(() => finishEdit(null));
      return;
    }
    const items = [...currentMeeting[field]];
    items[index] = value;
    settle(() => saveList(field, items, true));
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit();
    if (e.key === "Escape") settle(() => finishEdit(null));
  });
  input.addEventListener("blur", commit);
}

async function saveList(field, items, viaFinishEdit = false) {
  try {
    const m = await api(`/api/meetings/${currentMeeting.id}`, jsonOpts("PATCH", { [field]: items }));
    if (viaFinishEdit) finishEdit(m);
    else renderResult(m);
  } catch (e) {
    showToast(editError(e));
    if (viaFinishEdit) finishEdit(null);
  }
}

// --------------------------------------------------------------------------
// Transcript
// --------------------------------------------------------------------------
function buildColorMap(segments) {
  const ids = [...new Set(segments.map((s) => s.speaker))].sort();
  const map = {};
  ids.forEach((id, i) => (map[id] = `var(--speaker-${(i % 6) + 1})`));
  return map;
}

function renderLegend(m, colorOf) {
  const legend = $("speaker-legend");
  legend.innerHTML = "";
  for (const id of Object.keys(colorOf)) {
    const recognized = (m.recognized || []).includes(id);
    const chip = el("div", "chip");
    chip.style.setProperty("--speaker-color", colorOf[id]);
    chip.innerHTML = `<span class="swatch"></span>`;
    const input = document.createElement("input");
    input.value = m.speakers[id] || id;
    input.title = "Rename — saves this voice for future recordings";
    input.setAttribute("aria-label", `Rename ${m.speakers[id] || id}`);
    input.addEventListener("change", () => renameSpeaker(m.id, id, input.value));
    chip.appendChild(input);
    if (recognized) {
      const badge = el("span", "badge-recognized");
      badge.textContent = "known";
      badge.title = "Recognized from a saved voice";
      chip.appendChild(badge);
    }
    legend.appendChild(chip);
  }
}

function renderTranscript(m, colorOf, editable) {
  const t = $("transcript");
  t.innerHTML = "";
  (m.segments || []).forEach((seg, i) => {
    const div = el("div", "seg");
    div.style.setProperty("--speaker-color", colorOf[seg.speaker] || "var(--border-strong)");
    const name = m.speakers[seg.speaker] || seg.speaker;
    div.innerHTML = `<div class="meta"><span class="who">${escapeHtml(name)}</span><span class="ts">${fmtTs(seg.start)}</span></div>
      <div class="seg-text">${escapeHtml(seg.text)}</div>`;
    if (editable) {
      const actions = el("div", "seg-actions");
      actions.appendChild(iconBtn("i-pencil", "Edit segment", () => startSegmentEdit(i, div)));
      div.appendChild(actions);
    }
    t.appendChild(div);
  });
}

function startSegmentEdit(index, segEl) {
  if (editing || !currentMeeting) return;
  editing = `seg:${index}`;
  const seg = currentMeeting.segments[index];
  segEl.classList.add("editing");
  segEl.querySelector(".seg-text").classList.add("hidden");
  const segActions = segEl.querySelector(".seg-actions");
  if (segActions) segActions.classList.add("hidden");

  const editor = el("div", "editor");
  const ta = document.createElement("textarea");
  ta.rows = 3;
  ta.value = seg.text;
  ta.setAttribute("aria-label", "Segment text");
  editor.appendChild(ta);

  const select = document.createElement("select");
  select.setAttribute("aria-label", "Speaker");
  for (const [id, name] of Object.entries(currentMeeting.speakers)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = name || id;
    if (id === seg.speaker) opt.selected = true;
    select.appendChild(opt);
  }

  const cleanup = () => {
    editor.remove();
    segEl.classList.remove("editing");
    segEl.querySelector(".seg-text").classList.remove("hidden");
    if (segActions) segActions.classList.remove("hidden");
  };
  const commit = async () => {
    const body = {};
    const text = ta.value.trim();
    if (text && text !== seg.text) body.text = text;
    if (select.value !== seg.speaker) body.speaker = select.value;
    if (!Object.keys(body).length) {
      cleanup();
      finishEdit(null);
      return;
    }
    try {
      const m = await api(
        `/api/meetings/${currentMeeting.id}/segments/${index}`,
        jsonOpts("PATCH", body)
      );
      cleanup();
      finishEdit(m);
    } catch (e) {
      showToast(editError(e));
    }
  };
  const cancel = () => {
    cleanup();
    finishEdit(null);
  };
  const actions = editorActions(commit, cancel);
  actions.insertBefore(select, actions.firstChild);
  editor.appendChild(actions);
  wireEditorKeys(ta, commit, cancel);
  segEl.appendChild(editor);
  ta.focus();
}

async function renameSpeaker(meetingId, diarizedId, newName) {
  try {
    const m = await api(`/api/meetings/${meetingId}/speakers`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names: { [diarizedId]: newName } }),
    });
    renderResult(m);
    refreshSpeakers();
  } catch (e) {
    showToast(`Rename failed: ${e.message}`);
  }
}

// --------------------------------------------------------------------------
// Copy to clipboard
// --------------------------------------------------------------------------
function notesAsText(m) {
  const lines = [`# ${m.title || m.filename}`, ""];
  if (m.summary) lines.push(m.summary, "");
  if (m.takeaways && m.takeaways.length) {
    lines.push("Key takeaways:");
    m.takeaways.forEach((t) => lines.push(`- ${t}`));
    lines.push("");
  }
  if (m.action_items && m.action_items.length) {
    lines.push("Action items:");
    m.action_items.forEach((a) => lines.push(`- [ ] ${a}`));
  }
  return lines.join("\n").trim();
}

function transcriptAsText(m) {
  return (m.segments || [])
    .map((seg) => `${m.speakers[seg.speaker] || seg.speaker} [${fmtTs(seg.start)}]: ${seg.text}`)
    .join("\n");
}

async function copyText(text, okMsg) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    showToast(okMsg, 2500, true);
  } catch {
    showToast("Copy failed");
  }
}

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
    help: "Hint for Whisper models — Parakeet always detects the language itself.",
    languages: "Auto-detect",
  },
  SPEAKER_THRESHOLD: {
    section: "config-transcription",
    label: "Voice match strictness",
    help: "How closely a voice must match a saved speaker to reuse their name. Higher is stricter.",
  },
  DIAR_THRESHOLD: {
    section: "config-transcription",
    label: "Speaker separation",
    help: "How different voices must sound to count as separate people. Higher means fewer speakers.",
  },
  SPEAKER_MERGE_THRESHOLD: {
    section: "config-transcription",
    label: "Merge similar speakers",
    help: "After separation, voices more similar than this are folded back into one person.",
  },
  MIN_SPEAKER_SECONDS: {
    section: "config-transcription",
    label: "Minimum speaking time",
    help: "Speakers with fewer seconds of speech than this are merged into the closest main voice.",
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
    help: "The language the notes are written in.",
    languages: "Match the meeting",
  },
  USE_GPU: {
    section: "config-processing",
    label: "Hardware acceleration",
    help: "Run the summary model on the GPU where possible. Falls back to the CPU automatically. Transcription and speaker detection always use the CPU, where they run fastest.",
    toggle: true,
  },
  NUM_THREADS: {
    section: "config-processing",
    label: "CPU threads",
    help: "Threads for the parts that run on the CPU. More finish faster but occupy more of the computer.",
    number: true,
  },
  KEEP_AUDIO: {
    section: "config-processing",
    label: "Keep original recordings",
    help: "Keep uploaded audio/video files after processing instead of deleting them.",
    toggle: true,
  },
};

let modelsPollTimer = null;
let lastModels = null;

async function openSettings() {
  $("app").classList.add("hidden");
  $("settings").classList.remove("hidden");
  refreshSpeakers();
  try {
    const [cfg, data, models] = await Promise.all([
      api("/api/config"),
      api("/api/data"),
      api("/api/models"),
    ]);
    renderConfigForm(cfg, models);
    renderDataSection(data);
    maybePollModels();
  } catch (e) {
    showToast(`Couldn't load settings: ${e.message}`);
  }
}

function closeSettings() {
  if (modelsPollTimer) clearInterval(modelsPollTimer);
  modelsPollTimer = null;
  $("settings").classList.add("hidden");
  $("app").classList.remove("hidden");
}

function renderConfigForm(cfg, models) {
  lastModels = models;
  for (const id of ["config-transcription", "config-summary", "config-processing"]) {
    $(id).innerHTML = "";
  }
  for (const [name, meta] of Object.entries(CONFIG_META)) {
    if (!(name in cfg.values)) continue;
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
        if (opt.id === cfg.values[name]) o.selected = true;
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
      if (locked) row.appendChild(lockedHint());
      $(meta.section).appendChild(row);
      continue;
    }

    if (meta.languages) {
      const select = document.createElement("select");
      select.dataset.config = name;
      select.disabled = locked;
      for (const [code, lang] of [["auto", meta.languages], ...LANGUAGES]) {
        const o = document.createElement("option");
        o.value = code;
        o.textContent = lang;
        if (code === cfg.values[name]) o.selected = true;
        select.appendChild(o);
      }
      control.appendChild(select);
    } else if (meta.toggle) {
      const sw = el("label", "switch");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!cfg.values[name];
      input.disabled = locked;
      input.dataset.config = name;
      sw.appendChild(input);
      sw.appendChild(el("span", "track"));
      control.appendChild(sw);
    } else if (meta.number) {
      const input = document.createElement("input");
      input.type = "number";
      input.min = limits.min;
      input.max = limits.max;
      input.step = 1;
      input.value = cfg.values[name];
      input.disabled = locked;
      input.dataset.config = name;
      control.appendChild(input);
    } else {
      const input = document.createElement("input");
      input.type = "range";
      input.min = limits.min;
      input.max = limits.max;
      input.step = meta.step || 0.05;
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
    if (locked) row.appendChild(lockedHint());
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
  if (!opt || opt.ready) return;
  if (opt.downloading) {
    const pct = opt.total ? Math.round((opt.downloaded / opt.total) * 100) : 0;
    status.textContent = `Downloading… ${pct}%`;
    const bar = el("span", "bar");
    bar.innerHTML = `<span style="width:${pct}%"></span>`;
    status.appendChild(bar);
  } else if (opt.error) {
    status.textContent = `Download failed: ${opt.error} `;
    const retry = el("button", "btn ghost");
    retry.type = "button";
    retry.textContent = "Retry";
    retry.addEventListener("click", async () => {
      try {
        await api("/api/models/download", jsonOpts("POST", { step, id: opt.id }));
        maybePollModels(true);
      } catch (e) {
        showToast(`Couldn't start download: ${e.message}`);
      }
    });
    status.appendChild(retry);
  } else {
    status.textContent = `Not downloaded yet (${opt.size || "size unknown"}) — the download starts when you save. The default model is used until it finishes.`;
  }
}

function refreshModelStatuses() {
  for (const status of document.querySelectorAll(".model-status")) {
    const select = status.parentElement.querySelector("select[data-config]");
    if (select) updateModelStatus(select, status);
  }
}

// Poll /api/models while any download is running so progress bars move.
function maybePollModels(force = false) {
  const anyDownloading = () =>
    lastModels &&
    ["asr", "summary"].some((s) => lastModels[s].options.some((o) => o.downloading));
  if (modelsPollTimer || (!force && !anyDownloading())) return;
  modelsPollTimer = setInterval(async () => {
    let models;
    try {
      models = await api("/api/models");
    } catch {
      return; // transient — keep polling
    }
    const before = new Set(
      ["asr", "summary"].flatMap((s) =>
        lastModels[s].options.filter((o) => o.downloading).map((o) => `${s}:${o.id}`)
      )
    );
    lastModels = models;
    refreshModelStatuses();
    for (const s of ["asr", "summary"]) {
      for (const o of models[s].options) {
        if (before.has(`${s}:${o.id}`) && o.ready) {
          showToast(`Model downloaded — it will be used for the next recording.`, 4000, true);
        }
      }
    }
    if (!anyDownloading()) {
      clearInterval(modelsPollTimer);
      modelsPollTimer = null;
    }
  }, 1200);
}

function collectConfigValues() {
  const body = {};
  for (const input of document.querySelectorAll("#settings [data-config]")) {
    if (input.disabled) continue;
    const name = input.dataset.config;
    if (input.tagName === "SELECT") body[name] = input.value;
    else if (input.type === "checkbox") body[name] = input.checked;
    else if (input.type === "number") body[name] = parseInt(input.value, 10);
    else body[name] = parseFloat(input.value);
  }
  return body;
}

async function saveConfig() {
  try {
    const cfg = await api("/api/config", jsonOpts("PUT", collectConfigValues()));
    const models = await api("/api/models");
    renderConfigForm(cfg, models);
    maybePollModels();
    showToast("Settings saved — they apply to the next recording.", 3500, true);
  } catch (err) {
    showToast(`Couldn't save settings: ${err.message}`);
  }
}

async function resetConfig() {
  const body = {};
  for (const name of Object.keys(CONFIG_META)) body[name] = null;
  try {
    const cfg = await api("/api/config", jsonOpts("PUT", body));
    const models = await api("/api/models");
    renderConfigForm(cfg, models);
    showToast("Settings reset to defaults.", 3000, true);
  } catch (err) {
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
    await api("/api/data/reveal", { method: "POST" });
  } catch (e) {
    showToast(`Couldn't open folder: ${e.message}`);
  }
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
function el(tag, cls) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
}

function iconBtn(icon, title, onClick, cls = "icon-btn subtle") {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = cls;
  btn.title = title;
  btn.setAttribute("aria-label", title);
  btn.innerHTML = `<svg class="icon"><use href="#${icon}" /></svg>`;
  btn.addEventListener("click", onClick);
  return btn;
}

function editorActions(onSave, onCancel) {
  const wrap = el("div", "editor-actions");
  const save = el("button", "btn");
  save.type = "button";
  save.textContent = "Save";
  save.addEventListener("click", onSave);
  const cancel = el("button", "btn ghost");
  cancel.type = "button";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", onCancel);
  const hint = el("span", "editor-hint");
  hint.textContent = "⌘↵ save · esc cancel";
  wrap.append(save, cancel, hint);
  return wrap;
}

function wireEditorKeys(textarea, onSave, onCancel) {
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onSave();
    if (e.key === "Escape") onCancel();
  });
}

function showToast(msg, ttl = 5000, ok = false) {
  const div = el("div", "toast" + (ok ? " ok" : ""));
  div.textContent = msg;
  $("toasts").appendChild(div);
  setTimeout(() => div.remove(), ttl);
}

function showFatal(err) {
  $("boot").classList.add("hidden");
  $("setup").classList.add("hidden");
  $("app").classList.add("hidden");
  $("settings").classList.add("hidden");
  $("fatal").classList.remove("hidden");
  $("fatal-msg").textContent = err && err.message ? err.message : String(err);
}

// Two-click guard for destructive buttons: the first click arms the button for
// 3 s, the second runs the action. (Native confirm() is unreliable in webviews.)
// Buttons with a <span> label show the armed text there; icon-only buttons just
// turn red and change their tooltip.
function confirmThen(btn, armedText, action) {
  if (btn.dataset.armed) {
    clearTimeout(+btn.dataset.timer);
    disarm(btn);
    action();
    return;
  }
  btn.dataset.armed = "1";
  const label = btn.querySelector("span:not(.icon)");
  if (label) {
    btn.dataset.orig = label.textContent;
    label.textContent = armedText;
  } else {
    btn.dataset.origTitle = btn.title;
    btn.title = armedText;
  }
  btn.classList.add("armed");
  btn.dataset.timer = setTimeout(() => disarm(btn), 3000);
}

function disarm(btn) {
  if (!btn.dataset.armed) return;
  const label = btn.querySelector("span:not(.icon)");
  if (label && btn.dataset.orig !== undefined) label.textContent = btn.dataset.orig;
  if (btn.dataset.origTitle !== undefined) btn.title = btn.dataset.origTitle;
  delete btn.dataset.armed;
  delete btn.dataset.orig;
  delete btn.dataset.origTitle;
  delete btn.dataset.timer;
  btn.classList.remove("armed");
}

function fmtTs(seconds) {
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  const m = Math.floor(seconds / 60) % 60;
  const h = Math.floor(seconds / 3600);
  return h ? `${h}:${m.toString().padStart(2, "0")}:${s}` : `${m}:${s}`;
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { dateStyle: "long", timeStyle: "short" });
}

function fmtRelative(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const sec = Math.round((then - Date.now()) / 1000);
  const units = [
    ["year", 31536000],
    ["month", 2592000],
    ["week", 604800],
    ["day", 86400],
    ["hour", 3600],
    ["minute", 60],
  ];
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, s] of units) {
    if (Math.abs(sec) >= s) return rtf.format(Math.round(sec / s), unit);
  }
  return "just now";
}

function fmtBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n >= 10 || i === 0 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------
let appInitialized = false;

function initApp() {
  if (appInitialized) return;
  appInitialized = true;
  $("setup").classList.add("hidden");
  $("app").classList.remove("hidden");
  setupDropzone();
  $("settings-btn").addEventListener("click", openSettings);
  $("settings-back").addEventListener("click", closeSettings);
  $("config-save").addEventListener("click", saveConfig);
  $("config-reset").addEventListener("click", resetConfig);
  $("reveal-btn").addEventListener("click", revealDataDir);
  $("meeting-search").addEventListener("input", (e) => {
    searchQuery = e.target.value;
    renderMeetingList();
  });
  refreshMeetings();
}

async function main() {
  await resolveApiBase();
  $("boot").classList.add("hidden");
  $("setup-retry").addEventListener("click", retrySetup);
  $("setup-continue").addEventListener("click", initApp);
  $("dl-pill").addEventListener("click", retrySetup); // no-op unless a download failed
  await watchSetup();
  const backgrounded = appInitialized;
  initApp();
  $("dl-pill").classList.add("hidden");
  if (backgrounded) {
    showToast("All models ready — queued recordings are processing now.", 4000, true);
    refreshMeetings();
  }
}

main().catch(showFatal);
