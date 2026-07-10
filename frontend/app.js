// LazyMeet frontend — vanilla JS over Tauri commands + events (the backend
// runs in-process in Rust; there is no HTTP sidecar anymore).

const tauri = window.__TAURI__;

// Invoke a backend command. Command errors arrive as a structured
// { code, kind, message } object; surface it as an Error with `.status` so
// existing 404/409/422 checks keep working.
async function call(cmd, args) {
  try {
    return await tauri.core.invoke(cmd, args);
  } catch (e) {
    const err = new Error(e && e.message ? e.message : String(e));
    err.status = e && e.code;
    throw err;
  }
}

const listen = (event, handler) => tauri.event.listen(event, handler);
const convertFileSrc = (path) => tauri.core.convertFileSrc(path);

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
let uploading = false;
let meetings = []; // cached list — search re-filters without refetching
let searchQuery = "";
let currentMeeting = null;
let editing = null; // "title" | "summary" | "seg:3" | "takeaways:1" | … | null
let pendingMeeting = null;
let viewingVersion = null; // history version number rendered read-only, or null for current

// --------------------------------------------------------------------------
// First-run setup. The download runs in the backend either way; the user can
// enter the app immediately ("Use the app while it downloads") and follow
// progress in a header pill. Uploads made meanwhile wait in the processing
// queue until the models are on disk. Progress arrives as `setup-progress`
// events (the old 1 s polling loop is gone).
// --------------------------------------------------------------------------
let modelsReady = false;

function applySetupStatus(status) {
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
}

/// Resolves once all first-run models are on disk.
async function watchSetup() {
  const status = await call("get_setup");
  applySetupStatus(status);
  if (status.ready) return;
  await new Promise((resolve) => {
    setupReadyResolve = resolve;
  });
}

let setupReadyResolve = null;

let lastSetupStatus = null;

function renderDownloadPill(status) {
  lastSetupStatus = status;
  const pill = $("dl-pill");
  pill.classList.remove("hidden");
  if (status.error) {
    pill.classList.add("failed");
    pill.textContent = "Model download failed — click for details";
  } else {
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
  if (!$("dl-popover").classList.contains("hidden")) renderDownloadPopover(status);
}

const fmtMB = (bytes) => `${Math.round(bytes / (1 << 20))} MB`;

function renderDownloadPopover(status) {
  const pop = $("dl-popover");
  pop.innerHTML = "";
  for (const item of status.items) {
    const pct = item.total ? Math.round((item.downloaded / item.total) * 100) : (item.ready ? 100 : 0);
    const label = item.ready
      ? "✓"
      : item.total
        ? `${fmtMB(item.downloaded)} / ${fmtMB(item.total)} · ${pct}%`
        : "starting…";
    const div = el("div", "setup-item");
    div.innerHTML = `<div class="setup-name"><span>${escapeHtml(item.name)}</span><span>${label}</span></div>
      <div class="bar"><span style="width:${pct}%"></span></div>`;
    pop.appendChild(div);
  }
  if (status.error) {
    const err = el("p", "dl-error");
    err.textContent = status.error;
    pop.appendChild(err);
    const retry = el("button", "btn ghost");
    retry.textContent = "Retry download";
    retry.addEventListener("click", retrySetup);
    pop.appendChild(retry);
  }
}

function toggleDownloadPopover(open) {
  const pop = $("dl-popover");
  const show = open !== undefined ? open : pop.classList.contains("hidden");
  pop.classList.toggle("hidden", !show);
  $("dl-pill").setAttribute("aria-expanded", String(show));
  if (show && lastSetupStatus) renderDownloadPopover(lastSetupStatus);
}

async function retrySetup() {
  $("setup-error").classList.add("hidden");
  $("setup-retry").classList.add("hidden");
  try {
    await call("retry_setup");
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
  // Click/keyboard opens a native file dialog (webview File objects have no
  // filesystem path, so the old <input type="file"> can't feed the backend).
  const browse = async () => {
    try {
      const path = await call("pick_recording");
      if (path) upload(path);
    } catch (e) {
      showToast(`Couldn't open the file picker: ${e.message}`);
    }
  };
  dz.addEventListener("click", browse);
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      browse();
    }
  });
  // Drops are accepted anywhere in the window, not just on the dropzone.
  // Tauri's native drag-drop gives real filesystem paths (the HTML5 events no
  // longer fire once the native interception is on).
  tauri.webview.getCurrentWebview().onDragDropEvent((event) => {
    const type = event.payload.type;
    if (type === "enter" || type === "over") dz.classList.add("drag");
    else if (type === "leave") dz.classList.remove("drag");
    else if (type === "drop") {
      dz.classList.remove("drag");
      const path = event.payload.paths && event.payload.paths[0];
      if (path) upload(path);
    }
  });
}

async function upload(path) {
  if (uploading) return;
  const dz = $("dropzone");
  const label = dz.querySelector("strong");
  const originalLabel = label.textContent;
  uploading = true;
  dz.classList.add("uploading");
  label.textContent = "Uploading…";
  try {
    const numSpeakers = parseInt($("num-speakers")?.value ?? "-1", 10);
    const { id } = await call("create_meeting", {
      path,
      num_speakers: Number.isNaN(numSpeakers) ? -1 : numSpeakers,
    });
    await refreshMeetings();
    selectMeeting(id);
  } catch (e) {
    showToast(`Upload failed: ${e.message}`);
  } finally {
    uploading = false;
    dz.classList.remove("uploading");
    label.textContent = originalLabel;
  }
}

async function refreshMeetings() {
  try {
    meetings = await call("list_meetings");
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
    del.className = "btn ghost";
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

// --------------------------------------------------------------------------
// Meeting view. Progress arrives as `meeting-updated` events (no polling);
// selecting a meeting does one fetch for the current state.
// --------------------------------------------------------------------------
function selectMeeting(id) {
  activeId = id;
  editing = null; // abandon any open edit — it belonged to the previous meeting
  pendingMeeting = null;
  viewingVersion = null;
  releasePlayer(); // the playing audio belonged to the previous meeting
  $("empty").classList.add("hidden");
  renderMeetingList();
  showActiveMeeting();
}

function showEmpty() {
  activeId = null;
  currentMeeting = null;
  editing = null;
  pendingMeeting = null;
  viewingVersion = null;
  releasePlayer();
  $("processing").classList.add("hidden");
  $("result").classList.add("hidden");
  $("empty").classList.remove("hidden");
}

/// Fetch the active meeting and render whichever view its status calls for.
async function showActiveMeeting() {
  const id = activeId;
  if (!id) return;
  let m;
  try {
    m = await call("get_meeting", { meeting_id: id });
  } catch (e) {
    if (id !== activeId) return; // stale response for a deselected meeting
    if (e.status === 404) {
      showEmpty();
      showToast("Meeting no longer exists");
      refreshMeetings();
    } else {
      showToast(`Couldn't load meeting: ${e.message}`);
    }
    return;
  }
  if (id !== activeId) return; // user selected another meeting mid-request
  if (m.status === "done" || m.status === "error") {
    renderResult(m);
  } else {
    renderProcessing(m);
  }
}

/// `meeting-updated` event: {id, status} on every stage transition, plus the
/// full public meeting on done/error.
function onMeetingUpdated({ id, status, meeting }) {
  refreshMeetings();
  if (id !== activeId) return;
  if (meeting) {
    renderResult(meeting);
  } else {
    showActiveMeeting(); // intermediate stage — fetch for the stage render
  }
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

  // Version viewing: an older snapshot renders read-only in place of the
  // live content until "Back to current". The latest version IS the live
  // content, and entries from before snapshots existed have nothing to show.
  const versions = m.versions || [];
  const latestV = versions.length ? versions[versions.length - 1].v : null;
  let viewed = null;
  if (viewingVersion != null && viewingVersion !== latestV) {
    viewed = versions.find((v) => v.v === viewingVersion && v.snapshot) || null;
  }
  if (!viewed) viewingVersion = null;
  const shown = viewed ? { ...m, ...viewed.snapshot } : m;

  const banner = $("version-banner");
  banner.classList.toggle("hidden", !viewed);
  if (viewed) {
    $("version-banner-text").textContent =
      `Viewing v${viewed.v} · ${HISTORY_ACTIONS[viewed.action] || "Processed"} · ${fmtDate(viewed.created_at)} — read-only. The current version is v${latestV}.`;
  }
  $("version-back").onclick = () => {
    viewingVersion = null;
    renderResult(currentMeeting);
  };

  $("result-title").textContent = m.title || m.filename;
  $("result-meta").textContent = fmtDate(m.created_at);
  $("result-title").onclick = startTitleEdit;
  $("title-edit-btn").onclick = startTitleEdit;

  // Wire the toolbar first so Delete/Export always work — even for an errored
  // meeting with nothing else to show.
  const deleteBtn = $("delete-btn");
  disarm(deleteBtn);
  // Export always writes the current version — disabled while viewing an old
  // one so it can't silently export something other than what's on screen.
  $("export-btn").disabled = !!viewed;
  $("export-btn").title = viewed ? "Go back to the current version to export" : "";
  $("export-btn").onclick = async () => {
    try {
      const path = await call("export_meeting", { meeting_id: m.id });
      if (path) showToast(`Exported to ${path}`, 4000, true);
    } catch (e) {
      showToast(`Export failed: ${e.message}`);
    }
  };
  $("copy-notes-btn").onclick = () => copyText(notesAsText(shown), "Notes copied");
  $("copy-transcript-btn").onclick = () => copyText(transcriptAsText(shown), "Transcript copied");
  deleteBtn.onclick = () => {
    confirmThen(deleteBtn, "Sure?", async () => {
      try {
        releasePlayer(); // let go of the audio file first (Windows locks it)
        await call("delete_meeting", { meeting_id: m.id });
        showEmpty();
        refreshMeetings();
      } catch (e) {
        showToast(`Delete failed: ${e.message}`);
      }
    });
  };

  // Regeneration: notes need a transcript; a full re-run needs the kept
  // audio. Hidden while viewing an old version — they act on the current one.
  const regenNotesBtn = $("regen-notes-btn");
  const retranscribeBtn = $("retranscribe-btn");
  disarm(regenNotesBtn);
  disarm(retranscribeBtn);
  regenNotesBtn.classList.toggle("hidden", !!viewed || !(m.segments || []).length);
  retranscribeBtn.classList.toggle("hidden", !!viewed || !m.has_audio);
  regenNotesBtn.onclick = () =>
    confirmThen(regenNotesBtn, "Click again to replace the current notes", () =>
      reprocess(m.id, "notes"));
  retranscribeBtn.onclick = () =>
    confirmThen(retranscribeBtn, "Click again to replace transcript, speakers, and notes", () =>
      reprocess(m.id, "transcription"));

  const errNote = $("error-note");
  if (m.status === "error") {
    errNote.textContent = m.error || "Processing failed.";
    errNote.classList.remove("hidden");
  } else {
    errNote.classList.add("hidden");
  }

  // Content edits are only allowed once processing succeeded — mirrors the
  // backend, which 409s earlier (the worker would clobber the edit). An old
  // version on screen is a snapshot — never editable.
  const editable = m.status === "done" && !viewed;
  $("summary-text").textContent = shown.summary || "";
  $("summary-edit-btn").classList.toggle("hidden", !editable);
  $("summary-edit-btn").onclick = startSummaryEdit;
  renderNoteList(shown, "takeaways", "takeaways", editable);
  renderNoteList(shown, "action-items", "action_items", editable);

  const colorOf = buildColorMap(shown.segments || []);
  renderLegend(shown, colorOf, !!viewed);
  renderTranscript(shown, colorOf, editable);
  renderHistory(m, latestV);
}

/// Queue a regeneration; the meeting-updated event flow takes over from there.
async function reprocess(meetingId, mode) {
  try {
    await call("reprocess_meeting", { meeting_id: meetingId, mode });
    showToast(mode === "notes" ? "Regenerating the notes…" : "Re-transcribing the recording…", 3000, true);
  } catch (e) {
    showToast(
      e.status === 409
        ? "Still processing — try again when it's done."
        : `Couldn't start: ${e.message}`
    );
  }
}

const HISTORY_ACTIONS = {
  processed: "Processed",
  notes: "Notes regenerated",
  transcription: "Re-transcribed",
};

// One row per (re)generation, newest first: which models made that version.
// The newest row is the live content; older rows with a snapshot open it
// read-only. Models show "not recorded" for meetings processed before
// version history existed — that can't be reconstructed afterwards.
function renderHistory(m, latestV) {
  const versions = m.versions || [];
  $("history-section").classList.toggle("hidden", !versions.length);
  const list = $("history-list");
  list.innerHTML = "";
  for (const v of [...versions].reverse()) {
    const li = document.createElement("li");
    const badge = el("span", "history-v");
    badge.textContent = `v${v.v}`;
    const action = el("span", "history-action");
    action.textContent = HISTORY_ACTIONS[v.action] || v.action || "Processed";
    const when = el("span", "history-when");
    when.textContent = v.created_at ? fmtDate(v.created_at) : "";
    const models = el("span", "history-models");
    models.textContent =
      `transcription: ${v.asr_model || "not recorded"} · notes: ${v.summary_model || "not recorded"}`;
    li.append(badge, action, when, models);

    if (v.v === latestV) {
      const chip = el("span", "history-current");
      chip.textContent = viewingVersion == null ? "current" : "back to current";
      if (viewingVersion != null) {
        const btn = el("button", "btn ghost history-view");
        btn.type = "button";
        btn.textContent = "Back to current";
        btn.onclick = () => {
          viewingVersion = null;
          renderResult(currentMeeting);
        };
        li.appendChild(btn);
      } else {
        li.appendChild(chip);
      }
    } else if (v.snapshot) {
      const btn = el("button", "btn ghost history-view");
      btn.type = "button";
      btn.textContent = v.v === viewingVersion ? "Viewing" : "View";
      btn.disabled = v.v === viewingVersion;
      btn.onclick = () => {
        viewingVersion = v.v;
        renderResult(currentMeeting);
      };
      li.appendChild(btn);
    } else {
      const note = el("span", "history-nocopy");
      note.textContent = "no saved copy";
      note.title = "This version was replaced before LazyMeet kept snapshots.";
      li.appendChild(note);
    }
    list.appendChild(li);
  }
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
      const m = await call("patch_meeting", {
        meeting_id: currentMeeting.id,
        fields: { title: value },
      });
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
      const m = await call("patch_meeting", {
        meeting_id: currentMeeting.id,
        fields: { summary: ta.value },
      });
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

function renderNoteList(m, elId, field, editable) {
  const ul = $(elId);
  ul.innerHTML = "";
  const items = m[field] || [];
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
    const m = await call("patch_meeting", {
      meeting_id: currentMeeting.id,
      fields: { [field]: items },
    });
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

function renderLegend(m, colorOf, readOnly) {
  const legend = $("speaker-legend");
  legend.innerHTML = "";
  for (const id of Object.keys(colorOf)) {
    const recognized = (m.recognized || []).includes(id);
    const confirmed = (m.confirmed || []).includes(id);
    const name = m.speakers[id] || id;
    const chip = el("div", "chip");
    chip.dataset.speakerId = id;
    chip.style.setProperty("--speaker-color", colorOf[id]);
    chip.innerHTML = `<span class="swatch"></span>`;
    const input = document.createElement("input");
    input.value = name;
    input.title = readOnly
      ? "Read-only — speakers belong to the current version"
      : "Rename — saves this voice for future recordings";
    input.disabled = !!readOnly;
    input.setAttribute("aria-label", `Rename ${name}`);
    input.setAttribute("list", "speaker-names");
    input.addEventListener("change", () => renameSpeaker(m.id, id, input.value));
    chip.appendChild(input);
    if (readOnly) {
      legend.appendChild(chip);
      continue; // no verify buttons on a snapshot
    }
    if (recognized && confirmed) {
      const badge = el("span", "badge-recognized");
      badge.textContent = "known";
      badge.title = "Confirmed — this recording refined their saved voice";
      chip.appendChild(badge);
    } else if (recognized) {
      // Auto-matched but unverified: ask before the name is trusted, so a
      // wrong match can be unlinked without poisoning the saved voice.
      const badge = el("span", "badge-recognized unconfirmed");
      badge.textContent = "known?";
      badge.title = `Auto-recognized as ${name} from a saved voice — is that right?`;
      chip.appendChild(badge);
      chip.appendChild(
        iconBtn("i-check", `Yes, this is ${name}`, () => verifySpeaker(m.id, id, true), "icon-btn verify yes")
      );
      chip.appendChild(
        iconBtn("i-x", `Not ${name} — unlink and pick another name`, () => verifySpeaker(m.id, id, false), "icon-btn verify no")
      );
    }
    legend.appendChild(chip);
  }
}

async function verifySpeaker(meetingId, diarizedId, correct) {
  try {
    const m = await call("verify_speaker", {
      meeting_id: meetingId,
      diarized_id: diarizedId,
      correct,
    });
    renderResult(m);
    refreshSpeakers();
    if (!correct) {
      // The name is back to "Speaker N" — put the cursor there so the right
      // name (or a brand-new person) can be typed straight away.
      const input = $("speaker-legend").querySelector(
        `.chip[data-speaker-id="${CSS.escape(diarizedId)}"] input`
      );
      if (input) {
        input.focus();
        input.select();
      }
    }
  } catch (e) {
    showToast(`Couldn't update speaker: ${e.message}`);
  }
}

function renderTranscript(m, colorOf, editable) {
  stopSegmentPlayback(); // the buttons owning the playing state are re-created
  const t = $("transcript");
  t.innerHTML = "";
  (m.segments || []).forEach((seg, i) => {
    const div = el("div", "seg");
    div.style.setProperty("--speaker-color", colorOf[seg.speaker] || "var(--border-strong)");
    const name = m.speakers[seg.speaker] || seg.speaker;
    div.innerHTML = `<div class="meta"><span class="who">${escapeHtml(name)}</span><span class="ts">${fmtTs(seg.start)}</span></div>
      <div class="seg-text">${escapeHtml(seg.text)}</div>`;
    if (m.has_audio || editable) {
      const actions = el("div", "seg-actions");
      if (m.has_audio) {
        const btn = iconBtn("i-play", "Play this section", () => playSegment(m.id, seg, i, btn, div));
        actions.appendChild(btn);
      }
      if (editable) {
        actions.appendChild(iconBtn("i-pencil", "Edit segment", () => startSegmentEdit(i, div)));
      }
      div.appendChild(actions);
    }
    t.appendChild(div);
  });
}

// --------------------------------------------------------------------------
// Segment playback — hear a section to decide who's speaking before naming.
// One shared <audio> streams the kept recording through the asset protocol
// (Range/seek support); play seeks to the segment start and pauses at its
// end. Clicking the playing segment stops it.
// --------------------------------------------------------------------------
let player = null;
let playerSrc = "";
let playingIndex = null;
let playingBtn = null;
let playingSeg = null; // the .seg element, for the highlight class

function stopSegmentPlayback() {
  if (player) {
    player.pause();
    player.ontimeupdate = null;
    player.onended = null;
  }
  if (playingBtn) {
    playingBtn.querySelector("use").setAttribute("href", "#i-play");
    playingBtn.title = "Play this section";
    playingBtn.setAttribute("aria-label", "Play this section");
  }
  if (playingSeg) playingSeg.classList.remove("playing");
  playingIndex = null;
  playingBtn = null;
  playingSeg = null;
}

// Also drop the audio source so the OS file handle is released (deleting a
// meeting on Windows fails while the webview holds its recording open).
function releasePlayer() {
  stopSegmentPlayback();
  if (player) {
    player.removeAttribute("src");
    player.load();
  }
  playerSrc = "";
}

async function playSegment(meetingId, seg, index, btn, segEl) {
  if (playingIndex === index) {
    stopSegmentPlayback();
    return;
  }
  stopSegmentPlayback();
  if (!player) {
    player = new Audio();
    player.preload = "auto";
  }
  let src;
  try {
    const audio = await call("get_meeting_audio_path", { meeting_id: meetingId });
    src = convertFileSrc(audio.path);
  } catch (e) {
    showToast(
      e.status === 404 ? "The recording is no longer on disk." : `Couldn't load audio: ${e.message}`
    );
    return;
  }
  if (playerSrc !== src) {
    player.src = src;
    playerSrc = src;
  }
  playingIndex = index;
  playingBtn = btn;
  playingSeg = segEl;
  btn.querySelector("use").setAttribute("href", "#i-stop");
  btn.title = "Stop";
  btn.setAttribute("aria-label", "Stop");
  segEl.classList.add("playing");

  const begin = () => {
    player.currentTime = seg.start || 0;
    // Stop at the segment boundary (with a whisker of slack — timeupdate
    // fires only a few times a second).
    player.ontimeupdate = () => {
      if (seg.end && player.currentTime >= seg.end) stopSegmentPlayback();
    };
    player.onended = stopSegmentPlayback;
    player.play().catch((e) => {
      stopSegmentPlayback();
      showToast(`Couldn't play audio: ${e.message}`);
    });
  };
  // Seeking needs the media metadata; wait for it on first load.
  if (player.readyState >= 1) {
    begin();
  } else {
    player.onloadedmetadata = begin;
    player.onerror = () => {
      stopSegmentPlayback();
      showToast("Couldn't load the recording for playback.");
    };
  }
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
      const m = await call("patch_segment", {
        meeting_id: currentMeeting.id,
        index,
        fields: body,
      });
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
    const m = await call("rename_speakers", {
      meeting_id: meetingId,
      names: { [diarizedId]: newName },
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
    help: "Where the speech-to-text model runs. Its engine (onnxruntime) accelerates on the Neural Engine on Apple Silicon; the GPU is not available for it. CPU measures fastest on most machines.",
    providers: true,
  },
  DIAR_PROVIDER: {
    section: "config-processing",
    label: "Speaker separation compute",
    help: "Where the who-spoke-when model runs: CPU or Neural Engine.",
    providers: true,
  },
  SPEAKER_PROVIDER: {
    section: "config-processing",
    label: "Voice matching compute",
    help: "Where the voiceprint model runs: CPU or Neural Engine.",
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
  } catch (e) {
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
  } catch (e) {
    showToast(e.message);
  }
}

async function pickBenchmarkFile() {
  try {
    const path = await call("pick_recording");
    if (path) startBenchmark(path);
  } catch (e) {
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
        if (cls) td.className = cls;
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
      } else {
        cell(`${row.seconds} s`);
        cell(row.speed);
      }
      tbody.appendChild(tr);
    }
    $("bench-results").classList.remove("hidden");
  }
  if (p.status === "running") {
    $("bench-status").textContent = p.step || "Measuring…";
  } else if (p.status === "done") {
    benchRunning = false;
    $("bench-run").disabled = false;
    $("bench-run-file").disabled = false;
    $("bench-status").textContent = "Done.";
    renderBenchRecommendation(p.recommendation, p.current);
  } else if (p.status === "error") {
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
  return (
    `Fastest on this machine: transcription ${PROVIDER_LABELS[rec.ASR_PROVIDER]}` +
    ` · speaker separation ${PROVIDER_LABELS[rec.DIAR_PROVIDER]}` +
    ` · voice matching ${PROVIDER_LABELS[rec.SPEAKER_PROVIDER]}` +
    ` · summarizer ${PROVIDER_LABELS[rec.SUMMARY_PROVIDER]}.`
  );
}

function renderBenchRecommendation(rec, current) {
  if (!rec) return;
  benchRec = rec;
  const matches =
    current &&
    ["ASR_PROVIDER", "DIAR_PROVIDER", "SPEAKER_PROVIDER", "SUMMARY_PROVIDER"].every(
      (k) => current[k] === rec[k]
    );
  const line = $("bench-rec");
  line.textContent =
    benchSummaryText(rec) +
    (matches ? " Your settings already match." : " Your settings differ.");
  line.classList.remove("hidden");
  $("bench-apply").classList.toggle("hidden", matches);
}

async function applyBenchRecommendation() {
  if (!benchRec) return;
  try {
    await call("put_config", { fields: benchRec });
    $("bench-apply").classList.add("hidden");
    $("bench-rec").textContent =
      benchSummaryText(benchRec) + " Applied — used from the next recording on.";
    showToast("Fastest configuration applied.", 3500, true);
  } catch (e) {
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

    if (meta.providers) {
      const select = document.createElement("select");
      select.dataset.config = name;
      const options = (cfg.compute_options && cfg.compute_options[name]) || ["cpu"];
      select.disabled = locked || options.length < 2;
      for (const value of options) {
        const o = document.createElement("option");
        o.value = value;
        o.textContent = PROVIDER_LABELS[value] || value;
        if (value === cfg.values[name]) o.selected = true;
        select.appendChild(o);
      }
      control.appendChild(select);
    } else if (meta.languages) {
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
    } else if (meta.textarea) {
      row.classList.add("stacked");
      const ta = document.createElement("textarea");
      ta.dataset.config = name;
      ta.rows = 4;
      ta.value = cfg.values[name];
      ta.placeholder = cfg.defaults[name] || "";
      if (limits.max) ta.maxLength = limits.max;
      ta.disabled = locked;
      ta.setAttribute("aria-label", meta.label);
      control.appendChild(ta);
      // Stacked layout reads label → help → control.
      row.append(label, help, control);
      if (locked) row.appendChild(lockedHint());
      $(meta.section).appendChild(row);
      continue;
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
        await call("download_model", { step, id: opt.id });
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

// `models-progress` events keep the settings progress bars moving while any
// on-demand download runs (replaces the old 1.2 s polling loop).
function onModelsProgress(models) {
  const before = lastModels
    ? new Set(
        ["asr", "summary"].flatMap((s) =>
          lastModels[s].options.filter((o) => o.downloading).map((o) => `${s}:${o.id}`)
        )
      )
    : new Set();
  lastModels = models;
  refreshModelStatuses();
  for (const s of ["asr", "summary"]) {
    for (const o of models[s].options) {
      if (before.has(`${s}:${o.id}`) && o.ready) {
        showToast(`Model downloaded — it will be used for the next recording.`, 4000, true);
      }
    }
  }
}

function collectConfigValues() {
  const body = {};
  for (const input of document.querySelectorAll("#settings [data-config]")) {
    if (input.disabled) continue;
    const name = input.dataset.config;
    if (input.tagName === "SELECT") body[name] = input.value;
    else if (input.tagName === "TEXTAREA") body[name] = input.value.trim() || null; // empty resets to default
    else if (input.type === "checkbox") body[name] = input.checked;
    else if (input.type === "number") body[name] = parseInt(input.value, 10);
    else body[name] = parseFloat(input.value);
  }
  return body;
}

async function saveConfig() {
  try {
    const cfg = await call("put_config", { fields: collectConfigValues() });
    const models = await call("get_models");
    renderConfigForm(cfg, models);
    showToast("Settings saved — they apply to the next recording.", 3500, true);
  } catch (err) {
    showToast(`Couldn't save settings: ${err.message}`);
  }
}

async function resetConfig() {
  const body = {};
  for (const name of Object.keys(CONFIG_META)) body[name] = null;
  try {
    const cfg = await call("put_config", { fields: body });
    const models = await call("get_models");
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
    await call("reveal_data_dir");
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
  $("speakers").classList.add("hidden");
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
  $("speakers-btn").addEventListener("click", openSpeakers);
  $("speakers-back").addEventListener("click", closeSpeakers);
  $("bench-btn").addEventListener("click", openBench);
  $("bench-back").addEventListener("click", closeBench);
  $("bench-run").addEventListener("click", () => startBenchmark(null));
  $("bench-run-file").addEventListener("click", pickBenchmarkFile);
  $("bench-apply").addEventListener("click", applyBenchRecommendation);
  $("config-save").addEventListener("click", saveConfig);
  $("config-reset").addEventListener("click", resetConfig);
  $("reveal-btn").addEventListener("click", revealDataDir);
  $("meeting-search").addEventListener("input", (e) => {
    searchQuery = e.target.value;
    renderMeetingList();
  });
  refreshMeetings();
  refreshSpeakers();
}

async function main() {
  // Backend events replace the old polling loops; listeners must be armed
  // before the first fetches so no transition slips between them.
  await listen("meeting-updated", (e) => onMeetingUpdated(e.payload));
  await listen("models-progress", (e) => onModelsProgress(e.payload));
  await listen("benchmark-progress", (e) => onBenchProgress(e.payload));
  await listen("setup-progress", (e) => {
    applySetupStatus(e.payload);
    if (e.payload.ready && setupReadyResolve) {
      setupReadyResolve();
      setupReadyResolve = null;
    }
  });

  $("boot").classList.add("hidden");
  $("setup-retry").addEventListener("click", retrySetup);
  $("setup-continue").addEventListener("click", initApp);
  $("dl-pill").addEventListener("click", () => toggleDownloadPopover());
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".dl-wrap")) toggleDownloadPopover(false);
  });
  await watchSetup();
  const backgrounded = appInitialized;
  initApp();
  $("dl-pill").classList.add("hidden");
  toggleDownloadPopover(false);
  if (backgrounded) {
    showToast("All models ready — queued recordings are processing now.", 4000, true);
    refreshMeetings();
  }
}

main().catch(showFatal);
