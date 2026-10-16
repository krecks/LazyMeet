// --------------------------------------------------------------------------
// Meeting view. Progress arrives as `meeting-updated` events (no polling);
// selecting a meeting does one fetch for the current state.
// --------------------------------------------------------------------------
import { $, STAGES, call } from "./core";
import {
  confirmThen,
  disarm,
  editorActions,
  el,
  fmtDate,
  iconBtn,
  showToast,
  wireEditorKeys,
} from "./helpers";
import { refreshMeetings, renderMeetingList } from "./meetings";
import { copyText, notesAsText, releasePlayer, transcriptAsText } from "./player";
import { hideSearchResults, searchOpen } from "./search";
import { state } from "./state";
import { buildColorMap, renderLegend, renderTranscript } from "./transcript";

export function selectMeeting(id) {
  state.activeId = id;
  state.editing = null; // abandon any open edit — it belonged to the previous meeting
  state.pendingMeeting = null;
  state.viewingVersion = null;
  releasePlayer(); // the playing audio belonged to the previous meeting
  $("empty").classList.add("hidden");
  hideSearchResults(); // the meeting takes the pane back from the result list
  renderMeetingList();
  showActiveMeeting();
}

function showEmpty() {
  state.activeId = null;
  state.currentMeeting = null;
  state.editing = null;
  state.pendingMeeting = null;
  state.viewingVersion = null;
  releasePlayer();
  hideSearchResults();
  $("processing").classList.add("hidden");
  $("result").classList.add("hidden");
  $("empty").classList.remove("hidden");
}

/// Fetch the active meeting and render whichever view its status calls for.
export async function showActiveMeeting() {
  const id = state.activeId;
  if (!id) return;
  let m;
  try {
    m = await call("get_meeting", { meeting_id: id });
  } catch (e) {
    if (id !== state.activeId) return; // stale response for a deselected meeting
    if (e.status === 404) {
      showEmpty();
      showToast("Meeting no longer exists");
      refreshMeetings();
    } else {
      showToast(`Couldn't load meeting: ${e.message}`);
    }
    return;
  }
  if (id !== state.activeId) return; // user selected another meeting mid-request
  if (m.status === "done" || m.status === "error") {
    renderResult(m);
  } else {
    renderProcessing(m);
  }
}

/// `meeting-updated` event: {id, status} on every stage transition, plus the
/// full public meeting on done/error.
export function onMeetingUpdated({ id, status, meeting }) {
  refreshMeetings();
  if (id !== state.activeId) return;
  if (meeting) {
    renderResult(meeting);
  } else {
    showActiveMeeting(); // intermediate stage — fetch for the stage render
  }
}

function renderProcessing(m) {
  $("result").classList.add("hidden");
  $("processing").classList.toggle("hidden", searchOpen());
  const title = m.title || m.filename;
  $("proc-title").textContent =
    m.status === "queued"
      ? state.modelsReady
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

export function renderResult(m) {
  if (state.editing) {
    // An inline edit is open — re-rendering now would destroy it. Apply once
    // the edit commits or cancels (finishEdit).
    state.pendingMeeting = m;
    return;
  }
  state.currentMeeting = m;
  $("processing").classList.add("hidden");
  // A background update must not push the meeting back over the search
  // results; the render still happens, it just stays behind them.
  $("result").classList.toggle("hidden", searchOpen());

  // Version viewing: an older snapshot renders read-only in place of the
  // live content until "Back to current". The latest version IS the live
  // content, and entries from before snapshots existed have nothing to show.
  const versions = m.versions || [];
  const latestV = versions.length ? versions[versions.length - 1].v : null;
  let viewed = null;
  if (state.viewingVersion != null && state.viewingVersion !== latestV) {
    viewed = versions.find((v) => v.v === state.viewingVersion && v.snapshot) || null;
  }
  if (!viewed) state.viewingVersion = null;
  const shown = viewed ? { ...m, ...viewed.snapshot } : m;

  const banner = $("version-banner");
  banner.classList.toggle("hidden", !viewed);
  if (viewed) {
    $("version-banner-text").textContent =
      `Viewing v${viewed.v} · ${HISTORY_ACTIONS[viewed.action] || "Processed"} · ${fmtDate(viewed.created_at)} — read-only. The current version is v${latestV}.`;
  }
  $("version-back").onclick = () => {
    state.viewingVersion = null;
    renderResult(state.currentMeeting);
  };

  $("result-title").textContent = m.title || m.filename;
  const rangeCount = Array.isArray(m.ranges) ? m.ranges.length : 0;
  $("result-meta").textContent =
    fmtDate(m.created_at) +
    (rangeCount ? ` · transcribed ${rangeCount} selected segment${rangeCount === 1 ? "" : "s"}` : "");
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
      chip.textContent = state.viewingVersion == null ? "current" : "back to current";
      if (state.viewingVersion != null) {
        const btn = el("button", "btn ghost history-view");
        btn.type = "button";
        btn.textContent = "Back to current";
        btn.onclick = () => {
          state.viewingVersion = null;
          renderResult(state.currentMeeting);
        };
        li.appendChild(btn);
      } else {
        li.appendChild(chip);
      }
    } else if (v.snapshot) {
      const btn = el("button", "btn ghost history-view");
      btn.type = "button";
      btn.textContent = v.v === state.viewingVersion ? "Viewing" : "View";
      btn.disabled = v.v === state.viewingVersion;
      btn.onclick = () => {
        state.viewingVersion = v.v;
        renderResult(state.currentMeeting);
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
export function finishEdit(fresh) {
  state.editing = null;
  const m = fresh || state.pendingMeeting || state.currentMeeting;
  state.pendingMeeting = null;
  if (m && m.id === state.activeId) renderResult(m);
}

export function editError(e) {
  return e.status === 409
    ? "Still processing — try again when it's done."
    : `Couldn't save: ${e.message}`;
}

// --------------------------------------------------------------------------
// Inline editing: title, summary, takeaway/action lists, segments
// --------------------------------------------------------------------------
function startTitleEdit() {
  if (state.editing || !state.currentMeeting) return;
  state.editing = "title";
  const h2 = $("result-title");
  const btn = $("title-edit-btn");
  h2.classList.add("hidden");
  btn.classList.add("hidden");
  const input = document.createElement("input");
  input.className = "title-input";
  input.maxLength = 200;
  input.value = state.currentMeeting.title || state.currentMeeting.filename;
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
    if (!value || value === (state.currentMeeting.title || state.currentMeeting.filename)) {
      settle(() => finishEdit(null));
      return;
    }
    try {
      const m = await call("patch_meeting", {
        meeting_id: state.currentMeeting.id,
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
  if (state.editing || !state.currentMeeting) return;
  state.editing = "summary";
  const p = $("summary-text");
  p.classList.add("hidden");
  $("summary-edit-btn").classList.add("hidden");
  const editor = el("div", "editor");
  const ta = document.createElement("textarea");
  ta.rows = 6;
  ta.value = state.currentMeeting.summary || "";
  ta.setAttribute("aria-label", "Summary");
  editor.appendChild(ta);
  const cleanup = () => {
    editor.remove();
    p.classList.remove("hidden");
  };
  const commit = async () => {
    try {
      const m = await call("patch_meeting", {
        meeting_id: state.currentMeeting.id,
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
  if (state.editing || !state.currentMeeting) return;
  state.editing = `${field}:${index}`;
  span.classList.add("hidden");
  const actions = li.querySelector(".item-actions");
  if (actions) actions.classList.add("hidden");
  const input = document.createElement("input");
  input.type = "text";
  input.className = "item-text";
  input.value = state.currentMeeting[field][index];
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
    if (!value || value === state.currentMeeting[field][index]) {
      settle(() => finishEdit(null));
      return;
    }
    const items = [...state.currentMeeting[field]];
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
      meeting_id: state.currentMeeting.id,
      fields: { [field]: items },
    });
    if (viaFinishEdit) finishEdit(m);
    else renderResult(m);
  } catch (e) {
    showToast(editError(e));
    if (viaFinishEdit) finishEdit(null);
  }
}

