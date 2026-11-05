// --------------------------------------------------------------------------
// Upload + meeting list
// --------------------------------------------------------------------------
import { $, call, getCurrentWebview } from "./core";
import { confirmThen, el, escapeHtml, fmtRelative, type MenuItem, openMenu, plural, prefs, showToast } from "./helpers";
import { exportMeetings, pickExportFormat } from "./exports";
import { selectMeeting, showEmpty } from "./meeting";
import { openPrepare } from "./prepare";
import { gsearch } from "./search";
import { state } from "./state";

export function setupDropzone() {
  const dz = $("dropzone");
  // Click/keyboard opens a native file dialog (webview File objects have no
  // filesystem path, so the old <input type="file"> can't feed the backend).
  const browse = async () => {
    try {
      const path = await call("pick_recording");
      if (path) openPrepare(path);
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
  getCurrentWebview().onDragDropEvent((event) => {
    const type = event.payload.type;
    if (type === "enter" || type === "over") dz.classList.add("drag");
    else if (type === "leave") dz.classList.remove("drag");
    else if (type === "drop") {
      dz.classList.remove("drag");
      const path = event.payload.paths && event.payload.paths[0];
      if (path) openPrepare(path); // replaces the current prep file if one is open
    }
  });
}

export async function refreshMeetings() {
  try {
    state.meetings = await call("list_meetings");
  } catch (e) {
    showToast(`Couldn't load meetings: ${e.message}`);
    return;
  }
  renderMeetingList();
}

export function badgeFor(status) {
  if (status === "done") return ["done", "done"];
  if (status === "error") return ["error", "failed"];
  if (status === "queued") return ["queued", "queued"];
  return ["processing", status];
}

// ---- sorting and date groups ------------------------------------------------

export const SORTS = [
  { key: "newest", label: "Newest first" },
  { key: "oldest", label: "Oldest first" },
  { key: "title", label: "Title" },
  { key: "length", label: "Longest first" },
];

export function currentSort(): string {
  const saved = prefs.get("sort");
  return SORTS.some((s) => s.key === saved) ? (saved as string) : "newest";
}

export function sortMeetings(list: any[], sort: string): any[] {
  const time = (m) => new Date(m.created_at || 0).getTime() || 0;
  const title = (m) => (m.title || m.filename || "").toLocaleLowerCase();
  const out = [...list];
  if (sort === "oldest") out.sort((a, b) => time(a) - time(b));
  else if (sort === "title") out.sort((a, b) => title(a).localeCompare(title(b)) || time(b) - time(a));
  else if (sort === "length") out.sort((a, b) => (b.duration || 0) - (a.duration || 0) || time(b) - time(a));
  else out.sort((a, b) => time(b) - time(a));
  return out;
}

/// "Today", "Yesterday", "Last 7 days", "Earlier this month", then the
/// month ("October 2026") — headers for the date-sorted list.
export function dateGroup(iso: string, now = new Date()): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Undated";
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((day(now) - day(d)) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Last 7 days";
  if (d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth()) return "Earlier this month";
  return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

/// Every tag in the list, most used first (case-insensitive, first spelling).
export function allTags(meetings: any[]): { tag: string; count: number }[] {
  const counts = new Map<string, { tag: string; count: number }>();
  for (const m of meetings) {
    for (const tag of m.tags || []) {
      const key = tag.toLocaleLowerCase();
      const entry = counts.get(key) || { tag, count: 0 };
      entry.count++;
      counts.set(key, entry);
    }
  }
  return [...counts.values()].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
}

const hasTag = (m, tag: string) => (m.tags || []).some((t) => t.toLocaleLowerCase() === tag.toLocaleLowerCase());

/// The meetings the sidebar shows, in display order.
export function visibleMeetings(): any[] {
  const q = state.searchQuery.trim().toLowerCase();
  // Titles filter instantly; meetings whose *content* matched join them once
  // the global search comes back, so the list can't contradict the results.
  const withHits = new Set(gsearch.results.map((g) => g.id));
  let shown = q
    ? state.meetings.filter(
        (m) => withHits.has(m.id) || (m.title || m.filename || "").toLowerCase().includes(q)
      )
    : state.meetings;
  if (state.tagFilter.size) shown = shown.filter((m) => [...state.tagFilter].every((t) => hasTag(m, t)));
  return sortMeetings(shown, currentSort());
}

// ---- rendering ----------------------------------------------------------------

export function setupSidebar() {
  $("sort-btn").addEventListener("click", () => {
    const sort = currentSort();
    openMenu(
      $("sort-btn"),
      SORTS.map((s) => ({
        label: s.label,
        active: s.key === sort,
        onPick: () => {
          prefs.set("sort", s.key);
          renderMeetingList();
        },
      })),
      "Sort meetings"
    );
  });
  $("select-btn").addEventListener("click", () => {
    state.selectMode = !state.selectMode;
    if (!state.selectMode) state.selected.clear();
    renderMeetingList();
  });
  $("bulk-clear").addEventListener("click", () => {
    state.selected.clear();
    state.selectMode = false;
    renderMeetingList();
  });
  $("bulk-export").addEventListener("click", () =>
    pickExportFormat($("bulk-export"), (format) => exportMeetings([...state.selected], format))
  );
  $("bulk-tag").addEventListener("click", openBulkTagMenu);
  const tagInput = $("bulk-tag-input") as HTMLInputElement;
  const hideInput = () => tagInput.classList.add("hidden");
  tagInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && tagInput.value.trim()) {
      tagSelected([...state.selected], [tagInput.value.trim()], []);
      hideInput();
    } else if (e.key === "Escape") hideInput();
  });
  tagInput.addEventListener("blur", hideInput);
  const del = $("bulk-delete") as HTMLButtonElement;
  del.addEventListener("click", () =>
    confirmThen(del, "Sure?", async () => {
      const ids = [...state.selected];
      try {
        const r = await call("delete_meetings", { meeting_ids: ids });
        showToast(`Deleted ${plural(r.deleted, "meeting", "meetings")}`, 3000, true);
      } catch (e) {
        showToast(`Delete failed: ${e.message}`);
      }
      state.selected.clear();
      state.selectMode = false;
      if (state.activeId && ids.includes(state.activeId)) showEmpty();
      await refreshMeetings();
    })
  );
}

function openBulkTagMenu() {
  const ids = [...state.selected];
  const chosen = state.meetings.filter((m) => state.selected.has(m.id));
  const items: MenuItem[] = allTags(state.meetings).map(({ tag }) => {
    const onAll = chosen.every((m) => hasTag(m, tag));
    return {
      label: tag,
      hint: onAll ? "remove" : "add",
      active: onAll,
      onPick: () => void tagSelected(ids, onAll ? [] : [tag], onAll ? [tag] : []),
    };
  });
  items.push({
    label: "New tag…",
    onPick: () => {
      const input = $("bulk-tag-input") as HTMLInputElement;
      input.classList.remove("hidden");
      input.value = "";
      input.focus();
    },
  });
  openMenu($("bulk-tag"), items, "Tag selected meetings");
}

export async function tagSelected(ids: string[], add: string[], remove: string[]) {
  try {
    await call("tag_meetings", { meeting_ids: ids, add, remove });
  } catch (e) {
    showToast(`Couldn't change tags: ${e.message}`);
  }
  await refreshMeetings();
  if (state.activeId && ids.includes(state.activeId)) selectMeeting(state.activeId);
}

function renderTagFilter() {
  const bar = $("tag-filter");
  bar.innerHTML = "";
  const tags = allTags(state.meetings);
  // A filter on a tag nobody carries any more would hide everything.
  for (const t of [...state.tagFilter]) {
    if (!tags.some((x) => x.tag.toLocaleLowerCase() === t.toLocaleLowerCase())) state.tagFilter.delete(t);
  }
  bar.classList.toggle("hidden", !tags.length);
  for (const { tag, count } of tags) {
    const on = [...state.tagFilter].some((t) => t.toLocaleLowerCase() === tag.toLocaleLowerCase());
    const chip = el("button", "tag-chip" + (on ? " on" : ""));
    chip.type = "button";
    chip.setAttribute("aria-pressed", String(on));
    chip.title = on ? `Showing meetings tagged ${tag}` : `Show only meetings tagged ${tag}`;
    chip.innerHTML = `<span></span><span class="tag-count"></span>`;
    (chip.firstChild as HTMLElement).textContent = tag;
    (chip.lastChild as HTMLElement).textContent = String(count);
    chip.addEventListener("click", () => {
      if (on) {
        for (const t of [...state.tagFilter]) if (t.toLocaleLowerCase() === tag.toLocaleLowerCase()) state.tagFilter.delete(t);
      } else state.tagFilter.add(tag);
      renderMeetingList();
    });
    bar.appendChild(chip);
  }
}

function renderBulkBar() {
  const known = new Set(state.meetings.map((m) => m.id));
  for (const id of [...state.selected]) if (!known.has(id)) state.selected.delete(id);
  const n = state.selected.size;
  $("bulk-bar").classList.toggle("hidden", n === 0);
  $("bulk-count").textContent = `${n} selected`;
  $("select-btn").classList.toggle("on", state.selectMode);
  $("select-btn").setAttribute("aria-pressed", String(state.selectMode));
  $("sort-label").textContent = SORTS.find((s) => s.key === currentSort())!.label;
}

/// Plain click opens; Cmd/Ctrl-click (or any click in select mode) toggles;
/// Shift-click selects the range from the last toggled meeting.
function onItemClick(e: MouseEvent, id: string, order: string[]) {
  const toggle = e.metaKey || e.ctrlKey || state.selectMode;
  if (e.shiftKey && state.selectAnchor && order.includes(state.selectAnchor)) {
    const [a, b] = [order.indexOf(state.selectAnchor), order.indexOf(id)].sort((x, y) => x - y);
    for (const rid of order.slice(a, b + 1)) state.selected.add(rid);
    renderMeetingList();
    return;
  }
  if (toggle) {
    if (state.selected.has(id)) state.selected.delete(id);
    else state.selected.add(id);
    state.selectAnchor = id;
    renderMeetingList();
    return;
  }
  state.selectAnchor = id;
  selectMeeting(id);
}

export function renderMeetingList() {
  renderTagFilter();
  renderBulkBar();
  const list = $("meeting-list");
  list.innerHTML = "";
  list.classList.toggle("selecting", state.selectMode || state.selected.size > 0);
  const shown = visibleMeetings();
  if (!shown.length) {
    const filtered = state.meetings.length && (state.searchQuery.trim() || state.tagFilter.size);
    list.innerHTML = `<li class="list-empty muted">${filtered ? "No matches" : "No meetings yet"}</li>`;
    return;
  }
  const order = shown.map((m) => m.id);
  const sort = currentSort();
  const grouped = sort === "newest" || sort === "oldest";
  let lastGroup = "";
  for (const m of shown) {
    if (grouped) {
      const group = dateGroup(m.created_at);
      if (group !== lastGroup) {
        const head = el("li", "list-group");
        head.textContent = group;
        list.appendChild(head);
        lastGroup = group;
      }
    }
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    if (m.id === state.activeId) btn.classList.add("active");
    const selected = state.selected.has(m.id);
    if (selected) btn.classList.add("selected");
    btn.setAttribute("aria-pressed", String(selected));
    const [cls, label] = badgeFor(m.status);
    const tags = (m.tags || []).slice(0, 2).map((t) => `<span class="tag-pill">${escapeHtml(t)}</span>`).join("");
    const more = (m.tags || []).length > 2 ? `<span class="tag-pill more">+${m.tags.length - 2}</span>` : "";
    btn.innerHTML = `<span class="m-check" aria-hidden="true"></span>
      <span class="m-title">${escapeHtml(m.title || m.filename)}</span>
      <span class="m-meta"><span class="badge ${cls}">${label}</span><span>${fmtRelative(m.created_at)}</span>${tags}${more}</span>`;
    btn.addEventListener("click", (e) => onItemClick(e, m.id, order));
    li.appendChild(btn);
    list.appendChild(li);
  }
}
