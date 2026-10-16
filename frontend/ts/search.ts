// --------------------------------------------------------------------------
// Global search. The sidebar box searches every meeting on disk, not just the
// cached titles: the backend scans the meeting files (`search_meetings`) and
// returns hits grouped per meeting, which render in the main pane. Matching is
// literal and case-insensitive — "Paket" finds "Arbeitspaket" — using the same
// matcher as the transcript find bar, so highlights agree.
// --------------------------------------------------------------------------
import { $, call } from "./core";
import { el, escapeHtml, fmtRelative, fmtTs, plural, showToast } from "./helpers";
import { selectMeeting, showActiveMeeting } from "./meeting";
import { badgeFor, renderMeetingList } from "./meetings";
import { state } from "./state";
import { find, gotoMatch, literalRegex, markHits, openFind } from "./transcript";

export const gsearch = {
  query: "", // the query the rendered results belong to
  results: [],
  token: 0, // stale-response guard, same idea as activeId in showActiveMeeting
  timer: null,
  collapsed: new Set(), // meeting ids the user folded away
};

// One letter would scan the whole library to match nearly every meeting.
const MIN_QUERY = 2;
const SNIPPET_PAD = 60; // characters of context kept either side of the hit

const HIT_KINDS = {
  title: "Title",
  speaker: "Speaker",
  summary: "Summary",
  takeaway: "Key takeaway",
  action_item: "Action item",
};

/// Debounced entry point from the sidebar input. The sidebar's own title
/// filter stays instant; only the disk scan waits for a pause in typing.
export function queueGlobalSearch(query) {
  clearTimeout(gsearch.timer);
  if (query.trim().length < MIN_QUERY) {
    // Backspacing out of a search puts the previous view back.
    if (searchOpen() || gsearch.query) clearGlobalSearch();
    return;
  }
  gsearch.timer = setTimeout(() => runGlobalSearch(query), 200);
}

async function runGlobalSearch(query) {
  const token = ++gsearch.token;
  let results;
  try {
    results = await call("search_meetings", { query });
  } catch (e) {
    showToast(`Couldn't search: ${e.message}`);
    return;
  }
  if (token !== gsearch.token) return; // a newer query already ran
  gsearch.query = query;
  gsearch.results = results;
  gsearch.collapsed.clear();
  renderMeetingList(); // content matches join the sidebar's title matches
  renderSearchResults();
}

/// Take the results over the main pane. The meeting views stay in the DOM so
/// clearing the search can restore whatever was open.
function renderSearchResults() {
  $("empty").classList.add("hidden");
  $("processing").classList.add("hidden");
  $("result").classList.add("hidden");
  $("search-results").classList.remove("hidden");

  const re = literalRegex(gsearch.query, false);
  const total = gsearch.results.reduce((n, g) => n + g.count, 0);
  $("search-summary").textContent = total
    ? `“${gsearch.query}” — ${plural(total, "match", "matches")} in ${plural(
        gsearch.results.length,
        "meeting",
        "meetings"
      )}`
    : `No matches for “${gsearch.query}”`;

  const box = $("search-groups");
  box.innerHTML = "";
  for (const group of gsearch.results) {
    box.appendChild(renderSearchGroup(group, re));
  }
}

function renderSearchGroup(group, re) {
  const wrap = el("section", "search-group");
  const collapsed = gsearch.collapsed.has(group.id);
  const [cls, label] = badgeFor(group.status);

  const head = document.createElement("button");
  head.type = "button";
  head.className = "search-group-head";
  head.setAttribute("aria-expanded", String(!collapsed));
  head.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#i-chev-${
    collapsed ? "down" : "up"
  }" /></svg>
    <span class="g-title">${escapeHtml(group.title || group.filename || "Meeting")}</span>
    <span class="badge ${cls}">${label}</span>
    <span class="g-count">${plural(group.count, "match", "matches")} · ${escapeHtml(
      fmtRelative(group.created_at)
    )}</span>`;
  head.addEventListener("click", () => {
    if (collapsed) gsearch.collapsed.delete(group.id);
    else gsearch.collapsed.add(group.id);
    renderSearchResults();
  });
  wrap.appendChild(head);
  if (collapsed) return wrap;

  const hits = el("div", "search-hits");
  for (const hit of group.hits) {
    hits.appendChild(renderSearchHit(group, hit, re));
  }
  if (group.truncated) {
    const more = el("p", "search-more muted");
    more.textContent = `Showing the first ${group.hits.length} places — open the meeting to see the rest.`;
    hits.appendChild(more);
  }
  wrap.appendChild(hits);
  return wrap;
}

function renderSearchHit(group, hit, re) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "search-hit";
  const meta =
    hit.kind === "segment"
      ? `<span class="who">${escapeHtml(hit.speaker_name || hit.speaker)}</span>
         <span class="ts">${fmtTs(hit.start)}</span>`
      : `<span class="hit-kind">${escapeHtml(HIT_KINDS[hit.kind] || hit.kind)}</span>`;
  btn.innerHTML = `<span class="meta">${meta}</span>
    <span class="snippet">${snippetHtml(hit.text, re)}</span>`;
  btn.addEventListener("click", () => openHit(group, hit));
  return btn;
}

/// A window around the first match, with the query highlighted throughout.
/// The backend sends whole segments; long ones would otherwise bury the hit.
function snippetHtml(text, re) {
  re.lastIndex = 0;
  const first = re.exec(text);
  if (!first) return escapeHtml(text); // matcher disagreement — show it plain
  const from = Math.max(0, first.index - SNIPPET_PAD);
  const to = Math.min(text.length, first.index + first[0].length + SNIPPET_PAD);
  const body = markHits(text.slice(from, to), re);
  return (from ? "…" : "") + body + (to < text.length ? "…" : "");
}

/// Open the meeting the hit came from. A transcript hit also carries the
/// segment to land on, handled once that transcript is on screen.
function openHit(group, hit) {
  pendingScrollSeg =
    hit.kind === "segment" ? { id: group.id, index: hit.index, query: gsearch.query } : null;
  selectMeeting(group.id);
}

export const searchOpen = () => !$("search-results").classList.contains("hidden");

/// Step aside for a meeting view. The rendered groups stay in the DOM so
/// returning to them (Enter in the search box) costs nothing.
export function hideSearchResults() {
  $("search-results").classList.add("hidden");
}

/// Set by openHit, consumed by the next renderTranscript for that meeting.
let pendingScrollSeg = null;

export function consumePendingScroll(m) {
  if (!pendingScrollSeg || pendingScrollSeg.id !== m.id) return;
  const { index, query } = pendingScrollSeg;
  pendingScrollSeg = null;
  // Arm the find bar with the same term and make the clicked occurrence the
  // current match, so next/prev carries on from where the user landed.
  $("find-input").value = query;
  find.current = 0;
  openFind();
  const k = find.matches.findIndex((m) => m.seg === index);
  if (k >= 0) gotoMatch(k); // scrolls the mark itself into view
  const seg = $("transcript").children[index];
  if (!seg) return;
  if (k < 0) seg.scrollIntoView({ block: "center" });
  seg.classList.remove("flash");
  void seg.offsetWidth; // restart the animation if the same segment is re-opened
  seg.classList.add("flash");
}

export function setupGlobalSearch() {
  $("search-clear").addEventListener("click", clearSearchBox);
  $("meeting-search").addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      clearSearchBox();
      return;
    }
    // Enter brings back the results after following a hit into a meeting.
    if (e.key === "Enter") {
      e.preventDefault();
      const q = e.currentTarget.value;
      if (q === gsearch.query && gsearch.results.length) renderSearchResults();
      else queueGlobalSearch(q);
    }
  });
}

function clearSearchBox() {
  $("meeting-search").value = "";
  state.searchQuery = "";
  renderMeetingList();
  clearGlobalSearch();
}

/// Drop the results and put back whichever meeting view was open.
function clearGlobalSearch() {
  clearTimeout(gsearch.timer);
  gsearch.token++; // any in-flight response is now stale
  gsearch.query = "";
  gsearch.results = [];
  gsearch.collapsed.clear();
  hideSearchResults();
  $("search-groups").innerHTML = "";
  renderMeetingList();
  if (state.activeId) showActiveMeeting();
  else $("empty").classList.remove("hidden");
}

