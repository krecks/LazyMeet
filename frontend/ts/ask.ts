// --------------------------------------------------------------------------
// Ask — questions about one meeting (under its notes) or all of them (from
// the search results), answered on this machine by the summary model.
// Answers stream in as `ask-progress` events; [mm:ss] / [M2 mm:ss]
// citations become links to that moment.
// --------------------------------------------------------------------------
import { $, call } from "./core";
import { el, escapeHtml, showToast } from "./helpers";
import { jumpToTime, openMeetingAt } from "./search";
import { state } from "./state";

/// Live answers by request id: where their text goes as it streams.
const live = new Map<string, (delta: string) => void>();

/// `ask-progress` from the backend (wired in boot).
export function onAskProgress(p) {
  if (p?.delta) live.get(p.request_id)?.(p.delta);
}

const newRequestId = () =>
  globalThis.crypto?.randomUUID?.() || `ask-${Date.now()}-${Math.random().toString(36).slice(2)}`;

/// [12:34], [1:02:03], [M2 12:34], and ranges like [23:28-24:00] (linked to
/// their start).
const CITE = /\[(?:(M\d+)[\s,]+)?(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*[-–]\s*\d{1,2}:\d{2}(?::\d{2})?)?\]/g;

function seconds(ts: string): number {
  return ts.split(":").reduce((acc, part) => acc * 60 + Number(part), 0);
}

/// The answer as HTML: escaped text, paragraphs, citations as links, and —
/// in an all-meetings answer — bare "M2" mentions as the meeting's title.
/// `sources` maps M1… to meetings.
export function answerHtml(text: string, sources: Record<string, { id: string; title: string }> = {}) {
  const pattern = new RegExp(`${CITE.source}|\\b(M\\d+)\\b`, "g");
  return escapeHtml(text.trim())
    .split(/\n{2,}/)
    .map((para) =>
      `<p>${para
        .replace(pattern, (whole, label, ts, bare) => {
          if (bare) {
            const src = sources[bare];
            return src ? `<span class="cite-title">${escapeHtml(src.title)}</span>` : whole;
          }
          const src = label ? sources[label] : null;
          if (label && !src) return whole;
          const title = src ? `${escapeHtml(src.title)}, ${ts}` : `Jump to ${ts}`;
          return `<button type="button" class="cite" data-t="${seconds(ts)}"${
            src ? ` data-meeting="${src.id}"` : ""
          } title="${title}">${src ? `${escapeHtml(label)} · ` : ""}${ts}</button>`;
        })
        .replace(/\n/g, "<br>")}</p>`
    )
    .join("");
}

function wireCitations(node: HTMLElement) {
  node.addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>("button.cite");
    if (!btn) return;
    const t = Number(btn.dataset.t);
    if (btn.dataset.meeting) openMeetingAt(btn.dataset.meeting, t);
    else if (state.currentMeeting) jumpToTime(state.currentMeeting, t);
  });
}

function qaItem(q: string, answer: string, sources = {}): { item: HTMLElement; body: HTMLElement } {
  const item = el("div", "qa-item");
  const question = el("p", "qa-q");
  question.textContent = q;
  const body = el("div", "qa-a serif");
  body.innerHTML = answerHtml(answer, sources);
  item.append(question, body);
  return { item, body };
}

// ---- one meeting ----------------------------------------------------------

export function renderAsk(m, readOnly: boolean) {
  const block = $("ask-block");
  const done = m.status === "done" && (m.segments || []).length > 0;
  block.classList.toggle("hidden", !done);
  if (!done) return;
  const list = $("qa-list");
  list.innerHTML = "";
  for (const entry of m.qa || []) list.appendChild(qaItem(entry.q, entry.a).item);
  ($("ask-input") as HTMLInputElement).disabled = readOnly;
  ($("ask-btn") as HTMLButtonElement).disabled = readOnly;
}

let running: string | null = null;

async function askMeeting() {
  const input = $("ask-input") as HTMLInputElement;
  const question = input.value.trim();
  const m = state.currentMeeting;
  if (!question || !m || running) return;
  const requestId = newRequestId();
  running = requestId;
  const { item, body } = qaItem(question, "");
  item.classList.add("pending");
  body.innerHTML = '<p class="muted">Reading the meeting…</p>';
  $("qa-list").appendChild(item);
  item.scrollIntoView({ block: "nearest" });
  let text = "";
  live.set(requestId, (delta) => {
    text += delta;
    body.innerHTML = answerHtml(text);
  });
  setAskBusy(true);
  try {
    const entry = await call("ask_meeting", { meeting_id: m.id, question, request_id: requestId });
    body.innerHTML = answerHtml(entry.a);
    if (state.currentMeeting?.id === m.id) {
      state.currentMeeting.qa = [...(state.currentMeeting.qa || []), entry];
    }
    input.value = "";
  } catch (e) {
    item.remove();
    showToast(e.status === 409 ? e.message : `Couldn't answer: ${e.message}`);
  } finally {
    item.classList.remove("pending");
    live.delete(requestId);
    running = null;
    setAskBusy(false);
  }
}

function setAskBusy(busy: boolean) {
  $("ask-btn").classList.toggle("hidden", busy);
  $("ask-stop").classList.toggle("hidden", !busy);
  ($("ask-input") as HTMLInputElement).disabled = busy;
}

// ---- all meetings -----------------------------------------------------------

async function askAll() {
  const question = ($("meeting-search") as HTMLInputElement).value.trim();
  if (!question || running) return;
  const requestId = newRequestId();
  running = requestId;
  const card = $("search-answer");
  card.classList.remove("hidden");
  card.innerHTML = "";
  const { item, body } = qaItem(question, "");
  item.classList.add("pending");
  body.innerHTML = '<p class="muted">Reading your meetings…</p>';
  const stop = el("button", "btn ghost small");
  stop.type = "button";
  stop.textContent = "Stop";
  stop.addEventListener("click", () => call("cancel_ask", { request_id: requestId }));
  card.append(item, stop);
  let text = "";
  live.set(requestId, (delta) => {
    text += delta;
    body.innerHTML = answerHtml(text);
  });
  ($("search-ask") as HTMLButtonElement).disabled = true;
  try {
    const out = await call("ask_all", { question, request_id: requestId });
    const sources = Object.fromEntries((out.sources || []).map((s) => [s.label, s]));
    body.innerHTML = answerHtml(out.a, sources);
  } catch (e) {
    body.innerHTML = `<p class="muted">${escapeHtml(e.status === 409 ? e.message : `Couldn't answer: ${e.message}`)}</p>`;
  } finally {
    item.classList.remove("pending");
    stop.remove();
    live.delete(requestId);
    running = null;
    ($("search-ask") as HTMLButtonElement).disabled = false;
  }
}

export function setupAsk() {
  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    askMeeting();
  });
  $("ask-stop").addEventListener("click", () => running && call("cancel_ask", { request_id: running }));
  wireCitations($("qa-list"));
  wireCitations($("search-answer"));
  $("search-ask").addEventListener("click", askAll);
}
