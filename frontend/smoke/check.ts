// Assertions for the headless smoke test (see scripts/smoke_frontend.sh).
// `main.ts` runs this right after boot; the result is one <pre id="smoke-result">
// that the shell greps for.
import { closePrepare, openPrepare, openPrepareMany } from "../ts/prepare";
import { onBenchProgress, openBench } from "../ts/bench";
import { openSettings, openSpeakers } from "../ts/settings";
import { renderResult, selectMeeting } from "../ts/meeting";
import { DOCS_MEETING_ID, docsMeetings } from "./docs";
import { currentSort, dateGroup, renderMeetingList } from "../ts/meetings";
import { prefs } from "../ts/helpers";
import { state } from "../ts/state";
import { LIMITS } from "./fixtures";
import { calls } from "./tauri-api";

export async function runSmokeChecks(): Promise<void> {
  const problems: string[] = [];
  const finish = () => {
    if (document.getElementById("smoke-result")) return;
    const result = document.createElement("pre");
    result.id = "smoke-result";
    result.textContent = problems.length ? "SMOKE_FAIL: " + problems.join("; ") : "SMOKE_OK";
    document.body.appendChild(result);
  };

  window.addEventListener("error", (e) => problems.push("uncaught error: " + e.message));
  window.addEventListener("unhandledrejection", (e) => {
    problems.push("unhandled rejection: " + ((e.reason && e.reason.message) || e.reason));
  });

  // Boot is async (event listeners + the setup status round-trip). Wait for it
  // to settle before judging the DOM: #app visible means it finished, #fatal
  // means it failed. The budget is virtual time under --virtual-time-budget.
  const hidden = (id: string) => document.getElementById(id)!.classList.contains("hidden");
  const settled = () => !hidden("app") || !hidden("fatal");
  for (let i = 0; i < 100 && !settled(); i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  if (!hidden("boot")) problems.push("boot screen still visible");
  if (!hidden("setup")) problems.push("setup screen visible despite ready models");
  if (!hidden("fatal")) {
    problems.push("fatal screen: " + document.getElementById("fatal-msg")!.textContent!.trim());
  }
  if (hidden("app")) problems.push("main screen not visible");
  if (
    location.hash !== "#meeting" &&
    !document.getElementById("meeting-list")!.textContent!.includes("No meetings yet")
  ) {
    problems.push("meeting-list empty state missing");
  }
  if (
    location.hash !== "#speakers" &&
    !document.getElementById("speaker-list")!.textContent!.includes("No saved voices")
  ) {
    problems.push("speaker-list empty state missing");
  }

  try {
    await openPrepare("/tmp/smoke.wav");
    if (hidden("prepare")) problems.push("prepare screen did not open");
    if ((document.getElementById("prep-transcribe") as HTMLButtonElement).disabled) {
      problems.push("transcribe button stayed disabled after a successful probe");
    }
    if (!document.querySelector("#prep-range-list .range-row")) {
      problems.push("preselected range row missing");
    }
  } catch (e: any) {
    problems.push("openPrepare threw: " + e.message);
  }

  // README screenshots: `#meeting` (a finished meeting) and `#docs-prepare`
  // (a 28-minute recording with a break cut out).
  if (location.hash === "#meeting") {
    closePrepare();
    selectMeeting(DOCS_MEETING_ID);
    await new Promise((resolve) => setTimeout(resolve, 50));
    finish();
    return;
  }
  if (location.hash === "#docs-prepare") {
    closePrepare();
    await openPrepare("/Users/you/Recordings/website-relaunch-sync.m4a");
    const setTime = (input: HTMLInputElement | undefined, value: string) => {
      if (!input) return problems.push("prep time input missing");
      input.value = value;
      input.dispatchEvent(new Event("blur"));
    };
    setTime(document.querySelectorAll<HTMLInputElement>("#prep-range-list .time-input")[1], "0:21:52.0");
    document.getElementById("prep-add-range")!.click();
    setTime(document.querySelectorAll<HTMLInputElement>("#prep-range-list .time-input")[2], "0:23:05.0");
    setTime(document.querySelectorAll<HTMLInputElement>("#prep-range-list .time-input")[3], "0:28:32.4");
    await new Promise((resolve) => setTimeout(resolve, 300));
    finish();
    return;
  }
  // `#prepare`: leave the prepare screen open for screenshots.
  if (location.hash === "#prepare") {
    finish();
    return;
  }
  // `#speakers`: the Known speakers page (the stub returns saved voices).
  if (location.hash === "#speakers") {
    closePrepare();
    openSpeakers();
    finish();
    return;
  }
  // `#docs`: the clean Settings page for the README screenshot.
  if (location.hash === "#docs") {
    closePrepare();
    await openSettings();
    finish();
    return;
  }
  try {
    closePrepare();
    await checkSettings(problems, hidden);
  } catch (e: any) {
    problems.push("settings check threw: " + e.message);
  }
  try {
    await checkNotes(problems);
  } catch (e: any) {
    problems.push("notes check threw: " + e.message);
  }
  try {
    await checkBatchPrepare(problems, hidden);
  } catch (e: any) {
    problems.push("batch prepare check threw: " + e.message);
  }
  try {
    checkSidebar(problems, hidden);
  } catch (e: any) {
    problems.push("sidebar check threw: " + e.message);
  }
  if (location.hash === "#bench") {
    try {
      checkBench(problems, hidden);
    } catch (e: any) {
      problems.push("bench check threw: " + e.message);
    }
  }
  finish();
}

async function checkSettings(problems: string[], hidden: (id: string) => boolean) {
  await openSettings();
  if (hidden("settings")) problems.push("settings did not open");
  const q = <T extends Element = HTMLElement>(sel: string) => document.querySelector<T>(sel);
  const row = (name: string) => q(`[data-row="${name}"]`);
  // Dead compute selects are gone; the summary one stays (two options).
  for (const gone of ["ASR_PROVIDER", "DIAR_PROVIDER", "SPEAKER_PROVIDER"]) {
    if (row(gone)) problems.push(`${gone} row still rendered`);
  }
  if (!row("SUMMARY_PROVIDER")) problems.push("summary compute row missing");
  // Threads: "Auto" is the default and comes first, the counts follow.
  const threads = q<HTMLSelectElement>("#cfg-NUM_THREADS");
  if (!threads) problems.push("CPU threads select missing");
  else {
    if (threads.options[0]?.value !== "0") problems.push("CPU threads: Auto is not first");
    if (!/Auto — 7 of 10 threads/.test(threads.options[0].textContent || ""))
      problems.push(`CPU threads: Auto label is "${threads.options[0].textContent}"`);
    // Auto + one option per explicit thread count.
    const expected = LIMITS.NUM_THREADS.max + 1;
    if (threads.options.length !== expected)
      problems.push(`CPU threads has ${threads.options.length} options, expected ${expected}`);
  }
  // Classic-model sliders only with pyannote.
  if (!row("DIAR_THRESHOLD")?.hidden) problems.push("pyannote slider visible with Nemotron");
  q<HTMLElement>('[data-model-id="pyannote-3.0"]')!.click();
  if (row("DIAR_THRESHOLD")?.hidden) problems.push("pyannote slider hidden with pyannote");
  if (q("#settings-savebar")!.hidden) problems.push("save bar not shown after a change");
  // Cohere: no auto-detect, only its languages.
  q<HTMLElement>('[data-model-id="cohere-transcribe"]')!.click();
  const lang = q<HTMLSelectElement>("#cfg-ASR_LANGUAGE")!;
  const codes = [...lang.options].map((o) => o.value);
  if (codes.includes("auto") || codes.includes("ru")) problems.push("Cohere offers auto/ru");
  if (!codes.includes("vi")) problems.push("Cohere misses vi");
  // Labels are tied to their controls.
  if (!q('label[for="cfg-SPEAKER_THRESHOLD"]')) problems.push("slider label not associated");
  // Cards are a radio group with one checked card.
  const checked = document.querySelectorAll('#config-transcription [role="radio"][aria-checked="true"]');
  if (checked.length !== 1) problems.push(`ASR radio group has ${checked.length} checked`);
  // Disk list: removable models get a button, the default does not.
  const disk = q("#model-disk-list")!;
  if (!disk.querySelector('[aria-label="Remove Whisper Base"]')) problems.push("no remove button");
  if (disk.querySelector('[aria-label="Remove Parakeet TDT 0.6B v3"]')) problems.push("default removable");
  // Watch folder: off by default, set with the folder picker.
  const watchRow = q('[data-row="WATCH_DIR"]');
  if (!watchRow) problems.push("watch folder row missing");
  else if (location.hash !== "#settings") {
    if (watchRow.querySelector(".folder-path")!.textContent !== "Off") problems.push("watch folder not off");
    [...watchRow.querySelectorAll<HTMLButtonElement>("button")].find((b) => b.textContent === "Choose…")!.click();
    await new Promise((r) => setTimeout(r, 20));
    if (watchRow.querySelector(".folder-path")!.textContent !== "/Users/you/Movies/OBS") problems.push("folder not shown");
    if (q("#settings-savebar")!.hidden) problems.push("picking a folder did not mark settings dirty");
  }
  // `#settings` keeps the page as-is for screenshots.
  if (location.hash !== "#settings") {
    q<HTMLElement>("#config-discard")!.click();
    if (!q("#settings-savebar")!.hidden) problems.push("discard left the save bar");
  }
}

// A finished run as the backend emits it (M3 Pro numbers).
export const BENCH_DONE = {
  status: "done",
  threads: 7,
  cores: 12,
  source: "synthetic audio",
  models: { asr: "Parakeet TDT 0.6B v3", diar: "Nemotron 3 Diarization", summary: "Qwen 3.5 4B" },
  rows: [
    { stage: "Transcription + speaker separation", backend: "CPU", model: "Parakeet TDT 0.6B v3 + Nemotron 3 Diarization", seconds: 22.5, speed: "13.4x realtime" },
    { stage: "Voice matching", backend: "CPU", model: "TitaNet", seconds: 1.3, speed: "2 speakers" },
    { stage: "Summary", backend: "GPU", provider: "gpu", model: "Qwen 3.5 4B", seconds: 12.6, speed: "reads 499 tokens/s, writes 30.6 tokens/s" },
    { stage: "Summary", backend: "CPU", provider: "cpu", model: "Qwen 3.5 4B", seconds: 119.0, speed: "reads 87 tokens/s, writes 2.5 tokens/s" },
  ],
  estimate: { per_hour_secs: 316, transcription_secs: 270, voice_secs: 1, summary_secs: 45 },
  recommendation: { SUMMARY_PROVIDER: "gpu" },
  current: { SUMMARY_PROVIDER: "cpu" },
};

function checkBench(problems: string[], hidden: (id: string) => boolean) {
  document.getElementById("settings")!.classList.add("hidden");
  document.getElementById("app")!.classList.remove("hidden");
  openBench("app");
  onBenchProgress(BENCH_DONE);
  if (hidden("bench")) problems.push("bench did not open");
  const head = document.getElementById("bench-estimate-head")!.textContent || "";
  if (!head.includes("5 min")) problems.push(`estimate headline: ${head}`);
  if (document.querySelectorAll("#bench-results tbody tr").length !== 4) problems.push("bench rows");
  if (!document.querySelector("#bench-results tr.best")) problems.push("fastest summary not marked");
  if (hidden("bench-apply")) problems.push("apply hidden although settings differ");
}

// A finished meeting in the {short, detail} notes shape the backend hands out.
const MEETING = {
  id: "smoke-meeting",
  title: "Weekly sync",
  filename: "sync.mp4",
  status: "done",
  created_at: "2026-11-05T10:00:00Z",
  summary: "Short summary.",
  summary_long: "First paragraph.\n\nSecond paragraph.",
  takeaways: [{ short: "Diarization works.", detail: "Runs end to end." }],
  action_items: [{ short: "Anna: ship it", detail: "" }],
  timings: { extract: 1, transcribe: 124, diarize: 20, identify: 11, summarize: 22, total: 178 },
  segments: [
    { start: 0, end: 2, speaker: "spk_0", text: "Hello." },
    { start: 2, end: 8, speaker: "spk_1", text: "Hi Anna, all good here." },
  ],
  speakers: { spk_0: "Anna", spk_1: "Ben" },
  recognized: [],
  confirmed: [],
  has_audio: false,
  versions: [],
};

async function checkNotes(problems: string[]) {
  state.activeId = MEETING.id;
  state.currentMeeting = MEETING;
  renderResult(MEETING);
  if (document.querySelectorAll("#summary-long-text p").length !== 2) problems.push("long summary paragraphs");
  const timings = document.getElementById("timings-note")!.textContent || "";
  if (!timings.startsWith("Took 2m 58s — transcription 2m 04s")) problems.push(`timings line: ${timings}`);
  const talk = [...document.querySelectorAll("#speaker-legend .talk-time")].map((n) => n.textContent);
  if (talk.join(" | ") !== "2.0s · 25 % | 6.0s · 75 %") problems.push(`talk time: ${talk.join(" | ")}`);
  if (document.querySelector("#takeaways .item-detail")?.textContent !== "Runs end to end.") {
    problems.push("takeaway detail missing");
  }
  // Editing an item opens both fields; cancelling restores the list.
  document.querySelector<HTMLElement>('#takeaways [aria-label="Edit"]')!.click();
  const editor = document.querySelector("#takeaways .item-editor");
  if (!editor) problems.push("item editor did not open");
  else if ((editor.querySelector("textarea") as HTMLTextAreaElement).value !== "Runs end to end.") {
    problems.push("item editor detail not prefilled");
  }
  [...document.querySelectorAll<HTMLButtonElement>("#takeaways .editor-actions button")]
    .find((b) => b.textContent === "Cancel")
    ?.click();
  if (state.editing) problems.push(`still editing after cancel: ${state.editing}`);
  if (document.querySelector("#takeaways .item-editor")) problems.push("item editor left behind");
  // The summary editor replaces the long block rather than showing it twice.
  document.getElementById("summary-edit-btn")!.click();
  if (!document.getElementById("summary-long-block")!.hidden) problems.push("long summary shown while editing");
  if (document.querySelectorAll("#summary-block textarea").length < 2) {
    problems.push("summary editor missing the long-form box");
  }
  [...document.querySelectorAll<HTMLButtonElement>(".editor-actions button")]
    .find((b) => b.textContent === "Cancel")
    ?.click();
  if (document.getElementById("summary-long-block")!.hidden) problems.push("long summary not restored");
  // The segment editor splits at the cursor (in characters) and merges.
  document.querySelector<HTMLElement>('#transcript [aria-label="Edit segment"]')!.click();
  const segBox = document.querySelector<HTMLTextAreaElement>("#transcript .editor textarea");
  if (!segBox) problems.push("segment editor did not open");
  else {
    const tools = [...document.querySelectorAll<HTMLButtonElement>("#transcript .seg-tools button")];
    if (tools.map((b) => b.textContent).join(",") !== "Split at cursor,Merge with next") {
      problems.push(`segment tools: ${tools.map((b) => b.textContent).join(",")}`);
    }
    segBox.setSelectionRange(3, 3);
    tools[0].click();
    await new Promise((r) => setTimeout(r, 10));
    const split = calls.filter((c) => c.cmd === "split_segment").pop();
    if (split?.args?.at !== 3 || split?.args?.index !== 0) problems.push(`split call: ${JSON.stringify(split?.args)}`);
    state.editing = null;
    renderResult(MEETING);
  }
  // Regenerate notes opens a panel with the template and language.
  document.getElementById("regen-notes-btn")!.click();
  await new Promise((r) => setTimeout(r, 30));
  const panel = document.querySelector(".regen-panel");
  if (!panel) problems.push("regenerate panel did not open");
  else {
    const [tpl] = panel.querySelectorAll("select");
    (tpl as HTMLSelectElement).value = "standup";
    [...panel.querySelectorAll<HTMLButtonElement>("button")].find((b) => b.textContent === "Regenerate")!.click();
    await new Promise((r) => setTimeout(r, 10));
    const regen = calls.filter((c) => c.cmd === "reprocess_meeting").pop();
    if (regen?.args?.notes_template !== "standup" || regen?.args?.mode !== "notes") {
      problems.push(`regenerate call: ${JSON.stringify(regen?.args)}`);
    }
  }
  // Export opens a format menu; the pick goes to the backend and is remembered.
  document.getElementById("export-btn")!.click();
  const formats = [...document.querySelectorAll<HTMLButtonElement>(".menu .menu-item")];
  if (formats.length !== 7) problems.push(`export menu items: ${formats.length}`);
  formats.find((b) => b.textContent!.startsWith("PDF"))?.click();
  if (document.querySelector(".menu")) problems.push("export menu stayed open");
  const exported = calls.filter((c) => c.cmd === "export_meeting").pop();
  if (exported?.args?.format !== "pdf") problems.push(`export call: ${JSON.stringify(exported)}`);
  document.getElementById("export-btn")!.click();
  if (!document.querySelector(".menu .menu-item.active")?.textContent!.startsWith("PDF")) {
    problems.push("last export format not remembered");
  }
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
  document.querySelector<HTMLElement>(".menu .menu-item")?.dispatchEvent(
    new KeyboardEvent("keydown", { key: "Escape", bubbles: true })
  );
  if (document.querySelector(".menu")) problems.push("Escape did not close the menu");
  state.activeId = null;
  state.currentMeeting = null;
}

function checkSidebar(problems: string[], hidden: (id: string) => boolean) {
  const titles = () => [...document.querySelectorAll("#meeting-list .m-title")].map((n) => n.textContent);
  const groups = () => [...document.querySelectorAll("#meeting-list .list-group")].map((n) => n.textContent);
  state.activeId = null; // the batch check opened the meeting it created
  state.meetings = docsMeetings();
  prefs.set("sort", "newest");
  renderMeetingList();
  const expected = [...new Set(state.meetings.map((m) => dateGroup(m.created_at)))].join("|");
  if (groups().join("|") !== expected || groups().length < 3) {
    problems.push(`date groups: ${groups().join("|")}`);
  }
  const chips = [...document.querySelectorAll<HTMLButtonElement>("#tag-filter .tag-chip")];
  if (chips.map((c) => c.textContent).join(",") !== "Weekly2,Nordwind1,Relaunch1") {
    problems.push(`tag chips: ${chips.map((c) => c.textContent).join(",")}`);
  }
  chips[0].click(); // Weekly
  if (titles().length !== 2) problems.push(`tag filter shows ${titles().length}`);
  document.querySelector<HTMLButtonElement>("#tag-filter .tag-chip.on")!.click();
  if (titles().length !== 4) problems.push("tag filter did not clear");

  // Sorting by length drops the date headers.
  prefs.set("sort", "length");
  renderMeetingList();
  if (groups().length || titles()[0] !== "Q3 roadmap planning") problems.push(`length sort: ${titles()[0]}`);
  if (document.getElementById("sort-label")!.textContent !== "Longest first") problems.push("sort label");
  prefs.set("sort", "newest");
  renderMeetingList();
  if (currentSort() !== "newest") problems.push("sort not restored");

  // Cmd-click selects without opening; Shift-click extends the range.
  const items = () => [...document.querySelectorAll<HTMLButtonElement>("#meeting-list button")];
  items()[0].dispatchEvent(new MouseEvent("click", { bubbles: true, metaKey: true }));
  items()[2].dispatchEvent(new MouseEvent("click", { bubbles: true, shiftKey: true }));
  if (state.selected.size !== 3) problems.push(`selected after shift-click: ${state.selected.size}`);
  if (hidden("bulk-bar")) problems.push("bulk bar hidden with a selection");
  if (document.getElementById("bulk-count")!.textContent !== "3 selected") problems.push("bulk count");
  if (state.activeId) problems.push("a cmd-click opened a meeting");
  document.getElementById("bulk-tag")!.click();
  const tagItems = [...document.querySelectorAll(".menu .menu-item")].map((n) => n.textContent);
  if (!tagItems.includes("New tag…")) problems.push(`bulk tag menu: ${tagItems.join(",")}`);
  document.querySelector<HTMLElement>(".menu .menu-item")?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  document.getElementById("bulk-clear")!.click();
  if (state.selected.size || !hidden("bulk-bar")) problems.push("selection not cleared");

  state.meetings = [];
  renderMeetingList();
}

async function checkBatchPrepare(problems: string[], hidden: (id: string) => boolean) {
  const created = () => calls.filter((c) => c.cmd === "create_meeting");
  const before = created().length;
  const settle = () => new Promise((r) => setTimeout(r, 50));
  await openPrepareMany(["/tmp/a.mp4", "/tmp/b.m4a", "/tmp/c.mkv"]);
  await settle();
  const file = () => document.getElementById("prep-file")!.textContent;
  if (file() !== "a.mp4 · file 1 of 3") problems.push(`batch header: ${file()}`);
  if (hidden("prep-skip") || hidden("prep-rest")) problems.push("batch buttons hidden");
  document.getElementById("prep-transcribe")!.click();
  await settle();
  if (created().length !== before + 1) problems.push("first file not queued");
  if (file() !== "b.m4a · file 2 of 3") problems.push(`after transcribe: ${file()}`);
  document.getElementById("prep-skip")!.click();
  await settle();
  if (file() !== "c.mkv · file 3 of 3") problems.push(`after skip: ${file()}`);
  if (!hidden("prep-rest")) problems.push("'Transcribe all' shown on the last file");
  document.getElementById("prep-transcribe")!.click();
  await settle();
  const first = created()[before]?.args || {};
  if (first.notes_template !== "general" || first.notes_language !== "auto") {
    problems.push(`notes options not sent: ${JSON.stringify(first)}`);
  }
  const paths = created().slice(before).map((c) => c.args!.path);
  if (paths.join(",") !== "/tmp/a.mp4,/tmp/c.mkv") problems.push(`queued: ${paths.join(",")}`);
  if (!hidden("prepare")) problems.push("prepare still open after the batch");

  // "Transcribe all" queues the rest whole.
  await openPrepareMany(["/tmp/d.mp4", "/tmp/e.mp4", "/tmp/f.mp4"]);
  await settle();
  document.getElementById("prep-rest")!.click();
  await settle();
  await settle();
  const rest = created().slice(before + 2);
  if (rest.map((c) => c.args!.path).join(",") !== "/tmp/d.mp4,/tmp/e.mp4,/tmp/f.mp4") {
    problems.push(`transcribe all queued: ${rest.map((c) => c.args!.path).join(",")}`);
  }
  if (rest.slice(1).some((c) => c.args!.ranges !== null)) problems.push("waiting files not queued whole");
  if (!hidden("prepare")) problems.push("prepare still open after transcribe all");
}
