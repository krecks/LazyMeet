// Assertions for the headless smoke test (see scripts/smoke_frontend.sh).
// `main.ts` runs this right after boot; the result is one <pre id="smoke-result">
// that the shell greps for.
import { closePrepare, openPrepare } from "../ts/prepare";
import { onBenchProgress, openBench } from "../ts/bench";
import { openSettings } from "../ts/settings";
import { renderResult } from "../ts/meeting";
import { state } from "../ts/state";
import { LIMITS } from "./fixtures";

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
  if (!document.getElementById("meeting-list")!.textContent!.includes("No meetings yet")) {
    problems.push("meeting-list empty state missing");
  }
  if (!document.getElementById("speaker-list")!.textContent!.includes("No saved voices")) {
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
    checkNotes(problems);
  } catch (e: any) {
    problems.push("notes check threw: " + e.message);
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
  segments: [{ start: 0, end: 2, speaker: "spk_0", text: "Hello." }],
  speakers: { spk_0: "Anna" },
  recognized: [],
  confirmed: [],
  has_audio: false,
  versions: [],
};

function checkNotes(problems: string[]) {
  state.activeId = MEETING.id;
  state.currentMeeting = MEETING;
  renderResult(MEETING);
  if (document.querySelectorAll("#summary-long-text p").length !== 2) problems.push("long summary paragraphs");
  const timings = document.getElementById("timings-note")!.textContent || "";
  if (!timings.startsWith("Took 2m 58s — transcription 2m 04s")) problems.push(`timings line: ${timings}`);
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
  state.activeId = null;
  state.currentMeeting = null;
}
