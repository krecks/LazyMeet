// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------

import { $, listen } from "./core";
import { showFatal, showToast } from "./helpers";
import { onMeetingUpdated } from "./meeting";
import { refreshMeetings, renderMeetingList, setupDropzone, setupSidebar } from "./meetings";
import { onAskProgress, setupAsk } from "./ask";
import { setupPrepare } from "./prepare";
import { queueGlobalSearch, setupGlobalSearch } from "./search";
import {
  applyBenchRecommendation,
  closeBench,
  onBenchProgress,
  openBench,
  pickBenchmarkFile,
  startBenchmark,
} from "./bench";
import {
  closeSettings,
  closeSpeakers,
  discardConfig,
  onModelsProgress,
  openSettings,
  openSpeakers,
  refreshSpeakers,
  resetConfig,
  revealDataDir,
  saveConfig,
} from "./settings";
import { applySetupStatus, retrySetup, toggleDownloadPopover, watchSetup } from "./setup";
import { state } from "./state";
import { setupFind } from "./transcript";

function initApp() {
  if (state.appInitialized) return;
  state.appInitialized = true;
  $("setup").classList.add("hidden");
  $("app").classList.remove("hidden");
  setupDropzone();
  setupSidebar();
  setupAsk();
  setupPrepare();
  setupFind();
  setupGlobalSearch();
  $("settings-btn").addEventListener("click", () => openSettings());
  $("settings-back").addEventListener("click", closeSettings);
  $("speakers-btn").addEventListener("click", openSpeakers);
  $("speakers-back").addEventListener("click", closeSpeakers);
  $("bench-btn").addEventListener("click", () => openBench("app"));
  $("bench-back").addEventListener("click", closeBench);
  $("bench-run").addEventListener("click", () => startBenchmark(null));
  $("bench-run-file").addEventListener("click", pickBenchmarkFile);
  $("bench-apply").addEventListener("click", applyBenchRecommendation);
  $("config-save").addEventListener("click", saveConfig);
  $("config-discard").addEventListener("click", discardConfig);
  $("config-reset").addEventListener("click", resetConfig);
  // Esc leaves the full-screen pages, like the prep screen.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || e.defaultPrevented) return;
    if ((e.target as HTMLElement)?.closest?.("input, textarea, select")) return;
    if (!$("settings").classList.contains("hidden")) closeSettings();
    else if (!$("speakers").classList.contains("hidden")) closeSpeakers();
    else if (!$("bench").classList.contains("hidden")) closeBench();
  });
  $("reveal-btn").addEventListener("click", revealDataDir);
  $("meeting-search").addEventListener("input", (e) => {
    state.searchQuery = e.target.value;
    renderMeetingList(); // the sidebar title filter stays instant
    queueGlobalSearch(state.searchQuery); // the disk scan waits for a pause
  });
  refreshMeetings();
  refreshSpeakers();
}

async function main() {
  // Backend events replace the old polling loops; listeners must be armed
  // before the first fetches so no transition slips between them.
  await listen("meeting-updated", (e) => onMeetingUpdated(e.payload));
  await listen("ask-progress", (e) => onAskProgress(e.payload));
  // The watch folder imported a recording (or failed to).
  await listen("meetings-changed", (e: any) => {
    const p = e.payload || {};
    if (p.imported) showToast(`Imported ${p.imported.filename} from the watch folder`, 4000, true);
    if (p.error) showToast(p.error);
    refreshMeetings();
  });
  await listen("models-progress", (e) => onModelsProgress(e.payload));
  await listen("benchmark-progress", (e) => onBenchProgress(e.payload));
  await listen("setup-progress", (e) => {
    applySetupStatus(e.payload);
    if (e.payload.ready && state.setupReadyResolve) {
      state.setupReadyResolve();
      state.setupReadyResolve = null;
    }
  });

  $("boot").classList.add("hidden");
  $("setup-retry").addEventListener("click", retrySetup);
  $("setup-continue").addEventListener("click", initApp);
  $("dl-pill").addEventListener("click", () => toggleDownloadPopover());
  document.addEventListener("click", (e) => {
    const outside = !(e.target instanceof Element) || !e.target.closest(".dl-wrap");
    if (outside) toggleDownloadPopover(false);
  });
  await watchSetup();
  const backgrounded = state.appInitialized;
  initApp();
  $("dl-pill").classList.add("hidden");
  toggleDownloadPopover(false);
  if (backgrounded) {
    showToast("All models ready — queued recordings are processing now.", 4000, true);
    refreshMeetings();
  }
}

main().catch(showFatal);
