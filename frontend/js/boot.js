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
  setupPrepare();
  setupFind();
  setupGlobalSearch();
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
    renderMeetingList(); // the sidebar title filter stays instant
    queueGlobalSearch(searchQuery); // the disk scan waits for a pause
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
