// --------------------------------------------------------------------------
// First-run setup. The download runs in the backend either way; the user can
// enter the app immediately ("Use the app while it downloads") and follow
// progress in a header pill. Uploads made meanwhile wait in the processing
// queue until the models are on disk. Progress arrives as `setup-progress`
// events (the old 1 s polling loop is gone).
// --------------------------------------------------------------------------

import { $, call } from "./core";
import { el, escapeHtml, showToast } from "./helpers";
import { state } from "./state";

export function applySetupStatus(status) {
  if (status.ready) {
    state.modelsReady = true;
    return;
  }
  if (state.appInitialized) {
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
export async function watchSetup() {
  const status = await call("get_setup");
  applySetupStatus(status);
  if (status.ready) return;
  await new Promise<void>((resolve) => {
    state.setupReadyResolve = () => resolve();
  });
}


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

export function toggleDownloadPopover(open?: boolean) {
  const pop = $("dl-popover");
  const show = open !== undefined ? open : pop.classList.contains("hidden");
  pop.classList.toggle("hidden", !show);
  $("dl-pill").setAttribute("aria-expanded", String(show));
  if (show && lastSetupStatus) renderDownloadPopover(lastSetupStatus);
}

export async function retrySetup() {
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

