// --------------------------------------------------------------------------
// Prepare a recording — preview + waveform + time-range selection before the
// pipeline runs. Opens on every file add; the whole file is preselected, so
// the fast path stays one click ("Transcribe"). A selection that isn't the
// whole file goes to create_meeting as ranges = [[start, end], ...] seconds.
// --------------------------------------------------------------------------
import { $, call, convertFileSrc } from "./core";
import { el, fmtTs, iconBtn, showToast } from "./helpers";
import { selectMeeting } from "./meeting";
import { refreshMeetings } from "./meetings";
import { LANGUAGES } from "./settings";
import { state } from "./state";

let prep = null; // { path, filename, duration, peaks, ranges, probeFailed,
//                    hoverIndex, view: {start, end} | null (zoom window),
//                    fallbackTried, fallbackPending (audio-preview fallback) }
let prepDrag = null; // transient pointer interaction on the waveform
let prepStopAt = null; // pause the preview at this time (segment play)
let prepPlayingIndex = null; // range row currently previewing (⏵/⏹ icon)

const MIN_SEG_SECS = 0.5;
const WAVE_H = 110; // main waveform area height (px)
const OV_TOP = 118; // overview strip: full-file minimap for pan/zoom
const OV_H = 14;
const CANVAS_H = OV_TOP + OV_H;
const MIN_VIEW_SECS = 2; // deepest zoom

function basename(path) {
  return String(path).split(/[\\/]/).pop();
}

export async function openPrepare(path) {
  releasePrepMedia();
  prep = {
    path,
    filename: basename(path),
    duration: 0,
    peaks: null,
    ranges: [],
    probeFailed: false,
    hoverIndex: null,
    view: null,
    fallbackTried: false,
    fallbackPending: false,
  };
  for (const screen of ["app", "settings", "speakers", "bench"]) {
    $(screen).classList.add("hidden");
  }
  $("prepare").classList.remove("hidden");
  $("prep-file").textContent = prep.filename;
  $("prep-nopreview").classList.add("hidden");
  $("prep-video").classList.remove("audio-only", "hidden");
  $("prep-wave-status").textContent = "Analyzing audio…";
  $("prep-wave-status").classList.remove("hidden");
  $("prep-range-list").innerHTML = "";
  $("prep-hint").classList.add("hidden");
  $("prep-transcribe").disabled = true;
  drawWave();

  // Fast: whitelist the original file for the asset protocol, then preview.
  // Formats the webview can't play (mkv, avi, …) fail over to an extracted
  // audio-only preview so listening still works.
  const video = $("prep-video");
  try {
    await call("prepare_media_preview", { path });
    if (!prep || prep.path !== path) return;
    video.onerror = () => tryAudioFallback();
    video.onloadedmetadata = () => {
      // Audio-only files: collapse the black video box to bare controls.
      if (!video.videoWidth) video.classList.add("audio-only");
    };
    video.src = convertFileSrc(path);
  } catch {
    tryAudioFallback();
  }

  // Slow: decode for duration + waveform (a few seconds on long recordings).
  try {
    const probe = await call("probe_media", { path });
    if (!prep || prep.path !== path) return; // replaced or cancelled mid-probe
    prep.duration = probe.duration;
    prep.peaks = probe.peaks;
    prep.ranges = [{ start: 0, end: probe.duration }]; // default: everything
    prep.view = { start: 0, end: probe.duration };
    $("prep-wave-status").classList.add("hidden");
    renderPrep();
  } catch (e) {
    if (!prep || prep.path !== path) return;
    // Still transcribable: the button sends ranges = null (the whole file).
    prep.probeFailed = true;
    $("prep-wave-status").textContent =
      `Couldn't analyze this file: ${e.message} — you can still transcribe the whole recording.`;
    $("prep-transcribe").disabled = false;
  }
}

function showNoPreview() {
  const note = $("prep-nopreview");
  note.textContent =
    "Preview isn't available for this format — selecting and transcribing still work.";
  note.classList.remove("hidden");
  $("prep-video").classList.add("hidden");
}

// The webview can't play some containers at all (mkv, avi, wma, …). ffmpeg
// decodes them fine, so fall back to an extracted audio-only preview: the
// picture is lost, but listening — and the segment play buttons — work.
async function tryAudioFallback() {
  if (!prep || prep.fallbackTried) {
    showNoPreview();
    return;
  }
  const path = prep.path;
  prep.fallbackTried = true;
  prep.fallbackPending = true;
  const note = $("prep-nopreview");
  note.textContent = "This format can't be played directly — preparing an audio preview…";
  note.classList.remove("hidden");
  $("prep-video").classList.add("hidden");
  try {
    const preview = await call("extract_preview_audio", { path });
    if (!prep || prep.path !== path) return;
    prep.fallbackPending = false;
    const video = $("prep-video");
    video.classList.remove("hidden");
    video.classList.add("audio-only");
    video.src = convertFileSrc(preview.path); // errors → fallbackTried → note
    note.classList.add("hidden");
  } catch {
    if (!prep || prep.path !== path) return;
    prep.fallbackPending = false;
    showNoPreview();
  }
}

// Drop the preview source so the OS releases the file handle (same reason as
// releasePlayer) and clear any in-flight interaction state.
function releasePrepMedia() {
  const video = $("prep-video");
  video.onerror = null;
  video.pause();
  video.removeAttribute("src");
  video.load();
  prepStopAt = null;
  prepDrag = null;
  prepPlayingIndex = null;
}

function closePrepare() {
  releasePrepMedia();
  prep = null;
  $("prepare").classList.add("hidden");
  $("app").classList.remove("hidden");
}

async function startTranscription() {
  if (state.uploading || !prep) return;
  if (!prep.probeFailed && prep.ranges.length === 0) return;
  // A whole-file selection (and a failed probe) goes as null — the backend
  // then behaves exactly as before this screen existed.
  const whole =
    prep.probeFailed ||
    (prep.ranges.length === 1 &&
      prep.ranges[0].start <= 0.05 &&
      prep.ranges[0].end >= prep.duration - 0.05);
  const ranges = whole ? null : prep.ranges.map((r) => [r.start, r.end]);
  const numSpeakers = parseInt($("num-speakers")?.value ?? "-1", 10);
  state.uploading = true;
  $("prep-transcribe").disabled = true;
  try {
    const { id } = await call("create_meeting", {
      path: prep.path,
      num_speakers: Number.isNaN(numSpeakers) ? -1 : numSpeakers,
      ranges,
      language: $("prep-language")?.value || "auto",
    });
    closePrepare();
    await refreshMeetings();
    selectMeeting(id);
  } catch (e) {
    showToast(`Couldn't start transcription: ${e.message}`);
    $("prep-transcribe").disabled = false;
  } finally {
    state.uploading = false;
  }
}

// All range mutations funnel through here: clamp to the recording, drop
// sub-0.5 s slivers, sort, merge overlaps — mirroring the backend's
// normalize_ranges so what's shown is what gets stored.
function setPrepRanges(next) {
  const dur = prep.duration;
  const cleaned = next
    .map((r) => ({
      start: Math.max(0, Math.min(Math.min(r.start, r.end), dur)),
      end: Math.max(0, Math.min(Math.max(r.start, r.end), dur)),
    }))
    .filter((r) => r.end - r.start >= MIN_SEG_SECS)
    .sort((a, b) => a.start - b.start);
  const merged = [];
  for (const r of cleaned) {
    const last = merged[merged.length - 1];
    if (last && r.start <= last.end + 0.01) last.end = Math.max(last.end, r.end);
    else merged.push({ ...r });
  }
  prep.ranges = merged;
  prep.hoverIndex = null;
  renderPrep();
}

function renderPrep() {
  drawWave();
  renderRangeList();
  const canTranscribe = prep.probeFailed || prep.ranges.length > 0;
  $("prep-transcribe").disabled = !canTranscribe || state.uploading;
  $("prep-hint").classList.toggle("hidden", canTranscribe);
}

// ------------------------------------------------------------- waveform ---
// The main area shows the current zoom window (prep.view); the strip below
// it is a full-file minimap whose outlined window can be dragged to pan.
function prepView() {
  return prep.view || { start: 0, end: prep.duration };
}

function setView(start, span) {
  span = Math.max(MIN_VIEW_SECS, Math.min(span, prep.duration));
  const s = Math.max(0, Math.min(start, prep.duration - span));
  prep.view = { start: s, end: s + span };
  drawWave();
}

function zoomAt(t, factor) {
  const view = prepView();
  setView(t - (t - view.start) * factor, (view.end - view.start) * factor);
}

function drawWave() {
  const wrap = $("prep-wave-wrap");
  const canvas = $("prep-wave");
  const w = canvas.clientWidth || wrap.clientWidth;
  if (!w) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(CANVAS_H * dpr);
  canvas.style.height = `${CANVAS_H}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, CANVAS_H);
  if (!prep || !prep.peaks) return;

  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const accent = color("--accent");
  const rest = color("--border-strong");
  const { peaks, ranges, duration } = prep;
  const view = prepView();
  const span = view.end - view.start;
  const toX = (t) => ((t - view.start) / span) * w;
  const clampX = (x) => Math.max(0, Math.min(x, w));
  const inSelection = (t) => ranges.some((r) => t >= r.start && t <= r.end);
  const secsPerPeak = duration / peaks.length;

  // ---- main area: selection fill under the bars ----
  ctx.fillStyle = color("--accent-soft");
  for (const r of ranges) {
    const x1 = clampX(toX(r.start));
    const x2 = clampX(toX(r.end));
    if (x2 > x1) ctx.fillRect(x1, 0, x2 - x1, WAVE_H);
  }

  // Peak bars for the visible window: accent inside the selection.
  const mid = WAVE_H / 2;
  const first = Math.max(0, Math.floor(view.start / secsPerPeak));
  const last = Math.min(peaks.length, Math.ceil(view.end / secsPerPeak));
  if (last - first >= w) {
    // More peaks than pixels: fold to one bar per column.
    const cols = new Float32Array(w);
    for (let p = first; p < last; p++) {
      const x = Math.floor(toX((p + 0.5) * secsPerPeak));
      if (x >= 0 && x < w && peaks[p] > cols[x]) cols[x] = peaks[p];
    }
    for (let x = 0; x < w; x++) {
      const t = view.start + ((x + 0.5) / w) * span;
      ctx.fillStyle = inSelection(t) ? accent : rest;
      const bh = Math.max(1, cols[x] * (mid - 6));
      ctx.fillRect(x, mid - bh, 1, bh * 2);
    }
  } else {
    // Zoomed in: one (wider) bar per peak.
    const barW = w / Math.max(1, last - first);
    for (let p = first; p < last; p++) {
      const t = (p + 0.5) * secsPerPeak;
      ctx.fillStyle = inSelection(t) ? accent : rest;
      const bh = Math.max(1, peaks[p] * (mid - 6));
      ctx.fillRect(toX(p * secsPerPeak), mid - bh, Math.max(1, barW * 0.7), bh * 2);
    }
  }

  // Range edges (drag handles) + hovered-range outline, where visible.
  ctx.fillStyle = accent;
  for (const [i, r] of ranges.entries()) {
    for (const t of [r.start, r.end]) {
      const x = toX(t);
      if (x >= -3 && x <= w + 3) ctx.fillRect(x - 1.5, 0, 3, WAVE_H);
    }
    if (i === prep.hoverIndex) {
      const x1 = clampX(toX(r.start));
      const x2 = clampX(toX(r.end));
      if (x2 > x1) {
        ctx.strokeStyle = accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, 1, x2 - x1, WAVE_H - 2);
      }
    }
  }

  // Transient rectangle while dragging out a new segment.
  if (prepDrag && prepDrag.kind === "create" && prepDrag.current != null) {
    const a = clampX(toX(Math.min(prepDrag.anchor, prepDrag.current)));
    const b = clampX(toX(Math.max(prepDrag.anchor, prepDrag.current)));
    ctx.fillStyle = color("--accent-soft");
    ctx.fillRect(a, 0, b - a, WAVE_H);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1;
    ctx.strokeRect(a + 0.5, 0.5, b - a - 1, WAVE_H - 1);
  }

  // Playhead.
  const video = $("prep-video");
  if (!video.classList.contains("hidden") && video.currentTime > 0) {
    const x = toX(Math.min(video.currentTime, duration));
    if (x >= 0 && x <= w) {
      ctx.fillStyle = color("--text");
      ctx.fillRect(x - 0.5, 0, 1, WAVE_H);
    }
  }

  // Visible window's timestamps in the bottom corners (the top-right corner
  // belongs to the zoom buttons).
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.fillStyle = color("--text-muted");
  ctx.textBaseline = "bottom";
  ctx.fillText(fmtTs(view.start), 4, WAVE_H - 3);
  ctx.textAlign = "right";
  ctx.fillText(fmtTs(view.end), w - 4, WAVE_H - 3);
  ctx.textAlign = "left";

  // ---- overview strip: whole file, selection tint, draggable window ----
  const ovMid = OV_TOP + OV_H / 2;
  const fx = (t) => (t / duration) * w;
  const ovCols = new Float32Array(w);
  for (let p = 0; p < peaks.length; p++) {
    const x = Math.floor(((p + 0.5) / peaks.length) * w);
    if (x >= 0 && x < w && peaks[p] > ovCols[x]) ovCols[x] = peaks[p];
  }
  ctx.fillStyle = rest;
  for (let x = 0; x < w; x++) {
    const bh = Math.max(0.5, ovCols[x] * (OV_H / 2 - 1));
    ctx.fillRect(x, ovMid - bh, 1, bh * 2);
  }
  ctx.fillStyle = color("--accent-soft");
  for (const r of ranges) {
    ctx.fillRect(fx(r.start), OV_TOP, fx(r.end) - fx(r.start), OV_H);
  }
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1;
  ctx.strokeRect(
    fx(view.start) + 0.5,
    OV_TOP + 0.5,
    Math.max(2, fx(view.end) - fx(view.start)) - 1,
    OV_H - 1
  );
}

function setupWaveInteractions() {
  const canvas = $("prep-wave");
  const relPos = (e) => {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top, w: rect.width };
  };
  const posT = (e) => {
    const { x, w } = relPos(e);
    const view = prepView();
    return view.start + (x / w) * (view.end - view.start);
  };
  const pxToSecs = (px) => {
    const rect = canvas.getBoundingClientRect();
    const view = prepView();
    return (px / rect.width) * (view.end - view.start);
  };
  const hitTest = (e) => {
    const t = posT(e);
    const edge = pxToSecs(6);
    for (let i = 0; i < prep.ranges.length; i++) {
      const r = prep.ranges[i];
      if (Math.abs(t - r.start) <= edge) return { kind: "edge", index: i, side: "start" };
      if (Math.abs(t - r.end) <= edge) return { kind: "edge", index: i, side: "end" };
    }
    for (let i = 0; i < prep.ranges.length; i++) {
      const r = prep.ranges[i];
      if (t >= r.start && t <= r.end) return { kind: "move", index: i };
    }
    return { kind: "create" };
  };

  canvas.addEventListener("pointerdown", (e) => {
    if (!prep || !prep.peaks || state.uploading) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const { x, y, w } = relPos(e);
    if (y > WAVE_H) {
      // Overview strip: grab the view window, or jump it to the click.
      const view = prepView();
      const span = view.end - view.start;
      const tf = (x / w) * prep.duration;
      const start = tf >= view.start && tf <= view.end ? view.start : tf - span / 2;
      prepDrag = { kind: "view", grab: tf - start, span };
      setView(start, span);
      return;
    }
    const t = posT(e);
    const hit = hitTest(e);
    if (hit.kind === "edge") {
      prepDrag = { kind: "edge", index: hit.index, side: hit.side };
    } else if (hit.kind === "move") {
      prepDrag = { kind: "move", index: hit.index, grab: t - prep.ranges[hit.index].start };
    } else {
      prepDrag = { kind: "create", anchor: t, current: null };
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!prep || !prep.peaks) return;
    if (!prepDrag) {
      const { y } = relPos(e);
      if (y > WAVE_H) {
        canvas.style.cursor = "grab";
        return;
      }
      const hit = hitTest(e);
      canvas.style.cursor =
        hit.kind === "edge" ? "ew-resize" : hit.kind === "move" ? "grab" : "crosshair";
      const hover = hit.kind === "create" ? null : hit.index;
      if (hover !== prep.hoverIndex) {
        prep.hoverIndex = hover;
        drawWave();
      }
      return;
    }
    if (prepDrag.kind === "view") {
      const { x, w } = relPos(e);
      const tf = (x / w) * prep.duration;
      setView(tf - prepDrag.grab, prepDrag.span);
      return;
    }
    const t = Math.max(0, Math.min(posT(e), prep.duration));
    if (prepDrag.kind === "create") {
      prepDrag.current = t;
    } else if (prepDrag.kind === "edge") {
      const r = prep.ranges[prepDrag.index];
      if (prepDrag.side === "start") r.start = Math.min(t, r.end - MIN_SEG_SECS);
      else r.end = Math.max(t, r.start + MIN_SEG_SECS);
    } else {
      const r = prep.ranges[prepDrag.index];
      const len = r.end - r.start;
      const start = Math.max(0, Math.min(t - prepDrag.grab, prep.duration - len));
      r.start = start;
      r.end = start + len;
    }
    drawWave(); // the list re-syncs on pointerup
  });

  canvas.addEventListener("pointerup", (e) => {
    if (!prep || !prepDrag) return;
    const drag = prepDrag;
    prepDrag = null;
    if (drag.kind === "view") return;
    if (drag.kind === "create") {
      const t = Math.max(0, Math.min(posT(e), prep.duration));
      if (drag.current == null || Math.abs(t - drag.anchor) < pxToSecs(4)) {
        seekPreview(t); // a plain click scrubs the preview
        drawWave();
        return;
      }
      setPrepRanges([
        ...prep.ranges,
        { start: Math.min(drag.anchor, t), end: Math.max(drag.anchor, t) },
      ]);
    } else {
      setPrepRanges(prep.ranges); // normalize whatever the drag left behind
    }
  });

  canvas.addEventListener("pointercancel", () => {
    prepDrag = null;
    if (prep && prep.peaks) renderPrep();
  });

  // Pinch (macOS trackpads report it as ctrl+wheel) or ⌘-scroll zooms at the
  // cursor; a plain scroll pans the view window.
  canvas.addEventListener(
    "wheel",
    (e) => {
      if (!prep || !prep.peaks) return;
      e.preventDefault();
      const view = prepView();
      const span = view.end - view.start;
      if (e.ctrlKey || e.metaKey) {
        zoomAt(posT(e), Math.exp(e.deltaY * 0.01));
      } else {
        const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        const rect = canvas.getBoundingClientRect();
        setView(view.start + d * (span / rect.width), span);
      }
    },
    { passive: false }
  );
}

function seekPreview(t) {
  const video = $("prep-video");
  if (video.classList.contains("hidden") || !video.src) return;
  try {
    video.currentTime = t;
  } catch {
    /* metadata not ready yet — ignore */
  }
}

// ----------------------------------------------------------- range list ---
function renderRangeList() {
  const list = $("prep-range-list");
  list.innerHTML = "";
  prep.ranges.forEach((r, i) => {
    const li = el("li", "range-row");
    // Guard on prep: pointerleave can fire as the screen closes under the cursor.
    li.addEventListener("pointerenter", () => {
      if (!prep) return;
      prep.hoverIndex = i;
      drawWave();
    });
    li.addEventListener("pointerleave", () => {
      if (!prep) return;
      prep.hoverIndex = null;
      drawWave();
    });

    const label = el("span", "range-label");
    label.textContent = `Segment ${i + 1}`;
    const sep = el("span", "muted");
    sep.textContent = "–";
    const startInput = timeInput(r.start, `Segment ${i + 1} start`, (v) =>
      setPrepRanges(prep.ranges.map((x, j) => (j === i ? { start: v, end: x.end } : x)))
    );
    const endInput = timeInput(r.end, `Segment ${i + 1} end`, (v) =>
      setPrepRanges(prep.ranges.map((x, j) => (j === i ? { start: x.start, end: v } : x)))
    );
    const dur = el("span", "range-dur muted small");
    dur.textContent = fmtTs(r.end - r.start);

    const playingThis = prepPlayingIndex === i;
    const play = iconBtn(playingThis ? "i-stop" : "i-play", playingThis ? "Stop" : "Play this segment", () => {
      const video = $("prep-video");
      if (prepPlayingIndex === i && !video.paused) {
        video.pause(); // the pause handler resets the icon
        return;
      }
      if (!video.getAttribute("src")) {
        showToast(
          prep.fallbackPending
            ? "The audio preview is still being prepared — try again in a moment."
            : "No audio preview is available for this file."
        );
        return;
      }
      prepPlayingIndex = i;
      seekPreview(r.start);
      prepStopAt = r.end;
      video.play().catch((err) => {
        prepPlayingIndex = null;
        showToast(`Couldn't play the preview: ${err.message}`);
      });
      renderRangeList();
    });
    const remove = iconBtn("i-x", "Remove segment", () =>
      setPrepRanges(prep.ranges.filter((_, j) => j !== i))
    );

    li.append(label, startInput, sep, endInput, dur, play, remove);
    list.appendChild(li);
  });
}

function timeInput(seconds, label, onCommit) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "time-input";
  input.value = fmtTsInput(seconds);
  input.setAttribute("aria-label", label);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") {
      e.stopPropagation(); // revert the field, don't close the screen
      input.value = fmtTsInput(seconds);
      input.blur();
    }
  });
  input.addEventListener("blur", () => {
    if (input.value === fmtTsInput(seconds)) return;
    const v = parseTs(input.value);
    if (v == null) {
      showToast("Times look like 1:23 or 1:02:03.5");
      input.value = fmtTsInput(seconds);
      return;
    }
    onCommit(Math.max(0, Math.min(v, prep.duration)));
  });
  return input;
}

// h:mm:ss.t — always with hours so the columns line up.
function fmtTsInput(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
  const s = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${h}:${m}:${s}`;
}

// Accepts "90", "1:30", "1:02:03", each with an optional ".d" fraction.
function parseTs(text) {
  const parts = String(text).trim().split(":");
  if (parts.length < 1 || parts.length > 3) return null;
  if (parts.some((p) => !/^\d+(\.\d+)?$/.test(p))) return null;
  let secs = 0;
  for (const p of parts) secs = secs * 60 + Number(p);
  return Number.isFinite(secs) ? secs : null;
}

// "+ Add segment": a segment in the middle of the largest unselected gap.
function addRangeInGap() {
  if (!prep || !prep.peaks) return;
  const edges = [0, ...prep.ranges.flatMap((r) => [r.start, r.end]), prep.duration];
  let best = null;
  for (let i = 0; i < edges.length; i += 2) {
    const gap = edges[i + 1] - edges[i];
    if (gap > (best?.gap ?? 0)) best = { gap, from: edges[i], to: edges[i + 1] };
  }
  if (!best || best.gap < MIN_SEG_SECS * 2) {
    showToast("No room left — adjust or remove a segment first.");
    return;
  }
  const len = Math.min(30, best.gap / 2);
  const mid = (best.from + best.to) / 2;
  setPrepRanges([...prep.ranges, { start: mid - len / 2, end: mid + len / 2 }]);
}

// One-time wiring for the prep screen (called from initApp).
export function setupPrepare() {
  // Spoken-language hint, per recording. Starts on the saved ASR_LANGUAGE
  // setting, then keeps whatever was chosen last (like the Speakers select).
  const langSelect = $("prep-language");
  for (const [code, name] of [["auto", "Auto-detect"], ...LANGUAGES]) {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = name;
    langSelect.appendChild(opt);
  }
  call("get_config")
    .then((cfg) => {
      const saved = cfg?.values?.ASR_LANGUAGE;
      if (saved && [...langSelect.options].some((o) => o.value === saved)) {
        langSelect.value = saved;
      }
    })
    .catch(() => {});

  $("prep-back").addEventListener("click", closePrepare);
  $("prep-cancel").addEventListener("click", closePrepare);
  $("prep-transcribe").addEventListener("click", startTranscription);
  $("prep-add-range").addEventListener("click", addRangeInGap);
  $("prep-select-all").addEventListener("click", () => {
    if (prep && prep.peaks) setPrepRanges([{ start: 0, end: prep.duration }]);
  });
  $("wave-zoom-in").addEventListener("click", () => {
    if (!prep || !prep.peaks) return;
    const v = prepView();
    zoomAt((v.start + v.end) / 2, 0.5);
  });
  $("wave-zoom-out").addEventListener("click", () => {
    if (!prep || !prep.peaks) return;
    const v = prepView();
    zoomAt((v.start + v.end) / 2, 2);
  });
  $("wave-zoom-fit").addEventListener("click", () => {
    if (prep && prep.peaks) setView(0, prep.duration);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && prep && !$("prepare").classList.contains("hidden")) {
      closePrepare();
    }
  });
  window.addEventListener("resize", () => {
    if (prep && !$("prepare").classList.contains("hidden")) drawWave();
  });

  // Keep the playhead moving while the preview plays; stop segment playback
  // at its end (prepStopAt is set by the per-segment play buttons).
  const video = $("prep-video");
  let raf = null;
  const tick = () => {
    drawWave();
    raf = video.paused || video.ended ? null : requestAnimationFrame(tick);
  };
  video.addEventListener("play", () => {
    if (prep) tick();
  });
  video.addEventListener("pause", () => {
    prepStopAt = null;
    prepPlayingIndex = null;
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    if (prep && !prepDrag) {
      renderRangeList(); // flip the row's ⏹ back to ⏵
      drawWave();
    }
  });
  video.addEventListener("seeked", () => {
    if (prep) drawWave();
  });
  video.addEventListener("timeupdate", () => {
    if (prepStopAt != null && video.currentTime >= prepStopAt) video.pause();
  });

  setupWaveInteractions();
}
