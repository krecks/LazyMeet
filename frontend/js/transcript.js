// --------------------------------------------------------------------------
// Transcript
// --------------------------------------------------------------------------
function buildColorMap(segments) {
  const ids = [...new Set(segments.map((s) => s.speaker))].sort();
  const map = {};
  ids.forEach((id, i) => (map[id] = `var(--speaker-${(i % 6) + 1})`));
  return map;
}

function renderLegend(m, colorOf, readOnly) {
  const legend = $("speaker-legend");
  legend.innerHTML = "";
  for (const id of Object.keys(colorOf)) {
    const recognized = (m.recognized || []).includes(id);
    const confirmed = (m.confirmed || []).includes(id);
    const name = m.speakers[id] || id;
    const chip = el("div", "chip");
    chip.dataset.speakerId = id;
    chip.style.setProperty("--speaker-color", colorOf[id]);
    chip.innerHTML = `<span class="swatch"></span>`;
    const input = document.createElement("input");
    input.value = name;
    input.title = readOnly
      ? "Read-only — speakers belong to the current version"
      : "Rename — saves this voice for future recordings";
    input.disabled = !!readOnly;
    input.setAttribute("aria-label", `Rename ${name}`);
    input.setAttribute("list", "speaker-names");
    input.addEventListener("change", () => renameSpeaker(m.id, id, input.value));
    chip.appendChild(input);
    if (readOnly) {
      legend.appendChild(chip);
      continue; // no verify buttons on a snapshot
    }
    if (recognized && confirmed) {
      const badge = el("span", "badge-recognized");
      badge.textContent = "known";
      badge.title = "Confirmed — this recording refined their saved voice";
      chip.appendChild(badge);
    } else if (recognized) {
      // Auto-matched but unverified: ask before the name is trusted, so a
      // wrong match can be unlinked without poisoning the saved voice.
      const badge = el("span", "badge-recognized unconfirmed");
      badge.textContent = "known?";
      badge.title = `Auto-recognized as ${name} from a saved voice — is that right?`;
      chip.appendChild(badge);
      chip.appendChild(
        iconBtn("i-check", `Yes, this is ${name}`, () => verifySpeaker(m.id, id, true), "icon-btn verify yes")
      );
      chip.appendChild(
        iconBtn("i-x", `Not ${name} — unlink and pick another name`, () => verifySpeaker(m.id, id, false), "icon-btn verify no")
      );
    }
    legend.appendChild(chip);
  }
}

async function verifySpeaker(meetingId, diarizedId, correct) {
  try {
    const m = await call("verify_speaker", {
      meeting_id: meetingId,
      diarized_id: diarizedId,
      correct,
    });
    renderResult(m);
    refreshSpeakers();
    if (!correct) {
      // The name is back to "Speaker N" — put the cursor there so the right
      // name (or a brand-new person) can be typed straight away.
      const input = $("speaker-legend").querySelector(
        `.chip[data-speaker-id="${CSS.escape(diarizedId)}"] input`
      );
      if (input) {
        input.focus();
        input.select();
      }
    }
  } catch (e) {
    showToast(`Couldn't update speaker: ${e.message}`);
  }
}

function renderTranscript(m, colorOf, editable) {
  stopSegmentPlayback(); // the buttons owning the playing state are re-created
  const t = $("transcript");
  t.innerHTML = "";
  (m.segments || []).forEach((seg, i) => {
    const div = el("div", "seg");
    div.style.setProperty("--speaker-color", colorOf[seg.speaker] || "var(--border-strong)");
    const name = m.speakers[seg.speaker] || seg.speaker;
    div.innerHTML = `<div class="meta"><span class="who">${escapeHtml(name)}</span><span class="ts">${fmtTs(seg.start)}</span></div>
      <div class="seg-text">${escapeHtml(seg.text)}</div>`;
    if (m.has_audio || editable) {
      const actions = el("div", "seg-actions");
      if (m.has_audio) {
        const btn = iconBtn("i-play", "Play this section", () => playSegment(m.id, seg, i, btn, div));
        actions.appendChild(btn);
      }
      if (editable) {
        actions.appendChild(iconBtn("i-pencil", "Edit segment", () => startSegmentEdit(i, div)));
      }
      div.appendChild(actions);
    }
    t.appendChild(div);
  });
  // Whatever is on screen is what find searches — including read-only
  // snapshots (where the replace row hides).
  find.meeting = m;
  find.editable = editable;
  refreshFind();
  consumePendingScroll(m); // a global-search hit asked for this segment
}

// --------------------------------------------------------------------------
// Find & replace in the transcript. Matching lives entirely here (the same
// matcher drives highlights, the counter, and replace); the backend only gets
// the resulting segment texts via the bulk patch_segments command, so a
// replace-all is one locked write instead of one per segment.
// --------------------------------------------------------------------------
const find = {
  open: false,
  matchCase: false,
  matches: [], // { seg, start, end } in transcript order
  current: -1, // index into matches
  marks: [], // <mark> elements, parallel to matches
  meeting: null, // the meeting whose segments are rendered (may be a snapshot)
  editable: false,
};

const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Literal search as a regex: the 'i' flag gives Unicode-aware case folding
// without the index drift of comparing toLowerCase()d strings. Shared with
// global search so highlights can never disagree between the two surfaces.
function literalRegex(query, matchCase) {
  if (!query) return null;
  return new RegExp(escapeRegExp(query), matchCase ? "g" : "gi");
}

function findRegex() {
  return literalRegex($("find-input").value, find.matchCase);
}

/// `text` as HTML with every match wrapped in a <mark>. `currentAt` marks one
/// hit as the active one (by its ordinal within this text); global search
/// passes nothing and gets all hits styled the same.
function markHits(text, re, offset = 0, currentAt = -1) {
  let html = "";
  let pos = 0;
  let k = offset;
  re.lastIndex = 0;
  for (const hit of text.matchAll(re)) {
    html += escapeHtml(text.slice(pos, hit.index));
    html += `<mark class="find-hit${k === currentAt ? " current" : ""}">${escapeHtml(hit[0])}</mark>`;
    pos = hit.index + hit[0].length;
    k += 1;
  }
  return html + escapeHtml(text.slice(pos));
}

function openFind() {
  find.open = true;
  $("find-bar").classList.remove("hidden");
  $("find-toggle-btn").setAttribute("aria-expanded", "true");
  const input = $("find-input");
  input.focus();
  input.select();
  refreshFind();
}

function closeFind() {
  find.open = false;
  $("find-bar").classList.add("hidden");
  $("find-toggle-btn").setAttribute("aria-expanded", "false");
  refreshFind(); // clears the highlights
}

/// Recompute matches for the rendered transcript and repaint. Runs on every
/// transcript render, query edit, and case toggle. `find.current` is clamped,
/// not reset — after a replace the position carries over to the next match.
function refreshFind() {
  const re = find.open ? findRegex() : null;
  const segs = (find.meeting && find.meeting.segments) || [];
  find.matches = [];
  if (re) {
    segs.forEach((seg, i) => {
      for (const hit of seg.text.matchAll(re)) {
        find.matches.push({ seg: i, start: hit.index, end: hit.index + hit[0].length });
      }
    });
  }
  find.current = find.matches.length
    ? Math.min(Math.max(find.current, 0), find.matches.length - 1)
    : -1;
  paintFindMarks();
  updateFindControls();
}

// Rewrite each segment's text HTML with <mark>s around the matches. Segments
// that had no marks and get none are left untouched.
function paintFindMarks() {
  const t = $("transcript");
  const segs = (find.meeting && find.meeting.segments) || [];
  const bySeg = new Map();
  find.matches.forEach((m, k) => {
    if (!bySeg.has(m.seg)) bySeg.set(m.seg, []);
    bySeg.get(m.seg).push({ ...m, k });
  });
  const re = find.matches.length ? findRegex() : null;
  segs.forEach((seg, i) => {
    const node = t.children[i] && t.children[i].querySelector(".seg-text");
    if (!node) return;
    const hits = bySeg.get(i);
    if (!hits) {
      if (node.querySelector("mark")) node.innerHTML = escapeHtml(seg.text);
      return;
    }
    // hits are contiguous within a segment, so the first one's ordinal places
    // the whole run against find.current
    node.innerHTML = markHits(seg.text, re, hits[0].k, find.current);
  });
  // matches are generated in document order, so this lines up with find.matches
  find.marks = [...t.querySelectorAll("mark.find-hit")];
}

function updateFindControls() {
  const n = find.matches.length;
  $("find-count").textContent = $("find-input").value ? `${n ? find.current + 1 : 0}/${n}` : "";
  $("find-prev").disabled = !n;
  $("find-next").disabled = !n;
  $("find-replace-row").classList.toggle("hidden", !find.editable);
  $("find-replace").disabled = !n;
  const allBtn = $("find-replace-all");
  if (allBtn.disabled !== !n) disarm(allBtn); // a stale confirm must not survive
  allBtn.disabled = !n;
}

function gotoMatch(k) {
  const n = find.matches.length;
  if (!n) return;
  find.current = ((k % n) + n) % n;
  find.marks.forEach((el, i) => el.classList.toggle("current", i === find.current));
  const el = find.marks[find.current];
  if (el) el.scrollIntoView({ block: "center" });
  updateFindControls();
}

/// Replace the current match (all = false) or every match (all = true).
/// Only reachable when `find.editable`, i.e. the live version of a "done"
/// meeting is on screen — so `find.meeting === currentMeeting`.
async function replaceMatches(all) {
  if (!find.editable || !find.matches.length || !currentMeeting) return;
  const replacement = $("find-replace-input").value;
  const targets = all ? find.matches : [find.matches[find.current]];
  const bySeg = new Map();
  for (const m of targets) {
    if (!bySeg.has(m.seg)) bySeg.set(m.seg, []);
    bySeg.get(m.seg).push(m);
  }
  // Within a segment, splice from the end so earlier offsets stay valid.
  const edits = [];
  for (const [i, hits] of bySeg) {
    let text = currentMeeting.segments[i].text;
    for (const h of [...hits].sort((a, b) => b.start - a.start)) {
      text = text.slice(0, h.start) + replacement + text.slice(h.end);
    }
    edits.push([i, text]);
  }
  const keep = all ? 0 : find.current;
  try {
    const m = await call("patch_segments", { meeting_id: currentMeeting.id, edits });
    find.current = keep; // refreshFind (via renderResult) clamps this onto the next match
    renderResult(m);
    if (all) showToast(`Replaced ${targets.length} match${targets.length === 1 ? "" : "es"}`, 2500, true);
    else if (find.marks[find.current]) find.marks[find.current].scrollIntoView({ block: "center" });
  } catch (e) {
    showToast(editError(e));
  }
}

// One-time wiring for the find bar (called from initApp).
function setupFind() {
  $("find-toggle-btn").addEventListener("click", () => (find.open ? closeFind() : openFind()));
  $("find-close").addEventListener("click", closeFind);
  $("find-prev").addEventListener("click", () => gotoMatch(find.current - 1));
  $("find-next").addEventListener("click", () => gotoMatch(find.current + 1));
  $("find-case").addEventListener("click", () => {
    find.matchCase = !find.matchCase;
    const btn = $("find-case");
    btn.classList.toggle("on", find.matchCase);
    btn.setAttribute("aria-pressed", String(find.matchCase));
    find.current = 0;
    refreshFind();
  });
  $("find-input").addEventListener("input", () => {
    find.current = 0;
    refreshFind();
    if (find.matches.length) gotoMatch(0); // follow the first match while typing
  });
  $("find-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      gotoMatch(find.current + (e.shiftKey ? -1 : 1));
    }
    if (e.key === "Escape") {
      e.stopPropagation();
      closeFind();
    }
  });
  $("find-replace-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      replaceMatches(false);
    }
    if (e.key === "Escape") {
      e.stopPropagation();
      closeFind();
    }
  });
  $("find-replace").addEventListener("click", () => replaceMatches(false));
  // Replace-all has no undo — same two-click guard as the other destructive buttons.
  $("find-replace-all").addEventListener("click", (e) => {
    confirmThen(e.currentTarget, `Replace all ${find.matches.length}?`, () => replaceMatches(true));
  });
  document.addEventListener("keydown", (e) => {
    if (
      (e.metaKey || e.ctrlKey) &&
      e.key.toLowerCase() === "f" &&
      !$("app").classList.contains("hidden") &&
      !$("result").classList.contains("hidden")
    ) {
      e.preventDefault();
      openFind();
    }
  });
}

