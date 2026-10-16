// --------------------------------------------------------------------------
// Segment playback — hear a section to decide who's speaking before naming.
// One shared <audio> streams the kept recording through the asset protocol
// (Range/seek support); play seeks to the segment start and pauses at its
// end. Clicking the playing segment stops it.
// --------------------------------------------------------------------------
let player = null;
let playerSrc = "";
let playingIndex = null;
let playingBtn = null;
let playingSeg = null; // the .seg element, for the highlight class

function stopSegmentPlayback() {
  if (player) {
    player.pause();
    player.ontimeupdate = null;
    player.onended = null;
  }
  if (playingBtn) {
    playingBtn.querySelector("use").setAttribute("href", "#i-play");
    playingBtn.title = "Play this section";
    playingBtn.setAttribute("aria-label", "Play this section");
  }
  if (playingSeg) playingSeg.classList.remove("playing");
  playingIndex = null;
  playingBtn = null;
  playingSeg = null;
}

// Also drop the audio source so the OS file handle is released (deleting a
// meeting on Windows fails while the webview holds its recording open).
function releasePlayer() {
  stopSegmentPlayback();
  if (player) {
    player.removeAttribute("src");
    player.load();
  }
  playerSrc = "";
}

async function playSegment(meetingId, seg, index, btn, segEl) {
  if (playingIndex === index) {
    stopSegmentPlayback();
    return;
  }
  stopSegmentPlayback();
  if (!player) {
    player = new Audio();
    player.preload = "auto";
  }
  let src;
  try {
    const audio = await call("get_meeting_audio_path", { meeting_id: meetingId });
    src = convertFileSrc(audio.path);
  } catch (e) {
    showToast(
      e.status === 404 ? "The recording is no longer on disk." : `Couldn't load audio: ${e.message}`
    );
    return;
  }
  if (playerSrc !== src) {
    player.src = src;
    playerSrc = src;
  }
  playingIndex = index;
  playingBtn = btn;
  playingSeg = segEl;
  btn.querySelector("use").setAttribute("href", "#i-stop");
  btn.title = "Stop";
  btn.setAttribute("aria-label", "Stop");
  segEl.classList.add("playing");

  const begin = () => {
    player.currentTime = seg.start || 0;
    // Stop at the segment boundary (with a whisker of slack — timeupdate
    // fires only a few times a second).
    player.ontimeupdate = () => {
      if (seg.end && player.currentTime >= seg.end) stopSegmentPlayback();
    };
    player.onended = stopSegmentPlayback;
    player.play().catch((e) => {
      stopSegmentPlayback();
      showToast(`Couldn't play audio: ${e.message}`);
    });
  };
  // Seeking needs the media metadata; wait for it on first load.
  if (player.readyState >= 1) {
    begin();
  } else {
    player.onloadedmetadata = begin;
    player.onerror = () => {
      stopSegmentPlayback();
      showToast("Couldn't load the recording for playback.");
    };
  }
}

function startSegmentEdit(index, segEl) {
  if (editing || !currentMeeting) return;
  editing = `seg:${index}`;
  const seg = currentMeeting.segments[index];
  segEl.classList.add("editing");
  segEl.querySelector(".seg-text").classList.add("hidden");
  const segActions = segEl.querySelector(".seg-actions");
  if (segActions) segActions.classList.add("hidden");

  const editor = el("div", "editor");
  const ta = document.createElement("textarea");
  ta.rows = 3;
  ta.value = seg.text;
  ta.setAttribute("aria-label", "Segment text");
  editor.appendChild(ta);

  const select = document.createElement("select");
  select.setAttribute("aria-label", "Speaker");
  for (const [id, name] of Object.entries(currentMeeting.speakers)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = name || id;
    if (id === seg.speaker) opt.selected = true;
    select.appendChild(opt);
  }

  const cleanup = () => {
    editor.remove();
    segEl.classList.remove("editing");
    segEl.querySelector(".seg-text").classList.remove("hidden");
    if (segActions) segActions.classList.remove("hidden");
  };
  const commit = async () => {
    const body = {};
    const text = ta.value.trim();
    if (text && text !== seg.text) body.text = text;
    if (select.value !== seg.speaker) body.speaker = select.value;
    if (!Object.keys(body).length) {
      cleanup();
      finishEdit(null);
      return;
    }
    try {
      const m = await call("patch_segment", {
        meeting_id: currentMeeting.id,
        index,
        fields: body,
      });
      cleanup();
      finishEdit(m);
    } catch (e) {
      showToast(editError(e));
    }
  };
  const cancel = () => {
    cleanup();
    finishEdit(null);
  };
  const actions = editorActions(commit, cancel);
  actions.insertBefore(select, actions.firstChild);
  editor.appendChild(actions);
  wireEditorKeys(ta, commit, cancel);
  segEl.appendChild(editor);
  ta.focus();
}

async function renameSpeaker(meetingId, diarizedId, newName) {
  try {
    const m = await call("rename_speakers", {
      meeting_id: meetingId,
      names: { [diarizedId]: newName },
    });
    renderResult(m);
    refreshSpeakers();
  } catch (e) {
    showToast(`Rename failed: ${e.message}`);
  }
}

// --------------------------------------------------------------------------
// Copy to clipboard
// --------------------------------------------------------------------------
function notesAsText(m) {
  const lines = [`# ${m.title || m.filename}`, ""];
  if (m.summary) lines.push(m.summary, "");
  if (m.takeaways && m.takeaways.length) {
    lines.push("Key takeaways:");
    m.takeaways.forEach((t) => lines.push(`- ${t}`));
    lines.push("");
  }
  if (m.action_items && m.action_items.length) {
    lines.push("Action items:");
    m.action_items.forEach((a) => lines.push(`- [ ] ${a}`));
  }
  return lines.join("\n").trim();
}

function transcriptAsText(m) {
  return (m.segments || [])
    .map((seg) => `${m.speakers[seg.speaker] || seg.speaker} [${fmtTs(seg.start)}]: ${seg.text}`)
    .join("\n");
}

async function copyText(text, okMsg) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    showToast(okMsg, 2500, true);
  } catch {
    showToast("Copy failed");
  }
}

