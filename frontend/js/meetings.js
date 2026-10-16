// --------------------------------------------------------------------------
// Upload + meeting list
// --------------------------------------------------------------------------
function setupDropzone() {
    const dz = $("dropzone");
    // Click/keyboard opens a native file dialog (webview File objects have no
    // filesystem path, so the old <input type="file"> can't feed the backend).
    const browse = async () => {
        try {
            const path = await call("pick_recording");
            if (path)
                openPrepare(path);
        }
        catch (e) {
            showToast(`Couldn't open the file picker: ${e.message}`);
        }
    };
    dz.addEventListener("click", browse);
    dz.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            browse();
        }
    });
    // Drops are accepted anywhere in the window, not just on the dropzone.
    // Tauri's native drag-drop gives real filesystem paths (the HTML5 events no
    // longer fire once the native interception is on).
    tauri.webview.getCurrentWebview().onDragDropEvent((event) => {
        const type = event.payload.type;
        if (type === "enter" || type === "over")
            dz.classList.add("drag");
        else if (type === "leave")
            dz.classList.remove("drag");
        else if (type === "drop") {
            dz.classList.remove("drag");
            const path = event.payload.paths && event.payload.paths[0];
            if (path)
                openPrepare(path); // replaces the current prep file if one is open
        }
    });
}
async function refreshMeetings() {
    try {
        meetings = await call("list_meetings");
    }
    catch (e) {
        showToast(`Couldn't load meetings: ${e.message}`);
        return;
    }
    renderMeetingList();
}
function badgeFor(status) {
    if (status === "done")
        return ["done", "done"];
    if (status === "error")
        return ["error", "failed"];
    if (status === "queued")
        return ["queued", "queued"];
    return ["processing", status];
}
function renderMeetingList() {
    const list = $("meeting-list");
    list.innerHTML = "";
    const q = searchQuery.trim().toLowerCase();
    // Titles filter instantly; meetings whose *content* matched join them once
    // the global search comes back, so the list can't contradict the results.
    const withHits = new Set(gsearch.results.map((g) => g.id));
    const shown = q
        ? meetings.filter((m) => withHits.has(m.id) || (m.title || m.filename || "").toLowerCase().includes(q))
        : meetings;
    if (!shown.length) {
        list.innerHTML = `<li class="list-empty muted">${meetings.length ? "No matches" : "No meetings yet"}</li>`;
        return;
    }
    for (const m of shown) {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        if (m.id === activeId)
            btn.classList.add("active");
        const [cls, label] = badgeFor(m.status);
        btn.innerHTML = `<span class="m-title">${escapeHtml(m.title || m.filename)}</span>
      <span class="m-meta"><span class="badge ${cls}">${label}</span><span>${fmtRelative(m.created_at)}</span></span>`;
        btn.addEventListener("click", () => selectMeeting(m.id));
        li.appendChild(btn);
        list.appendChild(li);
    }
}
