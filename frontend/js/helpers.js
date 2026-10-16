// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
function el(tag, cls) {
    const node = document.createElement(tag);
    if (cls)
        node.className = cls;
    return node;
}
function iconBtn(icon, title, onClick, cls = "icon-btn subtle") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = cls;
    btn.title = title;
    btn.setAttribute("aria-label", title);
    btn.innerHTML = `<svg class="icon"><use href="#${icon}" /></svg>`;
    btn.addEventListener("click", onClick);
    return btn;
}
function editorActions(onSave, onCancel) {
    const wrap = el("div", "editor-actions");
    const save = el("button", "btn");
    save.type = "button";
    save.textContent = "Save";
    save.addEventListener("click", onSave);
    const cancel = el("button", "btn ghost");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", onCancel);
    const hint = el("span", "editor-hint");
    hint.textContent = "⌘↵ save · esc cancel";
    wrap.append(save, cancel, hint);
    return wrap;
}
function wireEditorKeys(textarea, onSave, onCancel) {
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey))
            onSave();
        if (e.key === "Escape")
            onCancel();
    });
}
function showToast(msg, ttl = 5000, ok = false) {
    const div = el("div", "toast" + (ok ? " ok" : ""));
    div.textContent = msg;
    $("toasts").appendChild(div);
    setTimeout(() => div.remove(), ttl);
}
function showFatal(err) {
    $("boot").classList.add("hidden");
    $("setup").classList.add("hidden");
    $("app").classList.add("hidden");
    $("prepare").classList.add("hidden");
    $("settings").classList.add("hidden");
    $("speakers").classList.add("hidden");
    $("fatal").classList.remove("hidden");
    $("fatal-msg").textContent = err && err.message ? err.message : String(err);
}
// Two-click guard for destructive buttons: the first click arms the button for
// 3 s, the second runs the action. (Native confirm() is unreliable in webviews.)
// Buttons with a <span> label show the armed text there; icon-only buttons just
// turn red and change their tooltip.
function confirmThen(btn, armedText, action) {
    if (btn.dataset.armed) {
        clearTimeout(+btn.dataset.timer);
        disarm(btn);
        action();
        return;
    }
    btn.dataset.armed = "1";
    const label = btn.querySelector("span:not(.icon)");
    if (label) {
        btn.dataset.orig = label.textContent;
        label.textContent = armedText;
    }
    else {
        btn.dataset.origTitle = btn.title;
        btn.title = armedText;
    }
    btn.classList.add("armed");
    btn.dataset.timer = setTimeout(() => disarm(btn), 3000);
}
function disarm(btn) {
    if (!btn.dataset.armed)
        return;
    const label = btn.querySelector("span:not(.icon)");
    if (label && btn.dataset.orig !== undefined)
        label.textContent = btn.dataset.orig;
    if (btn.dataset.origTitle !== undefined)
        btn.title = btn.dataset.origTitle;
    delete btn.dataset.armed;
    delete btn.dataset.orig;
    delete btn.dataset.origTitle;
    delete btn.dataset.timer;
    btn.classList.remove("armed");
}
function fmtTs(seconds) {
    const s = Math.floor(seconds % 60).toString().padStart(2, "0");
    const m = Math.floor(seconds / 60) % 60;
    const h = Math.floor(seconds / 3600);
    return h ? `${h}:${m.toString().padStart(2, "0")}:${s}` : `${m}:${s}`;
}
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
function fmtDate(iso) {
    if (!iso)
        return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime()))
        return "";
    return d.toLocaleString(undefined, { dateStyle: "long", timeStyle: "short" });
}
function fmtRelative(iso) {
    if (!iso)
        return "";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then))
        return "";
    const sec = Math.round((then - Date.now()) / 1000);
    const units = [
        ["year", 31536000],
        ["month", 2592000],
        ["week", 604800],
        ["day", 86400],
        ["hour", 3600],
        ["minute", 60],
    ];
    const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
    for (const [unit, s] of units) {
        if (Math.abs(sec) >= s)
            return rtf.format(Math.round(sec / s), unit);
    }
    return "just now";
}
function fmtBytes(n) {
    if (!n)
        return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (n >= 1024 && i < units.length - 1) {
        n /= 1024;
        i++;
    }
    return `${n >= 10 || i === 0 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
}
function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
