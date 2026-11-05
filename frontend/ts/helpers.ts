// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
import { $ } from "./core";

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
}

export function iconBtn(icon, title, onClick, cls = "icon-btn subtle") {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = cls;
  btn.title = title;
  btn.setAttribute("aria-label", title);
  btn.innerHTML = `<svg class="icon"><use href="#${icon}" /></svg>`;
  btn.addEventListener("click", onClick);
  return btn;
}

export function editorActions(onSave, onCancel) {
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

export function wireEditorKeys(textarea, onSave, onCancel) {
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onSave();
    if (e.key === "Escape") onCancel();
  });
}

export function showToast(msg, ttl = 5000, ok = false) {
  const div = el("div", "toast" + (ok ? " ok" : ""));
  div.textContent = msg;
  $("toasts").appendChild(div);
  setTimeout(() => div.remove(), ttl);
}

export function showFatal(err) {
  $("boot").classList.add("hidden");
  $("setup").classList.add("hidden");
  $("app").classList.add("hidden");
  $("prepare").classList.add("hidden");
  $("settings").classList.add("hidden");
  $("speakers").classList.add("hidden");
  $("bench").classList.add("hidden");
  $("fatal").classList.remove("hidden");
  $("fatal-msg").textContent = err && err.message ? err.message : String(err);
}

// Two-click guard for destructive buttons: the first click arms the button for
// 3 s, the second runs the action. (Native confirm() is unreliable in webviews.)
// Buttons with a <span> label show the armed text there; icon-only buttons just
// turn red and change their tooltip.
export function confirmThen(btn, armedText, action) {
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
  } else {
    btn.dataset.origTitle = btn.title;
    btn.title = armedText;
  }
  btn.classList.add("armed");
  btn.dataset.timer = setTimeout(() => disarm(btn), 3000);
}

export function disarm(btn) {
  if (!btn.dataset.armed) return;
  const label = btn.querySelector("span:not(.icon)");
  if (label && btn.dataset.orig !== undefined) label.textContent = btn.dataset.orig;
  if (btn.dataset.origTitle !== undefined) btn.title = btn.dataset.origTitle;
  delete btn.dataset.armed;
  delete btn.dataset.orig;
  delete btn.dataset.origTitle;
  delete btn.dataset.timer;
  btn.classList.remove("armed");
}

export function fmtTs(seconds) {
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  const m = Math.floor(seconds / 60) % 60;
  const h = Math.floor(seconds / 3600);
  return h ? `${h}:${m.toString().padStart(2, "0")}:${s}` : `${m}:${s}`;
}

export const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/// A duration for people: "45.2s" under a minute, else "12m 05s"
/// (minutes keep counting past the hour, like the timings line).
export function fmtSpan(s: number): string {
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s` : `${s.toFixed(1)}s`;
}

export function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { dateStyle: "long", timeStyle: "short" });
}

export function fmtRelative(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const sec = Math.round((then - Date.now()) / 1000);
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ["year", 31536000],
    ["month", 2592000],
    ["week", 604800],
    ["day", 86400],
    ["hour", 3600],
    ["minute", 60],
  ];
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, s] of units) {
    if (Math.abs(sec) >= s) return rtf.format(Math.round(sec / s), unit);
  }
  return "just now";
}

export function fmtBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n >= 10 || i === 0 ? Math.round(n) : n.toFixed(1)} ${units[i]}`;
}

export function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}


export interface MenuItem {
  label: string;
  hint?: string;
  active?: boolean;
  danger?: boolean;
  onPick: () => void;
}

let openMenuEl: HTMLElement | null = null;

export function closeMenu() {
  openMenuEl?.remove();
  openMenuEl = null;
}

/// A small dropdown under `anchor`: click or Enter picks, Escape or a click
/// outside closes, arrow keys move between items.
export function openMenu(anchor: HTMLElement, items: MenuItem[], label = "Menu") {
  closeMenu();
  const menu = el("div", "menu");
  menu.setAttribute("role", "menu");
  menu.setAttribute("aria-label", label);
  for (const item of items) {
    const btn = el("button", "menu-item" + (item.active ? " active" : "") + (item.danger ? " danger" : ""));
    btn.type = "button";
    btn.setAttribute("role", "menuitem");
    btn.innerHTML = `<span></span>${item.hint ? '<span class="menu-hint"></span>' : ""}`;
    (btn.firstChild as HTMLElement).textContent = item.label;
    if (item.hint) (btn.lastChild as HTMLElement).textContent = item.hint;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeMenu();
      item.onPick();
    });
    menu.appendChild(btn);
  }
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  const width = menu.offsetWidth;
  menu.style.top = `${r.bottom + 6}px`;
  menu.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - width - 8))}px`;
  openMenuEl = menu;
  const buttons = [...menu.querySelectorAll<HTMLButtonElement>(".menu-item")];
  (buttons.find((b) => b.classList.contains("active")) || buttons[0])?.focus();
  menu.addEventListener("keydown", (e) => {
    const i = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (e.key === "ArrowDown") buttons[(i + 1) % buttons.length].focus();
    else if (e.key === "ArrowUp") buttons[(i - 1 + buttons.length) % buttons.length].focus();
    else if (e.key === "Escape") {
      closeMenu();
      anchor.focus();
    } else return;
    e.preventDefault();
  });
  // Close on the next click anywhere else (registered after this click ends).
  setTimeout(() => {
    const away = (e: MouseEvent) => {
      if (openMenuEl === menu && !menu.contains(e.target as Node)) closeMenu();
      if (openMenuEl !== menu) document.removeEventListener("mousedown", away, true);
    };
    document.addEventListener("mousedown", away, true);
  });
}

/// localStorage that never throws (private windows, blocked storage).
export const prefs = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(`lazymeet.${key}`);
    } catch {
      return null;
    }
  },
  set(key: string, value: string) {
    try {
      localStorage.setItem(`lazymeet.${key}`, value);
    } catch {
      /* not persisted — fine */
    }
  },
};
