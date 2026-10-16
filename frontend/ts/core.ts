// LazyMeet frontend — vanilla JS over Tauri commands + events (the backend
// runs in-process in Rust; there is no HTTP sidecar anymore).

const tauri = window.__TAURI__;

// Invoke a backend command. Command errors arrive as a structured
// { code, kind, message } object; surface it as an Error with `.status` so
// existing 404/409/422 checks keep working.
async function call(cmd: string, args?: Record<string, unknown>): Promise<any> {
  try {
    return await tauri.core.invoke(cmd, args);
  } catch (e) {
    const err: ApiError = new Error(e && e.message ? e.message : String(e));
    err.status = e && e.code;
    throw err;
  }
}

const listen = (event: string, handler: (event: TauriEvent) => void) =>
  tauri.event.listen(event, handler);
const convertFileSrc = (path: string) => tauri.core.convertFileSrc(path);

const STAGES = [
  ["extracting", "Extracting audio"],
  ["transcribing", "Transcribing (Parakeet)"],
  ["diarizing", "Separating speakers"],
  ["identifying", "Identifying speakers"],
  ["summarizing", "Summarizing"],
];

// Ids are static and known-good; the return type stays `any` so the 200-odd
// call sites don't need per-element casts. Cross-file calls are still checked.
const $ = (id: string): any => document.getElementById(id);

// ---------------------------------------------------------------------------
// State. Renders are imperative; `editing` is the anti-clobber guard: while an
// inline edit is open, fresh meeting data is stashed in `pendingMeeting`
// instead of re-rendering (which would destroy the user's half-typed edit).
// ---------------------------------------------------------------------------
let activeId = null;
let uploading = false;
let meetings = []; // cached list — search re-filters without refetching
let searchQuery = "";
let currentMeeting = null;
let editing = null; // "title" | "summary" | "seg:3" | "takeaways:1" | … | null
let pendingMeeting = null;
let viewingVersion = null; // history version number rendered read-only, or null for current

