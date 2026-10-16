// LazyMeet frontend — Tauri commands + events (the backend runs in-process in
// Rust; there is no HTTP sidecar anymore). This module is the Tauri seam:
// everything else imports `call`, `listen`, `convertFileSrc`,
// `getCurrentWebview`, and `$` from here.
import { invoke, convertFileSrc as tauriConvertFileSrc } from "@tauri-apps/api/core";
import { listen as tauriListen } from "@tauri-apps/api/event";
import { getCurrentWebview as tauriGetCurrentWebview } from "@tauri-apps/api/webview";

// Backend failures arrive as a structured { code, kind, message } object;
// `call` re-throws them as an Error carrying the backend's status code, so
// the existing 404/409/422 checks keep working.
interface ApiError extends Error {
  status?: number;
}

export async function call(cmd: string, args?: Record<string, unknown>): Promise<any> {
  try {
    return await invoke(cmd, args);
  } catch (e) {
    const err: ApiError = new Error(e && e.message ? e.message : String(e));
    err.status = e && e.code;
    throw err;
  }
}

export const listen = (event: string, handler: (event: { payload: any }) => void) =>
  tauriListen(event, handler);
export const convertFileSrc = (path: string) => tauriConvertFileSrc(path);
export const getCurrentWebview = () => tauriGetCurrentWebview();

export const STAGES = [
  ["extracting", "Extracting audio"],
  ["transcribing", "Transcribing (Parakeet)"],
  ["diarizing", "Separating speakers"],
  ["identifying", "Identifying speakers"],
  ["summarizing", "Summarizing"],
];

// Ids are static and known-good; the return type stays `any` so the 200-odd
// call sites don't need per-element casts. Cross-file calls are still checked.
export const $ = (id: string): any => document.getElementById(id);
