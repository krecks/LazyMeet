// Smoke-test stand-in for the @tauri-apps/api surface the app imports.
//
// `vite build --mode smoke` aliases "@tauri-apps/api/{core,event,webview}" to
// this module (see vite.config.ts), so the smoke test runs the real bundle
// against a fake backend instead of the Rust one. Unknown commands resolve to
// null, so a new command at boot shows up as a smoke failure rather than a
// crash in the real app.
import { config, models } from "./fixtures";

export const calls: { cmd: string; args?: Record<string, unknown> }[] = [];

export async function invoke(cmd: string, args?: Record<string, unknown>): Promise<any> {
  calls.push({ cmd, args });
  if (cmd === "get_setup") return { ready: true, items: [] };
  if (cmd === "list_meetings") return [];
  if (cmd === "get_speakers") return [];
  if (cmd === "get_config" || cmd === "put_config") return config();
  if (cmd === "get_models") return models();
  if (cmd === "remove_model") return { ok: true };
  if (cmd === "get_data_info") {
    return {
      data_dir: "/Users/you/Library/Application Support/LazyMeet",
      total_bytes: 9.1e9,
      models_bytes: 8.6e9,
      meetings_bytes: 2.1e6,
      uploads_bytes: 4.9e8,
      meeting_count: 5,
    };
  }
  if (cmd === "probe_media") {
    return { duration: 3, peaks: Array.from({ length: 64 }, (_, i) => Math.abs(Math.sin(i))) };
  }
  return null;
}

export function convertFileSrc(path: string): string {
  return "asset://localhost/" + path;
}

export async function listen(_event: string, _handler: (event: any) => void): Promise<() => void> {
  return () => {};
}

export function getCurrentWebview() {
  return { onDragDropEvent: async (_handler: (event: any) => void) => () => {} };
}
