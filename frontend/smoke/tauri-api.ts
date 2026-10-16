// Smoke-test stand-in for the @tauri-apps/api surface the app imports.
//
// `vite build --mode smoke` aliases "@tauri-apps/api/{core,event,webview}" to
// this module (see vite.config.ts), so the smoke test runs the real bundle
// against a fake backend instead of the Rust one. Unknown commands resolve to
// null, so a new command at boot shows up as a smoke failure rather than a
// crash in the real app.
export async function invoke(cmd: string, _args?: Record<string, unknown>): Promise<any> {
  if (cmd === "get_setup") return { ready: true, items: [] };
  if (cmd === "list_meetings") return [];
  if (cmd === "get_speakers") return [];
  if (cmd === "get_config") return { values: {} };
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
