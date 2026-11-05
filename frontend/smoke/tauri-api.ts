// Smoke-test stand-in for the @tauri-apps/api surface the app imports.
//
// `vite build --mode smoke` aliases "@tauri-apps/api/{core,event,webview}" to
// this module (see vite.config.ts), so the smoke test runs the real bundle
// against a fake backend instead of the Rust one. Unknown commands resolve to
// null, so a new command at boot shows up as a smoke failure rather than a
// crash in the real app.
import { config, models, speakers } from "./fixtures";
import { docsMeeting, docsMeetings, docsPeaks, DOCS_PREP_DURATION, silentWavUrl } from "./docs";

const hash = () => location.hash;

export const calls: { cmd: string; args?: Record<string, unknown> }[] = [];

export async function invoke(cmd: string, args?: Record<string, unknown>): Promise<any> {
  calls.push({ cmd, args });
  if (cmd === "get_setup") return { ready: true, items: [] };
  if (cmd === "list_meetings") return hash() === "#meeting" ? docsMeetings() : [];
  if (cmd === "get_meeting") return docsMeeting();
  if (cmd === "get_speakers") return speakers();
  if (cmd === "get_config" || cmd === "put_config") return config();
  if (cmd === "get_models") return models();
  if (cmd === "remove_model") return { ok: true };
  if (cmd === "create_meeting") return { id: `smoke-${calls.filter((c) => c.cmd === "create_meeting").length}` };
  if (cmd === "pick_folder") return "/Users/you/Movies/OBS";
  if (cmd === "list_audio_inputs") {
    return { inputs: [{ name: "MacBook Pro Microphone", default: true }, { name: "USB Mic", default: false }], system_audio: true };
  }
  if (cmd === "start_recording") return { recording: true, sources: args!.system_audio ? ["mic", "system"] : ["mic"], warnings: [] };
  if (cmd === "stop_recording") return { path: "/Users/you/Library/Application Support/LazyMeet/recordings/2026-11-06 10-00-00/recording.m4a" };
  if (cmd === "leftover_recordings") {
    return hash() === "#meeting" ? [] : [{ name: "2026-11-05 16-30-12", path: "/tmp/rec/recording.m4a" }];
  }
  if (cmd === "ask_meeting") {
    return { q: args!.question, a: "Launch holds at August 14 [00:28].\n\nLegal reviews first [0:52].", at: "2026-11-06T10:00:00Z" };
  }
  if (cmd === "ask_all") {
    return {
      q: args!.question,
      a: "M2 set the relaunch for August 14 [M2 00:28]; the roadmap moved it once [M1 12:05-12:40].",
      sources: [
        { label: "M1", id: "docs-roadmap", title: "Q3 roadmap planning" },
        { label: "M2", id: "smoke-meeting", title: "Weekly sync" },
      ],
    };
  }
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
    if (hash() === "#docs-prepare") return { duration: DOCS_PREP_DURATION, peaks: docsPeaks() };
    return { duration: 3, peaks: Array.from({ length: 64 }, (_, i) => Math.abs(Math.sin(i))) };
  }
  return null;
}

export function convertFileSrc(path: string): string {
  if (hash() === "#docs-prepare") return silentWavUrl();
  return "asset://localhost/" + path;
}

export async function listen(_event: string, _handler: (event: any) => void): Promise<() => void> {
  return () => {};
}

export function getCurrentWebview() {
  return { onDragDropEvent: async (_handler: (event: any) => void) => () => {} };
}
