// --------------------------------------------------------------------------
// Export formats — one meeting (save dialog) or several (folder picker).
// --------------------------------------------------------------------------
import { call } from "./core";
import { openMenu, plural, prefs, showToast } from "./helpers";

export const EXPORT_FORMATS = [
  { ext: "md", label: "Markdown", hint: ".md" },
  { ext: "docx", label: "Word", hint: ".docx" },
  { ext: "pdf", label: "PDF", hint: ".pdf" },
  { ext: "txt", label: "Plain text", hint: ".txt" },
  { ext: "srt", label: "Subtitles (SRT)", hint: ".srt" },
  { ext: "vtt", label: "Subtitles (WebVTT)", hint: ".vtt" },
  { ext: "json", label: "JSON", hint: ".json" },
];

export function lastExportFormat(): string {
  const saved = prefs.get("exportFormat");
  return EXPORT_FORMATS.some((f) => f.ext === saved) ? (saved as string) : "md";
}

/// Show the format menu under `anchor`; the picked format is remembered and
/// handed to `run`.
export function pickExportFormat(anchor: HTMLElement, run: (ext: string) => void) {
  const last = lastExportFormat();
  openMenu(
    anchor,
    EXPORT_FORMATS.map((f) => ({
      label: f.label,
      hint: f.hint,
      active: f.ext === last,
      onPick: () => {
        prefs.set("exportFormat", f.ext);
        run(f.ext);
      },
    })),
    "Export format"
  );
}

export async function exportMeeting(meetingId: string, format: string) {
  try {
    const path = await call("export_meeting", { meeting_id: meetingId, format });
    if (path) showToast(`Exported to ${path}`, 4000, true);
  } catch (e) {
    showToast(`Export failed: ${e.message}`);
  }
}

export async function exportMeetings(meetingIds: string[], format: string) {
  try {
    const paths = await call("export_meetings", { meeting_ids: meetingIds, format });
    if (paths) showToast(`Exported ${plural(paths.length, "meeting", "meetings")}`, 4000, true);
  } catch (e) {
    showToast(`Export failed: ${e.message}`);
  }
}
