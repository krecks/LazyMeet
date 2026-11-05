// --------------------------------------------------------------------------
// Notes options — the template and notes language, picked per recording on
// the prepare screen and again when regenerating notes.
// --------------------------------------------------------------------------
import { call } from "./core";
import { el, openPopover, showToast } from "./helpers";
import { fillLanguageSelect } from "./settings";

export interface NotesChoice {
  notes_template: string;
  notes_language: string;
}

/// Fill a template select and a notes-language select, starting on `current`
/// (falling back to the saved settings).
export async function fillNotesSelects(
  template: HTMLSelectElement,
  language: HTMLSelectElement,
  current: Partial<NotesChoice> = {}
) {
  let cfg;
  try {
    cfg = await call("get_config");
  } catch {
    return;
  }
  template.innerHTML = "";
  for (const t of cfg.notes_templates || []) template.appendChild(new Option(t.label, t.id));
  template.value = current.notes_template || cfg.values.NOTES_TEMPLATE || "general";
  fillLanguageSelect(
    language,
    cfg.languages,
    current.notes_language || cfg.values.SUMMARY_LANGUAGE || "auto",
    null,
    "Match the meeting"
  );
}

export function readNotesSelects(template: HTMLSelectElement, language: HTMLSelectElement): NotesChoice {
  return { notes_template: template.value || "general", notes_language: language.value || "auto" };
}

/// "Regenerate notes": a small panel with both choices and a button.
export async function openRegenPanel(anchor: HTMLElement, meeting, run: (choice: NotesChoice) => void) {
  const panel = el("div", "menu regen-panel");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Regenerate notes");
  const head = el("p", "regen-head");
  head.textContent = "Write the notes again";
  const field = (label: string, select: HTMLSelectElement) => {
    const wrap = el("label", "regen-field");
    const span = el("span", "field-label");
    span.textContent = label;
    wrap.append(span, select);
    return wrap;
  };
  const template = document.createElement("select");
  const language = document.createElement("select");
  const go = el("button", "btn small");
  go.type = "button";
  go.textContent = "Regenerate";
  const note = el("p", "muted small");
  note.textContent = "The current notes stay in the version history.";
  panel.append(head, field("Kind of meeting", template), field("Notes language", language), note, go);
  openPopover(anchor, panel);
  await fillNotesSelects(template, language, meeting);
  template.focus();
  go.addEventListener("click", () => {
    const choice = readNotesSelects(template, language);
    panel.remove();
    try {
      run(choice);
    } catch (e) {
      showToast(`Couldn't start: ${e.message}`);
    }
  });
}
