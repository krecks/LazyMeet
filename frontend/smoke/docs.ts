// Content for the README screenshots (`#meeting`, `#docs-prepare`): a
// believable finished meeting in the current notes shape, and a 28-minute
// audio file for the prepare screen. Rendered by the real bundle against
// the smoke stub, so the screenshots always show the current UI.

const ago = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString();

export const DOCS_MEETING_ID = "docs-relaunch";

export function docsMeetings() {
  return [
    // Still processing: no notes yet, so it still carries its file name.
    { id: "docs-nordwind", filename: "customer-interview-nordwind.mp4", title: "customer-interview-nordwind.mp4", status: "transcribing", created_at: ago(4), tags: ["Nordwind"] },
    { id: DOCS_MEETING_ID, filename: "2026-07-10 13-15-02.mkv", title: "Website relaunch go-live check", status: "done", created_at: ago(120), duration: 1712, tags: ["Relaunch", "Weekly"] },
    { id: "docs-roadmap", filename: "2026-07-07 10-00-41.mkv", title: "Q3 roadmap planning", status: "done", created_at: ago(3 * 1440), duration: 2830, tags: ["Weekly"] },
    { id: "docs-support", filename: "support-handover.m4a", title: "Support handover to the June team", status: "done", created_at: ago(14 * 1440), duration: 960 },
  ];
}

const item = (short: string, detail: string) => ({ short, detail });

export function docsMeeting() {
  const segments = [
    { start: 12, end: 19, speaker: "spk_0", text: "Alright, quick status round on the relaunch. Ben, where did we land with the translation hand-off?" },
    { start: 28, end: 41, speaker: "spk_1", text: "English and Spanish are done and already in the CMS. German is at about eighty percent — the agency gets the last batch on Wednesday, so we're still inside the buffer." },
    { start: 52, end: 66, speaker: "spk_2", text: "One thing from my side: legal still wants to review the imprint and the cookie banner copy. That usually takes them two days, so please don't treat the German pages as final before Friday." },
    { start: 75, end: 88, speaker: "spk_0", text: "Good catch. Then August 14 still holds, but I want a go/no-go on the 11th. Ben, the pricing page — is that the last design task?" },
    { start: 92, end: 104, speaker: "spk_1", text: "Yes. Mobile spacing and the currency toggle, I'll have both fixed by tomorrow. After the accessibility audit is merged we freeze the component library." },
  ];
  return {
    id: DOCS_MEETING_ID,
    filename: "2026-07-10 13-15-02.mkv",
    title: "Website relaunch go-live check",
    status: "done",
    created_at: ago(120),
    tags: ["Relaunch", "Weekly"],
    summary:
      "The team confirmed August 14 as the launch date for the relaunched website, with a go/no-go call on August 11. German translations and a legal review of the imprint and cookie banner are the open threads, so the German pages are final on Friday at the earliest.",
    summary_long:
      "Anna opened with the translation hand-off. Ben reported English and Spanish complete in the CMS and German at about 80 %, with the agency receiving the last batch on Wednesday — still inside the planned buffer.\n\nSpeaker 3 flagged that legal has yet to review the imprint and the cookie banner copy, which usually takes two days. The team agreed not to treat the German pages as final before Friday, and Anna kept August 14 as the launch date on the condition of a go/no-go call on August 11.\n\nOn design, the pricing page is the last open task: Ben fixes mobile spacing and the currency toggle by tomorrow, and the component library freezes once the accessibility audit is merged.",
    takeaways: [
      item("Launch holds at August 14.", "Confirmed by a go/no-go call on August 11, after Monday's load test."),
      item("German pages can't be final before Friday.", "Legal still reviews the imprint and cookie banner copy, which takes about two days."),
      item("The pricing page is the last design task.", "The component library freezes once the accessibility audit is merged."),
    ],
    action_items: [
      item("Ben: fix mobile spacing and the currency toggle on the pricing page", "Due tomorrow; blocks the component-library freeze."),
      item("Ben: run the load test against staging on Monday", "Beta traffic profile plus video assets; feeds the go/no-go call."),
      item("Anna: add the legal review to the launch checklist and send the go/no-go invite", "Invite for August 11."),
    ],
    timings: { extract: 4.1, transcribe: 124.2, diarize: 6.8, identify: 2.4, summarize: 25.3, total: 162.8 },
    segments,
    speakers: { spk_0: "Anna Weber", spk_1: "Ben Ortiz", spk_2: "Speaker 3" },
    recognized: ["spk_0", "spk_1"],
    confirmed: ["spk_0"],
    has_audio: false,
    versions: [],
    asr_model: "parakeet-tdt-0.6b-v3",
  };
}

// The prepare screen's recording: 28:32 of speech-like peaks with a break.
export const DOCS_PREP_DURATION = 1712.4;

export function docsPeaks(n = 1200) {
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  return Array.from({ length: n }, (_, i) => {
    const t = (i / n) * DOCS_PREP_DURATION;
    if (t > 1312 && t < 1385) return 0.04 + rand() * 0.05; // coffee break
    const pause = rand() < 0.05 ? 0.15 : 1;
    return Math.min(1, (0.35 + rand() * 0.55) * pause);
  });
}

// A silent WAV of exactly that duration (8 kHz, 8-bit: ~14 MB), so the
// player shows the real length instead of an error.
let silentUrl: string | null = null;
export function silentWavUrl(): string {
  if (silentUrl) return silentUrl;
  const rate = 8000;
  const n = Math.round(DOCS_PREP_DURATION * rate);
  const buf = new Uint8Array(44 + n);
  const view = new DataView(buf.buffer);
  const str = (o: number, s: string) => [...s].forEach((c, i) => (buf[o + i] = c.charCodeAt(0)));
  str(0, "RIFF");
  view.setUint32(4, 36 + n, true);
  str(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate, true);
  view.setUint16(32, 1, true);
  view.setUint16(34, 8, true);
  str(36, "data");
  view.setUint32(40, n, true);
  buf.fill(128, 44);
  silentUrl = URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
  return silentUrl;
}
