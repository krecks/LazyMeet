//! Meeting exports: Markdown, plain text, SRT/VTT subtitles, JSON, Word and
//! PDF. Everything renders from the public record (what the UI shows), so an
//! export never leaks voiceprints or internal markers.
//!
//! Word and PDF share one block model (`blocks`), so the two documents carry
//! the same content in the same order.

use std::io::Cursor;

use serde_json::Value;

use crate::store::{self, Record};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Md,
    Txt,
    Srt,
    Vtt,
    Json,
    Docx,
    Pdf,
}

impl Format {
    pub const ALL: [Format; 7] = [
        Format::Md,
        Format::Txt,
        Format::Srt,
        Format::Vtt,
        Format::Json,
        Format::Docx,
        Format::Pdf,
    ];

    pub fn parse(s: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.ext() == s.trim().to_ascii_lowercase())
    }

    pub fn ext(self) -> &'static str {
        match self {
            Format::Md => "md",
            Format::Txt => "txt",
            Format::Srt => "srt",
            Format::Vtt => "vtt",
            Format::Json => "json",
            Format::Docx => "docx",
            Format::Pdf => "pdf",
        }
    }

    /// The save dialog's filter name.
    pub fn label(self) -> &'static str {
        match self {
            Format::Md => "Markdown",
            Format::Txt => "Plain text",
            Format::Srt => "SubRip subtitles",
            Format::Vtt => "WebVTT subtitles",
            Format::Json => "JSON",
            Format::Docx => "Word document",
            Format::Pdf => "PDF",
        }
    }
}

/// The exported bytes and a suggested file name, from a public record.
pub fn export(record: &Record, format: Format) -> Result<(Vec<u8>, String), String> {
    let bytes = match format {
        Format::Md => to_markdown(record).into_bytes(),
        Format::Txt => to_text(record).into_bytes(),
        Format::Srt => to_subtitles(record, false).into_bytes(),
        Format::Vtt => to_subtitles(record, true).into_bytes(),
        Format::Json => to_json(record)?,
        Format::Docx => to_docx(record)?,
        Format::Pdf => to_pdf(record)?,
    };
    Ok((bytes, format!("{}.{}", file_stem(record), format.ext())))
}

/// A file-system-safe name from the title: letters (any script), digits,
/// `_`, `-`; spaces become `-`. Falls back to the id.
pub fn file_stem(record: &Record) -> String {
    let safe: String = title(record)
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == ' ')
        .collect::<String>()
        .trim()
        .replace(' ', "-");
    if safe.is_empty() {
        record.get("id").and_then(|v| v.as_str()).unwrap_or("meeting").to_string()
    } else {
        safe
    }
}

fn title(record: &Record) -> &str {
    record
        .get("title")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| record.get("filename").and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
        .unwrap_or("Meeting")
}

fn text_field<'a>(record: &'a Record, key: &str) -> Option<&'a str> {
    record.get(key).and_then(|v| v.as_str()).map(str::trim).filter(|s| !s.is_empty())
}

/// One transcript line, with the speaker's display name resolved.
struct Line<'a> {
    name: &'a str,
    start: f64,
    end: f64,
    text: &'a str,
}

fn lines(record: &Record) -> Vec<Line<'_>> {
    let speakers = record.get("speakers").and_then(|v| v.as_object());
    let Some(segments) = record.get("segments").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    segments
        .iter()
        .map(|seg| {
            let id = seg.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
            let name = speakers.and_then(|m| m.get(id)).and_then(|v| v.as_str()).unwrap_or(id);
            let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Line {
                name,
                start,
                end: seg.get("end").and_then(|v| v.as_f64()).unwrap_or(start).max(start),
                text: seg.get("text").and_then(|v| v.as_str()).unwrap_or("").trim(),
            }
        })
        .collect()
}

/// (short, detail) pairs of a notes list; a detail that only repeats the
/// short line is dropped, as in the UI.
fn items(record: &Record, key: &str) -> Vec<(String, String)> {
    store::note_items(record.get(key))
        .iter()
        .map(|item| {
            let field = |k: &str| item.get(k).and_then(|v| v.as_str()).unwrap_or("").trim();
            let (short, detail) = (field("short"), field("detail"));
            let detail = if detail == short { "" } else { detail };
            (short.to_string(), detail.to_string())
        })
        .filter(|(short, _)| !short.is_empty())
        .collect()
}

/// `mm:ss`, or `hh:mm:ss` past the hour.
pub fn fmt_ts(seconds: f64) -> String {
    let total = seconds.max(0.0) as i64;
    let (m, s) = (total / 60, total % 60);
    let (h, m) = (m / 60, m % 60);
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

// --------------------------------------------------------------------------
// Markdown and plain text
// --------------------------------------------------------------------------

pub fn to_markdown(record: &Record) -> String {
    let mut out = format!("# {}\n\n", title(record));
    if let Some(summary) = text_field(record, "summary") {
        out.push_str(&format!("## Summary\n\n{summary}\n\n"));
    }
    if let Some(detail) = text_field(record, "summary_long") {
        out.push_str(&format!("## In detail\n\n{detail}\n\n"));
    }
    for (key, heading, mark) in
        [("takeaways", "Key takeaways", "- "), ("action_items", "Action items", "- [ ] ")]
    {
        let list = items(record, key);
        if list.is_empty() {
            continue;
        }
        out.push_str(&format!("## {heading}\n\n"));
        for (short, detail) in &list {
            out.push_str(&format!("{mark}{short}\n"));
            if !detail.is_empty() {
                // Indented continuation keeps the list item intact.
                out.push_str(&format!("  {detail}\n"));
            }
        }
        out.push('\n');
    }
    out.push_str("## Transcript\n\n");
    for line in lines(record) {
        out.push_str(&format!("**{}** [{}]: {}\n\n", line.name, fmt_ts(line.start), line.text));
    }
    out
}

/// The notes and transcript as plain text — the same shape as the app's
/// "Copy" buttons, for pasting into mail or chat.
pub fn to_text(record: &Record) -> String {
    let mut out = format!("{}\n\n", title(record));
    if let Some(summary) = text_field(record, "summary") {
        out.push_str(&format!("{summary}\n\n"));
    }
    if let Some(detail) = text_field(record, "summary_long") {
        out.push_str(&format!("{detail}\n\n"));
    }
    for (key, heading, mark) in
        [("takeaways", "Key takeaways:", "- "), ("action_items", "Action items:", "- [ ] ")]
    {
        let list = items(record, key);
        if list.is_empty() {
            continue;
        }
        out.push_str(&format!("{heading}\n"));
        for (short, detail) in &list {
            out.push_str(&format!("{mark}{short}\n"));
            if !detail.is_empty() {
                out.push_str(&format!("  {detail}\n"));
            }
        }
        out.push('\n');
    }
    let transcript = lines(record);
    if !transcript.is_empty() {
        out.push_str("Transcript:\n");
        for line in transcript {
            out.push_str(&format!("{} [{}]: {}\n", line.name, fmt_ts(line.start), line.text));
        }
    }
    out
}

fn to_json(record: &Record) -> Result<Vec<u8>, String> {
    let mut public = record.clone();
    // Version snapshots are the app's undo history, not part of the meeting.
    public.remove("versions");
    serde_json::to_vec_pretty(&Value::Object(public)).map_err(|e| e.to_string())
}

// --------------------------------------------------------------------------
// Subtitles
// --------------------------------------------------------------------------

/// Readable cues: at most ~7 s and two subtitle lines of text each.
const CUE_MAX_SECS: f64 = 7.0;
const CUE_MAX_CHARS: usize = 84;

/// One subtitle cue.
#[derive(Debug, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
    pub text: String,
}

/// Transcript segments as subtitle cues: a long segment is split at word
/// boundaries into pieces of at most `CUE_MAX_CHARS`, and each piece gets a
/// share of the segment's time proportional to its length (the transcript
/// has no word timings).
pub fn cues(record: &Record) -> Vec<Cue> {
    let mut out = Vec::new();
    for line in lines(record) {
        if line.text.is_empty() {
            continue;
        }
        let span = line.end - line.start;
        let chars = line.text.chars().count();
        let parts = if span <= CUE_MAX_SECS && chars <= CUE_MAX_CHARS {
            vec![line.text.to_string()]
        } else {
            let by_time = (span / CUE_MAX_SECS).ceil() as usize;
            let by_chars = chars.div_ceil(CUE_MAX_CHARS);
            split_words(line.text, by_time.max(by_chars).max(1))
        };
        let total: usize = parts.iter().map(|p| p.chars().count()).sum::<usize>().max(1);
        let mut at = line.start;
        let n = parts.len();
        for (i, part) in parts.into_iter().enumerate() {
            let share = span * part.chars().count() as f64 / total as f64;
            let end = if i + 1 == n { line.end } else { at + share };
            out.push(Cue { start: at, end, speaker: line.name.to_string(), text: part });
            at = end;
        }
    }
    out
}

/// Split `text` at spaces into `n` pieces of roughly equal length (never
/// more pieces than words).
fn split_words(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = n.min(words.len()).max(1);
    let target = text.chars().count() as f64 / n as f64;
    let mut parts = Vec::with_capacity(n);
    let mut cur = String::new();
    for (i, word) in words.iter().enumerate() {
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
        let words_left = words.len() - i - 1;
        let parts_left = n - parts.len() - 1;
        let full = cur.chars().count() as f64 >= target;
        // Close the piece once it's long enough — but leave a word for each
        // remaining piece, and keep the last piece open to the end.
        if parts_left > 0 && (full || words_left == parts_left) {
            parts.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

fn fmt_cue_ts(seconds: f64, vtt: bool) -> String {
    let ms_total = (seconds.max(0.0) * 1000.0).round() as u64;
    let (s_total, ms) = (ms_total / 1000, ms_total % 1000);
    let (h, m, s) = (s_total / 3600, (s_total / 60) % 60, s_total % 60);
    let sep = if vtt { '.' } else { ',' };
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

pub fn to_subtitles(record: &Record, vtt: bool) -> String {
    let mut out = String::new();
    if vtt {
        out.push_str("WEBVTT\n\n");
    }
    for (i, cue) in cues(record).iter().enumerate() {
        let (a, b) = (fmt_cue_ts(cue.start, vtt), fmt_cue_ts(cue.end, vtt));
        if vtt {
            // <v Name> is WebVTT's voice tag; players show it as the speaker.
            let name = cue.speaker.replace(['<', '>', '&'], "");
            out.push_str(&format!("{a} --> {b}\n<v {name}>{}\n\n", vtt_escape(&cue.text)));
        } else {
            out.push_str(&format!("{}\n{a} --> {b}\n{}: {}\n\n", i + 1, cue.speaker, cue.text));
        }
    }
    out
}

fn vtt_escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// --------------------------------------------------------------------------
// Word and PDF: one block model
// --------------------------------------------------------------------------

enum Block {
    Title(String),
    Meta(String),
    Heading(String),
    Para(String),
    Item { short: String, detail: String, checkbox: bool },
    Line { name: String, ts: String, text: String },
}

fn blocks(record: &Record) -> Vec<Block> {
    let mut out = vec![Block::Title(title(record).to_string())];
    if let Some(date) = record.get("created_at").and_then(|v| v.as_str()) {
        let day = date.get(..10).unwrap_or(date);
        let duration = lines(record).iter().map(|l| l.end).fold(0.0, f64::max);
        let meta =
            if duration > 0.0 { format!("{day} · {}", fmt_ts(duration)) } else { day.to_string() };
        out.push(Block::Meta(meta));
    }
    if let Some(summary) = text_field(record, "summary") {
        out.push(Block::Heading("Summary".into()));
        out.push(Block::Para(summary.to_string()));
    }
    if let Some(detail) = text_field(record, "summary_long") {
        out.push(Block::Heading("In detail".into()));
        for para in detail.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
            out.push(Block::Para(para.to_string()));
        }
    }
    for (key, heading, checkbox) in
        [("takeaways", "Key takeaways", false), ("action_items", "Action items", true)]
    {
        let list = items(record, key);
        if list.is_empty() {
            continue;
        }
        out.push(Block::Heading(heading.into()));
        for (short, detail) in list {
            out.push(Block::Item { short, detail, checkbox });
        }
    }
    let transcript = lines(record);
    if !transcript.is_empty() {
        out.push(Block::Heading("Transcript".into()));
        for line in transcript {
            out.push(Block::Line {
                name: line.name.to_string(),
                ts: fmt_ts(line.start),
                text: line.text.to_string(),
            });
        }
    }
    out
}

fn to_docx(record: &Record) -> Result<Vec<u8>, String> {
    use docx_rs::{Docx, Paragraph, Run, RunFonts};

    // Sizes are half-points.
    let run = |text: &str| Run::new().add_text(text);
    let mut doc = Docx::new().default_fonts(RunFonts::new().ascii("Calibri").hi_ansi("Calibri"));
    for block in blocks(record) {
        let para = match block {
            Block::Title(t) => Paragraph::new().add_run(run(&t).bold().size(36)).outline_lvl(0),
            Block::Meta(m) => Paragraph::new().add_run(run(&m).color("666666").size(20)),
            Block::Heading(h) => Paragraph::new().add_run(run(&h).bold().size(28)).outline_lvl(1),
            Block::Para(p) => Paragraph::new().add_run(run(&p).size(22)),
            Block::Item { short, detail, checkbox } => {
                let mark = if checkbox { "☐ " } else { "• " };
                let mut p = Paragraph::new().add_run(run(&format!("{mark}{short}")).size(22));
                if !detail.is_empty() {
                    p = p.add_run(Run::new().add_break(docx_rs::BreakType::TextWrapping));
                    p = p.add_run(run(&detail).color("555555").size(20));
                }
                p.indent(Some(360), Some(docx_rs::SpecialIndentType::Hanging(240)), None, None)
            }
            Block::Line { name, ts, text } => Paragraph::new()
                .add_run(run(&name).bold().size(22))
                .add_run(run(&format!("  {ts}")).color("888888").size(18))
                .add_run(Run::new().add_break(docx_rs::BreakType::TextWrapping))
                .add_run(run(&text).size(22)),
        };
        doc = doc.add_paragraph(para);
    }
    let mut buf = Cursor::new(Vec::new());
    doc.build().pack(&mut buf).map_err(|e| format!("could not build the Word file: {e}"))?;
    Ok(buf.into_inner())
}

// Noto Sans covers Latin, Greek and Cyrillic — every language Parakeet
// transcribes. krilla embeds only the glyphs a document uses.
static NOTO_REGULAR: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");
static NOTO_BOLD: &[u8] = include_bytes!("../assets/fonts/NotoSans-Bold.ttf");

struct Face {
    font: krilla::text::Font,
    data: &'static [u8],
}

impl Face {
    fn load(data: &'static [u8]) -> Result<Face, String> {
        let font = krilla::text::Font::new(data.into(), 0).ok_or("bundled font is unreadable")?;
        Ok(Face { font, data })
    }

    /// Advance width of `text` at `size` points (no kerning — wrapping keeps
    /// a small margin for it).
    fn width(&self, text: &str, size: f32) -> f32 {
        use skrifa::instance::{LocationRef, Size};
        use skrifa::MetadataProvider;
        let Ok(font) = skrifa::FontRef::new(self.data) else { return 0.0 };
        let cmap = font.charmap();
        let metrics = font.glyph_metrics(Size::new(size), LocationRef::default());
        text.chars()
            .map(|c| cmap.map(c).and_then(|g| metrics.advance_width(g)).unwrap_or(size * 0.5))
            .sum()
    }

    /// Greedy word wrap to `max` points; a word wider than a line is broken
    /// by characters.
    fn wrap(&self, text: &str, size: f32, max: f32) -> Vec<String> {
        let fits = |s: &str| self.width(s, size) * 1.02 <= max;
        let mut lines = Vec::new();
        for para in text.split('\n') {
            let mut cur = String::new();
            for word in para.split_whitespace() {
                let candidate =
                    if cur.is_empty() { word.to_string() } else { format!("{cur} {word}") };
                if fits(&candidate) {
                    cur = candidate;
                    continue;
                }
                if !cur.is_empty() {
                    lines.push(std::mem::take(&mut cur));
                }
                if fits(word) {
                    cur = word.to_string();
                } else {
                    for c in word.chars() {
                        let mut next = cur.clone();
                        next.push(c);
                        if !fits(&next) && !cur.is_empty() {
                            lines.push(std::mem::take(&mut cur));
                            next = c.to_string();
                        }
                        cur = next;
                    }
                }
            }
            lines.push(cur);
        }
        lines
    }
}

fn to_pdf(record: &Record) -> Result<Vec<u8>, String> {
    use krilla::color::rgb;
    use krilla::geom::{PathBuilder, Point, Rect};
    use krilla::page::PageSettings;
    use krilla::paint::{Fill, Stroke};
    use krilla::text::TextDirection;

    const W: f32 = 595.0; // A4 in points
    const H: f32 = 842.0;
    const MARGIN: f32 = 56.0;
    const TEXT_W: f32 = W - 2.0 * MARGIN;

    let regular = Face::load(NOTO_REGULAR)?;
    let bold = Face::load(NOTO_BOLD)?;
    let ink = rgb::Color::new(28, 28, 30);
    let muted = rgb::Color::new(110, 110, 116);

    // Lay everything out first, one row per output line.
    struct Row<'a> {
        face: &'a Face,
        size: f32,
        color: rgb::Color,
        x: f32,
        text: String,
        /// Space above the line; None = same baseline as the previous row
        /// (a list marker and its item's first line).
        gap: Option<f32>,
        /// Draw an empty checkbox square instead of text (Noto Sans has no ☐).
        checkbox: bool,
    }
    fn push<'a>(
        rows: &mut Vec<Row<'a>>,
        face: &'a Face,
        size: f32,
        color: rgb::Color,
        x: f32,
        text: &str,
        gap: f32,
    ) {
        for (i, line) in face.wrap(text, size, TEXT_W - (x - MARGIN)).into_iter().enumerate() {
            let gap = Some(if i == 0 { gap } else { 0.0 });
            rows.push(Row { face, size, color, x, text: line, gap, checkbox: false });
        }
    }
    let mut rows: Vec<Row> = Vec::new();
    for block in blocks(record) {
        let r = &mut rows;
        match block {
            Block::Title(t) => push(r, &bold, 20.0, ink, MARGIN, &t, 0.0),
            Block::Meta(m) => push(r, &regular, 9.5, muted, MARGIN, &m, 2.0),
            Block::Heading(h) => push(r, &bold, 13.0, ink, MARGIN, &h, 14.0),
            Block::Para(p) => push(r, &regular, 10.5, ink, MARGIN, &p, 5.0),
            Block::Item { short, detail, checkbox } => {
                push(r, &regular, 10.5, ink, MARGIN, "•", 4.0);
                if let Some(marker) = r.last_mut() {
                    marker.checkbox = checkbox;
                }
                let first = r.len();
                push(r, &regular, 10.5, ink, MARGIN + 14.0, &short, 0.0);
                r[first].gap = None;
                if !detail.is_empty() {
                    push(r, &regular, 9.5, muted, MARGIN + 14.0, &detail, 1.0);
                }
            }
            Block::Line { name, ts, text } => {
                push(r, &bold, 10.0, ink, MARGIN, &format!("{name}  ·  {ts}"), 7.0);
                push(r, &regular, 10.5, ink, MARGIN, &text, 1.0);
            }
        }
    }

    let mut doc = krilla::Document::new();
    doc.set_metadata(krilla::metadata::Metadata::new().title(title(record).to_string()));
    let a4 = || PageSettings::from_wh(W, H).expect("A4 is a valid size");
    let mut page = doc.start_page_with(a4());
    let mut surface = page.surface();
    let mut y = MARGIN;
    for row in &rows {
        if let Some(gap) = row.gap {
            let advance = gap + row.size * 1.35;
            if y + advance > H - MARGIN {
                surface.finish();
                page.finish();
                page = doc.start_page_with(a4());
                surface = page.surface();
                y = MARGIN;
            } else {
                y += gap;
            }
            y += row.size * 1.35;
        }
        if row.checkbox {
            let side = row.size * 0.68;
            let mut path = PathBuilder::new();
            if let Some(rect) = Rect::from_xywh(row.x, y - side - 0.5, side, side) {
                path.push_rect(rect);
            }
            if let Some(path) = path.finish() {
                surface.set_fill(None);
                surface.set_stroke(Some(Stroke {
                    paint: row.color.into(),
                    width: 0.8,
                    ..Default::default()
                }));
                surface.draw_path(&path);
                surface.set_stroke(None);
            }
            continue;
        }
        surface.set_fill(Some(Fill { paint: row.color.into(), ..Default::default() }));
        surface.draw_text(
            Point::from_xy(row.x, y),
            row.face.font.clone(),
            row.size,
            &row.text,
            false,
            TextDirection::Auto,
        );
    }
    surface.finish();
    page.finish();
    doc.finish().map_err(|e| format!("could not build the PDF: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_words_balances_and_keeps_every_word() {
        let text = "one two three four five six seven eight nine ten";
        let parts = split_words(text, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts.join(" "), text);
        assert_eq!(split_words("a b", 5), vec!["a", "b"]);
        assert_eq!(split_words("single", 1), vec!["single"]);
    }

    #[test]
    fn cue_timestamps() {
        assert_eq!(fmt_cue_ts(3661.5, false), "01:01:01,500");
        assert_eq!(fmt_cue_ts(0.0, true), "00:00:00.000");
    }
}
