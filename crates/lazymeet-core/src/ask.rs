//! Ask questions about one meeting or all of them, answered by the local
//! summary model from the transcript.
//!
//! A meeting that fits the model's context goes in whole. A longer one — or
//! every meeting at once — is cut into overlapping windows of transcript
//! lines, and the windows that best match the question (BM25) fill the
//! context. No embedding model to download: plain word overlap finds the
//! right stretch of a meeting well enough when the question names its topic.
//!
//! Answers cite `[mm:ss]` (one meeting) or `[M2 mm:ss]` (all meetings); the
//! UI turns those into links that play or open that moment.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::{ApiError, ApiResult};
use crate::export::fmt_ts;
use crate::pipeline::{self, PipelineCtx};
use crate::store::{self, Record};
use crate::Ctx;

/// Transcript lines per retrieval window, and how far windows step.
const WINDOW: usize = 6;
const STRIDE: usize = 3;
/// Most questions and answers kept on a meeting.
pub const MAX_QA: usize = 50;
pub const MAX_QUESTION_CHARS: usize = 1000;
/// Tokens the answer may use.
pub const ANSWER_TOKENS: usize = 700;

pub const ONE_SYSTEM: &str = "You answer questions about one meeting, using only its transcript \
(and, for long meetings, its notes) below. Each transcript line starts with the time it was said, like [12:34]. Rules:\n\
- Use only what the meeting says. If it doesn't answer the question, say so in one sentence.\n\
- Be brief: a few sentences or a short list.\n\
- End every sentence with the time of the transcript line it comes from, like [12:34] — \
one time, not a range. A fact found only in the notes gets no time.\n\
- Use the speakers' names as written.\n\
- Answer in the language of the question. Plain text, no markdown headings.";

pub const ALL_SYSTEM: &str = "You answer questions across several meetings, using only the \
excerpts below. Each meeting is headed like \"M2: Title (date)\", and each line starts with the \
time it was said. Rules:\n\
- Use only what the excerpts say. If they don't answer the question, say so in one sentence.\n\
- Be brief: a few sentences or a short list; say which meeting each point comes from.\n\
- End every sentence with the meeting and the time of the line it comes from, like \
[M2 12:34] — one time, not a range.\n\
- Answer in the language of the question. Plain text, no markdown headings.";

// --------------------------------------------------------------------------
// Retrieval
// --------------------------------------------------------------------------

/// Words that carry no topic.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "what", "who", "when", "where", "which", "how",
    "why", "did", "does", "do", "is", "it", "in", "on", "of", "to", "a", "an", "we", "you", "they",
    "i", "about", "that", "this", "with", "be", "or", "at", "by", "from", "as", "so", "der", "die",
    "das", "und", "ist", "wer", "was", "wie", "wann", "wo", "ein", "eine", "zu", "den", "dem",
    "mit", "von", "für", "auf", "im", "es", "wir", "sie", "ich",
];

/// Lowercase words of 2+ letters/digits, minus stopwords.
pub fn terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .map(|w| w.to_lowercase())
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .collect()
}

/// BM25 scores of `docs` for `query` (k1 = 1.2, b = 0.75).
pub fn bm25(docs: &[Vec<String>], query: &[String]) -> Vec<f64> {
    let n = docs.len() as f64;
    if docs.is_empty() || query.is_empty() {
        return vec![0.0; docs.len()];
    }
    let avg = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / n;
    let mut df: HashMap<&str, f64> = HashMap::new();
    for doc in docs {
        let mut seen: Vec<&str> = doc.iter().map(String::as_str).collect();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            *df.entry(t).or_default() += 1.0;
        }
    }
    let (k1, b) = (1.2, 0.75);
    docs.iter()
        .map(|doc| {
            let len = doc.len() as f64;
            query
                .iter()
                .map(|q| {
                    let tf = doc.iter().filter(|t| *t == q).count() as f64;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    let d = df.get(q.as_str()).copied().unwrap_or(0.0);
                    let idf = ((n - d + 0.5) / (d + 0.5) + 1.0).ln();
                    idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / avg.max(1.0)))
                })
                .sum()
        })
        .collect()
}

/// Overlapping windows over `n` lines: (first, last exclusive).
pub fn windows(n: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + WINDOW).min(n);
        out.push((start, end));
        if end == n {
            return out;
        }
        start += STRIDE;
    }
}

/// The transcript lines to show the model for `question`: every line when
/// they all fit, else the best-matching windows (best first until the next
/// one wouldn't fit), in their original order with "…" between gaps.
pub fn pick_lines(lines: &[String], question: &str, fits: &dyn Fn(&str) -> bool) -> String {
    let all = lines.join("\n");
    if fits(&all) {
        return all;
    }
    let spans = windows(lines.len());
    let docs: Vec<Vec<String>> =
        spans.iter().map(|(a, b)| terms(&lines[*a..*b].join(" "))).collect();
    let scores = bm25(&docs, &terms(question));
    let mut order: Vec<usize> = (0..spans.len()).collect();
    // Best first; ties (and all-zero scores) keep the meeting's order.
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    let mut keep = vec![false; lines.len()];
    for w in order {
        let (a, b) = spans[w];
        let mut next = keep.clone();
        next[a..b].iter_mut().for_each(|k| *k = true);
        if fits(&render_kept(lines, &next)) {
            keep = next;
        }
    }
    render_kept(lines, &keep)
}

fn render_kept(lines: &[String], keep: &[bool]) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut gap = false;
    for (line, k) in lines.iter().zip(keep) {
        if *k {
            if gap && !out.is_empty() {
                out.push("…");
            }
            out.push(line);
            gap = false;
        } else {
            gap = true;
        }
    }
    out.join("\n")
}

/// `[mm:ss] Name: text` for each segment.
pub fn timed_lines(record: &Record) -> Vec<String> {
    let speakers = record.get("speakers").and_then(|v| v.as_object());
    record
        .get("segments")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .map(|seg| {
            let id = seg.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
            let name = speakers.and_then(|m| m.get(id)).and_then(|v| v.as_str()).unwrap_or(id);
            let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
            format!("[{}] {name}: {text}", fmt_ts(start))
        })
        .collect()
}

fn notes_text(record: &Record) -> String {
    let field = |k: &str| record.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let mut out = field("summary");
    let long = field("summary_long");
    if !long.is_empty() {
        out.push_str("\n\n");
        out.push_str(&long);
    }
    out
}

/// The user message for a question about one meeting: the whole transcript
/// when it fits. Otherwise the notes (the overview of what was left out)
/// plus the transcript windows that best match the question.
pub fn one_meeting_prompt(record: &Record, question: &str, fits: &dyn Fn(&str) -> bool) -> String {
    let tail = format!("\n\nQuestion: {question}");
    let lines = timed_lines(record);
    let whole = format!("Transcript:\n{}{tail}", lines.join("\n"));
    if fits(&whole) {
        return whole;
    }
    let head = format!("Notes:\n{}\n\nTranscript excerpts:\n", notes_text(record));
    let body = pick_lines(&lines, question, &|t| fits(&format!("{head}{t}{tail}")));
    format!("{head}{body}{tail}")
}

/// One meeting's part of an all-meetings question.
pub struct Source {
    pub label: String,
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub lines: Vec<String>,
}

/// The user message for a question across meetings: the best windows from
/// every meeting, grouped under "M1: Title (date)" headers in date order.
pub fn all_meetings_prompt(
    sources: &[Source],
    question: &str,
    fits: &dyn Fn(&str) -> bool,
) -> String {
    let tail = format!("\n\nQuestion: {question}");
    // Every window of every meeting competes in one ranking.
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    for (m, src) in sources.iter().enumerate() {
        spans.extend(windows(src.lines.len()).into_iter().map(|(a, b)| (m, a, b)));
    }
    let docs: Vec<Vec<String>> =
        spans.iter().map(|(m, a, b)| terms(&sources[*m].lines[*a..*b].join(" "))).collect();
    let scores = bm25(&docs, &terms(question));
    let mut order: Vec<usize> = (0..spans.len()).filter(|i| scores[*i] > 0.0).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    let mut keep: Vec<Vec<bool>> = sources.iter().map(|s| vec![false; s.lines.len()]).collect();
    let render = |keep: &[Vec<bool>]| -> String {
        let mut parts = Vec::new();
        for (src, k) in sources.iter().zip(keep) {
            if !k.iter().any(|x| *x) {
                continue;
            }
            let day = src.created_at.get(..10).unwrap_or(&src.created_at);
            parts.push(format!(
                "{}: {} ({day})\n{}",
                src.label,
                src.title,
                render_kept(&src.lines, k)
            ));
        }
        parts.join("\n\n")
    };
    for w in order {
        let (m, a, b) = spans[w];
        let mut next = keep.clone();
        next[m][a..b].iter_mut().for_each(|k| *k = true);
        if fits(&format!("{}{tail}", render(&next))) {
            keep = next;
        }
    }
    let body = render(&keep);
    if body.is_empty() {
        format!("(No meeting mentions this.){tail}")
    } else {
        format!("{body}{tail}")
    }
}

// --------------------------------------------------------------------------
// Running a question
// --------------------------------------------------------------------------

/// Cancel flags of the questions being answered, by request id.
#[derive(Default)]
pub struct AskState {
    running: std::sync::Mutex<HashMap<String, Arc<AtomicBool>>>,
}

/// Stop answering `request_id` (no-op if it already finished).
pub fn cancel(ctx: &Ctx, request_id: &str) {
    if let Some(flag) = ctx.ask.running.lock().unwrap().get(request_id) {
        flag.store(true, Ordering::SeqCst);
    }
}

fn check_question(question: &str) -> ApiResult<String> {
    let q = question.trim();
    if q.is_empty() {
        return Err(ApiError::invalid("ask a question"));
    }
    if q.chars().count() > MAX_QUESTION_CHARS {
        return Err(ApiError::invalid(format!(
            "questions are at most {MAX_QUESTION_CHARS} characters"
        )));
    }
    Ok(q.to_string())
}

/// Stream an answer: `ask-progress` events {request_id, delta} while the
/// model writes, then the whole answer is returned.
fn run(
    ctx: &Ctx,
    request_id: &str,
    system: &str,
    build_prompt: &dyn Fn(&dyn Fn(&str) -> bool) -> String,
) -> ApiResult<String> {
    if ctx.bench_running.load(Ordering::SeqCst) {
        return Err(ApiError::conflict("the benchmark is running — ask again once it's done"));
    }
    if !crate::models::is_ready(ctx) {
        return Err(ApiError::conflict("the models are still downloading"));
    }
    let flag = Arc::new(AtomicBool::new(false));
    ctx.ask.running.lock().unwrap().insert(request_id.to_string(), flag.clone());
    let config = ctx.config.read().unwrap().clone();
    let pctx = PipelineCtx {
        fake: ctx.fake,
        config: &config,
        models_dir: &ctx.paths.models_dir,
        ffmpeg: ctx.ffmpeg.clone(),
    };
    let rid = request_id.to_string();
    let mut on_delta = |delta: &str| {
        ctx.events.emit("ask-progress", json!({"request_id": rid, "delta": delta}));
    };
    let result = pipeline::ask(&pctx, system, build_prompt, ANSWER_TOKENS, &mut on_delta, &flag);
    ctx.ask.running.lock().unwrap().remove(request_id);
    let answer = result.map_err(ApiError::internal)?;
    let cancelled = flag.load(Ordering::SeqCst);
    ctx.events.emit(
        "ask-progress",
        json!({"request_id": request_id, "done": true, "cancelled": cancelled}),
    );
    Ok(answer.trim().to_string())
}

/// Answer a question about one finished meeting; the exchange is kept on
/// the meeting (newest last, at most `MAX_QA`).
pub fn ask_meeting(
    ctx: &Ctx,
    meeting_id: &str,
    question: &str,
    request_id: &str,
) -> ApiResult<Value> {
    let question = check_question(question)?;
    let record = store::get_meeting(&ctx.paths, &ctx.locks, meeting_id)
        .ok_or_else(|| ApiError::not_found("meeting not found"))?;
    if record.get("segments").and_then(|v| v.as_array()).is_none_or(|a| a.is_empty()) {
        return Err(ApiError::conflict("there is no transcript to ask about yet"));
    }
    let answer =
        run(ctx, request_id, ONE_SYSTEM, &|fits| one_meeting_prompt(&record, &question, fits))?;
    let entry = json!({"q": question, "a": answer, "at": store::now_iso()});
    store::push_qa(&ctx.paths, &ctx.locks, meeting_id, entry.clone(), MAX_QA);
    Ok(entry)
}

/// Answer a question across every finished meeting. Returns the answer and
/// the meetings it could cite, as {label: "M1", id, title}.
pub fn ask_all(ctx: &Ctx, question: &str, request_id: &str) -> ApiResult<Value> {
    let question = check_question(question)?;
    let mut records: Vec<Record> = store::list_meetings(&ctx.paths, &ctx.locks)
        .into_iter()
        .filter(|m| m.get("status").and_then(|v| v.as_str()) == Some("done"))
        .filter_map(|m| store::get_meeting(&ctx.paths, &ctx.locks, m.get("id")?.as_str()?))
        .collect();
    if records.is_empty() {
        return Err(ApiError::conflict("there are no finished meetings to ask about yet"));
    }
    // Oldest first, so M1, M2… read in date order.
    records.sort_by(|a, b| {
        let key =
            |r: &Record| r.get("created_at").and_then(|v| v.as_str()).unwrap_or("").to_string();
        key(a).cmp(&key(b))
    });
    let sources: Vec<Source> = records
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let public = store::public_meeting(&ctx.paths, r);
            let text = |k: &str| public.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
            Source {
                label: format!("M{}", i + 1),
                id: text("id"),
                title: text("title"),
                created_at: text("created_at"),
                lines: timed_lines(r),
            }
        })
        .collect();
    let answer =
        run(ctx, request_id, ALL_SYSTEM, &|fits| all_meetings_prompt(&sources, &question, fits))?;
    let cited: Vec<Value> =
        sources.iter().map(|s| json!({"label": s.label, "id": s.id, "title": s.title})).collect();
    Ok(json!({"q": question, "a": answer, "sources": cited}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_overlap_and_cover_everything() {
        assert_eq!(windows(0), vec![]);
        assert_eq!(windows(4), vec![(0, 4)]);
        assert_eq!(windows(10), vec![(0, 6), (3, 9), (6, 10)]);
    }

    #[test]
    fn bm25_prefers_the_window_that_names_the_topic() {
        let docs = vec![
            terms("we talked about the weather and lunch"),
            terms("the pricing for Acme goes up in March, pricing is final"),
            terms("Acme sent the contract"),
        ];
        let scores = bm25(&docs, &terms("What did we decide on pricing for Acme?"));
        assert!(scores[1] > scores[2] && scores[2] > scores[0], "{scores:?}");
    }

    #[test]
    fn long_transcripts_keep_the_matching_part_in_order() {
        let mut lines: Vec<String> =
            (0..60).map(|i| format!("[00:{i:02}] A: filler line {i}")).collect();
        lines[40] = "[00:40] B: the budget for the relaunch is 40k".into();
        let fits = |t: &str| t.lines().count() <= 8;
        let picked = pick_lines(&lines, "what is the relaunch budget?", &fits);
        assert!(picked.contains("budget for the relaunch"), "{picked}");
        assert!(picked.lines().count() <= 8);
        let all = pick_lines(&lines[..5], "anything", &fits);
        assert_eq!(all, lines[..5].join("\n"), "short transcripts go in whole");
    }
}
