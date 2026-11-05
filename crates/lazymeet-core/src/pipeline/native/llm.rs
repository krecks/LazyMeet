//! Summarization: llama.cpp chat, language detection, map-reduce over long
//! transcripts (port of `summarize.py`).

use super::*;
use crate::pipeline::clean_title;

/// Tokens reserved for the model's response plus prompt scaffolding. The
/// answer budget (`ANSWER_TOKENS`) plus the format/language instruction has to
/// fit inside it, or a long meeting would push the prompt out of the window.
const RESERVE: usize = 3072;
/// Room for summary + summary_long + up to 6 detailed takeaways and action
/// items (~1300 tokens in practice; the rest is headroom for verbose models).
const ANSWER_TOKENS: usize = 1600;
/// Prompt tokens decoded per `llama_decode` call. One call for the whole
/// prompt forces llama.cpp to allocate a compute buffer for every token at
/// once (a 13k-token meeting spikes memory and is slower than a few 512-token
/// decodes); chunking is what llama.cpp's own examples do.
const PREFILL_CHUNK: usize = 512;

/// What a meeting title looks like — shared by the summary's JSON format and
/// the title-only pass, so both name meetings the same way. (A macro because
/// `concat!` only takes literals.)
macro_rules! title_rule {
    () => {
        "3-8 words naming what the meeting was about, like a calendar entry (e.g. \
        \"Q3 pricing review with Acme\"); no date, no quotes, no file name, no trailing period."
    };
}

/// Fixed response-format instruction, appended after the (user-editable)
/// SUMMARY_PROMPT so a customized prompt can't break summary parsing.
/// `summary`/`summary_long` are two lengths of the same story, and every list
/// item carries a one-line `short` plus a `detail` — that is what makes the
/// notes scannable and still complete.
/// The title rule lives here rather than in the editable prompt so a custom
/// prompt still names the meeting.
pub(super) const FORMAT: &str = concat!(
    "Respond with ONLY a JSON object of this exact shape, with no prose and no code fences: \
    {\"title\": string, \"summary\": string, \"summary_long\": string, \"takeaways\": \
    [{\"short\": string, \"detail\": string}], \"action_items\": [{\"short\": string, \
    \"detail\": string}]}. title: ",
    title_rule!(),
);

/// Title-only pass over existing notes (meetings processed before titles).
const TITLE_SYSTEM: &str = concat!(
    "You name meetings. Given a meeting's notes, reply with ONLY the title, nothing else. \
    The title: ",
    title_rule!(),
);

const NOTES_SYSTEM: &str =
    "You are a meeting assistant. Turn this portion of a meeting transcript into compact \
    notes: topics, decisions, exact numbers and dates, open questions, and every commitment \
    (who does what by when). Keep the speakers' names. Return plain text bullets only, no \
    preamble.";

fn language_name(code: &str) -> &str {
    super::super::LANGUAGE_NAMES.iter().find(|(c, _)| *c == code).map(|(_, n)| *n).unwrap_or(code)
}

/// Detect the transcript's language, restricted to the languages the
/// summarizer can be instructed in. None when detection isn't reliable
/// (very short or heavily mixed-language transcripts).
pub(super) fn detect_language(text: &str) -> Option<&'static str> {
    use whatlang::Lang;
    const LANGS: &[(Lang, &str)] = &[
        (Lang::Eng, "en"),
        (Lang::Deu, "de"),
        (Lang::Fra, "fr"),
        (Lang::Spa, "es"),
        (Lang::Ita, "it"),
        (Lang::Por, "pt"),
        (Lang::Nld, "nl"),
        (Lang::Pol, "pl"),
        (Lang::Rus, "ru"),
        (Lang::Ukr, "uk"),
        (Lang::Tur, "tr"),
        (Lang::Cmn, "zh"),
        (Lang::Jpn, "ja"),
        (Lang::Kor, "ko"),
    ];
    // A few thousand characters are plenty for a confident call.
    let sample: String = text.chars().take(4000).collect();
    let detector = whatlang::Detector::with_allowlist(LANGS.iter().map(|(l, _)| *l).collect());
    let info = detector.detect(&sample)?;
    if !info.is_reliable() {
        return None;
    }
    LANGS.iter().find(|(l, _)| *l == info.lang()).map(|(_, code)| *code)
}

/// The language order given to the summarizer. An explicit setting wins;
/// "auto" detects the language from the transcript text. Naming the language
/// outright ("Write everything in German.") is what small models actually
/// follow — the former "same language as the meeting" phrasing made them
/// drift into English on non-English meetings.
fn lang_instruction(ctx: &PipelineCtx, transcript: &str) -> String {
    let code = ctx.config.summary_language();
    let code = if !code.is_empty() && code != "auto" {
        Some(code)
    } else {
        detect_language(transcript).map(str::to_string)
    };
    match code {
        Some(code) => format!("Write everything in {}.", language_name(&code)),
        None => "Write in the same language the meeting was held in.".into(),
    }
}

pub(super) fn llama_backend() -> Result<&'static LlamaBackend, String> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            // With dynamic backends the ggml kernels (CPU variants, Vulkan)
            // are loadable modules that must be registered before init.
            // LAZYMEET_LLM_BACKENDS_DIR points at the shipped modules (set
            // by the desktop shell from its resource dir); without it, fall
            // back to the build-time directory, which is valid on the
            // machine that compiled the binary (dev builds, CI benches).
            #[cfg(feature = "llama-dynamic-backends")]
            match std::env::var_os("LAZYMEET_LLM_BACKENDS_DIR") {
                Some(dir) => {
                    llama_cpp_2::llama_backend::load_backends_from_path(std::path::Path::new(&dir))
                }
                None => llama_cpp_2::llama_backend::load_backends(),
            }
            LlamaBackend::init().map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| e.clone())
}

pub(super) struct ChatTiming {
    pub(super) prompt_tokens: usize,
    pub(super) prefill_secs: f64,
    pub(super) generated_tokens: usize,
    pub(super) generate_secs: f64,
}

pub(super) struct Llm {
    pub(super) model: LlamaModel,
    pub(super) n_ctx: u32,
    pub(super) threads: i32,
}

impl Llm {
    pub(super) fn load(ctx: &PipelineCtx) -> Result<Self, String> {
        let backend = llama_backend()?;
        let path =
            crate::models::resolve_llm_at(ctx.models_dir, ctx.fake, &ctx.config.summary_model());
        // Offload all layers to the GPU (Metal on Apple Silicon) when built
        // with GPU support; if that load fails (driver/VRAM), fall back to a
        // CPU-only load rather than failing the meeting. SUMMARY_PROVIDER
        // "cpu" must set n_gpu_layers to 0 explicitly — llama.cpp's *default*
        // on Metal builds is full offload.
        let cpu = || LlamaModelParams::default().with_n_gpu_layers(0);
        let model = if ctx.config.use_gpu() {
            let gpu = LlamaModelParams::default().with_n_gpu_layers(1_000_000);
            LlamaModel::load_from_file(backend, &path, &gpu)
                .or_else(|_| LlamaModel::load_from_file(backend, &path, &cpu()))
        } else {
            LlamaModel::load_from_file(backend, &path, &cpu())
        }
        .map_err(|e| format!("llm load: {e}"))?;
        Ok(Self { model, n_ctx: config::llm_n_ctx(), threads: ctx.config.num_threads() as i32 })
    }

    pub(super) fn count_tokens(&self, text: &str) -> usize {
        self.model.str_to_token(text, AddBos::Never).map(|t| t.len()).unwrap_or(0)
    }

    fn detokenize(&self, tokens: &[LlamaToken]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            #[allow(deprecated)]
            if let Ok(b) = self.model.token_to_bytes(t, Special::Tokenize) {
                bytes.extend(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub(super) fn truncate_to_tokens(&self, text: &str, budget: usize) -> String {
        match self.model.str_to_token(text, AddBos::Never) {
            Ok(tokens) if tokens.len() > budget => self.detokenize(&tokens[..budget]),
            _ => text.to_string(),
        }
    }

    /// One chat completion: apply the model's template, prefill an empty
    /// think-block (the C-API template matcher doesn't disable Qwen thinking),
    /// decode greedily at temp 0.2.
    pub(super) fn chat(
        &self,
        system: &str,
        user: &str,
        max_tokens: usize,
    ) -> Result<String, String> {
        self.chat_timed(system, user, max_tokens).map(|(text, _)| text)
    }

    /// `chat`, also reporting how long the prompt (prefill) and the answer
    /// (generation) took.
    pub(super) fn chat_timed(
        &self,
        system: &str,
        user: &str,
        max_tokens: usize,
    ) -> Result<(String, ChatTiming), String> {
        self.chat_stream(system, user, max_tokens, &mut |_| {}, None)
    }

    /// `chat_timed`, handing the answer to `on_delta` as it is written (in
    /// whole UTF-8 characters) and stopping early once `cancel` is set.
    pub(super) fn chat_stream(
        &self,
        system: &str,
        user: &str,
        max_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(String, ChatTiming), String> {
        let backend = llama_backend()?;
        let template = self.model.chat_template(None).map_err(|e| format!("chat template: {e}"))?;
        let mut prompt = self
            .model
            .apply_chat_template(
                &template,
                &[
                    LlamaChatMessage::new("system".into(), system.into())
                        .map_err(|e| e.to_string())?,
                    LlamaChatMessage::new("user".into(), user.into()).map_err(|e| e.to_string())?,
                ],
                true,
            )
            .map_err(|e| e.to_string())?;
        // Qwen's jinja template inserts an empty think block when thinking is
        // disabled; prefill it or the model burns the budget on reasoning.
        prompt.push_str("<think>\n\n</think>\n\n");

        let tokens = self.model.str_to_token(&prompt, AddBos::Always).map_err(|e| e.to_string())?;
        let needed = tokens.len() + max_tokens + 8;
        if needed > self.n_ctx as usize {
            return Err(format!(
                "prompt does not fit the context window ({} + {max_tokens} tokens > {})",
                tokens.len(),
                self.n_ctx
            ));
        }
        // Fixed: llama.cpp requires n_batch >= n_ubatch (512), so a smaller
        // batch for short prompts would be rejected at context creation.
        let n_batch = PREFILL_CHUNK;
        let mut ctx = self
            .model
            .new_context(
                backend,
                LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(self.n_ctx))
                    .with_n_batch(n_batch as u32)
                    .with_n_threads(self.threads)
                    .with_n_threads_batch(self.threads),
            )
            .map_err(|e| format!("llm context: {e}"))?;

        // Prefill in chunks: logits are requested only for the last prompt
        // token, so every chunk before it decodes without an output head.
        let mut batch = LlamaBatch::new(n_batch, 1);
        let last_idx = tokens.len() as i32 - 1;
        let started = std::time::Instant::now();
        let mut pos = 0i32;
        for chunk in tokens.chunks(n_batch) {
            batch.clear();
            for token in chunk {
                batch.add(*token, pos, &[0], pos == last_idx).map_err(|e| e.to_string())?;
                pos += 1;
            }
            ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        }
        let prefill_secs = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.2), LlamaSampler::dist(42)]);
        let mut out: Vec<u8> = Vec::new();
        // Positions are absolute: the KV cache already holds every prompt
        // token, so generation starts after the whole prompt — not after the
        // last chunk, which is all `batch` still holds.
        let mut n_cur = tokens.len() as i32;
        let mut n_gen = 0usize;
        // Bytes of `out` already handed to on_delta (a token can end inside
        // a multi-byte character; that part waits for the next token).
        let mut emitted = 0usize;
        loop {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst)) {
                break;
            }
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) || n_gen >= max_tokens {
                break;
            }
            #[allow(deprecated)]
            if let Ok(bytes) = self.model.token_to_bytes(token, Special::Tokenize) {
                out.extend(bytes);
            }
            let valid = match std::str::from_utf8(&out[emitted..]) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                on_delta(std::str::from_utf8(&out[emitted..emitted + valid]).unwrap_or(""));
                emitted += valid;
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true).map_err(|e| e.to_string())?;
            n_cur += 1;
            n_gen += 1;
            ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        }
        let timing = ChatTiming {
            prompt_tokens: tokens.len(),
            prefill_secs,
            generated_tokens: n_gen,
            generate_secs: started.elapsed().as_secs_f64(),
        };
        Ok((String::from_utf8_lossy(&out).into_owned(), timing))
    }
}

fn transcript_lines(segments: &[MergedSegment], speakers: &Map<String, Value>) -> Vec<String> {
    segments
        .iter()
        .map(|seg| {
            let name = speakers.get(&seg.speaker).and_then(|v| v.as_str()).unwrap_or(&seg.speaker);
            format!("{name}: {}", seg.text)
        })
        .collect()
}

fn chunk_lines(llm: &Llm, lines: &[String], budget: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut cur_tok = 0usize;
    for line in lines {
        let t = llm.count_tokens(line) + 1;
        if !cur.is_empty() && cur_tok + t > budget {
            chunks.push(cur.join("\n"));
            cur.clear();
            cur_tok = 0;
        }
        cur.push(line);
        cur_tok += t;
    }
    if !cur.is_empty() {
        chunks.push(cur.join("\n"));
    }
    chunks
}

/// Drop <think>…</think> blocks some Qwen variants emit before the answer.
fn strip_thinking(content: &str) -> String {
    let re = regex::Regex::new(r"(?s)<think>.*?</think>").unwrap();
    re.replace_all(content, "").trim().to_string()
}

fn parse_summary(content: &str) -> Summary {
    use super::super::NoteItem;
    let content = strip_thinking(content);
    let candidate = match (content.find('{'), content.rfind('}')) {
        (Some(a), Some(b)) if b > a => &content[a..=b],
        _ => content.as_str(),
    };
    let data: Option<Value> =
        serde_json::from_str(&content).ok().or_else(|| serde_json::from_str(candidate).ok());
    let Some(Value::Object(data)) = data else {
        return Summary { summary: content.trim().to_string(), ..Default::default() };
    };
    // `text_of` accepts the plain-string form older prompts (and some models)
    // produce, so a schema slip costs the detail, never the whole note.
    let text_of = |v: &Value| -> String {
        match v {
            Value::String(s) => s.trim().to_string(),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    };
    let items = |key: &str| -> Vec<NoteItem> {
        data.get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| match x {
                        Value::String(s) => {
                            let s = s.trim();
                            (!s.is_empty()).then_some(NoteItem::line(s))
                        }
                        Value::Object(o) => {
                            let short = o.get("short").map(text_of).unwrap_or_default();
                            let detail = o.get("detail").map(text_of).unwrap_or_default();
                            // Some models put the whole point in `detail`.
                            let short = if short.is_empty() { detail.clone() } else { short };
                            let detail = if short == detail { String::new() } else { detail };
                            (!short.is_empty()).then_some(NoteItem { short, detail })
                        }
                        _ => None,
                    })
                    .take(20)
                    .collect()
            })
            .unwrap_or_default()
    };
    Summary {
        title: data.get("title").map(|v| clean_title(&text_of(v))).unwrap_or_default(),
        summary: data.get("summary").map(text_of).unwrap_or_default(),
        summary_long: data.get("summary_long").map(text_of).unwrap_or_default(),
        takeaways: items("takeaways"),
        action_items: items("action_items"),
    }
}

/// One model on the GPU at a time: summaries, titles and questions take
/// this lock for as long as they hold a loaded model.
static LLM_LOCK: Mutex<()> = Mutex::new(());

/// The model questions keep loaded between follow-ups: (model id, provider,
/// model, last used). Dropped after `CACHE_IDLE`, when the settings pick
/// another model, and before a summary loads its own copy.
#[allow(clippy::type_complexity)]
static ASK_CACHE: Mutex<Option<(String, String, Llm, std::time::Instant)>> = Mutex::new(None);
const CACHE_IDLE: std::time::Duration = std::time::Duration::from_secs(120);

fn evict_ask_cache() {
    ASK_CACHE.lock().unwrap_or_else(|e| e.into_inner()).take();
}

/// Free the cached model (app exit). Waits for a question in progress.
pub fn release_llm() {
    let _gpu = LLM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    evict_ask_cache();
}

/// Answer `build_prompt`'s question, streaming to `on_delta`. The prompt
/// builder gets a `fits` check in real tokens, so it can fill the context
/// with as much transcript as there is room for.
pub fn ask(
    ctx: &PipelineCtx,
    system: &str,
    build_prompt: crate::pipeline::PromptBuilder<'_>,
    max_tokens: usize,
    on_delta: &mut dyn FnMut(&str),
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<String, String> {
    let _gpu = LLM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let key = (ctx.config.summary_model(), ctx.config.summary_provider());
    let mut cache = ASK_CACHE.lock().unwrap();
    if cache.as_ref().map_or(true, |(m, p, _, _)| (m, p) != (&key.0, &key.1)) {
        *cache = None; // free the old one before loading the next
        *cache = Some((key.0.clone(), key.1.clone(), Llm::load(ctx)?, std::time::Instant::now()));
        start_cache_janitor();
    }
    let (_, _, llm, used) = cache.as_mut().expect("loaded above");
    *used = std::time::Instant::now();
    let budget = (llm.n_ctx as usize).saturating_sub(max_tokens + 600).max(512);
    let fits = |text: &str| llm.count_tokens(text) <= budget;
    let user = build_prompt(&fits);
    let (answer, _) = llm.chat_stream(system, &user, max_tokens, on_delta, Some(cancel))?;
    *used = std::time::Instant::now();
    Ok(strip_thinking(&answer))
}

/// Once per process: a thread that drops the cached model when idle.
fn start_cache_janitor() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        std::thread::spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(20));
            // try_lock: a question in progress is using it — not idle.
            if let Ok(mut cache) = ASK_CACHE.try_lock() {
                if cache.as_ref().is_some_and(|(_, _, _, used)| used.elapsed() > CACHE_IDLE) {
                    *cache = None;
                }
            }
        });
    });
}

/// Long meetings are chunked to fit the context window (map-reduce: summarize
/// each chunk to notes, then summarize the notes). The final answer is strict
/// JSON parsed defensively.
pub fn summarize(
    ctx: &PipelineCtx,
    segments: &[MergedSegment],
    speakers: &Map<String, Value>,
) -> Result<Summary, String> {
    let _gpu = LLM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    evict_ask_cache();
    let llm = Llm::load(ctx)?;
    let lines = transcript_lines(segments, speakers);
    let budget = (llm.n_ctx as usize).saturating_sub(RESERVE).max(1024);
    let full = lines.join("\n");
    let lang = lang_instruction(ctx, &full);
    let custom = ctx.config.summary_prompt();
    let task = custom.trim();
    let task = if task.is_empty() { config::DEFAULT_SUMMARY_PROMPT } else { task };
    // The template's focus and the vocabulary go after the (editable) task
    // and before the fixed format, so neither can break parsing.
    let guidance = config::notes_guidance(ctx.config);

    // The language order is repeated at the END of every user message: small
    // models weigh trailing instructions far more than the system prompt.
    let body = if llm.count_tokens(&full) <= budget {
        format!("Transcript:\n{full}\n\n{lang}")
    } else {
        // Map-reduce for long meetings.
        let mut notes = Vec::new();
        for chunk in chunk_lines(&llm, &lines, budget) {
            let note = llm.chat(
                &format!("{NOTES_SYSTEM} {guidance} {lang}"),
                &format!("{chunk}\n\n{lang}"),
                640,
            )?;
            notes.push(strip_thinking(&note));
        }
        let combined = llm.truncate_to_tokens(&notes.join("\n"), budget);
        format!("Meeting notes:\n{combined}\n\n{lang}")
    };
    let response = llm.chat(&format!("{task} {guidance} {FORMAT} {lang}"), &body, ANSWER_TOKENS)?;
    Ok(parse_summary(&response))
}

/// A title from a meeting's existing notes: a short prompt and a ~20-token
/// answer, so a few seconds even on the CPU.
pub fn title_from_notes(ctx: &PipelineCtx, notes: &str) -> Result<String, String> {
    let _gpu = LLM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    evict_ask_cache();
    let llm = Llm::load(ctx)?;
    let notes = llm.truncate_to_tokens(notes, 3000);
    let lang = lang_instruction(ctx, &notes);
    let vocab = ctx.config.vocabulary();
    let spelling = if vocab.is_empty() {
        String::new()
    } else {
        format!("Spell these names exactly like this: {}.", vocab.join(", "))
    };
    let answer = llm.chat(
        &format!("{TITLE_SYSTEM} {spelling} {lang}"),
        &format!("Notes:\n{notes}\n\n{lang}"),
        48,
    )?;
    Ok(clean_title(&strip_thinking(&answer)))
}
