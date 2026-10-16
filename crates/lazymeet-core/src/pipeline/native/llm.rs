//! Summarization: llama.cpp chat, language detection, map-reduce over long
//! transcripts (port of `summarize.py`).

use super::*;

/// Tokens reserved for the model's response plus prompt scaffolding.
const RESERVE: usize = 2560;

/// Fixed response-format instruction, appended after the (user-editable)
/// SUMMARY_PROMPT so a customized prompt can't break summary parsing.
pub(super) const FORMAT: &str = "Respond with ONLY a JSON object of the form \
    {\"summary\": str, \"takeaways\": [str], \"action_items\": [str]}.";

const NOTES_SYSTEM: &str =
    "You are a meeting assistant. Summarize this portion of a meeting transcript \
    into concise bullet notes capturing decisions, topics, and any action items. \
    Return plain text bullets only.";

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
        let n_batch = (tokens.len() + max_tokens + 32).max(512) as u32;
        let mut ctx = self
            .model
            .new_context(
                backend,
                LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(self.n_ctx))
                    .with_n_batch(n_batch)
                    .with_n_threads(self.threads)
                    .with_n_threads_batch(self.threads),
            )
            .map_err(|e| format!("llm context: {e}"))?;

        let mut batch = LlamaBatch::new(n_batch as usize, 1);
        let last_idx = tokens.len() as i32 - 1;
        for (i, token) in (0i32..).zip(&tokens) {
            batch.add(*token, i, &[0], i == last_idx).map_err(|e| e.to_string())?;
        }
        let started = std::time::Instant::now();
        ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        let prefill_secs = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.2), LlamaSampler::dist(42)]);
        let mut out: Vec<u8> = Vec::new();
        let mut n_cur = batch.n_tokens();
        let mut n_gen = 0usize;
        loop {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) || n_gen >= max_tokens {
                break;
            }
            #[allow(deprecated)]
            if let Ok(bytes) = self.model.token_to_bytes(token, Special::Tokenize) {
                out.extend(bytes);
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
    let content = strip_thinking(content);
    let candidate = match (content.find('{'), content.rfind('}')) {
        (Some(a), Some(b)) if b > a => &content[a..=b],
        _ => content.as_str(),
    };
    let data: Option<Value> =
        serde_json::from_str(&content).ok().or_else(|| serde_json::from_str(candidate).ok());
    let Some(Value::Object(data)) = data else {
        return Summary {
            summary: content.trim().to_string(),
            takeaways: vec![],
            action_items: vec![],
        };
    };
    let strings = |key: &str| -> Vec<String> {
        data.get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .map(|x| match x.as_str() {
                        Some(s) => s.to_string(),
                        None => x.to_string(),
                    })
                    .filter(|s| !s.trim().is_empty())
                    .collect()
            })
            .unwrap_or_default()
    };
    Summary {
        summary: data
            .get("summary")
            .map(|v| match v.as_str() {
                Some(s) => s.trim().to_string(),
                None => v.to_string(),
            })
            .unwrap_or_default(),
        takeaways: strings("takeaways"),
        action_items: strings("action_items"),
    }
}

/// Long meetings are chunked to fit the context window (map-reduce: summarize
/// each chunk to notes, then summarize the notes). The final answer is strict
/// JSON parsed defensively.
pub fn summarize(
    ctx: &PipelineCtx,
    segments: &[MergedSegment],
    speakers: &Map<String, Value>,
) -> Result<Summary, String> {
    let llm = Llm::load(ctx)?;
    let lines = transcript_lines(segments, speakers);
    let budget = (llm.n_ctx as usize).saturating_sub(RESERVE).max(1024);
    let full = lines.join("\n");
    let lang = lang_instruction(ctx, &full);
    let custom = ctx.config.summary_prompt();
    let task = custom.trim();
    let task = if task.is_empty() { config::DEFAULT_SUMMARY_PROMPT } else { task };

    // The language order is repeated at the END of every user message: small
    // models weigh trailing instructions far more than the system prompt.
    let body = if llm.count_tokens(&full) <= budget {
        format!("Transcript:\n{full}\n\n{lang}")
    } else {
        // Map-reduce for long meetings.
        let mut notes = Vec::new();
        for chunk in chunk_lines(&llm, &lines, budget) {
            let note =
                llm.chat(&format!("{NOTES_SYSTEM} {lang}"), &format!("{chunk}\n\n{lang}"), 512)?;
            notes.push(strip_thinking(&note));
        }
        let combined = llm.truncate_to_tokens(&notes.join("\n"), budget);
        format!("Meeting notes:\n{combined}\n\n{lang}")
    };
    let response = llm.chat(&format!("{task} {FORMAT} {lang}"), &body, 1024)?;
    Ok(parse_summary(&response))
}
