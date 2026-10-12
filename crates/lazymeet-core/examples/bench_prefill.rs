//! Measure the summarizer LLM's prompt-processing (prefill) and generation
//! speed separately. The summarize stage feeds thousands of transcript
//! tokens through the model, so prefill dominates it on long meetings — and
//! prefill is the compute-bound path where SIMD width matters, while
//! generation is typically memory-bandwidth-bound. CI companion of issue #1:
//!   cargo run --release -p lazymeet-core --example bench_prefill

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use lazymeet_core::{models, Ctx};

struct NoEvents;

impl lazymeet_core::events::Events for NoEvents {
    fn emit(&self, _: &str, _: serde_json::Value) {}
}

fn main() {
    let ctx = Ctx::from_env(Arc::new(NoEvents));
    let model_path = models::resolve_llm_at(&ctx.paths.models_dir, false, "qwen3.5-4b");

    // Same registration the pipeline's llama_backend() does: with dynamic
    // backends the kernels are loadable modules that must be found first.
    #[cfg(feature = "llama-dynamic-backends")]
    match std::env::var_os("LAZYMEET_LLM_BACKENDS_DIR") {
        Some(dir) => {
            llama_cpp_2::llama_backend::load_backends_from_path(std::path::Path::new(&dir))
        }
        None => llama_cpp_2::llama_backend::load_backends(),
    }
    let backend = LlamaBackend::init().expect("backend init");
    // CPU on purpose — this benchmark exists to compare CPU kernel builds.
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &params).expect("model load");

    // A ~2k-token prompt of repeated fake transcript lines.
    let mut text = String::new();
    while text.len() < 60_000 {
        text.push_str(
            "Alice: We agreed to move the rollout to Thursday so QA can sign off first.\n",
        );
        text.push_str(
            "Bob: The migration script still needs a dry run against the staging database.\n",
        );
    }
    let mut tokens = model.str_to_token(&text, AddBos::Always).expect("tokenize");
    tokens.truncate(2048);

    let threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4).max(1);
    let mut lctx = model
        .new_context(
            &backend,
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(4096))
                .with_n_batch(tokens.len() as u32)
                .with_n_threads(threads)
                .with_n_threads_batch(threads),
        )
        .expect("context");

    let mut batch = LlamaBatch::new(tokens.len(), 1);
    let last = tokens.len() as i32 - 1;
    for (i, t) in (0i32..).zip(&tokens) {
        batch.add(*t, i, &[0], i == last).expect("batch add");
    }
    let t0 = Instant::now();
    lctx.decode(&mut batch).expect("prefill decode");
    let prefill = t0.elapsed().as_secs_f64();

    // Short generation pass as the bandwidth-bound reference point.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.2), LlamaSampler::dist(42)]);
    let gen_n = 64usize;
    let mut n_cur = batch.n_tokens();
    let t1 = Instant::now();
    for _ in 0..gen_n {
        let token = sampler.sample(&lctx, batch.n_tokens() - 1);
        sampler.accept(token);
        batch.clear();
        batch.add(token, n_cur, &[0], true).expect("batch add");
        n_cur += 1;
        lctx.decode(&mut batch).expect("decode");
    }
    let gen = t1.elapsed().as_secs_f64();

    println!(
        "{{\"prompt_tokens\":{},\"prefill_secs\":{:.2},\"prefill_tps\":{:.1},\"gen_tps\":{:.1},\"threads\":{}}}",
        tokens.len(),
        prefill,
        tokens.len() as f64 / prefill,
        gen_n as f64 / gen,
        threads
    );
}
