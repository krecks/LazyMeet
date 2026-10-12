//! First-run download machinery against a local HTTP server: concurrent
//! fetches, progress bookkeeping, atomic tar.bz2 extraction, ready-flip.
//! (Replaces test_first_run_downloads_run_in_parallel, which monkeypatched
//! the downloader; this exercises the real one.)
//!
//! Sets LAZYMEET_*_URL env vars — the tests in this binary share one lock,
//! and cargo runs test binaries sequentially, so nothing races the process env.

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use common::TestApp;
use lazymeet_core::models;

/// Serializes the tests: they set process-wide env vars.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Minimal HTTP/1.1 server: every path resolves through `body_for`.
fn spawn_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
                let body = body_for(&path);
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(&body);
            });
        }
    });
    port
}

fn body_for(path: &str) -> Vec<u8> {
    match path {
        "/parakeet.tar.bz2" => tar_bz2(&[
            ("model/encoder.int8.onnx", b"parakeet-encoder"),
            ("model/decoder.int8.onnx", b"parakeet-decoder"),
            ("model/joiner.int8.onnx", b"parakeet-joiner"),
            ("model/tokens.txt", b"tokens"),
        ]),
        "/diar.onnx" => b"nemotron-graph".to_vec(),
        "/diar.onnx_data" => b"nemotron-weights".to_vec(),
        "/vad.onnx" => b"vad-model".to_vec(),
        "/embedding.onnx" => b"embedding-model".to_vec(),
        "/llm.gguf" => b"gguf-model".to_vec(),
        // The layout of the real sherpa-onnx release archive.
        "/qwen3-asr.tar.bz2" => tar_bz2(&[
            ("q/tokenizer/vocab.json", b"vocab"),
            ("q/tokenizer/tokenizer_config.json", b"tok-config"),
            ("q/tokenizer/merges.txt", b"merges"),
            ("q/conv_frontend.onnx", b"frontend"),
            ("q/encoder.int8.onnx", b"qwen-encoder"),
            ("q/test_wavs/de.wav", b"wav"),
            ("q/test_wavs/README.md", b"readme"),
            ("q/decoder.int8.onnx", b"qwen-decoder"),
            ("q/README.md", b"readme"),
        ]),
        _ => b"not found".to_vec(),
    }
}

fn tar_bz2(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
    {
        let mut tar = tar::Builder::new(&mut encoder);
        for (name, content) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, *content).unwrap();
        }
        tar.finish().unwrap();
    }
    encoder.finish().unwrap()
}

struct EnvGuard(Vec<&'static str>);
impl Drop for EnvGuard {
    fn drop(&mut self) {
        for name in &self.0 {
            std::env::remove_var(name);
        }
    }
}

#[test]
fn first_run_download_end_to_end() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let port = spawn_server();
    let url = |p: &str| format!("http://127.0.0.1:{port}{p}");
    std::env::set_var("LAZYMEET_ASR_URL", url("/parakeet.tar.bz2"));
    std::env::set_var("LAZYMEET_VAD_URL", url("/vad.onnx"));
    std::env::set_var("LAZYMEET_DIAR_URL", url("/diar.onnx"));
    std::env::set_var("LAZYMEET_DIAR_DATA_URL", url("/diar.onnx_data"));
    std::env::set_var("LAZYMEET_DIAR_EMB_URL", url("/embedding.onnx"));
    std::env::set_var("LAZYMEET_LLM_URL", url("/llm.gguf"));
    let _guard = EnvGuard(vec![
        "LAZYMEET_ASR_URL",
        "LAZYMEET_VAD_URL",
        "LAZYMEET_DIAR_URL",
        "LAZYMEET_DIAR_DATA_URL",
        "LAZYMEET_DIAR_EMB_URL",
        "LAZYMEET_LLM_URL",
    ]);

    let app = TestApp::with_fake(false);
    assert!(!models::is_ready(&app.ctx));
    models::ensure_models_async(&app.ctx);

    let deadline = Instant::now() + Duration::from_secs(20);
    while !models::is_ready(&app.ctx) {
        assert!(
            Instant::now() < deadline,
            "downloads never finished: {:?}",
            models::status(&app.ctx)
        );
        std::thread::sleep(Duration::from_millis(25));
    }

    let status = models::status(&app.ctx);
    assert_eq!(status["ready"], true);
    assert_eq!(status["state"], "ready");
    assert!(status["error"].is_null());

    // Files landed where the pipeline expects them, extracted from archives.
    let m = &app.ctx.paths.models_dir;
    assert_eq!(std::fs::read(m.join("parakeet/encoder.onnx")).unwrap(), b"parakeet-encoder");
    assert_eq!(std::fs::read(m.join("parakeet/tokens.txt")).unwrap(), b"tokens");
    // The diarizer's graph and its external weights, side by side.
    let diar = m.join("nemotron-3-diarization");
    assert_eq!(std::fs::read(diar.join("model_quantized.onnx")).unwrap(), b"nemotron-graph");
    assert_eq!(std::fs::read(diar.join("model_quantized.onnx_data")).unwrap(), b"nemotron-weights");
    assert_eq!(std::fs::read(m.join("silero_vad.onnx")).unwrap(), b"vad-model");
    assert_eq!(std::fs::read(m.join("speaker_embedding.onnx")).unwrap(), b"embedding-model");
    assert_eq!(std::fs::read(m.join("qwen3.5-4b-q4_k_m.gguf")).unwrap(), b"gguf-model");

    // No temp/staging leftovers.
    for entry in std::fs::read_dir(m).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(!name.ends_with(".download") && !name.ends_with(".part"), "leftover {name}");
    }

    // Progress events were emitted, ending in a ready state.
    let payloads = app.events.payloads("setup-progress");
    assert!(!payloads.is_empty());
    assert_eq!(payloads.last().unwrap()["state"], "ready");
}

#[test]
fn qwen3_asr_archive_lands_where_the_recognizer_looks() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let port = spawn_server();
    std::env::set_var(
        "LAZYMEET_QWEN3_ASR_URL",
        format!("http://127.0.0.1:{port}/qwen3-asr.tar.bz2"),
    );
    let _guard = EnvGuard(vec!["LAZYMEET_QWEN3_ASR_URL"]);

    let app = TestApp::with_fake(false);
    assert!(models::request_download(&app.ctx, "asr", "qwen3-asr-0.6b"));
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let paths = models::resolve_asr(&app.ctx, "qwen3-asr-0.6b");
        if paths.kind == "qwen3_asr" {
            break;
        }
        let status = models::models_status(&app.ctx);
        let opt = status["asr"]["options"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["id"] == "qwen3-asr-0.6b")
            .cloned()
            .unwrap();
        assert!(opt["error"].is_null(), "download failed: {opt}");
        assert!(Instant::now() < deadline, "download never finished: {opt}");
        std::thread::sleep(Duration::from_millis(25));
    }

    let paths = models::resolve_asr(&app.ctx, "qwen3-asr-0.6b");
    assert_eq!(std::fs::read(&paths.conv_frontend).unwrap(), b"frontend");
    assert_eq!(std::fs::read(&paths.encoder).unwrap(), b"qwen-encoder");
    assert_eq!(std::fs::read(&paths.decoder).unwrap(), b"qwen-decoder");
    for (file, content) in [
        ("vocab.json", &b"vocab"[..]),
        ("merges.txt", b"merges"),
        ("tokenizer_config.json", b"tok-config"),
    ] {
        assert_eq!(std::fs::read(paths.tokenizer.join(file)).unwrap(), content);
    }
}
