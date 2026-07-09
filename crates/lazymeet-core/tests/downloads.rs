//! First-run download machinery against a local HTTP server: concurrent
//! fetches, progress bookkeeping, atomic tar.bz2 extraction, ready-flip.
//! (Replaces test_first_run_downloads_run_in_parallel, which monkeypatched
//! the downloader; this exercises the real one.)
//!
//! Sets LAZYMEET_*_URL env vars — single test in this binary, and cargo runs
//! test binaries sequentially, so nothing races the process env.

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use common::TestApp;
use lazymeet_core::models;

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
        "/seg.tar.bz2" => tar_bz2(&[("seg/model.onnx", b"segmentation")]),
        "/vad.onnx" => b"vad-model".to_vec(),
        "/embedding.onnx" => b"embedding-model".to_vec(),
        "/llm.gguf" => b"gguf-model".to_vec(),
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
    let port = spawn_server();
    let url = |p: &str| format!("http://127.0.0.1:{port}{p}");
    std::env::set_var("LAZYMEET_ASR_URL", url("/parakeet.tar.bz2"));
    std::env::set_var("LAZYMEET_VAD_URL", url("/vad.onnx"));
    std::env::set_var("LAZYMEET_DIAR_SEG_URL", url("/seg.tar.bz2"));
    std::env::set_var("LAZYMEET_DIAR_EMB_URL", url("/embedding.onnx"));
    std::env::set_var("LAZYMEET_LLM_URL", url("/llm.gguf"));
    let _guard = EnvGuard(vec![
        "LAZYMEET_ASR_URL",
        "LAZYMEET_VAD_URL",
        "LAZYMEET_DIAR_SEG_URL",
        "LAZYMEET_DIAR_EMB_URL",
        "LAZYMEET_LLM_URL",
    ]);

    let app = TestApp::with_fake(false);
    assert!(!models::is_ready(&app.ctx));
    models::ensure_models_async(&app.ctx);

    let deadline = Instant::now() + Duration::from_secs(20);
    while !models::is_ready(&app.ctx) {
        assert!(Instant::now() < deadline, "downloads never finished: {:?}", models::status(&app.ctx));
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
    assert_eq!(std::fs::read(m.join("diarization_segmentation.onnx")).unwrap(), b"segmentation");
    assert_eq!(std::fs::read(m.join("silero_vad.onnx")).unwrap(), b"vad-model");
    assert_eq!(std::fs::read(m.join("speaker_embedding.onnx")).unwrap(), b"embedding-model");
    assert_eq!(std::fs::read(m.join("qwen3.5-4b-q4_k_m.gguf")).unwrap(), b"gguf-model");

    // No temp/staging leftovers.
    for entry in std::fs::read_dir(m).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(
            !name.ends_with(".download") && !name.ends_with(".part"),
            "leftover {name}"
        );
    }

    // Progress events were emitted, ending in a ready state.
    let payloads = app.events.payloads("setup-progress");
    assert!(!payloads.is_empty());
    assert_eq!(payloads.last().unwrap()["state"], "ready");
}
