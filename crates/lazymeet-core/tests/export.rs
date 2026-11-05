//! Meeting exports in every format, subtitle timing, and bulk export.

mod common;

use common::{changes, TestApp};
use lazymeet_core::{api, export};
use serde_json::{json, Map, Value};

fn record(segments: Value) -> Map<String, Value> {
    json!({
        "id": "m1",
        "title": "Größe sync",
        "created_at": "2026-11-05T10:00:00.000000+00:00",
        "summary": "Short.",
        "summary_long": "Para one.\n\nPara two.",
        "takeaways": [{"short": "A point", "detail": "Why it matters"}],
        "action_items": [{"short": "Anna: ship", "detail": ""}],
        "speakers": {"spk_0": "Anna", "spk_1": "Ben <PM>"},
        "segments": segments,
    })
    .as_object()
    .unwrap()
    .clone()
}

#[test]
fn every_format_exports_from_a_processed_meeting() {
    let app = TestApp::fake();
    let mid = app.upload("weekly.mp4", b"fake video bytes");
    app.wait_terminal(&mid);
    api::patch_meeting(&app.ctx, &mid, &changes("title", json!("Weekly sync"))).unwrap();

    for format in export::Format::ALL {
        let (bytes, filename) = api::export_meeting(&app.ctx, &mid, format.ext()).unwrap();
        assert!(!bytes.is_empty(), "{format:?} is empty");
        assert_eq!(filename, format!("Weekly-sync.{}", format.ext()));
        match format {
            export::Format::Docx => assert!(bytes.starts_with(b"PK"), "docx is a zip"),
            export::Format::Pdf => assert!(bytes.starts_with(b"%PDF"), "pdf header"),
            export::Format::Json => {
                let v: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(v["title"], "Weekly sync");
                assert!(v.get("versions").is_none(), "no undo history in the export");
                assert!(v.get("voiceprints").is_none(), "never leak voiceprints");
            }
            export::Format::Srt => assert!(String::from_utf8(bytes).unwrap().starts_with("1\n")),
            export::Format::Vtt => {
                assert!(String::from_utf8(bytes).unwrap().starts_with("WEBVTT\n\n"))
            }
            _ => assert!(String::from_utf8(bytes).unwrap().contains("Weekly sync")),
        }
    }
    let err = api::export_meeting(&app.ctx, &mid, "exe").unwrap_err();
    assert_eq!(err.code, 422);
    assert_eq!(api::export_meeting(&app.ctx, "nope", "md").unwrap_err().code, 404);
}

#[test]
fn long_segments_become_several_readable_cues() {
    let long = "word ".repeat(60); // 300 chars over 20 s
    let rec = record(json!([
        {"start": 1.0, "end": 3.5, "speaker": "spk_0", "text": "Hello there."},
        {"start": 4.0, "end": 24.0, "speaker": "spk_1", "text": long.trim()},
    ]));
    let cues = export::cues(&rec);
    assert_eq!(
        cues[0],
        export::Cue { start: 1.0, end: 3.5, speaker: "Anna".into(), text: "Hello there.".into() }
    );
    let rest = &cues[1..];
    assert!(rest.len() >= 4, "20 s / 300 chars split into at least 4 cues, got {}", rest.len());
    assert_eq!(rest.first().unwrap().start, 4.0);
    assert_eq!(rest.last().unwrap().end, 24.0);
    for pair in rest.windows(2) {
        assert!((pair[0].end - pair[1].start).abs() < 1e-9, "cues are contiguous");
    }
    for cue in rest {
        assert!(cue.text.chars().count() <= 84 + 5, "cue too long: {}", cue.text);
        assert!(cue.end - cue.start <= 7.0 + 1e-9);
    }
    let words: Vec<String> = rest.iter().map(|c| c.text.clone()).collect();
    assert_eq!(words.join(" "), long.trim(), "no word lost or duplicated");

    let srt = export::to_subtitles(&rec, false);
    assert!(srt.starts_with("1\n00:00:01,000 --> 00:00:03,500\nAnna: Hello there.\n\n2\n"));
    let vtt = export::to_subtitles(&rec, true);
    assert!(vtt.contains("00:00:01.000 --> 00:00:03.500\n<v Anna>Hello there.\n"));
    assert!(vtt.contains("<v Ben PM>"), "markup stripped from voice names: {vtt}");
}

#[test]
fn documents_carry_the_notes_and_transcript() {
    let rec =
        record(json!([{"start": 0.0, "end": 2.0, "speaker": "spk_0", "text": "Grüße aus Mainz."}]));
    let txt = export::to_text(&rec);
    assert!(txt.starts_with("Größe sync\n\nShort.\n\nPara one.\n\nPara two.\n\n"));
    assert!(txt.contains("Key takeaways:\n- A point\n  Why it matters\n"));
    assert!(txt.contains("Action items:\n- [ ] Anna: ship\n"));
    assert!(txt.contains("Anna [00:00]: Grüße aus Mainz."));

    let (pdf, name) = export::export(&rec, export::Format::Pdf).unwrap();
    assert_eq!(name, "Größe-sync.pdf");
    assert!(pdf.len() < 200_000, "fonts are subset, not embedded whole: {} bytes", pdf.len());
    let (docx, _) = export::export(&rec, export::Format::Docx).unwrap();
    assert!(docx.starts_with(b"PK"));
}

#[test]
fn bulk_export_writes_one_file_each_and_never_overwrites() {
    let app = TestApp::fake();
    let a = app.upload("a.mp4", b"fake video bytes");
    let b = app.upload("b.mp4", b"fake video bytes");
    app.wait_terminal(&a);
    app.wait_terminal(&b);
    // Same title for both: the second gets a suffix.
    for id in [&a, &b] {
        api::patch_meeting(&app.ctx, id, &changes("title", json!("Standup"))).unwrap();
    }
    let dir = app.tmp.path().join("out");
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("Standup.txt"), "already here").unwrap();

    let ids = vec![a.clone(), "gone".to_string(), b.clone()];
    let written = api::export_meetings(&app.ctx, &ids, "txt", &dir).unwrap();
    let names: Vec<String> = written
        .iter()
        .map(|p| std::path::Path::new(p).file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, vec!["Standup (2).txt", "Standup (3).txt"]);
    assert_eq!(std::fs::read_to_string(dir.join("Standup.txt")).unwrap(), "already here");

    let missing = app.tmp.path().join("nope");
    assert_eq!(api::export_meetings(&app.ctx, &ids, "txt", &missing).unwrap_err().code, 422);
}
