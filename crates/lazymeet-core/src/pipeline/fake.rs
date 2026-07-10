//! Deterministic FAKE-mode stubs (`LAZYMEET_FAKE=1`), byte-for-byte ports of
//! the Python stubs so the ported behavioral test suite asserts identical
//! values. The whole app can be exercised without any models on disk.

use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::{MediaProbe, Segment, Summary, TimeRange, Turn, Voiceprints};
use crate::config::FAKE_EMBED_DIM;

/// With `ranges`, the canonical utterances are clipped to the selected spans
/// (dropped when they don't overlap) — the same "only selected audio shows
/// up, timestamps stay in original time" contract as the native pipeline.
pub fn transcribe(ranges: Option<&[TimeRange]>) -> Vec<Segment> {
    let s = |start: f64, end: f64, text: &str| Segment { start, end, text: text.into() };
    let all = vec![
        s(0.0, 3.2, "Hi everyone, thanks for joining the weekly sync."),
        s(3.4, 7.1, "Let's start with the transcription pipeline status."),
        s(7.3, 11.0, "I finished wiring up the diarization step yesterday."),
        s(11.2, 15.4, "Great. Can you also add the speaker library before Friday?"),
        s(15.6, 19.0, "Yes, I'll have it ready and send the demo recording."),
    ];
    let Some(ranges) = ranges else { return all };
    let mut out = Vec::new();
    for seg in all {
        for &(s, e) in ranges {
            let lo = seg.start.max(s);
            let hi = seg.end.min(e);
            if hi - lo > 1e-6 {
                out.push(Segment { start: lo, end: hi, text: seg.text.clone() });
            }
        }
    }
    out
}

pub fn diarize(ranges: Option<&[TimeRange]>) -> Vec<Turn> {
    // Two alternating speakers matching the fake transcript timeline.
    let t = |start: f64, end: f64, speaker: &str| Turn { start, end, speaker: speaker.into() };
    let all = vec![
        t(0.0, 7.2, "SPEAKER_00"),
        t(7.2, 11.1, "SPEAKER_01"),
        t(11.1, 15.5, "SPEAKER_00"),
        t(15.5, 19.2, "SPEAKER_01"),
    ];
    let Some(ranges) = ranges else { return all };
    let mut out = Vec::new();
    for turn in all {
        for &(s, e) in ranges {
            let lo = turn.start.max(s);
            let hi = turn.end.min(e);
            if hi - lo > 1e-6 {
                out.push(Turn { start: lo, end: hi, speaker: turn.speaker.clone() });
            }
        }
    }
    out
}

/// Deterministic probe matching the fake timeline (which ends at 19.2 s).
pub fn probe() -> MediaProbe {
    MediaProbe {
        duration: 19.2,
        peaks: (0..192).map(|i| (i as f32 * 0.7).sin().abs() * 0.8 + 0.1).collect(),
    }
}

/// Deterministic unit-ish vector seeded by the speaker label. Same label →
/// same vector, so enrolling a fake speaker and re-processing a stub with the
/// same label exercises the real match/enrol path.
pub fn embedding(speaker_label: &str) -> Vec<f64> {
    let seed = Sha256::digest(speaker_label.as_bytes());
    (0..FAKE_EMBED_DIM)
        .map(|i| {
            let b = seed[i % seed.len()] as f64;
            (b / 255.0) * 2.0 - 1.0
        })
        .collect()
}

pub fn voiceprints(turns: &[Turn]) -> Voiceprints {
    let mut out = Voiceprints::new();
    for t in turns {
        out.entry(t.speaker.clone()).or_insert_with(|| embedding(&t.speaker));
    }
    out
}

pub fn summary() -> Summary {
    Summary {
        summary: "Weekly sync covering the transcription pipeline. The diarization step is \
                  complete and the speaker library is the remaining item before Friday."
            .into(),
        takeaways: vec![
            "Diarization step is implemented.".into(),
            "Speaker library is the next deliverable.".into(),
        ],
        action_items: vec![
            "Add the speaker library before Friday.".into(),
            "Send the demo recording to the team.".into(),
        ],
    }
}

/// Write a tiny valid (silent) 16 kHz mono 16-bit WAV so downstream stubs have
/// a real file.
pub fn write_silent_wav(out_wav: &Path) -> std::io::Result<()> {
    const RATE: u32 = 16000;
    const SECONDS: u32 = 1;
    let n_samples = RATE * SECONDS;
    let data_len = n_samples * 2; // 16-bit mono
    let mut f = std::fs::File::create(out_wav)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&RATE.to_le_bytes())?;
    f.write_all(&(RATE * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    f.write_all(&vec![0u8; data_len as usize])?;
    Ok(())
}
