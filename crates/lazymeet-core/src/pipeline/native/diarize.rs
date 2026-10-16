//! Diarization: Nemotron (via onnxruntime) or classic pyannote inside
//! sherpa-onnx, plus the shared speaker-embedding extractor (port of `diarize.py`).

use super::asr::{load_wav, sample_ranges};
use super::*;

/// The classic diarizer: pyannote segmentation + TitaNet embeddings +
/// clustering, inside sherpa-onnx.
pub(super) fn pyannote_diarizer(
    segmentation: &Path,
    models_dir: &Path,
    num_clusters: i32,
    threshold: f32,
    provider: &str,
    threads: i32,
) -> Result<OfflineSpeakerDiarization, String> {
    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: path_str(segmentation),
                ..Default::default()
            },
            num_threads: threads,
            debug: false,
            provider: Some(provider.to_string()),
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: path_str(&models_dir.join("speaker_embedding.onnx")),
            num_threads: threads,
            debug: false,
            provider: Some(provider.to_string()),
        },
        // num_clusters > 0 forces an exact speaker count; -1 lets the
        // threshold decide (higher => fewer speakers).
        clustering: FastClusteringConfig { num_clusters, threshold, compute_confidence: false },
        // smooth out sub-second fragments that would explode into hundreds
        // of spurious speakers
        min_duration_on: 0.3,
        min_duration_off: 0.5,
    };
    OfflineSpeakerDiarization::create(&config)
        .ok_or_else(|| "could not load the classic diarization models".into())
}

/// Diarize `samples` with the given engine: (start, end, speaker index).
fn diarize_samples(
    ctx: &PipelineCtx,
    engine: &str,
    model: &Path,
    samples: &[f32],
    num_speakers: i64,
    threads: i32,
) -> Result<Vec<(f64, f64, usize)>, String> {
    if engine == "nemotron" {
        let probs = Nemotron::load(model, threads)?.probabilities(samples)?;
        return Ok(nemotron::turns(&probs, num_speakers)
            .into_iter()
            .map(|t| (t.start, t.end, t.speaker))
            .collect());
    }
    let diarizer = pyannote_diarizer(
        model,
        ctx.models_dir,
        num_speakers as i32,
        ctx.config.diar_threshold() as f32,
        &ctx.config.diar_provider(),
        threads,
    )?;
    let result = diarizer.process(samples).ok_or("diarization failed")?;
    Ok(result
        .sort_by_start_time()
        .into_iter()
        .map(|s| (s.start as f64, s.end as f64, s.speaker.max(0) as usize))
        .collect())
}

pub fn diarize(
    ctx: &PipelineCtx,
    wav: &Path,
    num_speakers: i64,
    ranges: Option<&[(f64, f64)]>,
    threads: i32,
) -> Result<Vec<Turn>, String> {
    // A meeting-supplied count (>0) wins; otherwise the env default (-1 auto).
    let n = if num_speakers > 0 { num_speakers } else { config::env_num_speakers() };
    let (engine, model) = resolve_diar_at(ctx.models_dir, ctx.fake, &ctx.config.diar_model(), n);

    let samples = load_wav(wav)?;
    // With ranges, the diarizer runs once over the spliced selection (speaker
    // clustering must see all spans together); the concat-time turns are then
    // remapped — splitting at splice boundaries — back to original time.
    let (input, remap) = match ranges {
        None => (samples, None),
        Some(rs) => {
            let spans = sample_ranges(rs, samples.len());
            let mut concat = Vec::with_capacity(spans.iter().map(|(a, b)| b - a).sum());
            for &(a, b) in &spans {
                concat.extend_from_slice(&samples[a..b]);
            }
            if concat.is_empty() {
                return Ok(Vec::new());
            }
            // Remap against the *clamped* spans so a range past the end of
            // the audio maps consistently with what was actually fed.
            let secs: Vec<(f64, f64)> = spans
                .iter()
                .map(|&(a, b)| (a as f64 / SAMPLE_RATE as f64, b as f64 / SAMPLE_RATE as f64))
                .collect();
            (concat, Some(secs))
        }
    };
    let mut turns: Vec<Turn> = diarize_samples(ctx, engine, &model, &input, n, threads)?
        .into_iter()
        .map(|(start, end, speaker)| Turn {
            start: round3(start),
            end: round3(end),
            speaker: format!("SPEAKER_{speaker:02}"),
        })
        .collect();
    if let Some(secs) = remap {
        turns = remap_turns(&secs, &turns);
    }
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    Ok(turns)
}

// --------------------------------------------------------------------------
// Voiceprints (port of identify.py's extractor half)
// --------------------------------------------------------------------------

pub(super) fn new_extractor(
    models_dir: &Path,
    provider: &str,
    threads: i32,
) -> Result<SpeakerEmbeddingExtractor, String> {
    let model = models_dir.join("speaker_embedding.onnx");
    if !model.exists() {
        return Err(format!("speaker model not found at {}", model.display()));
    }
    SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig {
        model: path_str(&model),
        num_threads: threads,
        debug: false,
        provider: Some(provider.to_string()),
    })
    .ok_or_else(|| "could not load the speaker model".into())
}

/// The voice embedding of one stretch of audio; None if it is too short.
pub(super) fn speaker_embedding(
    extractor: &SpeakerEmbeddingExtractor,
    samples: &[f32],
) -> Option<Vec<f32>> {
    let stream = extractor.create_stream()?;
    stream.accept_waveform(SAMPLE_RATE as i32, samples);
    stream.input_finished();
    if !extractor.is_ready(&stream) {
        return None;
    }
    extractor.compute(&stream)
}
