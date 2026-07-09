"""Unit tests for the diarization consolidation pass (no models needed)."""


def _turn(spk, start, dur):
    return {"start": start, "end": start + dur, "speaker": spk}


def test_oversplit_labels_merge_to_real_speakers():
    from backend.pipeline.identify import consolidate_speakers

    # A and A2 are the same voice; B is distinct; N is sub-second noise.
    vp = {
        "A": [1.0, 0.0, 0.0],
        "A2": [0.95, 0.1, 0.0],
        "B": [0.0, 1.0, 0.0],
        "N": [0.0, 0.0, 1.0],
    }
    turns = [
        _turn("A", 0, 100),
        _turn("B", 100, 50),
        _turn("A2", 150, 10),
        _turn("N", 160, 1),
    ]
    out_turns, out_vp = consolidate_speakers(turns, vp)
    assert set(out_vp) == {"A", "B"}
    assert [t["speaker"] for t in out_turns] == ["A", "B", "A"]


def test_tiny_speaker_folds_into_closest_main_speaker():
    from backend.pipeline.identify import consolidate_speakers

    # cos(C, A) = 0.5 — below the merge threshold but above the tiny floor.
    vp = {
        "A": [1.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0],
        "C": [0.5, 0.0, 0.8660254],
    }
    turns = [_turn("A", 0, 60), _turn("B", 60, 60), _turn("C", 120, 8)]
    out_turns, out_vp = consolidate_speakers(turns, vp)
    assert set(out_vp) == {"A", "B"}
    assert [t["speaker"] for t in out_turns] == ["A", "B", "A"]


def test_quiet_distinct_speaker_survives():
    from backend.pipeline.identify import consolidate_speakers

    # 10 s of a genuinely different voice must not be folded or dropped.
    vp = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0]}
    turns = [_turn("A", 0, 60), _turn("B", 60, 10)]
    out_turns, out_vp = consolidate_speakers(turns, vp)
    assert set(out_vp) == {"A", "B"}
    assert len(out_turns) == 2


def test_short_clip_without_main_speaker_kept_as_is():
    from backend.pipeline.identify import consolidate_speakers

    # Both under MIN_SPEAKER_SECONDS: no main speaker, nothing gets touched.
    vp = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0]}
    turns = [_turn("A", 0, 8), _turn("B", 8, 6)]
    out_turns, out_vp = consolidate_speakers(turns, vp)
    assert set(out_vp) == {"A", "B"}
    assert len(out_turns) == 2


def test_fake_embeddings_stay_distinct():
    from backend import config
    from backend.pipeline.identify import _fake_embedding, cosine

    # The FAKE pipeline's two speakers must survive consolidation.
    sim = cosine(_fake_embedding("SPEAKER_00"), _fake_embedding("SPEAKER_01"))
    assert sim < config.SPEAKER_MERGE_THRESHOLD
