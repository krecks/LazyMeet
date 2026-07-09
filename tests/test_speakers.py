"""Speaker library: renames, enrolment learning, and match verification."""

import pytest


def test_rename_keeps_voiceprint(client):
    from backend import store

    store.upsert_speaker("Alice", [0.1, 0.2, 0.3])
    res = client.patch("/api/speakers/Alice", json={"name": "  Alice Weber "})
    assert res.status_code == 200
    names = res.json()
    assert "Alice Weber" in names and "Alice" not in names

    lib = store.load_speaker_library()
    assert lib["Alice Weber"]["embedding"] == [0.1, 0.2, 0.3]
    store.delete_speaker("Alice Weber")


def test_rename_missing_is_404(client):
    assert client.patch("/api/speakers/Nobody", json={"name": "X"}).status_code == 404


def test_rename_conflict_is_409(client):
    from backend import store

    store.upsert_speaker("Alice", [0.1])
    store.upsert_speaker("Bob", [0.2])
    res = client.patch("/api/speakers/Alice", json={"name": "Bob"})
    assert res.status_code == 409
    # both untouched
    assert set(store.get_speaker_names()) >= {"Alice", "Bob"}
    store.delete_speaker("Alice")
    store.delete_speaker("Bob")


def test_rename_invalid_names_are_422(client):
    from backend import store

    store.upsert_speaker("Alice", [0.1])
    assert client.patch("/api/speakers/Alice", json={"name": "   "}).status_code == 422
    assert client.patch("/api/speakers/Alice", json={"name": "x" * 101}).status_code == 422
    store.delete_speaker("Alice")


def test_meeting_rename_moves_library_entry(client):
    """Renaming an enrolled speaker in a meeting must not leave a duplicate."""
    from backend import store

    record = store.create_meeting("lib.mp4")
    store.update_meeting(
        record["id"],
        speakers={"spk_0": "Speaker 1"},
        voiceprints={"spk_0": [0.1, 0.2]},
    )
    store.merge_speaker_names(record["id"], {"spk_0": "Christine Schneider"})
    assert "Christine Schneider" in store.get_speaker_names()

    store.merge_speaker_names(record["id"], {"spk_0": "Christine Weber"})
    names = store.get_speaker_names()
    assert "Christine Weber" in names
    assert "Christine Schneider" not in names  # renamed, not duplicated

    store.delete_meeting(record["id"])
    store.delete_speaker("Christine Weber")


def test_meeting_rename_merges_into_existing_entry(client):
    """Renaming a voice to a name already in the library keeps one entry."""
    from backend import store

    store.upsert_speaker("Anna", [0.9, 0.9])
    record = store.create_meeting("merge.mp4")
    store.update_meeting(
        record["id"],
        speakers={"spk_0": "Speaker 1"},
        voiceprints={"spk_0": [0.1, 0.2]},
    )
    store.merge_speaker_names(record["id"], {"spk_0": "Nana"})
    store.merge_speaker_names(record["id"], {"spk_0": "Anna"})
    names = store.get_speaker_names()
    assert "Anna" in names and "Nana" not in names
    # the fresh voice refines the entry: matching uses the sample mean
    entry = store.load_speaker_library()["Anna"]
    assert entry["embedding"] == pytest.approx([0.5, 0.55])
    assert len(entry["samples"]) == 2

    store.delete_meeting(record["id"])
    store.delete_speaker("Anna")


def test_upsert_learns_running_mean(client):
    """Each enrolment adds a sample; matching uses their mean, so voiceprints
    get more robust with every file instead of the last voice overwriting."""
    from backend import store

    store.upsert_speaker("Olga", [1.0, 0.0])
    store.upsert_speaker("Olga", [0.0, 1.0])
    entry = store.load_speaker_library()["Olga"]
    assert entry["samples"] == [[1.0, 0.0], [0.0, 1.0]]
    assert entry["embedding"] == [0.5, 0.5]
    store.delete_speaker("Olga")


def test_correcting_wrong_match_keeps_victim_entry(client):
    """Renaming a wrongly auto-recognized speaker must not move or delete the
    matched person's library entry — only enrol the voice under the new name."""
    from backend import store

    store.upsert_speaker("Christine Schneider", [0.9, 0.9])
    record = store.create_meeting("wrong.mp4")
    store.update_meeting(
        record["id"],
        speakers={"spk_0": "Christine Schneider"},
        recognized=["spk_0"],
        voiceprints={"spk_0": [0.1, 0.2]},
    )
    store.merge_speaker_names(record["id"], {"spk_0": "Miriam"})

    lib = store.load_speaker_library()
    assert lib["Christine Schneider"]["embedding"] == [0.9, 0.9]  # untouched
    assert lib["Miriam"]["embedding"] == [0.1, 0.2]  # enrolled correctly
    assert store.get_meeting(record["id"])["recognized"] == []

    store.delete_meeting(record["id"])
    store.delete_speaker("Christine Schneider")
    store.delete_speaker("Miriam")


def test_confirm_recognized_speaker_learns(client):
    from backend import store

    store.upsert_speaker("Dimitri", [1.0, 0.0])
    record = store.create_meeting("confirm.mp4")
    store.update_meeting(
        record["id"],
        speakers={"spk_0": "Dimitri"},
        recognized=["spk_0"],
        voiceprints={"spk_0": [0.0, 1.0]},
    )
    res = client.post(
        f"/api/meetings/{record['id']}/speakers/spk_0/verify", json={"correct": True}
    )
    assert res.status_code == 200
    assert res.json()["confirmed"] == ["spk_0"]
    # this recording became another sample for Dimitri
    assert store.load_speaker_library()["Dimitri"]["embedding"] == [0.5, 0.5]

    store.delete_meeting(record["id"])
    store.delete_speaker("Dimitri")


def test_reject_recognized_speaker_unlinks_without_poison(client):
    from backend import store

    store.upsert_speaker("Christine Schneider", [0.9, 0.9])
    record = store.create_meeting("reject.mp4")
    store.update_meeting(
        record["id"],
        speakers={"spk_0": "Christine Schneider", "spk_1": "Speaker 1"},
        recognized=["spk_0"],
        voiceprints={"spk_0": [0.1, 0.2]},
        summary="Christine Schneider owns the rollout.",
    )
    res = client.post(
        f"/api/meetings/{record['id']}/speakers/spk_0/verify", json={"correct": False}
    )
    assert res.status_code == 200
    meeting = res.json()
    assert meeting["speakers"]["spk_0"] == "Speaker 2"  # "Speaker 1" was taken
    assert meeting["recognized"] == []
    assert meeting["summary"] == "Speaker 2 owns the rollout."
    # the wrongly matched person's voiceprint is untouched
    assert store.load_speaker_library()["Christine Schneider"]["embedding"] == [0.9, 0.9]

    # nothing left to confirm — the id no longer counts as recognized
    res = client.post(
        f"/api/meetings/{record['id']}/speakers/spk_0/verify", json={"correct": True}
    )
    assert res.status_code == 409

    store.delete_meeting(record["id"])
    store.delete_speaker("Christine Schneider")
