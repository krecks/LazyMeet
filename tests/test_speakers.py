"""Speaker library rename: PATCH /api/speakers/{name}."""


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
    # the freshly heard voice wins the entry
    assert store.load_speaker_library()["Anna"]["embedding"] == [0.1, 0.2]

    store.delete_meeting(record["id"])
    store.delete_speaker("Anna")
