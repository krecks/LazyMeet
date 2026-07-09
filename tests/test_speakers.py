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
