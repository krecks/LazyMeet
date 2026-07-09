"""End-to-end smoke tests over the FAKE_MODE pipeline (no models needed)."""

import threading
import time


def _wait_terminal(client, meeting_id, timeout=15.0):
    deadline = time.time() + timeout
    meeting = None
    while time.time() < deadline:
        res = client.get(f"/api/meetings/{meeting_id}")
        assert res.status_code == 200
        meeting = res.json()
        if meeting["status"] in ("done", "error"):
            return meeting
        time.sleep(0.2)
    raise AssertionError(f"meeting stuck in {meeting and meeting['status']!r}")


def test_full_lifecycle(client):
    res = client.post(
        "/api/meetings",
        files={"file": ("standup.mp4", b"fake video bytes")},
        data={"num_speakers": "-1"},
    )
    assert res.status_code == 200
    meeting_id = res.json()["id"]

    meeting = _wait_terminal(client, meeting_id)
    assert meeting["status"] == "done", meeting.get("error")
    assert meeting["segments"]
    assert meeting["speakers"]
    assert meeting["summary"]
    assert "voiceprints" not in meeting
    assert "worker_pid" not in meeting

    speaker_id = next(iter(meeting["speakers"]))
    res = client.patch(
        f"/api/meetings/{meeting_id}/speakers",
        json={"names": {speaker_id: "Alice"}},
    )
    assert res.status_code == 200
    assert res.json()["speakers"][speaker_id] == "Alice"
    assert "Alice" in client.get("/api/speakers").json()

    res = client.patch(f"/api/meetings/{meeting_id}", json={"title": "Daily Standup"})
    assert res.status_code == 200

    res = client.get(f"/api/meetings/{meeting_id}/export")
    assert res.status_code == 200
    assert res.text.startswith("# Daily Standup")
    assert "Alice" in res.text

    assert client.delete(f"/api/meetings/{meeting_id}").status_code == 200
    assert client.get(f"/api/meetings/{meeting_id}").status_code == 404
    assert client.delete("/api/speakers/Alice").status_code == 200


def test_worker_write_preserves_user_renames(client):
    from backend import store

    record = store.create_meeting("clobber.mp4")
    store.update_meeting(record["id"], speakers={"spk_0": "Alice"})
    # The worker's summarizing write must not overwrite user-set names.
    store.update_meeting(
        record["id"],
        preserve_speakers=True,
        speakers={"spk_0": "Speaker 1", "spk_1": "Speaker 2"},
    )
    meeting = store.get_meeting(record["id"])
    assert meeting["speakers"] == {"spk_0": "Alice", "spk_1": "Speaker 2"}
    store.delete_meeting(record["id"])


def test_concurrent_renames_both_survive(client):
    from backend import store

    record = store.create_meeting("race.mp4")
    store.update_meeting(record["id"], speakers={"a": "a", "b": "b"})
    threads = [
        threading.Thread(target=store.merge_speaker_names, args=(record["id"], names))
        for names in ({"a": "Alice"}, {"b": "Bob"})
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    meeting = store.get_meeting(record["id"])
    assert meeting["speakers"] == {"a": "Alice", "b": "Bob"}
    store.delete_meeting(record["id"])


def test_corrupt_meeting_json_is_tolerated(client):
    from backend import config

    bad = config.MEETINGS_DIR / "corrupt.json"
    bad.write_text("{not json", encoding="utf-8")
    try:
        assert client.get("/api/meetings").status_code == 200
        assert client.get("/api/meetings/corrupt").status_code == 404
    finally:
        bad.unlink()


def test_orphan_sweep_marks_dead_worker_error():
    from fastapi.testclient import TestClient

    from backend import store
    from backend.main import app

    record = store.create_meeting("orphan.mp4")
    store.update_meeting(record["id"], status="transcribing", worker_pid=99999999)
    with TestClient(app) as client:  # startup runs the sweep
        meeting = client.get(f"/api/meetings/{record['id']}").json()
    assert meeting["status"] == "error"
    assert "interrupted" in meeting["error"]
    store.delete_meeting(record["id"])
