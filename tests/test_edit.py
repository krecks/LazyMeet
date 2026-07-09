"""Editing endpoints: meeting PATCH, segment PATCH, title plumbing."""

import threading
import time


def _process_meeting(client, name="edit.mp4"):
    res = client.post(
        "/api/meetings",
        files={"file": (name, b"fake video bytes")},
        data={"num_speakers": "-1"},
    )
    assert res.status_code == 200
    meeting_id = res.json()["id"]
    deadline = time.time() + 15
    while time.time() < deadline:
        meeting = client.get(f"/api/meetings/{meeting_id}").json()
        if meeting["status"] in ("done", "error"):
            assert meeting["status"] == "done", meeting.get("error")
            return meeting
        time.sleep(0.2)
    raise AssertionError("meeting never finished")


# --------------------------------------------------------------------------- #
# Title
# --------------------------------------------------------------------------- #
def test_title_rename_flows_everywhere(client):
    meeting = _process_meeting(client)
    mid = meeting["id"]
    assert meeting["title"] == "edit.mp4"  # defaults to the upload filename

    res = client.patch(f"/api/meetings/{mid}", json={"title": "  Weekly Standup  "})
    assert res.status_code == 200
    assert res.json()["title"] == "Weekly Standup"
    assert res.json()["filename"] == "edit.mp4"  # original stays immutable

    listed = {m["id"]: m for m in client.get("/api/meetings").json()}
    assert listed[mid]["title"] == "Weekly Standup"

    export = client.get(f"/api/meetings/{mid}/export")
    assert export.text.startswith("# Weekly Standup")
    assert 'filename="Weekly-Standup.md"' in export.headers["content-disposition"]

    client.delete(f"/api/meetings/{mid}")


def test_title_rejects_empty_and_too_long(client):
    meeting = _process_meeting(client)
    mid = meeting["id"]
    assert client.patch(f"/api/meetings/{mid}", json={"title": "   "}).status_code == 422
    assert client.patch(f"/api/meetings/{mid}", json={"title": "x" * 201}).status_code == 422
    client.delete(f"/api/meetings/{mid}")


def test_title_editable_while_processing(client):
    from backend import store

    record = store.create_meeting("busy.mp4")
    store.update_meeting(record["id"], status="transcribing")
    res = client.patch(f"/api/meetings/{record['id']}", json={"title": "Renamed early"})
    assert res.status_code == 200
    assert res.json()["title"] == "Renamed early"
    store.delete_meeting(record["id"])


def test_old_records_without_title_get_fallback(client):
    from backend import store

    record = store.create_meeting("legacy.mp4")
    # Simulate a record from before titles existed.
    path = store._meeting_path(record["id"])
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    del data["title"]
    path.write_text(json.dumps(data), encoding="utf-8")

    assert client.get(f"/api/meetings/{record['id']}").json()["title"] == "legacy.mp4"
    listed = {m["id"]: m for m in client.get("/api/meetings").json()}
    assert listed[record["id"]]["title"] == "legacy.mp4"
    store.delete_meeting(record["id"])


# --------------------------------------------------------------------------- #
# Summary / takeaways / action items
# --------------------------------------------------------------------------- #
def test_content_edit_on_done_meeting(client):
    meeting = _process_meeting(client)
    mid = meeting["id"]
    res = client.patch(
        f"/api/meetings/{mid}",
        json={
            "summary": "We agreed on the plan.",
            "takeaways": ["Ship it", "  ", "", "Test it  "],
            "action_items": ["Alice: deploy"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["summary"] == "We agreed on the plan."
    assert data["takeaways"] == ["Ship it", "Test it"]  # empties dropped, stripped
    assert data["action_items"] == ["Alice: deploy"]

    export = client.get(f"/api/meetings/{mid}/export").text
    assert "We agreed on the plan." in export
    assert "- [ ] Alice: deploy" in export
    client.delete(f"/api/meetings/{mid}")


def test_content_edit_blocked_while_processing_or_error(client):
    from backend import store

    record = store.create_meeting("locked.mp4")
    for status in ("queued", "summarizing", "error"):
        store.update_meeting(record["id"], status=status)
        res = client.patch(f"/api/meetings/{record['id']}", json={"summary": "nope"})
        assert res.status_code == 409, status
    store.delete_meeting(record["id"])


def test_patch_with_no_fields_is_422(client):
    meeting = _process_meeting(client)
    assert client.patch(f"/api/meetings/{meeting['id']}", json={}).status_code == 422
    client.delete(f"/api/meetings/{meeting['id']}")


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #
def test_segment_text_edit(client):
    meeting = _process_meeting(client)
    mid = meeting["id"]
    res = client.patch(f"/api/meetings/{mid}/segments/0", json={"text": "Corrected words."})
    assert res.status_code == 200
    assert res.json()["segments"][0]["text"] == "Corrected words."
    # other segments untouched
    assert res.json()["segments"][1:] == meeting["segments"][1:]
    client.delete(f"/api/meetings/{mid}")


def test_segment_speaker_reassignment(client):
    meeting = _process_meeting(client)
    mid = meeting["id"]
    ids = list(meeting["speakers"])
    assert len(ids) >= 2, "fake pipeline should produce two speakers"
    other = ids[1] if meeting["segments"][0]["speaker"] == ids[0] else ids[0]

    res = client.patch(f"/api/meetings/{mid}/segments/0", json={"speaker": other})
    assert res.status_code == 200
    assert res.json()["segments"][0]["speaker"] == other
    # speakers map and rename history stay intact
    assert res.json()["speakers"] == meeting["speakers"]
    client.delete(f"/api/meetings/{mid}")


def test_segment_edit_error_cases(client):
    meeting = _process_meeting(client)
    mid = meeting["id"]
    base = f"/api/meetings/{mid}/segments"
    assert client.patch(f"{base}/999", json={"text": "x"}).status_code == 404
    assert client.patch(f"{base}/0", json={"speaker": "SPEAKER_99"}).status_code == 422
    assert client.patch(f"{base}/0", json={"text": "   "}).status_code == 422
    assert client.patch(f"{base}/0", json={}).status_code == 422
    assert client.patch("/api/meetings/nope/segments/0", json={"text": "x"}).status_code == 404
    client.delete(f"/api/meetings/{mid}")


def test_segment_edit_blocked_while_processing(client):
    from backend import store

    record = store.create_meeting("busyseg.mp4")
    store.update_meeting(
        record["id"],
        status="summarizing",
        segments=[{"start": 0, "end": 1, "speaker": "s0", "text": "hi"}],
        speakers={"s0": "Speaker 1"},
    )
    res = client.patch(f"/api/meetings/{record['id']}/segments/0", json={"text": "x"})
    assert res.status_code == 409
    store.delete_meeting(record["id"])


def test_concurrent_segment_edits_both_survive(client):
    from backend import store

    meeting = _process_meeting(client)
    mid = meeting["id"]
    assert len(meeting["segments"]) >= 2

    threads = [
        threading.Thread(target=store.update_segment, args=(mid, i), kwargs={"text": f"edit {i}"})
        for i in (0, 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record = store.get_meeting(mid)
    assert record["segments"][0]["text"] == "edit 0"
    assert record["segments"][1]["text"] == "edit 1"
    store.delete_meeting(mid)
