"""Model/language settings, the models catalog API, and the processing queue."""

import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _clean_overrides():
    from backend import config

    yield
    config.save_overrides({})
    config.refresh()


def test_models_catalog_shape(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    for step, default in (("asr", "parakeet-tdt-0.6b-v3"), ("summary", "qwen3.5-4b")):
        assert data[step]["active"] == default
        ids = [o["id"] for o in data[step]["options"]]
        assert default in ids
        for opt in data[step]["options"]:
            assert {"id", "label", "ready", "downloading", "error", "pros", "cons"} <= set(opt)
            # The settings UI renders a pros/cons overview for every model.
            assert opt["pros"] and opt["cons"]
    asr_ids = [o["id"] for o in data["asr"]["options"]]
    assert "whisper-small" in asr_ids and "whisper-base" in asr_ids
    llm_ids = [o["id"] for o in data["summary"]["options"]]
    assert llm_ids == ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]


def test_put_model_and_language_settings(client):
    from backend import config

    res = client.put(
        "/api/config",
        json={
            "ASR_MODEL": "whisper-small",
            "ASR_LANGUAGE": "de",
            "SUMMARY_MODEL": "qwen3.5-9b",
            "SUMMARY_LANGUAGE": "de",
        },
    )
    assert res.status_code == 200
    assert config.ASR_MODEL == "whisper-small"
    assert config.SUMMARY_LANGUAGE == "de"
    assert client.get("/api/models").json()["asr"]["active"] == "whisper-small"


@pytest.mark.parametrize(
    "body",
    [
        {"ASR_MODEL": "whisper-nonexistent"},
        {"SUMMARY_MODEL": "gpt-4"},
        {"ASR_LANGUAGE": "klingon"},
        {"SUMMARY_LANGUAGE": "xx"},
    ],
)
def test_put_rejects_unknown_models_and_languages(client, body):
    assert client.put("/api/config", json=body).status_code == 422


def test_download_endpoint_validates(client):
    assert client.post(
        "/api/models/download", json={"step": "asr", "id": "nope"}
    ).status_code == 404
    assert client.post(
        "/api/models/download", json={"step": "bogus", "id": "whisper-small"}
    ).status_code == 404
    # FAKE mode: everything counts as ready, request is a no-op success.
    assert client.post(
        "/api/models/download", json={"step": "asr", "id": "whisper-small"}
    ).json() == {"ok": True}


def test_resolvers_fall_back_to_defaults():
    from backend import config, models_setup

    # Unknown ids resolve to the default model rather than erroring.
    asr = models_setup.resolve_asr("does-not-exist")
    assert asr["kind"] == "nemo_transducer"
    assert asr["encoder"] == config.MODELS_DIR / "parakeet" / "encoder.onnx"
    assert (
        models_setup.resolve_llm("does-not-exist")
        == config.MODELS_DIR / "qwen3.5-4b-q4_k_m.gguf"
    )


def test_summary_language_instruction(monkeypatch):
    from backend import config
    from backend.pipeline import summarize

    monkeypatch.setattr(config, "SUMMARY_LANGUAGE", "de")
    assert "German" in summarize._lang_instruction()
    monkeypatch.setattr(config, "SUMMARY_LANGUAGE", "auto")
    assert "same language" in summarize._lang_instruction()


def test_uploads_process_sequentially(client, monkeypatch):
    from backend import main as backend_main

    release = threading.Event()
    started = []
    original = backend_main.run_worker

    def gated_worker(meeting_id, path):
        started.append(meeting_id)
        release.wait(10)
        original(meeting_id, path)

    monkeypatch.setattr(backend_main, "run_worker", gated_worker)

    ids = []
    for name in ("first.mp4", "second.mp4"):
        res = client.post(
            "/api/meetings",
            files={"file": (name, b"fake video bytes")},
            data={"num_speakers": "-1"},
        )
        ids.append(res.json()["id"])

    time.sleep(0.4)
    # Only the first worker may have started; the second waits in the queue.
    assert started == [ids[0]]
    assert client.get(f"/api/meetings/{ids[1]}").json()["status"] == "queued"

    release.set()
    deadline = time.time() + 15
    while time.time() < deadline:
        statuses = [client.get(f"/api/meetings/{i}").json()["status"] for i in ids]
        if all(s == "done" for s in statuses):
            break
        time.sleep(0.2)
    assert statuses == ["done", "done"]
    assert started == ids  # strictly in upload order
    for i in ids:
        client.delete(f"/api/meetings/{i}")


def test_uploads_wait_for_model_downloads(client, monkeypatch):
    """While first-run models download, uploads queue; the ready callback pumps."""
    from backend import main as backend_main
    from backend import models_setup

    monkeypatch.setattr(models_setup, "is_ready", lambda: False)
    res = client.post(
        "/api/meetings",
        files={"file": ("early-bird.mp4", b"fake video bytes")},
        data={"num_speakers": "-1"},
    )
    mid = res.json()["id"]
    time.sleep(0.3)
    assert client.get(f"/api/meetings/{mid}").json()["status"] == "queued"

    monkeypatch.setattr(models_setup, "is_ready", lambda: True)
    backend_main._pump_queue()  # what the download-complete callback does
    deadline = time.time() + 15
    while time.time() < deadline:
        status = client.get(f"/api/meetings/{mid}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.2)
    assert status == "done"
    client.delete(f"/api/meetings/{mid}")


def test_deleted_queued_meeting_is_skipped(client, monkeypatch):
    from backend import main as backend_main

    release = threading.Event()
    started = []
    original = backend_main.run_worker

    def gated_worker(meeting_id, path):
        started.append(meeting_id)
        release.wait(10)
        original(meeting_id, path)

    monkeypatch.setattr(backend_main, "run_worker", gated_worker)

    ids = []
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        res = client.post(
            "/api/meetings",
            files={"file": (name, b"fake video bytes")},
            data={"num_speakers": "-1"},
        )
        ids.append(res.json()["id"])
    time.sleep(0.2)
    # Delete the middle one while it waits.
    assert client.delete(f"/api/meetings/{ids[1]}").status_code == 200

    release.set()
    deadline = time.time() + 15
    while time.time() < deadline:
        statuses = [client.get(f"/api/meetings/{i}").json().get("status") for i in (ids[0], ids[2])]
        if all(s == "done" for s in statuses):
            break
        time.sleep(0.2)
    assert statuses == ["done", "done"]
    assert ids[1] not in started
    for i in (ids[0], ids[2]):
        client.delete(f"/api/meetings/{i}")
