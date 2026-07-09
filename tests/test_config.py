"""Settings API + config.json precedence (env > file > default)."""

import time

import pytest


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Tests share one data dir per session — leave no config.json behind."""
    from backend import config

    yield
    config.save_overrides({})
    config.refresh()


def test_get_config_shape(client):
    from backend import config

    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    for key in ("values", "overrides", "defaults", "env_locked", "limits"):
        assert key in data
    assert set(data["values"]) == set(config.TUNABLES)
    assert data["defaults"]["SPEAKER_THRESHOLD"] == 0.6
    assert data["limits"]["DIAR_THRESHOLD"] == {"min": 0.0, "max": 1.0}
    assert data["limits"]["KEEP_AUDIO"] == {"min": None, "max": None}
    # conftest sets none of the tunable env vars
    assert data["env_locked"] == []


def test_put_config_persists_and_applies(client):
    from backend import config

    res = client.put(
        "/api/config",
        json={"SPEAKER_THRESHOLD": 0.8, "KEEP_AUDIO": True, "NUM_THREADS": 2},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["values"]["SPEAKER_THRESHOLD"] == 0.8
    assert data["overrides"] == {
        "SPEAKER_THRESHOLD": 0.8,
        "KEEP_AUDIO": True,
        "NUM_THREADS": 2,
    }
    assert config.CONFIG_FILE.exists()
    assert config.SPEAKER_THRESHOLD == 0.8
    assert config.KEEP_AUDIO is True

    # A fresh GET agrees, and untouched values stay at their defaults.
    again = client.get("/api/config").json()
    assert again["values"]["SPEAKER_THRESHOLD"] == 0.8
    assert again["values"]["DIAR_THRESHOLD"] == 0.7


def test_put_null_removes_override(client):
    from backend import config

    client.put("/api/config", json={"DIAR_THRESHOLD": 0.9})
    assert config.DIAR_THRESHOLD == 0.9
    res = client.put("/api/config", json={"DIAR_THRESHOLD": None})
    assert res.status_code == 200
    assert "DIAR_THRESHOLD" not in res.json()["overrides"]
    assert config.DIAR_THRESHOLD == 0.7


@pytest.mark.parametrize(
    "body",
    [
        {"SPEAKER_THRESHOLD": 1.5},
        {"SPEAKER_THRESHOLD": -0.1},
        {"NUM_THREADS": 0},
        {"MIN_SPEAKER_SECONDS": 9999},
        {"KEEP_AUDIO": "sometimes"},
    ],
)
def test_put_config_rejects_invalid(client, body):
    assert client.put("/api/config", json=body).status_code == 422


def test_precedence_env_beats_file_beats_default(monkeypatch):
    from backend import config

    config.save_overrides({"SPEAKER_THRESHOLD": 0.9})
    monkeypatch.setenv("LAZYMEET_SPEAKER_THRESHOLD", "0.3")
    config.refresh()
    assert config.SPEAKER_THRESHOLD == 0.3
    assert "SPEAKER_THRESHOLD" in config.env_locked()

    monkeypatch.delenv("LAZYMEET_SPEAKER_THRESHOLD")
    config.refresh()
    assert config.SPEAKER_THRESHOLD == 0.9

    config.save_overrides({})
    config.refresh()
    assert config.SPEAKER_THRESHOLD == 0.6


def test_use_gpu_toggle_and_onnx_provider(monkeypatch):
    from backend import config

    # USE_GPU (summarizer Metal offload) accelerates by default and toggles.
    assert config.USE_GPU is True
    monkeypatch.setenv("LAZYMEET_USE_GPU", "0")
    config.refresh()
    assert config.USE_GPU is False
    monkeypatch.delenv("LAZYMEET_USE_GPU")
    config.refresh()
    assert config.USE_GPU is True

    # The ONNX models always run on the CPU: CoreML measured ~2x slower for
    # them, so USE_GPU deliberately does not affect the provider.
    assert config.onnx_provider() == "cpu"


def test_num_threads_default_tracks_cores():
    import os

    from backend import config

    cores = os.cpu_count() or 8
    default = config.TUNABLES["NUM_THREADS"][2]
    maximum = config.TUNABLES["NUM_THREADS"][4]
    assert maximum == min(16, cores)
    assert default == max(1, min(maximum, cores * 3 // 4))


def test_corrupt_config_json_is_tolerated():
    from backend import config

    config.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_FILE.write_text("{not json", encoding="utf-8")
    assert config.load_overrides() == {}
    config.refresh()
    assert config.SPEAKER_THRESHOLD == 0.6


def test_unknown_and_invalid_file_values_ignored():
    from backend import config

    config.save_overrides({})
    config.CONFIG_FILE.write_text(
        '{"BOGUS": 1, "DIAR_THRESHOLD": 5.0, "NUM_THREADS": 3}', encoding="utf-8"
    )
    assert config.load_overrides() == {"NUM_THREADS": 3}


def test_worker_picks_up_config_changes(client):
    """FAKE_MODE workers are threads — run_worker must refresh() explicitly."""
    from backend import config

    config.save_overrides({"MIN_SPEAKER_SECONDS": 42.0})
    assert config.MIN_SPEAKER_SECONDS != 42.0  # not refreshed yet

    res = client.post(
        "/api/meetings",
        files={"file": ("cfg.mp4", b"fake video bytes")},
        data={"num_speakers": "-1"},
    )
    meeting_id = res.json()["id"]
    deadline = time.time() + 15
    while time.time() < deadline:
        meeting = client.get(f"/api/meetings/{meeting_id}").json()
        if meeting["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert meeting["status"] == "done", meeting.get("error")
    assert config.MIN_SPEAKER_SECONDS == 42.0
    client.delete(f"/api/meetings/{meeting_id}")
