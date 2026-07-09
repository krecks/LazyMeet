"""Data-management endpoints: storage info and reveal-in-file-manager."""

import sys
from unittest.mock import patch


def test_data_info_shape(client):
    from backend import config

    res = client.get("/api/data")
    assert res.status_code == 200
    data = res.json()
    assert data["data_dir"] == str(config.DATA_DIR)
    for key in ("meetings_bytes", "uploads_bytes", "models_bytes", "total_bytes"):
        assert data[key] >= 0
    assert data["total_bytes"] == (
        data["meetings_bytes"] + data["uploads_bytes"] + data["models_bytes"]
    )
    assert data["meeting_count"] >= 0


def test_data_info_counts_meeting_files(client):
    from backend import store

    record = store.create_meeting("sized.mp4")
    data = client.get("/api/data").json()
    assert data["meetings_bytes"] > 0
    assert data["meeting_count"] >= 1
    store.delete_meeting(record["id"])


def test_reveal_opens_data_dir(client):
    from backend import config

    if sys.platform == "win32":
        with patch("backend.main.os.startfile", create=True) as started:
            assert client.post("/api/data/reveal").json() == {"ok": True}
            started.assert_called_once_with(str(config.DATA_DIR))
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    with patch("backend.main.subprocess.Popen") as popen:
        assert client.post("/api/data/reveal").json() == {"ok": True}
        popen.assert_called_once_with([opener, str(config.DATA_DIR)])


def test_reveal_failure_is_500(client):
    with patch("backend.main.subprocess.Popen", side_effect=OSError("no gui")), patch(
        "backend.main.os.startfile", side_effect=OSError("no gui"), create=True
    ):
        assert client.post("/api/data/reveal").status_code == 500
