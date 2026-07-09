"""Shared test setup.

The env must be configured before any ``backend.*`` import — ``backend.config``
reads ``LAZYMEET_FAKE`` and ``LAZYMEET_DATA_DIR`` at import time.
"""

import os
import tempfile

os.environ["LAZYMEET_FAKE"] = "1"
os.environ["LAZYMEET_DATA_DIR"] = tempfile.mkdtemp(prefix="lazymeet-test-")

import pytest


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:  # the with-block runs the lifespan handler
        yield c
