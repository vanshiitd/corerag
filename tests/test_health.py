"""Integration tests for the API (P0.9).

Require live Qdrant + Redis. Run with:  make test-int   (i.e. uv run pytest -m integration)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.mark.integration
def test_health_reports_ok_with_live_services() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["dependencies"]["qdrant"]["status"] == "up"
    assert body["dependencies"]["redis"]["status"] == "up"


@pytest.mark.integration
def test_root_banner() -> None:
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "corerag"
