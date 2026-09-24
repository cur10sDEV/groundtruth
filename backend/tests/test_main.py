import pytest
from fastapi.testclient import TestClient

import app.core.flags as flags_module
import app.core.redis_store as redis_store
import app.core.telemetry as telemetry_module
import app.db as db_module
from app.main import create_app


@pytest.fixture(autouse=True)
def fresh_singletons(monkeypatch):
    """Reset singletons so real /health probes never leak engines into other tests.

    Same pattern as test_health.py: monkeypatch restores the pre-test value at
    teardown, discarding any engine or client a probe created.
    """
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_sessionmaker", None)
    monkeypatch.setattr(redis_store, "_cache", None)
    monkeypatch.setattr(redis_store, "_limiter", None)
    monkeypatch.setattr(flags_module, "_flags", None)
    monkeypatch.setattr(telemetry_module, "_langfuse", None)


def test_cors_preflight_allowed_for_configured_origin():
    client = TestClient(create_app())
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_get_response_carries_allow_origin():
    client = TestClient(create_app())
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_preflight_rejects_unconfigured_origin():
    client = TestClient(create_app())
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_cors_origins_parse_comma_separated_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, http://other.example")
    client = TestClient(create_app())
    for origin in ("http://localhost:3000", "http://other.example"):
        resp = client.options(
            "/health",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == origin
