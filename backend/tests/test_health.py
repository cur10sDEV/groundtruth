import asyncio
import time

import pytest
from fastapi.testclient import TestClient

import app.api.routes_health as routes_health
import app.core.flags as flags_module
import app.core.redis_store as redis_store
import app.core.telemetry as telemetry_module
import app.db as db_module
from app.main import create_app

ALL_CHECKS = {"database", "qdrant", "redis", "flagsmith", "langfuse"}


@pytest.fixture(autouse=True)
def fresh_health_singletons(monkeypatch):
    """Reset singletons so real probes in this module never leak into other tests.

    monkeypatch restores the pre-test value at teardown, discarding any engine
    or client created during a probe.
    """
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_sessionmaker", None)
    monkeypatch.setattr(redis_store, "_cache", None)
    monkeypatch.setattr(redis_store, "_limiter", None)
    monkeypatch.setattr(flags_module, "_flags", None)
    monkeypatch.setattr(telemetry_module, "_langfuse", None)


async def _up() -> bool:
    return True


async def _down() -> bool:
    return False


def _stub_checks(monkeypatch, down: set[str]) -> None:
    for name in ALL_CHECKS:
        monkeypatch.setattr(routes_health, f"_check_{name}", _down if name in down else _up)


def test_health():
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "degraded"}


def test_health_includes_checks():
    from fastapi.testclient import TestClient

    from app.main import create_app

    resp = TestClient(create_app()).get("/health")
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["checks"]) >= {"database", "qdrant", "redis"}


def test_health_ok_when_all_checks_up(monkeypatch):
    _stub_checks(monkeypatch, set())
    resp = TestClient(create_app()).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "checks": {name: True for name in ALL_CHECKS}}


def test_health_degraded_when_database_down(monkeypatch):
    _stub_checks(monkeypatch, {"database"})
    resp = TestClient(create_app()).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"] is False
    assert body["checks"]["redis"] is True


def test_health_checks_are_isolated(monkeypatch):
    _stub_checks(monkeypatch, {"qdrant", "flagsmith"})
    body = TestClient(create_app()).get("/health").json()
    assert body["checks"] == {
        "database": True,
        "qdrant": False,
        "redis": True,
        "flagsmith": False,
        "langfuse": True,
    }
    assert body["status"] == "degraded"


def test_health_never_500s_when_probe_raises(monkeypatch):
    _stub_checks(monkeypatch, set())

    async def boom() -> bool:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(routes_health, "_check_redis", boom)
    resp = TestClient(create_app()).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"] is False
    assert body["checks"]["database"] is True


def test_health_probe_timeout_returns_promptly(monkeypatch):
    monkeypatch.setattr(routes_health, "_PROBE_TIMEOUT_SECONDS", 0.05)
    _stub_checks(monkeypatch, set())

    async def slow() -> bool:
        await asyncio.sleep(30)
        return True

    monkeypatch.setattr(routes_health, "_check_langfuse", slow)
    start = time.perf_counter()
    resp = TestClient(create_app()).get("/health")
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["langfuse"] is False
    assert elapsed < 5
