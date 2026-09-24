from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import (
    DomainError,
    NotFoundError,
    register_exception_handlers,
)
from app.core.logging import set_correlation_id
from app.main import create_app


def test_domain_error_status_and_detail():
    err = NotFoundError("missing")
    assert err.status_code == 404
    assert err.detail == "missing"


def test_register_exception_handlers_returns_json_with_trace_id():
    app = FastAPI()

    @app.get("/boom")
    async def boom():
        raise DomainError(500, "kaboom")

    register_exception_handlers(app)
    client = TestClient(app)
    resp = client.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "kaboom"
    assert "trace_id" in body


def test_app_factory_returns_trace_id_json_for_domain_errors():
    app = create_app()

    async def boom() -> None:
        raise NotFoundError(detail="missing")

    app.add_api_route("/boom", boom)
    with TestClient(app) as client:
        resp = client.get("/boom")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"] == "missing"
        assert "trace_id" in body


def test_domain_error_handler_reuses_request_correlation_id():
    app = FastAPI()

    async def boom() -> None:
        set_correlation_id("req-trace-42")
        raise NotFoundError(detail="missing")

    app.add_api_route("/boom", boom)
    register_exception_handlers(app)
    client = TestClient(app)

    resp = client.get("/boom")

    assert resp.status_code == 404
    body = resp.json()
    assert body["trace_id"] == "req-trace-42"
    set_correlation_id(None)
