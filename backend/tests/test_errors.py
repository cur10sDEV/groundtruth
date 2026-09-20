from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import (
    DomainError,
    NotFoundError,
    register_exception_handlers,
)


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
