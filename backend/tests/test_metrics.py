from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from app.main import create_app


def _sample(name: str) -> float:
    prefix = f"{name} "
    for line in generate_latest().decode().splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"metric sample not found: {name}")


def test_metrics_endpoint():
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "rag_requests_total" in resp.text


def test_middleware_counts_requests_and_latency():
    client = TestClient(create_app())
    requests_before = _sample("rag_requests_total")
    errors_before = _sample("rag_errors_total")
    inf_before = _sample('rag_latency_seconds_bucket{le="+Inf"}')

    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert _sample("rag_requests_total") == requests_before + 1
    assert _sample("rag_errors_total") == errors_before
    assert _sample('rag_latency_seconds_bucket{le="+Inf"}') == inf_before + 1


def test_middleware_counts_unhandled_exceptions_as_errors():
    app = create_app()

    @app.get("/boom")
    async def boom() -> dict:
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)
    requests_before = _sample("rag_requests_total")
    errors_before = _sample("rag_errors_total")

    resp = client.get("/boom")

    assert resp.status_code == 500
    assert _sample("rag_requests_total") == requests_before + 1
    assert _sample("rag_errors_total") == errors_before + 1


def test_middleware_does_not_count_4xx_as_errors():
    client = TestClient(create_app())
    errors_before = _sample("rag_errors_total")

    resp = client.get("/nope")

    assert resp.status_code == 404
    assert _sample("rag_errors_total") == errors_before
