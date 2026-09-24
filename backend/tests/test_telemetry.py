import pytest

import app.core.telemetry as telemetry_module
from app.core.telemetry import ensure_trace, get_langfuse, trace_step


class _FakeObservation:
    def __init__(self) -> None:
        self.id = "span-1"
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self, **kwargs):
        self.ended = True


class _FakeLangfuse:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    def start_observation(self, **kwargs):
        if self.fail:
            raise ValueError("langfuse unreachable")
        self.calls.append(kwargs)
        return _FakeObservation()


@pytest.fixture(autouse=True)
def reset_telemetry_singleton(monkeypatch):
    monkeypatch.setattr(telemetry_module, "_langfuse", None)


def test_get_langfuse_returns_client():
    # no-op safe when keys unset
    assert get_langfuse() is not None


def test_trace_step_is_context_manager():
    cm = trace_step(name="test", trace_id="t1")
    assert hasattr(cm, "__enter__") and hasattr(cm, "__exit__")


def test_get_langfuse_is_singleton():
    assert get_langfuse() is get_langfuse()


def test_get_langfuse_noop_client_is_safe_when_unconfigured():
    lf = get_langfuse()
    obs = lf.start_observation(name="x", trace_context={"trace_id": "ab" * 16})
    obs.update(input={"q": "hello"}, output={"a": "hi"}, metadata={"k": "v"})
    obs.end()


def test_trace_step_noop_roundtrip_when_unconfigured():
    with trace_step(name="retrieve", trace_id="550e8400-e29b-41d4-a716-446655440000") as span:
        span.update(input={"q": "hello"}, output={"docs": 2}, metadata={"k": "v"})
        assert span.name == "retrieve"
        assert span.trace_id == "550e8400-e29b-41d4-a716-446655440000"


def test_trace_step_passes_updates_and_ends_span(monkeypatch):
    fake = _FakeLangfuse()
    monkeypatch.setattr(telemetry_module, "get_langfuse", lambda: fake)
    with trace_step(name="retrieve", trace_id="ab" * 16) as span:
        span.update(output={"docs": 3}, input={"q": "hello"})
    assert fake.calls[0]["name"] == "retrieve"
    assert fake.calls[0]["trace_context"]["trace_id"] == "ab" * 16
    assert span.observation.updates[0] == {"output": {"docs": 3}, "input": {"q": "hello"}}
    assert span.observation.ended


def test_trace_step_marks_error_span_and_reraises(monkeypatch):
    fake = _FakeLangfuse()
    monkeypatch.setattr(telemetry_module, "get_langfuse", lambda: fake)
    span = None
    with (
        pytest.raises(RuntimeError, match="boom"),
        trace_step(name="generate", trace_id="ab" * 16) as span_ctx,
    ):
        span = span_ctx
        raise RuntimeError("boom")
    assert span.observation.updates[-1]["level"] == "ERROR"
    assert span.observation.updates[-1]["status_message"] == "boom"
    assert span.observation.ended


def test_trace_step_links_parent_span_id(monkeypatch):
    fake = _FakeLangfuse()
    monkeypatch.setattr(telemetry_module, "get_langfuse", lambda: fake)
    with (
        trace_step(name="root", trace_id="ab" * 16) as parent_span,
        trace_step(name="child", trace_id="ab" * 16, parent=parent_span),
    ):
        pass
    assert fake.calls[0]["trace_context"] == {"trace_id": "ab" * 16}
    assert fake.calls[1]["trace_context"]["parent_span_id"] == "span-1"


def test_trace_step_never_breaks_when_client_fails(monkeypatch):
    monkeypatch.setattr(telemetry_module, "get_langfuse", lambda: _FakeLangfuse(fail=True))
    with trace_step(name="retrieve", trace_id="t1") as span:
        span.update(input={"q": 1})
    assert span.observation is None


def test_ensure_trace_creates_root_context(monkeypatch):
    fake = _FakeLangfuse()
    monkeypatch.setattr(telemetry_module, "get_langfuse", lambda: fake)
    ctx = ensure_trace("550e8400-e29b-41d4-a716-446655440000")
    assert ctx.name == "query:550e8400-e29b-41d4-a716-446655440000"
    assert fake.calls[0]["trace_context"]["trace_id"] == "550e8400e29b41d4a716446655440000"
    assert fake.calls[0]["metadata"] == {"correlation_id": "550e8400-e29b-41d4-a716-446655440000"}


def test_ensure_trace_noop_when_unconfigured():
    ctx = ensure_trace("abc")
    assert ctx.trace_id == "abc"
    ctx.update(metadata={"x": 1})
    ctx.end()


def test_non_hex_trace_ids_are_normalized_to_valid_langfuse_ids(monkeypatch):
    fake = _FakeLangfuse()
    monkeypatch.setattr(telemetry_module, "get_langfuse", lambda: fake)
    with trace_step(name="step", trace_id="t1"):
        pass
    normalized = fake.calls[0]["trace_context"]["trace_id"]
    assert len(normalized) == 32
    assert all(c in "0123456789abcdef" for c in normalized)
