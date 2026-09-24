import io
import json
import logging

from app.core.logging import (
    JsonFormatter,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("test.json", logging.INFO, __file__, 1, msg, (), None)


def test_json_formatter_outputs_correlation_id():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test.json")
    logger.handlers = [handler]
    cid = new_correlation_id()
    rec = _record("hello")
    rec.correlation_id = cid
    handler.handle(rec)
    data = json.loads(stream.getvalue())
    assert data["msg"] == "hello"
    assert data["correlation_id"] == cid


def test_json_formatter_picks_up_contextvar_correlation_id():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    set_correlation_id("ctx-abc-123")
    try:
        handler.handle(_record("stage log"))
        data = json.loads(stream.getvalue())
        assert data["correlation_id"] == "ctx-abc-123"
    finally:
        set_correlation_id(None)
    # unset contextvar falls back to None without breaking log output
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.handle(_record("no ctx"))
    assert json.loads(stream.getvalue())["correlation_id"] is None


def test_new_correlation_id_unique():
    assert new_correlation_id() != new_correlation_id()


def test_set_and_get_correlation_id_roundtrip():
    set_correlation_id("roundtrip-1")
    try:
        assert get_correlation_id() == "roundtrip-1"
    finally:
        set_correlation_id(None)
    assert get_correlation_id() is None
