import contextlib
import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_langfuse: Any = None


class _NoopObservation:
    def __init__(self) -> None:
        self.id: str | None = None
        self.trace_id: str | None = None

    def update(self, **kwargs: Any) -> None:
        return

    def end(self, **kwargs: Any) -> None:
        return


class _NoopLangfuse:
    def start_observation(self, **kwargs: Any) -> _NoopObservation:
        return _NoopObservation()

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return


@dataclass
class SpanContext:
    name: str
    trace_id: str
    observation: Any = None

    def update(self, **kwargs: Any) -> None:
        if self.observation is not None:
            with contextlib.suppress(Exception):
                self.observation.update(**kwargs)

    def end(self, **kwargs: Any) -> None:
        if self.observation is not None:
            with contextlib.suppress(Exception):
                self.observation.end(**kwargs)


def _langfuse_trace_id(trace_id: str) -> str:
    cleaned = trace_id.replace("-", "").lower()
    if len(cleaned) == 32 and all(c in "0123456789abcdef" for c in cleaned):
        return cleaned
    return hashlib.sha256(trace_id.encode()).hexdigest()[:32]


def get_langfuse() -> Any:
    global _langfuse
    if _langfuse is None:
        s = get_settings()
        if not s.langfuse_public_key or not s.langfuse_secret_key:
            _langfuse = _NoopLangfuse()
        else:
            try:
                from langfuse import Langfuse

                _langfuse = Langfuse(
                    public_key=s.langfuse_public_key,
                    secret_key=s.langfuse_secret_key,
                    host=s.langfuse_host,
                )
            except Exception as exc:
                logger.warning("langfuse init failed, telemetry disabled", extra={"exc": str(exc)})
                _langfuse = _NoopLangfuse()
    return _langfuse


def ensure_trace(trace_id: str) -> SpanContext:
    lf = get_langfuse()
    observation = None
    try:
        observation = lf.start_observation(
            name=f"query:{trace_id}",
            trace_context={"trace_id": _langfuse_trace_id(trace_id)},
            metadata={"correlation_id": trace_id},
        )
    except Exception as exc:
        logger.warning(
            "trace creation failed, continuing without telemetry", extra={"exc": str(exc)}
        )
    return SpanContext(name=f"query:{trace_id}", trace_id=trace_id, observation=observation)


@contextmanager
def trace_step(
    name: str, trace_id: str, parent: SpanContext | None = None
) -> Iterator[SpanContext]:
    lf = get_langfuse()
    observation = None
    try:
        trace_context: dict[str, str] = {"trace_id": _langfuse_trace_id(trace_id)}
        if parent is not None and parent.observation is not None and parent.observation.id:
            trace_context["parent_span_id"] = parent.observation.id
        observation = lf.start_observation(name=name, trace_context=trace_context)
    except Exception as exc:
        logger.warning(
            "span creation failed, continuing without telemetry", extra={"exc": str(exc)}
        )
    ctx = SpanContext(name=name, trace_id=trace_id, observation=observation)
    try:
        yield ctx
    except Exception as exc:
        with contextlib.suppress(Exception):
            if observation is not None:
                observation.update(
                    metadata={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
                observation.end()
        raise
    with contextlib.suppress(Exception):
        if observation is not None:
            observation.end()


def flush_telemetry() -> None:
    with contextlib.suppress(Exception):
        get_langfuse().flush()
