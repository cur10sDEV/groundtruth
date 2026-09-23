import contextlib
import logging

from flagsmith import Flagsmith
from flagsmith.models import DefaultFlag

from app.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_FLAGS: dict[str, bool] = {
    "reranker.enabled": False,
    "cache.enabled": True,
    "multi_query.enabled": True,
    "filter_extraction.enabled": True,
    "faithfulness.enabled": True,
    "guard_model.enabled": False,
}


def _default_flag(name: str) -> DefaultFlag:
    return DefaultFlag(enabled=DEFAULT_FLAGS.get(name, False), value=None)


class FeatureFlags:
    def __init__(self) -> None:
        self._client: Flagsmith | None = None

    def init(self) -> None:
        s = get_settings()
        try:
            self._client = Flagsmith(
                environment_key=s.flagsmith_server_key,
                api_url=s.flagsmith_api_url,
                enable_local_evaluation=True,
                environment_refresh_interval_seconds=5,
                default_flag_handler=_default_flag,
                offline_mode=False,
            )
        except Exception as exc:
            logger.warning("flagsmith unavailable, using defaults", extra={"exc": str(exc)})
            self._client = None
            return
        # Warm the cache asynchronously-ish; failures fall back to defaults.
        try:
            self._client.get_environment_flags()
        except Exception as exc:
            logger.warning("flagsmith unavailable, using defaults", extra={"exc": str(exc)})

    async def get_flag(self, name: str) -> bool:
        if self._client is None:
            return DEFAULT_FLAGS.get(name, False)
        try:
            flags = self._client.get_environment_flags()
            return bool(flags.is_feature_enabled(name))
        except Exception:
            return DEFAULT_FLAGS.get(name, False)

    async def get_all(self) -> dict[str, bool]:
        out = dict(DEFAULT_FLAGS)
        if self._client is None:
            return out
        try:
            flags = self._client.get_environment_flags()
        except Exception:
            return out
        for name in DEFAULT_FLAGS:
            with contextlib.suppress(Exception):
                out[name] = bool(flags.is_feature_enabled(name))
        return out


_flags: FeatureFlags | None = None


def get_flags() -> FeatureFlags:
    global _flags
    if _flags is None:
        _flags = FeatureFlags()
        _flags.init()
    return _flags


async def get_feature_flags() -> dict[str, bool]:
    return await get_flags().get_all()
