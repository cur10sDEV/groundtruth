import asyncio

from fastapi import APIRouter
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.core.flags import get_flags
from app.core.redis_store import get_cache
from app.core.telemetry import get_langfuse
from app.db import _get_engine

router = APIRouter(tags=["health"])

_PROBE_TIMEOUT_SECONDS: float = 1.0
_REDIS_PROBE_KEY = "health:probe"

_CHECK_NAMES = ("database", "qdrant", "redis", "flagsmith", "langfuse")


async def _check_database() -> bool:
    try:
        async with _get_engine().connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        return True
    except Exception:
        return False


async def _check_qdrant() -> bool:
    s = get_settings()

    def probe() -> None:
        client = QdrantClient(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key or None,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        try:
            client.get_collections()
        finally:
            client.close()

    try:
        await asyncio.to_thread(probe)
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    try:
        await get_cache().get(_REDIS_PROBE_KEY)
        return True
    except Exception:
        return False


async def _check_flagsmith() -> bool:
    # Optional-availability: get_flags() falls back to built-in defaults when the
    # Flagsmith server is unreachable, so a working init means the flag system is up.
    try:
        await asyncio.to_thread(get_flags)
        return True
    except Exception:
        return False


async def _check_langfuse() -> bool:
    # Optional-availability: get_langfuse() degrades to a no-op client when
    # unconfigured, so a working init means telemetry is available.
    try:
        await asyncio.to_thread(get_langfuse)
        return True
    except Exception:
        return False


async def _readiness() -> dict:
    probes = {
        "database": _check_database,
        "qdrant": _check_qdrant,
        "redis": _check_redis,
        "flagsmith": _check_flagsmith,
        "langfuse": _check_langfuse,
    }
    checks: dict[str, bool] = {}
    for name, probe in probes.items():
        try:
            checks[name] = await asyncio.wait_for(probe(), _PROBE_TIMEOUT_SECONDS)
        except Exception:
            checks[name] = False
    return checks


@router.get("/health")
async def health() -> dict:
    try:
        checks = await _readiness()
    except Exception:
        checks = {name: False for name in _CHECK_NAMES}
    status = "ok" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}
