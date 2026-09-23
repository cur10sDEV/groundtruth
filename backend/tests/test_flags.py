import pytest
from flagsmith.models import Flag, Flags

import app.core.flags as flags_module
from app.core.flags import DEFAULT_FLAGS, FeatureFlags, get_feature_flags

EXPECTED_FLAG_NAMES = {
    "reranker.enabled",
    "cache.enabled",
    "multi_query.enabled",
    "filter_extraction.enabled",
    "faithfulness.enabled",
    "guard_model.enabled",
}

DEAD_URL = "http://127.0.0.1:1/api/v1/"


@pytest.fixture(autouse=True)
def reset_flags_singleton(monkeypatch):
    monkeypatch.setattr(flags_module, "_flags", None)


def test_default_flags_shape():
    assert set(DEFAULT_FLAGS) == EXPECTED_FLAG_NAMES


def test_default_flags_values_are_booleans():
    assert DEFAULT_FLAGS == {
        "reranker.enabled": False,
        "cache.enabled": True,
        "multi_query.enabled": True,
        "filter_extraction.enabled": True,
        "faithfulness.enabled": True,
        "guard_model.enabled": False,
    }
    assert all(isinstance(v, bool) for v in DEFAULT_FLAGS.values())


def test_feature_flags_client_constructs():
    ff = FeatureFlags()
    assert ff is not None


async def test_get_flag_falls_back_to_defaults_before_init():
    ff = FeatureFlags()
    assert await ff.get_flag("cache.enabled") is True
    assert await ff.get_flag("reranker.enabled") is False
    assert await ff.get_flag("unknown.flag") is False


async def test_get_all_falls_back_to_defaults_before_init():
    ff = FeatureFlags()
    assert await ff.get_all() == DEFAULT_FLAGS


async def test_provider_falls_back_when_no_server_key(monkeypatch):
    monkeypatch.setenv("FLAGSMITH_SERVER_KEY", "")
    flags = await get_feature_flags()
    assert flags == DEFAULT_FLAGS
    assert all(isinstance(v, bool) for v in flags.values())


async def test_outage_dead_url_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("FLAGSMITH_API_URL", DEAD_URL)
    monkeypatch.setenv("FLAGSMITH_SERVER_KEY", "ser.dead-url-test")
    ff = FeatureFlags()
    ff.init()
    assert await ff.get_flag("cache.enabled") is True
    assert await ff.get_flag("reranker.enabled") is False
    all_flags = await ff.get_all()
    assert all_flags == DEFAULT_FLAGS
    assert all(isinstance(v, bool) for v in all_flags.values())


class _StubFlagsmith:
    def __init__(self, enabled: dict[str, bool]) -> None:
        self._enabled = enabled

    def get_environment_flags(self) -> Flags:
        return Flags(
            flags={
                name: Flag(
                    enabled=value,
                    value=None,
                    feature_id=idx,
                    feature_name=name,
                )
                for idx, (name, value) in enumerate(self._enabled.items(), start=1)
            }
        )


async def test_get_flag_uses_flagsmith_value_over_default():
    ff = FeatureFlags()
    ff._client = _StubFlagsmith({"cache.enabled": False})
    assert await ff.get_flag("cache.enabled") is False
    assert await ff.get_flag("multi_query.enabled") is True


async def test_get_all_uses_flagsmith_values_with_per_flag_fallback():
    ff = FeatureFlags()
    ff._client = _StubFlagsmith({"cache.enabled": False, "reranker.enabled": True})
    result = await ff.get_all()
    assert result == {
        **DEFAULT_FLAGS,
        "cache.enabled": False,
        "reranker.enabled": True,
    }


def test_default_flag_handler_serves_defaults():
    flag = flags_module._default_flag("cache.enabled")
    assert flag.enabled is True
    assert flag.value is None
    assert flags_module._default_flag("unknown.flag").enabled is False
