from app.core.config import get_settings


def test_settings_loads_expected_fields():
    s = get_settings()
    assert s.app_name == "rag-prod"
    assert s.environment in {"development", "test", "production"}
    assert s.database_url.startswith("postgresql")


def test_settings_reads_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "override")
    assert get_settings().app_name == "override"
