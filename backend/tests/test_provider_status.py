from __future__ import annotations

import urllib.error

from app.config import Settings
from app.providers import registry as provider_module


def test_root_env_file_loads_server_side_only(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-openai\nGEMINI_API_KEY=test-gemini\nANTHROPIC_API_KEY=test-anthropic\n")
    settings = Settings(_env_file=str(env_file))
    assert settings.openai_api_key == "test-openai"
    assert settings.gemini_api_key == "test-gemini"
    assert settings.anthropic_api_key == "test-anthropic"


def test_provider_catalog_reports_missing_keys_without_secret_material(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(provider_module.settings, "openai_api_key", "")
    monkeypatch.setattr(provider_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(provider_module.settings, "google_api_key", "")
    monkeypatch.setattr(provider_module.settings, "anthropic_api_key", "")
    result = provider_module.ProviderRegistry().get("openai").to_dict()
    assert result["configured"] is False
    assert result["configuration_status"] == "missing"
    assert result["configuration_error"] == "API key missing"
    assert "test-" not in repr(result)


def test_provider_catalog_reports_configured_without_returning_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setattr(provider_module.settings, "openai_api_key", "")
    result = provider_module.ProviderRegistry().validate_configuration("openai")
    assert result["configured"] is True
    assert result["configuration_status"] == "configured"
    assert result["connection_status"] == "not_tested"
    assert "test-openai-secret" not in repr(result)


def test_explicit_provider_test_maps_authentication_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setattr(provider_module.settings, "openai_api_key", "")

    def fail_auth(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.openai.com/v1/models", 401, "unauthorized", {}, None)

    monkeypatch.setattr(provider_module.urllib.request, "urlopen", fail_auth)
    result = provider_module.ProviderRegistry().validate_configuration("openai", test_connection=True)
    assert result["connection_status"] == "authentication_failed"
    assert result["error_code"] == "AUTHENTICATION_FAILED"
    assert "test-openai-secret" not in repr(result)


def test_explicit_provider_test_maps_network_failure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-secret")
    monkeypatch.setattr(provider_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(provider_module.settings, "google_api_key", "")

    def fail_network(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(provider_module.urllib.request, "urlopen", fail_network)
    result = provider_module.ProviderRegistry().validate_configuration("gemini", test_connection=True)
    assert result["connection_status"] == "network_unavailable"
    assert result["error_code"] == "NETWORK_UNAVAILABLE"
    assert "test-gemini-secret" not in repr(result)
