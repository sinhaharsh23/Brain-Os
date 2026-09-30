from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

from app.config import settings
from app.models.registry import SUPPORTED_MODELS
from app.models.resolver import LocalModelError, LocalModelResolver
from app.providers.base import ModelDescriptor, ProviderCapabilities, ProviderDescriptor


DEEP = ProviderCapabilities(
    streaming_text=True, token_ids=True, embeddings=True, hidden_states=True,
    attention=True, qkv=True, mlp_activations=True, logits=True, probabilities=True,
    chat=True, logit_lens=True, kv_cache=True, streaming=True,
)
LIMITED = ProviderCapabilities(
    streaming_text=True, token_ids=False, embeddings=False, hidden_states=False,
    attention=False, qkv=False, mlp_activations=False, logits=False, probabilities=False,
    chat=True, api_usage=True, server_metadata=True, logprobs=False, streaming=True,
)
GGUF_DEEP = ProviderCapabilities(
    streaming_text=True, token_ids=True, embeddings=True, hidden_states=True,
    attention=True, qkv=True, mlp_activations=True, logits=True, probabilities=True,
    chat=True, api_usage=True, server_metadata=True, logprobs=True, logit_lens=False,
    kv_cache=False, streaming=True,
)


def _configured(*names: str) -> bool:
    return any(bool(str(os.getenv(name) or getattr(settings, name.lower(), "")).strip()) for name in names)


def _provider_key(provider_id: str) -> str:
    names = {
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(provider_id, ())
    for name in names:
        value = os.getenv(name) or getattr(settings, name.lower(), "")
        if str(value).strip():
            return str(value).strip()
    return ""


def _configured_models(env_name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(item.strip() for item in os.getenv(env_name, "").split(",") if item.strip())
    return values or defaults


def _ollama_url() -> str:
    return str(os.getenv("OLLAMA_URL") or os.getenv("BRAINOS_OLLAMA_URL") or settings.ollama_url).strip().rstrip("/")


def _ollama_models() -> tuple[str, ...]:
    configured = os.getenv("OLLAMA_MODELS") or os.getenv("BRAINOS_OLLAMA_MODELS") or settings.ollama_models
    return tuple(item.strip() for item in configured.split(",") if item.strip()) or ("llama3.2",)


def _cloud_models(provider: str, names: tuple[str, ...], configured: bool) -> tuple[ModelDescriptor, ...]:
    return tuple(ModelDescriptor(
        model_id=model_id, provider=provider, display_name=model_id, mode="cloud",
        capabilities=LIMITED, available=configured, supports_streaming=True,
        supports_introspection=False,
        configuration_error=None if configured else "API key missing",
    ) for model_id in names)


LOCAL_MODELS = tuple(item["model_id"] for item in SUPPORTED_MODELS)
OPENAI_MODELS = _configured_models("BRAINOS_OPENAI_MODELS", ("gpt-4o-mini",))
ANTHROPIC_MODELS = _configured_models("BRAINOS_ANTHROPIC_MODELS", ("claude-3-5-haiku-latest",))
GEMINI_MODELS = _configured_models("BRAINOS_GEMINI_MODELS", ("gemini-2.0-flash",))
OLLAMA_MODELS = _ollama_models()


def _ollama_model_descriptors(configured: bool) -> tuple[ModelDescriptor, ...]:
    return tuple(ModelDescriptor(
        model_id=model_id, provider="ollama", display_name=model_id, mode="local",
        architecture="Ollama GGUF via instrumented llama.cpp", capabilities=GGUF_DEEP,
        available=configured, supports_streaming=True, supports_introspection=True,
        configuration_error=None if configured else "Ollama is not configured",
    ) for model_id in _ollama_models())
_openai_configured = _configured("OPENAI_API_KEY")
_anthropic_configured = _configured("ANTHROPIC_API_KEY")
_gemini_configured = _configured("GOOGLE_API_KEY", "GEMINI_API_KEY")
_ollama_configured = bool(_ollama_url())
try:
    _local_resolution = LocalModelResolver().resolve_model()
    _local_available = True
    _local_error = None
except LocalModelError as exc:
    _local_resolution = None
    _local_available = False
    _local_error = str(exc)

PROVIDER_TEMPLATES = (
    ProviderDescriptor(
        provider_id="qwen-local", display_name="Local open-weight models", kind="local",
        provider_type="local", inspection_mode="deep", availability="available" if _local_available else "not-installed",
        configured=True, available=_local_available, configuration_error=_local_error,
        limitation="BrainOS executes the loaded PyTorch model and can expose captured transformer signals."
        if _local_available else "Install a complete local checkpoint before selecting local inference.",
        capabilities=DEEP, models=LOCAL_MODELS,
        model_descriptors=tuple(ModelDescriptor(
            model_id=item["model_id"], provider="qwen-local", display_name=item["model_id"].split("/")[-1], mode="local",
            architecture=item.get("adapter", "Hugging Face causal language model"), capabilities=DEEP,
            available=_local_available, configuration_error=_local_error,
            parameter_count=int(item.get("params_m", 0) * 1_000_000),
            supports_streaming=True, supports_introspection=True,
        ) for item in SUPPORTED_MODELS),
    ),
    ProviderDescriptor(
        provider_id="ollama", display_name="Ollama", kind="local", provider_type="local",
        inspection_mode="deep", availability="configured" if _ollama_configured else "missing-url",
        configured=_ollama_configured, available=_ollama_configured,
        configuration_error=None if _ollama_configured else "OLLAMA_URL is not configured",
        limitation="Ollama model responses use the installed GGUF checkpoint through BrainOS' instrumented llama.cpp runtime for local tensor inspection.",
        capabilities=GGUF_DEEP, models=OLLAMA_MODELS,
        model_descriptors=_ollama_model_descriptors(_ollama_configured),
    ),
    ProviderDescriptor(
        provider_id="openai", display_name="OpenAI GPT", kind="external", provider_type="cloud",
        inspection_mode="limited", availability="configured" if _openai_configured else "missing-api-key",
        configured=_openai_configured, available=_openai_configured,
        configuration_error=None if _openai_configured else "OPENAI_API_KEY is not configured",
        limitation="OpenAI APIs expose response/event metadata, not private model layers, attention, Q/K/V, MLP, residuals, hidden states, or internal KV cache.",
        capabilities=LIMITED, models=OPENAI_MODELS,
        model_descriptors=_cloud_models("openai", OPENAI_MODELS, _openai_configured),
    ),
    ProviderDescriptor(
        provider_id="anthropic", display_name="Anthropic Claude", kind="external", provider_type="cloud",
        inspection_mode="limited", availability="configured" if _anthropic_configured else "missing-api-key",
        configured=_anthropic_configured, available=_anthropic_configured,
        configuration_error=None if _anthropic_configured else "ANTHROPIC_API_KEY is not configured",
        limitation="Anthropic APIs expose messages, stream events, usage, and stop metadata; private transformer internals are not exposed.",
        capabilities=LIMITED, models=ANTHROPIC_MODELS,
        model_descriptors=_cloud_models("anthropic", ANTHROPIC_MODELS, _anthropic_configured),
    ),
    ProviderDescriptor(
        provider_id="gemini", display_name="Google Gemini", kind="external", provider_type="cloud",
        inspection_mode="limited", availability="configured" if _gemini_configured else "missing-api-key",
        configured=_gemini_configured, available=_gemini_configured,
        configuration_error=None if _gemini_configured else "GEMINI_API_KEY or GOOGLE_API_KEY is not configured",
        limitation="Gemini APIs expose response/event metadata and usage where returned; private transformer internals are not exposed.",
        capabilities=LIMITED, models=GEMINI_MODELS,
        model_descriptors=_cloud_models("gemini", GEMINI_MODELS, _gemini_configured),
    ),
)

PROVIDERS = PROVIDER_TEMPLATES


class ProviderRegistry:
    """Runtime registry for local and cloud provider families."""

    def __init__(self, providers: tuple[ProviderDescriptor, ...] | None = None) -> None:
        self._custom_descriptors = {provider.provider_id: provider for provider in providers} if providers is not None else None

    def _current_descriptors(self) -> dict[str, ProviderDescriptor]:
        if self._custom_descriptors is not None:
            return self._custom_descriptors
        # Re-read credentials on every catalog request. This keeps the backend
        # correct when tests or a long-running localhost process update the
        # root .env/environment without requiring a React rebuild.
        cloud_config = {
            "openai": (_configured("OPENAI_API_KEY"), _configured_models("BRAINOS_OPENAI_MODELS", ("gpt-4o-mini",))),
            "anthropic": (_configured("ANTHROPIC_API_KEY"), _configured_models("BRAINOS_ANTHROPIC_MODELS", ("claude-3-5-haiku-latest",))),
            "gemini": (_configured("GOOGLE_API_KEY", "GEMINI_API_KEY"), _configured_models("BRAINOS_GEMINI_MODELS", ("gemini-2.0-flash",))),
        }
        descriptors: dict[str, ProviderDescriptor] = {}
        for base in PROVIDER_TEMPLATES:
            if base.provider_id == "ollama":
                models = _ollama_models()
                configured = bool(_ollama_url())
                descriptors[base.provider_id] = replace(
                    base,
                    availability="configured" if configured else "missing-url",
                    configured=configured,
                    available=configured,
                    configuration_error=None if configured else "OLLAMA_URL is not configured",
                    configuration_status="configured" if configured else "missing",
                    connection_status="not_tested",
                    models=models,
                    model_descriptors=_ollama_model_descriptors(configured),
                )
                continue
            if base.provider_id == "qwen-local":
                # Local checkpoint resolution is performed during backend startup
                # and model-load validation. Do not rescan the Hugging Face cache
                # synchronously on every catalog request: a provider-health panel
                # must remain responsive while a local generation is running.
                from app.state import state
                loaded = state.adapter is not None and state.adapter.is_loaded
                descriptors[base.provider_id] = replace(
                    base,
                    available=loaded,
                    availability="available" if loaded else state.model_status,
                    configured=True,
                    configuration_status="configured" if loaded else "missing",
                    connection_status="available" if loaded else "unavailable",
                )
                continue
            configured, names = cloud_config.get(base.provider_id, (False, ()))
            descriptors[base.provider_id] = replace(
                base,
                availability="configured" if configured else "missing-api-key",
                configured=configured,
                available=configured,
                configuration_error=None if configured else "API key missing",
                configuration_status="configured" if configured else "missing",
                connection_status="not_tested",
                last_error_code=None if configured else "API_KEY_MISSING",
                models=names,
                model_descriptors=_cloud_models(base.provider_id, names, configured),
            )
        return descriptors

    @staticmethod
    def normalize_provider(provider_id: str) -> str:
        return {"google": "gemini", "local": "qwen-local"}.get(provider_id, provider_id)

    def get(self, provider_id: str) -> ProviderDescriptor:
        provider_id = self.normalize_provider(provider_id)
        try:
            return self._current_descriptors()[provider_id]
        except KeyError as exc:
            raise ValueError(f"unsupported provider: {provider_id}") from exc

    def list(self) -> list[ProviderDescriptor]:
        return list(self._current_descriptors().values())

    def list_models(self, provider_id: str | None = None) -> list[ModelDescriptor]:
        providers = [self.get(provider_id)] if provider_id else self.list()
        return [model for provider in providers for model in provider.model_descriptors]

    def model(self, provider_id: str, model_id: str) -> ModelDescriptor | None:
        return next((model for model in self.list_models(provider_id) if model.model_id == model_id), None)

    def validate_configuration(self, provider_id: str, *, test_connection: bool = False) -> dict[str, Any]:
        descriptor = self.get(provider_id)
        result = {
            "provider": descriptor.provider_id,
            "configured": descriptor.configured,
            "available": descriptor.available,
            "configuration_status": descriptor.configuration_status,
            "connection_status": descriptor.connection_status,
            "error_code": descriptor.last_error_code,
            "error": descriptor.configuration_error,
        }
        if descriptor.provider_id == "ollama" and test_connection:
            return _validate_ollama(_ollama_url(), result)
        if descriptor.provider_type != "cloud" or not test_connection:
            return result
        key = _provider_key(descriptor.provider_id)
        if not key:
            result.update({"configured": False, "available": False, "configuration_status": "missing", "connection_status": "not_tested", "error_code": "API_KEY_MISSING", "error": "API key missing"})
            return result
        connection, error_code, error = _test_provider_connection(descriptor.provider_id, key)
        result.update({"configured": True, "available": connection == "available", "configuration_status": "configured", "connection_status": connection, "error_code": error_code, "error": error})
        return result


def _validate_ollama(url: str, result: dict[str, Any]) -> dict[str, Any]:
    if not url:
        result.update({"configured": False, "available": False, "configuration_status": "missing", "connection_status": "not_tested", "error_code": "URL_MISSING", "error": "OLLAMA_URL is not configured"})
        return result
    request = urllib.request.Request(f"{url.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            if 200 <= int(response.status) < 300:
                result.update({"available": True, "connection_status": "available", "error_code": None, "error": None})
                return result
            result.update({"available": False, "connection_status": "unavailable", "error_code": "PROVIDER_UNAVAILABLE", "error": f"Ollama returned HTTP {response.status}"})
    except (urllib.error.URLError, TimeoutError, OSError):
        result.update({"available": False, "connection_status": "network_unavailable", "error_code": "NETWORK_UNAVAILABLE", "error": "Ollama is unavailable"})
    return result


def _test_provider_connection(provider_id: str, key: str) -> tuple[str, str | None, str | None]:
    """Perform an explicit, metadata-only provider check.

    This is never called during startup/catalog loading. The endpoints are
    chosen to avoid generating model output or incurring inference charges.
    """
    if provider_id == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider_id == "anthropic":
        url = "https://api.anthropic.com/v1/models"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    elif provider_id == "gemini":
        from urllib.parse import quote

        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={quote(key)}"
        headers = {}
    else:
        raise ValueError(f"unsupported provider connection test: {provider_id}")
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            if 200 <= int(response.status) < 300:
                return "available", None, None
            return "unavailable", "PROVIDER_UNAVAILABLE", f"Provider returned HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return "authentication_failed", "AUTHENTICATION_FAILED", "Authentication failed"
        if exc.code == 429:
            return "rate_limited", "RATE_LIMITED", "Provider rate limited the connection test"
        return "unavailable", "PROVIDER_UNAVAILABLE", f"Provider returned HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError):
        return "network_unavailable", "NETWORK_UNAVAILABLE", "Network unavailable"


provider_registry = ProviderRegistry()


def provider_for(provider_id: str) -> ProviderDescriptor:
    return provider_registry.get(provider_id)
