from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from app.providers.base import AIProvider, ChatMessage, ModelDescriptor, ProviderDescriptor
from app.providers.external import ExternalObservationEngine
from app.providers.registry import provider_registry


class _ExternalProvider(AIProvider):
    def __init__(self, provider_id: str, engine: ExternalObservationEngine | None = None) -> None:
        self.provider_id = provider_id
        self.engine = engine or ExternalObservationEngine()

    def get_provider_info(self) -> ProviderDescriptor:
        return provider_registry.get(self.provider_id)

    def list_models(self) -> list[ModelDescriptor]:
        return list(provider_registry.list_models(self.provider_id))

    def stream_chat(
        self,
        messages: Iterable[ChatMessage],
        model_id: str | None,
        params: Mapping[str, Any],
        on_event: Callable[[dict[str, Any]], None],
        cancel_event: Any | None = None,
    ) -> None:
        self.engine.run_unified(self.provider_id, list(messages), dict(params), on_event, cancel_event=cancel_event, model_id=model_id)

    def cancel_generation(self, generation_id: str | None = None) -> None:
        self.engine.cancel(generation_id)


class OpenAIProvider(_ExternalProvider):
    def __init__(self, engine: ExternalObservationEngine | None = None) -> None:
        super().__init__("openai", engine)


class AnthropicProvider(_ExternalProvider):
    def __init__(self, engine: ExternalObservationEngine | None = None) -> None:
        super().__init__("anthropic", engine)


class GeminiProvider(_ExternalProvider):
    def __init__(self, engine: ExternalObservationEngine | None = None) -> None:
        super().__init__("gemini", engine)


class OllamaProvider(_ExternalProvider):
    def __init__(self, engine: ExternalObservationEngine | None = None) -> None:
        super().__init__("ollama", engine)


class LocalModelProvider(AIProvider):
    """Descriptor/contract adapter for the existing local inference engine.

    The actual execution remains owned by InferenceEngine so all existing
    hooks, tensor stores, replay, cancellation, and scheduler behavior stay
    intact. This class provides the same metadata surface as cloud adapters.
    """

    provider_id = "qwen-local"

    def get_provider_info(self) -> ProviderDescriptor:
        return provider_registry.get(self.provider_id)

    def list_models(self) -> list[ModelDescriptor]:
        return list(provider_registry.list_models(self.provider_id))

    def stream_chat(self, messages, model_id, params, on_event, cancel_event=None) -> None:
        raise RuntimeError("LocalModelProvider is executed by the existing InferenceEngine scheduler")


class AIProviderRouter:
    def __init__(self, external_engine: ExternalObservationEngine | None = None) -> None:
        engine = external_engine or ExternalObservationEngine()
        self.providers: dict[str, AIProvider] = {
            "qwen-local": LocalModelProvider(),
            "openai": OpenAIProvider(engine),
            "anthropic": AnthropicProvider(engine),
            "gemini": GeminiProvider(engine),
            "ollama": OllamaProvider(engine),
        }

    @staticmethod
    def normalize_provider(provider_id: str) -> str:
        return provider_registry.normalize_provider(provider_id)

    def get_provider(self, provider_id: str) -> AIProvider:
        provider_id = self.normalize_provider(provider_id)
        try:
            return self.providers[provider_id]
        except KeyError as exc:
            raise ValueError(f"unsupported provider: {provider_id}") from exc

    def get_provider_info(self, provider_id: str) -> ProviderDescriptor:
        return self.get_provider(provider_id).get_provider_info()

    def list_models(self, provider_id: str | None = None) -> list[ModelDescriptor]:
        return self.get_provider(provider_id).list_models() if provider_id else provider_registry.list_models()

    def stream_chat(self, provider_id: str, messages, model_id, params, on_event, cancel_event=None) -> None:
        self.get_provider(provider_id).stream_chat(messages, model_id, params, on_event, cancel_event=cancel_event)


provider_router = AIProviderRouter()
