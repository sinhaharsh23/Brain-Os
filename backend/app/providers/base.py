from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class ProviderCapabilities:
    streaming_text: bool
    token_ids: bool
    embeddings: bool
    hidden_states: bool
    attention: bool
    qkv: bool
    mlp_activations: bool
    logits: bool
    probabilities: bool
    chat: bool = True
    api_usage: bool = False
    server_metadata: bool = False
    logprobs: bool = False
    logit_lens: bool = False
    kv_cache: bool = False
    streaming: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "forward_hooks": self.hidden_states and self.qkv,
            "architecture": self.server_metadata or self.hidden_states,
            "token_strings": self.token_ids,
            "kv_tensor_cache": self.kv_cache,
            "chat": self.chat,
            "streaming": self.streaming,
            "streaming_text": self.streaming_text,
            "api_usage": self.api_usage,
            "server_metadata": self.server_metadata,
            "logprobs": self.logprobs,
            "token_ids": self.token_ids,
            "embeddings": self.embeddings,
            "hidden_states": self.hidden_states,
            "attention": self.attention,
            "qkv": self.qkv,
            "mlp_activations": self.mlp_activations,
            "logits": self.logits,
            "probabilities": self.probabilities,
            "logit_lens": self.logit_lens,
            "kv_cache": self.kv_cache,
        }


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    display_name: str
    kind: str
    inspection_mode: str
    availability: str
    limitation: str
    capabilities: ProviderCapabilities
    models: tuple[str, ...] = ()
    provider_type: str | None = None
    configured: bool | None = None
    available: bool | None = None
    configuration_error: str | None = None
    configuration_status: str = "unknown"
    connection_status: str = "not_tested"
    last_error_code: str | None = None
    model_descriptors: tuple["ModelDescriptor", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "kind": self.kind,
            "provider_type": self.provider_type or ("local" if self.kind == "local" else "cloud"),
            "local_or_cloud": "local" if self.kind == "local" else "cloud",
            "inspection_mode": self.inspection_mode,
            "availability": self.availability,
            "configured": self.configured if self.configured is not None else self.availability == "configured",
            "available": self.available if self.available is not None else self.availability in {"available", "configured"},
            "configuration_error": self.configuration_error,
            "configuration_status": self.configuration_status,
            "connection_status": self.connection_status,
            "last_error_code": self.last_error_code,
            "limitation": self.limitation,
            "capabilities": self.capabilities.to_dict(),
            "models": list(self.models),
            "model_descriptors": [model.to_dict() for model in self.model_descriptors],
        }


@dataclass(frozen=True)
class ModelDescriptor:
    """Provider-neutral model metadata.

    Local transformer fields are optional by design. Cloud models should not
    be forced to masquerade as locally executed checkpoints.
    """

    model_id: str
    provider: str
    display_name: str
    mode: str
    architecture: str | None = None
    capabilities: ProviderCapabilities | Mapping[str, bool] = field(default_factory=lambda: ProviderCapabilities(False, False, False, False, False, False, False, False, False))
    context_length: int | None = None
    loaded: bool = False
    available: bool = True
    device: str | None = None
    parameter_count: int | None = None
    supports_streaming: bool = True
    supports_introspection: bool = False
    configuration_error: str | None = None

    @property
    def local_or_cloud(self) -> str:
        return "local" if self.mode == "local" else "cloud"

    def to_dict(self) -> dict[str, Any]:
        caps = self.capabilities.to_dict() if isinstance(self.capabilities, ProviderCapabilities) else dict(self.capabilities)
        return {
            "id": self.model_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "mode": self.mode,
            "local_or_cloud": self.local_or_cloud,
            "architecture": self.architecture,
            "capabilities": caps,
            "context_length": self.context_length,
            "loaded": self.loaded,
            "available": self.available,
            "device": self.device,
            "parameter_count": self.parameter_count,
            "supports_streaming": self.supports_streaming,
            "supports_introspection": self.supports_introspection,
            "configuration_error": self.configuration_error,
        }


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    id: str | None = None
    created_at: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "id": self.id, "created_at": self.created_at, "metadata": dict(self.metadata)}


class AIProvider(ABC):
    """Minimal adapter contract shared by local and cloud providers."""

    provider_id: str

    @abstractmethod
    def get_provider_info(self) -> ProviderDescriptor: ...

    @abstractmethod
    def list_models(self) -> list[ModelDescriptor]: ...

    def get_model_info(self, model_id: str) -> ModelDescriptor | None:
        return next((model for model in self.list_models() if model.model_id == model_id), None)

    def get_capabilities(self, model_id: str | None = None) -> dict[str, bool]:
        return self.get_provider_info().capabilities.to_dict()

    def validate_configuration(self) -> dict[str, Any]:
        info = self.get_provider_info()
        return {"configured": info.configured, "available": info.available, "error": info.configuration_error}

    @abstractmethod
    def stream_chat(self, messages: Iterable[ChatMessage], model_id: str | None, params: Mapping[str, Any], on_event: Callable[[dict[str, Any]], None], cancel_event: Any | None = None) -> None: ...

    def cancel_generation(self, generation_id: str | None = None) -> None:
        return None

    def get_usage(self) -> dict[str, int] | None:
        return None

    def get_generation_metadata(self) -> dict[str, Any]:
        return {}
