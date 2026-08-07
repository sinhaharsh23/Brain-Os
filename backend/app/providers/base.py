from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def to_dict(self) -> dict[str, bool]:
        return {
            "streaming_text": self.streaming_text,
            "token_ids": self.token_ids,
            "embeddings": self.embeddings,
            "hidden_states": self.hidden_states,
            "attention": self.attention,
            "qkv": self.qkv,
            "mlp_activations": self.mlp_activations,
            "logits": self.logits,
            "probabilities": self.probabilities,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "kind": self.kind,
            "inspection_mode": self.inspection_mode,
            "availability": self.availability,
            "limitation": self.limitation,
            "capabilities": self.capabilities.to_dict(),
            "models": list(self.models),
        }
