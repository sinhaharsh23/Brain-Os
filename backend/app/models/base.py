from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelMetadata:
    model_id: str
    architecture: str
    num_params: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    context_length: int
    activation_function: str
    dtype: str
    quantization: str
    device: str
    max_position_embeddings: int = 0
    tokenizer_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "num_params": self.num_params,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "max_position_embeddings": self.max_position_embeddings,
            "activation_function": self.activation_function,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "device": self.device,
            "tokenizer_name": self.tokenizer_name,
            "extra": self.extra,
            "capabilities": self.capabilities,
        }


@dataclass
class TokenInfo:
    text: str
    id: int
    position: int
    is_special: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "id": self.id, "position": self.position, "is_special": self.is_special}


class ModelAdapter(ABC):
    metadata: ModelMetadata

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def tokenize(self, text: str, max_tokens: int | None = None, use_chat_template: bool = True) -> tuple[list[TokenInfo], torch.Tensor, float]: ...

    @abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def embed_token(self, token_id: int) -> torch.Tensor: ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def decode(self, token_ids: list[int] | torch.Tensor) -> str: ...

    @abstractmethod
    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> tuple[int, float]: ...

    @abstractmethod
    def topk_candidates(self, logits: torch.Tensor, k: int = 10, temperature: float = 1.0) -> list[dict[str, Any]]: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...
