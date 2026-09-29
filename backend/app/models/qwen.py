from __future__ import annotations

import gc
import json
import logging
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.device import resolve_device
from app.models.base import BaseModelAdapter, ModelAdapter, ModelMetadata, TokenInfo
from app.models.resolver import LocalModelResolver, ResolvedLocalModel

log = logging.getLogger("brainos.model")

Q2_ARCH = "Qwen2ForCausalLM"


def pick_dtype(device: str, requested: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda"):
        return torch.bfloat16
    if device.startswith("mps"):
        return torch.float32
    # CPU bfloat16 kernels can silently produce non-finite logits on older
    # PyTorch/CPU combinations. Prefer a valid float32 inference path unless
    # a caller explicitly requests another dtype.
    return torch.float32


def pick_device(requested: str) -> str:
    return resolve_device(requested).device


class HuggingFaceCausalAdapter(BaseModelAdapter):
    name = "huggingface-causal"
    family = "causal-lm"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        model_path: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.resolved_model: ResolvedLocalModel | None = None
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device, dtype)
        self._model = None
        self._tokenizer = None
        self._load_time_s = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        t0 = time.time()
        try:
            # Resolve and validate on disk before Transformers sees the model.
            # Passing a repository ID directly can trigger Hub metadata/DNS
            # requests even when a usable local checkpoint exists.
            self.resolved_model = LocalModelResolver(
                model_id=self.model_id,
                model_path=self.model_path,
            ).resolve_model()
            resolved_path = str(self.resolved_model.path)
            checkpoint_config = json.loads((self.resolved_model.path / "config.json").read_text())
            self._config_dtype = checkpoint_config.get("dtype", checkpoint_config.get("torch_dtype"))
            self._tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                local_files_only=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                dtype=self.dtype,
                attn_implementation="eager",
                local_files_only=True,
            )
            try:
                self._model.to(self.device)
            except Exception as exc:
                if self.device == "mps":
                    log.warning("Failed to place model on MPS (%s), falling back to CPU", exc)
                    self.device = "cpu"
                    self.dtype = pick_dtype(self.device, "auto")
                    self._model.to(self.device)
                else:
                    raise
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._load_time_s = time.time() - t0
            self.metadata = self._build_metadata()
        except Exception as e:
            log.error(
                "\nBrainOS model initialization failed.\n"
                "Model: %s\n"
                "Device: %s\n"
                "Reason: %s\n"
                "Possible action: Verify Hugging Face model cache or set BRAINOS_DEVICE=cpu\n",
                self.model_id,
                self.device,
                e,
                exc_info=True,
            )
            raise

    def _build_metadata(self) -> ModelMetadata:
        cfg = self._model.config
        hidden_size = int(getattr(cfg, "hidden_size"))
        attention_heads = int(getattr(cfg, "num_attention_heads"))
        kv_heads = int(getattr(cfg, "num_key_value_heads", attention_heads))
        head_dim = int(getattr(cfg, "head_dim", None) or (hidden_size // attention_heads))
        layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 0)))
        intermediate_size = int(getattr(cfg, "intermediate_size", 0) or 0)
        context_length = int(
            getattr(cfg, "max_position_embeddings", 0)
            or getattr(cfg, "max_sequence_length", 0)
            or getattr(cfg, "sliding_window", 0)
            or 0
        )
        quant = "none"
        if hasattr(self._model, "hf_quantizer") and self._model.hf_quantizer is not None:
            quant = str(self._model.hf_quantizer.quantization_config.quant_method)
        model_memory_bytes = sum(p.numel() * p.element_size() for p in self._model.parameters()) + sum(
            b.numel() * b.element_size() for b in self._model.buffers()
        )
        rope_scaling = getattr(cfg, "rope_scaling", None)
        rope_theta = getattr(cfg, "rope_theta", None)
        uses_rope = hasattr(cfg, "rope_theta") or rope_scaling is not None
        if uses_rope:
            position_encoding_type = "RoPE"
            position_method = "Rotary Position Embedding (RoPE)"
        elif getattr(cfg, "alibi", False):
            position_encoding_type = "ALiBi"
            position_method = "ALiBi"
        elif hasattr(cfg, "max_position_embeddings"):
            position_encoding_type = "Absolute Position Embedding"
            position_method = "Absolute Position Embedding"
        else:
            position_encoding_type = "Unavailable"
            position_method = "Unavailable"
        return ModelMetadata(
            model_id=self.model_id,
            architecture=self._model.__class__.__name__,
            num_params=sum(p.numel() for p in self._model.parameters()),
            num_layers=layers,
            hidden_size=hidden_size,
            num_attention_heads=attention_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            vocab_size=cfg.vocab_size,
            context_length=context_length,
            max_position_embeddings=getattr(cfg, "max_position_embeddings", context_length),
            activation_function=getattr(cfg, "hidden_act", getattr(cfg, "hidden_activation", "unknown")),
            dtype=str(next(self._model.parameters()).dtype),
            quantization=quant,
            device=str(next(self._model.parameters()).device),
            tokenizer_name=self.model_id,
            model_memory_mb=round(model_memory_bytes / (1024 * 1024), 2),
            extra={
                "load_time_s": self._load_time_s,
                "config_dtype": getattr(self, "_config_dtype", str(getattr(cfg, "dtype", getattr(cfg, "torch_dtype", None)))),
                "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
                "use_cache": getattr(cfg, "use_cache", None),
                "trainable_parameters": sum(p.numel() for p in self._model.parameters() if p.requires_grad),
                "generation_defaults": self._model.generation_config.to_dict() if getattr(self._model, "generation_config", None) is not None else None,
                "revision": getattr(cfg, "_commit_hash", None),
                "attention_impl": "eager",
                "adapter_family": self.family,
                "resolved_model_path": str(self.resolved_model.path) if self.resolved_model else None,
                "model_source": self.resolved_model.source if self.resolved_model else "unknown",
                "network_required": False,
                "position_method": position_method,
                "position_encoding_type": position_encoding_type,
                "rotary_dim": head_dim if uses_rope else None,
                "rope_theta": rope_theta,
                "rope_scaling": rope_scaling,
            },
            capabilities=self._detect_capabilities(),
        )

    @property
    def _backbone(self):
        return getattr(self._model, "model", self._model)

    @property
    def _layers(self):
        return getattr(self._backbone, "layers", ())

    def _embedding_layer(self):
        embedding = getattr(self._backbone, "embed_tokens", None)
        if embedding is not None:
            return embedding
        return self._model.get_input_embeddings()

    def get_transformer_layers(self) -> list[torch.nn.Module]:
        return list(self._layers)

    def get_embedding_module(self) -> torch.nn.Module:
        return self._embedding_layer()

    def get_attention_module(self, layer_idx: int) -> torch.nn.Module | None:
        layers = self.get_transformer_layers()
        if 0 <= layer_idx < len(layers):
            return getattr(layers[layer_idx], "self_attn", None)
        return None

    def get_q_projection(self, layer_idx: int) -> torch.nn.Module | None:
        attn = self.get_attention_module(layer_idx)
        return getattr(attn, "q_proj", None) if attn is not None else None

    def get_k_projection(self, layer_idx: int) -> torch.nn.Module | None:
        attn = self.get_attention_module(layer_idx)
        return getattr(attn, "k_proj", None) if attn is not None else None

    def get_v_projection(self, layer_idx: int) -> torch.nn.Module | None:
        attn = self.get_attention_module(layer_idx)
        return getattr(attn, "v_proj", None) if attn is not None else None

    def get_o_projection(self, layer_idx: int) -> torch.nn.Module | None:
        attn = self.get_attention_module(layer_idx)
        return getattr(attn, "o_proj", None) if attn is not None else None

    def get_mlp_module(self, layer_idx: int) -> torch.nn.Module | None:
        layers = self.get_transformer_layers()
        if 0 <= layer_idx < len(layers):
            return getattr(layers[layer_idx], "mlp", None)
        return None

    def get_final_norm(self) -> torch.nn.Module | None:
        return getattr(self._backbone, "norm", None)

    def get_lm_head(self) -> torch.nn.Module | None:
        return getattr(self._model, "lm_head", None)

    def project_hidden_state_to_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Projects a hidden state through the final layer norm and language-model head."""
        if self._model is None:
            raise ValueError("Model is not loaded")
        norm = self.get_final_norm()
        lm_head = self.get_lm_head()
        if lm_head is None:
            raise ValueError("Model does not have an lm_head")

        with torch.no_grad():
            hs = hidden_state.to(device=self.device, dtype=self.dtype)
            if norm is not None:
                hs = norm(hs)
            logits = lm_head(hs)
        return logits.to(torch.float32).cpu()

    def get_architecture_tree(self, max_depth: int = 3) -> dict[str, Any]:
        """Returns the module hierarchy tree with parameter counts and tensor shapes."""
        if self._model is None:
            return {"name": "not_loaded", "type": "None", "total_params": 0, "children": []}

        def _build(mod: torch.nn.Module, name: str, depth: int) -> dict[str, Any]:
            direct_params = sum(p.numel() for p in mod.parameters(recurse=False))
            total_params = sum(p.numel() for p in mod.parameters())
            weight_shape = list(mod.weight.shape) if hasattr(mod, "weight") and hasattr(mod.weight, "shape") else None
            children = []
            if depth < max_depth:
                for child_name, child_mod in mod.named_children():
                    children.append(_build(child_mod, child_name, depth + 1))
            return {
                "name": name,
                "type": mod.__class__.__name__,
                "direct_params": direct_params,
                "total_params": total_params,
                "weight_shape": weight_shape,
                "children": children,
            }

        return _build(self._model, "model", 0)

    def _detect_capabilities(self) -> dict[str, bool]:
        sample_layer = self._layers[0] if len(self._layers) else None
        attn = getattr(sample_layer, "self_attn", None)
        mlp = getattr(sample_layer, "mlp", None)
        has_attention = attn is not None
        has_qkv = has_attention and hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj")
        has_mlp = mlp is not None
        return {
            "has_attentions": has_attention,
            "has_hidden_states": True,
            "has_qkv_hooks": has_qkv,
            "has_mlp_hooks": has_mlp,
            "has_embeddings": self._embedding_layer() is not None,
            "has_chat_template": bool(getattr(self._tokenizer, "chat_template", None)),
            "token_ids": True, "token_strings": True, "hidden_states": True, "mlp_activations": has_mlp,
            "kv_cache": True, "kv_tensor_cache": True, "forward_hooks": True, "architecture": True, "probabilities": True, "logprobs": True,
            "tokenization": True,
            "embeddings": self._embedding_layer() is not None,
            "hiddenStates": True,
            "attention": has_attention,
            "qkv": has_qkv,
            "mlp": has_mlp,
            "residual": has_attention and has_mlp,
            "logits": True,
            "logitLens": True,
            "kvCache": True,
            "activationPatching": False,
        }

    def tokenize(
        self,
        text: str,
        max_tokens: int | None = None,
        use_chat_template: bool = True,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[list[TokenInfo], torch.Tensor, float]:
        t0 = time.time()
        special_ids = set(self._tokenizer.all_special_ids)
        if use_chat_template and getattr(self._tokenizer, "chat_template", None):
            input_ids = self._tokenizer.apply_chat_template(
                messages or [{"role": "user", "content": text}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt",
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
        else:
            input_ids = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if max_tokens and input_ids.shape[1] > max_tokens:
            raise ValueError(f"Model input has {input_ids.shape[1]} tokens; limit is {max_tokens}. Shorten the conversation.")

        token_ids = input_ids[0].tolist()
        tokens = []
        for pos, tid in enumerate(token_ids):
            txt = self._tokenizer.decode([tid], skip_special_tokens=False)
            tokens.append(TokenInfo(text=txt, id=tid, position=pos, is_special=tid in special_ids))
        dt = time.time() - t0
        return tokens, input_ids.to(self.device), dt

    def user_text_token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def chat_template_snapshot(self, text: str, use_chat_template: bool = True, messages: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Return the exact serialized input used for a chat-template trace.

        This is intentionally a small, display-oriented payload. Token IDs and
        tensors remain the source of truth for inference; the string is only
        exposed so the UI can show the formatting step without inventing
        control tokens.
        """
        has_template = bool(getattr(self._tokenizer, "chat_template", None))
        if use_chat_template and has_template and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                serialized = self._tokenizer.apply_chat_template(
                    messages or [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return {
                    "available": True,
                    "name": "model tokenizer chat_template",
                    "serialized": str(serialized),
                    "message_count": len(messages or [{"role": "user", "content": text}]),
                    "has_system_message": any(item.get("role") == "system" for item in (messages or [])),
                    "generation_prompt": True,
                }
            except Exception as exc:
                return {
                    "available": False,
                    "name": "model tokenizer chat_template",
                    "serialized": "",
                    "message_count": 1,
                    "has_system_message": False,
                    "generation_prompt": False,
                    "error": str(exc),
                }
        return {
            "available": False,
            "name": "none",
            "serialized": text,
            "message_count": 1,
            "has_system_message": False,
            "generation_prompt": False,
            "reason": "This tokenizer does not expose a chat template.",
        }

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._embedding_layer()(input_ids)

    def embed_token(self, token_id: int) -> torch.Tensor:
        with torch.no_grad():
            t = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            return self._embedding_layer()(t).squeeze(0).squeeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        return {
            "logits": out.logits,
            "past_key_values": out.past_key_values,
            "attentions": list(out.attentions) if out.attentions else None,
            "hidden_states": list(out.hidden_states) if out.hidden_states else None,
        }

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def decode_output(self, token_ids: list[int] | torch.Tensor) -> str:
        """Decode the final human-readable response without control tokens.

        ``decode`` intentionally preserves special tokens because the token
        explorer needs to show the exact serialized sequence.  The response
        shown to a person is a separate representation and should omit chat
        protocol markers such as ``<|im_end|>``.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> tuple[int, float]:
        logits = logits.to(torch.float32)
        probs = self.sampling_distribution(logits, temperature, top_p, top_k)
        if temperature == 0.0 or (top_k and top_k == 1):
            token_id = int(probs.argmax().item())
            return token_id, float(probs[token_id])
        dist = torch.distributions.Categorical(probs)
        token_id = int(dist.sample().item())
        return token_id, float(probs[token_id])

    @staticmethod
    def sampling_distribution(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """Build the exact distribution used by instrumented sampling."""
        values = logits.to(torch.float32).clone()
        if temperature == 0.0:
            result = torch.zeros_like(values)
            result[int(values.argmax().item())] = 1.0
            return result
        values = values / temperature
        if top_k and top_k > 0:
            k = min(top_k, values.numel())
            threshold = torch.topk(values, k).values[-1]
            values[values < threshold] = float("-inf")
        probs = torch.softmax(values, dim=-1)
        if top_p < 1.0:
            sorted_p, sorted_idx = probs.sort(descending=True)
            cumulative = sorted_p.cumsum(dim=-1)
            remove = cumulative - sorted_p > top_p
            sorted_p[remove] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_p)
            probs = probs / probs.sum().clamp_min(1e-12)
        return probs

    def sampling_entropy(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> float:
        probs = self.sampling_distribution(logits, temperature, top_p, top_k)
        return float(-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum())

    def topk_candidates(self, logits: torch.Tensor, k: int = 10, temperature: float = 1.0) -> list[dict[str, Any]]:
        logits = logits.to(torch.float32)
        scaled = logits / temperature if temperature else logits
        k = min(k, logits.numel())
        vals, idx = torch.topk(scaled, k)
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        rank_map = {int(t): int(r) for r, t in enumerate(sorted_idx.tolist())}
        return [
            {
                "token_id": int(i),
                "text": self._tokenizer.decode([int(i)], skip_special_tokens=False),
                "logit": float(v),
                "probability": float(probs[int(i)]),
                "rank": rank_map[int(i)],
            }
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    def sampling_candidates(
        self,
        logits: torch.Tensor,
        k: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> list[dict[str, Any]]:
        """Return candidates with probabilities from the exact sampler distribution."""
        raw = logits.to(torch.float32)
        probs = self.sampling_distribution(raw, temperature, top_p, top_k)
        order = torch.argsort(probs, descending=True)[: min(k, raw.numel())]
        ranks = torch.argsort(probs, descending=True)
        rank_map = {int(token_id): int(rank) for rank, token_id in enumerate(ranks.tolist())}
        return [
            {
                "token_id": int(token_id),
                "text": self._tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logit": float(raw[int(token_id)]),
                "probability": float(probs[int(token_id)]),
                "rank": rank_map[int(token_id)],
            }
            for token_id in order.tolist()
        ]

    def sampling_rank(
        self,
        logits: torch.Tensor,
        token_id: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> int:
        probs = self.sampling_distribution(logits.to(torch.float32), temperature, top_p, top_k)
        order = torch.argsort(probs, descending=True)
        matches = torch.nonzero(order == int(token_id), as_tuple=False)
        return int(matches[0].item()) if matches.numel() else int(order.numel())

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()


class QwenAdapter(HuggingFaceCausalAdapter):
    name = "qwen2"
    family = "qwen2"
