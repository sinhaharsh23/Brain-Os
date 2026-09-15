from __future__ import annotations

import logging
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.device import resolve_device
from app.models.base import ModelAdapter, ModelMetadata, TokenInfo

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


class HuggingFaceCausalAdapter(ModelAdapter):
    name = "huggingface-causal"
    family = "causal-lm"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto", dtype: str = "auto") -> None:
        self.model_id = model_id
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device, dtype)
        self._model = None
        self._tokenizer = None
        self._load_time_s = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        t0 = time.time()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                attn_implementation="eager",
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
            device=self.device,
            tokenizer_name=self.model_id,
            extra={"load_time_s": self._load_time_s, "attention_impl": "eager", "adapter_family": self.family},
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

    def _detect_capabilities(self) -> dict[str, bool]:
        layers = list(self._layers)
        layer = layers[0] if layers else None
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        return {
            "tokenization": self._tokenizer is not None,
            "generation": True,
            "streaming": True,
            "embeddings": self._embedding_layer() is not None,
            "hidden_states": True,
            "attention": attention is not None,
            "qkv": attention is not None and all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")),
            "mlp_activations": mlp is not None and all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")),
            "logits": hasattr(self._model, "lm_head"),
            "probabilities": hasattr(self._model, "lm_head"),
        }

    def tokenize(
        self, text: str, max_tokens: int | None = None, use_chat_template: bool = True
    ) -> tuple[list[TokenInfo], torch.Tensor, float]:
        t0 = time.time()
        has_chat_template = bool(getattr(self._tokenizer, "chat_template", None))
        if use_chat_template and has_chat_template:
            messages = [{"role": "user", "content": text}]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = text
        enc = self._tokenizer(prompt, add_special_tokens=False if use_chat_template and has_chat_template else True)
        ids = enc["input_ids"]
        if max_tokens and len(ids) > max_tokens:
            ids = ids[:max_tokens]
        specials = set(self._tokenizer.all_special_tokens)
        token_strings = self._tokenizer.convert_ids_to_tokens(ids)
        tokens = [
            TokenInfo(
                text=token_strings[i],
                id=ids[i],
                position=i,
                is_special=token_strings[i] in specials or (token_strings[i].startswith("<") and token_strings[i].endswith(">")),
            )
            for i in range(len(ids))
        ]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return tokens, input_ids, time.time() - t0

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_layer()(input_ids)

    def embed_token(self, token_id: int) -> torch.Tensor:
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

    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> tuple[int, float]:
        logits = logits.to(torch.float32)
        if temperature == 0.0:
            token_id = int(logits.argmax().item())
            probs = torch.softmax(logits, dim=-1)
            return token_id, float(probs[token_id])
        if temperature != 1.0:
            logits = logits / temperature
        if top_k and top_k > 0:
            k = min(top_k, logits.numel())
            v, _ = torch.topk(logits, k)
            logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_p, sorted_idx = probs.sort(descending=True)
            cum = sorted_p.cumsum(dim=-1)
            mask = cum - sorted_p > top_p
            sorted_p[mask] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_p)
            probs = probs / probs.sum()
        if top_k and top_k == 1:
            token_id = int(logits.argmax().item())
            return token_id, float(probs[token_id])
        dist = torch.distributions.Categorical(probs)
        token_id = int(dist.sample().item())
        return token_id, float(probs[token_id])

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
                "text": self._tokenizer.convert_ids_to_tokens([int(i)])[0],
                "logit": float(v),
                "probability": float(probs[int(i)]),
                "rank": rank_map[int(i)],
            }
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenAdapter(HuggingFaceCausalAdapter):
    name = "qwen2"
    family = "qwen2"
