from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from app.events.types import Event, tensor_stats, token_view
from app.instrumentation.stats import (
    attention_topk,
    cosine_sim,
    pca_project_vector,
    pca_projection,
    qkv_head_slice,
    residual_metrics,
    sample_activations,
    tensor_info,
    topk_activations,
    vector_stats,
)


@dataclass
class TensorStore:
    session_id: str
    prompt_length: int = 0
    embeddings: torch.Tensor | None = None
    generated_embeddings: list[torch.Tensor] = field(default_factory=list)
    pca: dict[str, Any] = field(default_factory=dict)
    qkv: dict[tuple[int, str, int], torch.Tensor] = field(default_factory=dict)
    mlp: dict[tuple[int, str, int], torch.Tensor] = field(default_factory=dict)
    hidden: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    attention: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    logits: dict[int, torch.Tensor] = field(default_factory=dict)
    logit_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    logit_vocab_sizes: dict[int, int] = field(default_factory=dict)
    logit_indices: dict[int, torch.Tensor] = field(default_factory=dict)
    logit_candidates: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    raw_residual: dict[tuple[int, str, int], torch.Tensor] = field(default_factory=dict)
    residual: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    logit_lens: dict[tuple[int, int], list[dict[str, Any]]] = field(default_factory=dict)
    kv_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    steps_total: int = 0
    full_pass_steps: set[int] = field(default_factory=set)
    sequence_starts: dict[int, int] = field(default_factory=dict)
    qkv_starts: dict[tuple[int, str, int], int] = field(default_factory=dict)
    attention_starts: dict[int, int] = field(default_factory=dict)
    attention_column_starts: dict[int, int] = field(default_factory=dict)

    def position_of(self, position: int) -> tuple[int, int]:
        if position < self.prompt_length:
            return 0, position
        return position - self.prompt_length + 1, 0

    def attention_for_position(self, layer: int, position: int) -> torch.Tensor | None:
        step, row = self.position_of(position)
        m = self.attention.get((layer, step))
        if m is None:
            return None
        if step == 0:
            start = self.attention_starts.get(step, 0)
            relative_row = row - start
            if relative_row < 0 or relative_row >= m.shape[1]:
                return None
            return m[:, relative_row : relative_row + 1, :]
        return m

    def qkv_for_position(self, layer: int, name: str, position: int) -> torch.Tensor | None:
        step, row = self.position_of(position)
        m = self.qkv.get((layer, name, step))
        if m is None:
            return None
        start = self.qkv_starts.get((layer, name, step), self.sequence_starts.get(step, 0))
        relative_row = row - start if step == 0 else row
        if relative_row < 0 or relative_row >= m.shape[0]:
            return None
        return m[relative_row : relative_row + 1]

    def qkv_head_for_position(
        self, layer: int, name: str, head: int, position: int, head_dim: int, num_heads: int
    ) -> dict[str, Any] | None:
        t = self.qkv_for_position(layer, name, position)
        if t is None or head < 0 or head >= num_heads:
            return None
        head_vec = qkv_head_slice(t[0], head, head_dim, num_heads)
        return {
            "layer": layer,
            "name": name,
            "head": head,
            "position": position,
            "head_dim": head_dim,
            "stats": vector_stats(head_vec),
            "values": sample_activations(head_vec, limit=head_dim),
        }

    def mlp_for_position(self, layer: int, name: str, position: int) -> torch.Tensor | None:
        step, row = self.position_of(position)
        m = self.mlp.get((layer, name, step))
        if m is None:
            return None
        start = self.sequence_starts.get(step, 0)
        relative_row = row - start if step == 0 else row
        if relative_row < 0 or relative_row >= m.shape[0]:
            return None
        return m[relative_row : relative_row + 1]

    def hidden_for_position(self, layer: int, position: int) -> torch.Tensor | None:
        step, row = self.position_of(position)
        m = self.hidden.get((layer, step))
        if m is None:
            return None
        start = self.sequence_starts.get(step, 0)
        relative_row = row - start if step == 0 else row
        if relative_row < 0 or relative_row >= m.shape[0]:
            return None
        return m[relative_row : relative_row + 1]

    def hidden_for_step(self, layer: int, step: int) -> torch.Tensor | None:
        """Return the query row used to predict a generation step."""
        m = self.hidden.get((layer, step))
        if m is None or m.shape[0] == 0:
            return None
        return m[-1:] if step == 0 else m[:1]

    def residual_for_position(self, layer: int, position: int) -> dict[str, Any] | None:
        step, _ = self.position_of(position)
        return self.residual.get((layer, step))

    def logit_lens_for_position(self, layer: int, position: int) -> list[dict[str, Any]] | None:
        step, _ = self.position_of(position)
        return self.logit_lens.get((layer, step))

    def kv_cache_for_position(self, position: int) -> dict[str, Any] | None:
        step, _ = self.position_of(position)
        return self.kv_cache.get(step)


class SessionRecord:
    def __init__(
        self,
        session_id: str,
        prompt: str,
        model_id: str,
        params: dict[str, Any],
        metadata: dict[str, Any],
        adapter=None,
    ) -> None:
        self.session_id = session_id
        self.prompt = prompt
        self.model_id = model_id
        self.params = params
        self.metadata = metadata
        self.adapter = adapter
        self.adapter_name = str(metadata.get("adapter_name") or getattr(adapter, "name", "unknown"))
        self.created_at = time.time()
        self.tokens: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.output_tokens: list[dict[str, Any]] = []
        self.response = ""
        self.status = "running"
        self.store = TensorStore(session_id=session_id)
        self.timings: dict[str, float] = {}
        self.errors: list[str] = []
        self.step_stats: list[dict[str, Any]] = []

    def add_event(self, event: Event) -> None:
        self.events.append(event.to_dict())

    def summary(self) -> dict[str, Any]:
        from app.providers.telemetry import native_telemetry
        store = self.store
        capture_available = any(
            (
                store.embeddings is not None,
                bool(store.generated_embeddings),
                bool(store.qkv),
                bool(store.mlp),
                bool(store.hidden),
                bool(store.attention),
                bool(store.logits),
                bool(store.residual),
                bool(store.logit_lens),
                bool(store.kv_cache),
            )
        )
        return {
            "provider": self.metadata.get("provider", "qwen-local"),
            "inspection_mode": self.metadata.get("inspection_mode", "deep"),
            "telemetry": native_telemetry(self) if self.metadata.get("provider", "qwen-local") == "qwen-local" else None,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "model_id": self.model_id,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "num_tokens": len(self.tokens),
            "num_output_tokens": len(self.output_tokens),
            "tokens": self.tokens,
            "output_tokens": self.output_tokens,
            "response": self.response,
            "timings": self.timings,
            "step_stats": self.step_stats,
            "errors": self.errors,
            "metadata": self.metadata,
            "adapter_name": self.adapter_name,
            "capture_available": capture_available,
            "capture": {
                "qkv_entries": len(store.qkv),
                "mlp_entries": len(store.mlp),
                "hidden_entries": len(store.hidden),
                "attention_entries": len(store.attention),
                "residual_entries": len(store.residual),
                "logit_lens_entries": len(store.logit_lens),
                "kv_cache_steps": len(store.kv_cache),
                "capture_level": self.metadata.get("capture_level", "summary"),
                "attention_capture": self.metadata.get("attention_capture", True),
            },
        }


class InferenceEngine:
    def __init__(
        self,
        adapter,
        bus,
        max_prompt_tokens: int = 1024,
        capture_limit: int = 256,
        capture_level: str = "summary",
        attention_capture: bool = True,
    ) -> None:
        self.adapter = adapter
        self.bus = bus
        self.max_prompt_tokens = max_prompt_tokens
        self.capture_limit = max(1, int(capture_limit))
        self.capture_level = capture_level if capture_level in {"summary", "selected", "full"} else "summary"
        self.attention_capture = bool(attention_capture)
        self._lock = threading.Lock()
        self._current: SessionRecord | None = None
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._hooks_registered = False
        self._closed = False
        self._owner = None
        self._on_event = None
        self._archived: dict[str, SessionRecord] = {}
        # Restored captures can contain large CPU tensors. Keep a small LRU-like
        # working set so opening old sessions does not retain every capture for
        # the lifetime of the backend process. The source of truth remains the
        # replay directory and captures are reloaded on demand.
        self._max_archived_sessions = 3

    @property
    def current_session(self) -> SessionRecord | None:
        return self._current

    def restore_session(self, summary: dict[str, Any], capture: dict[str, Any]) -> SessionRecord:
        session_id = str(summary["session_id"])
        rec = SessionRecord(
            session_id,
            str(summary.get("prompt", "")),
            str(summary.get("model_id", "")),
            dict(summary.get("params", {})),
            dict(summary.get("metadata", {})),
        )
        rec.adapter_name = str(summary.get("adapter_name") or rec.metadata.get("adapter_name", "unknown"))
        rec.tokens = list(summary.get("tokens", []))
        rec.output_tokens = list(summary.get("output_tokens", []))
        rec.response = str(summary.get("response", ""))
        rec.status = str(summary.get("status", "complete"))
        rec.created_at = float(summary.get("created_at", rec.created_at))
        rec.timings = dict(summary.get("timings", {}))
        rec.step_stats = list(summary.get("step_stats", []))
        rec.errors = list(summary.get("errors", []))
        store = rec.store
        store.prompt_length = int(capture.get("prompt_length", len(rec.tokens)))
        store.embeddings = capture.get("embeddings")
        store.generated_embeddings = list(capture.get("generated_embeddings", []))
        store.pca = dict(capture.get("pca", {}))
        store.qkv = dict(capture.get("qkv", {}))
        store.mlp = dict(capture.get("mlp", {}))
        store.hidden = dict(capture.get("hidden", {}))
        store.attention = dict(capture.get("attention", {}))
        store.logits = dict(capture.get("logits", {}))
        store.logit_stats = dict(capture.get("logit_stats", {}))
        store.logit_vocab_sizes = dict(capture.get("logit_vocab_sizes", {}))
        store.logit_indices = dict(capture.get("logit_indices", {}))
        store.logit_candidates = dict(capture.get("logit_candidates", {}))
        store.residual = dict(capture.get("residual", {}))
        store.logit_lens = dict(capture.get("logit_lens", {}))
        store.kv_cache = dict(capture.get("kv_cache", {}))
        store.steps_total = int(capture.get("steps_total", 0))
        store.sequence_starts = dict(capture.get("sequence_starts", {}))
        store.qkv_starts = dict(capture.get("qkv_starts", {}))
        store.attention_starts = dict(capture.get("attention_starts", {}))
        store.attention_column_starts = dict(capture.get("attention_column_starts", {}))
        self._archived[session_id] = rec
        while len(self._archived) > self._max_archived_sessions:
            oldest_id = next(iter(self._archived))
            if oldest_id == session_id and len(self._archived) > 1:
                oldest_id = next(iter(list(self._archived)[1:]))
            self._archived.pop(oldest_id, None)
        return rec

    def cancel(self, owner=None) -> bool:
        if owner is not None and self._owner is not owner:
            return False
        self._cancel.set()
        self._pause.set()
        return True

    def close(self) -> None:
        self.cancel()
        with self._lock:
            self._closed = True
            self._cleanup_hooks()

    def _cleanup_hooks(self) -> None:
        """Remove per-generation hooks and release their session reference."""
        if self._hooks_registered:
            self._hook_manager.remove_all()
            self._hooks_registered = False
        if hasattr(self, "_store_ref"):
            self._store_ref["store"] = None

    def pause(self, owner=None) -> bool:
        if owner is not None and self._owner is not owner:
            return False
        self._pause.clear()
        return True

    def resume(self, owner=None) -> bool:
        if owner is not None and self._owner is not owner:
            return False
        self._pause.set()
        return True

    @property
    def paused(self) -> bool:
        return not self._pause.is_set()

    def _register_hooks(self) -> None:
        if self._hooks_registered:
            return
        model = self.adapter._model
        store_ref: dict[str, Any] = {"store": None}

        def on_qkv(layer: int, name: str, tensor: torch.Tensor) -> None:
            store = store_ref["store"]
            if store is None:
                return
            step = store.steps_total
            store.qkv[(layer, name, step)] = self._bounded_sequence_tensor(store, tensor, step)

        def on_mlp(layer: int, name: str, tensor: torch.Tensor) -> None:
            store = store_ref["store"]
            if store is None:
                return
            step = store.steps_total
            detached = self._bounded_sequence_tensor(store, tensor, step)
            store.mlp[(layer, name, step)] = detached
            if name == "up" and (layer, "gate_activation", step) in store.mlp:
                gate = store.mlp[(layer, "gate_activation", step)]
                if gate.shape == detached.shape:
                    store.mlp[(layer, "intermediate", step)] = gate * detached
            elif name == "gate_activation" and (layer, "up", step) in store.mlp:
                up = store.mlp[(layer, "up", step)]
                if up.shape == detached.shape:
                    store.mlp[(layer, "intermediate", step)] = detached * up

        def on_layer_out(layer: int, tensor: torch.Tensor) -> None:
            store = store_ref["store"]
            if store is None:
                return
            step = store.steps_total
            store.hidden[(layer + 1, step)] = self._bounded_sequence_tensor(store, tensor, step)

        def on_residual(layer: int, component: str, tensor: torch.Tensor) -> None:
            store = store_ref["store"]
            if store is None:
                return
            step = store.steps_total
            store.raw_residual[(layer, component, step)] = self._bounded_sequence_tensor(store, tensor, step)

        def on_module_event(category: str, layer: int, status: str) -> None:
            rec = self._current
            callback = self._on_event
            if rec is None or callback is None:
                return
            key = (rec.store.steps_total, category, layer)
            now = time.perf_counter()
            duration_ms = None
            if status == "started":
                module_starts[key] = now
            elif key in module_starts:
                duration_ms = round((now - module_starts.pop(key)) * 1000, 3)
                self._last_module_durations[key] = duration_ms
            self._emit(rec, callback, f"{category}.{'started' if status == 'started' else 'completed'}", {
                "step": rec.store.steps_total,
                "layer": layer,
                "module": category,
                "timestamp": time.time(),
                "duration_ms": duration_ms,
            })

        from app.instrumentation.hooks import HookManager

        module_starts: dict[tuple[int, str, int], float] = {}
        self._last_module_durations = {}
        self._hook_manager = HookManager(model)
        layers = getattr(getattr(model, "model", model), "layers", ())
        for i, layer in enumerate(layers):
            self._hook_manager.register_layer_hooks(
                layer, i, on_qkv, on_mlp, on_layer_out, on_residual=on_residual, on_module_event=on_module_event
            )
        self._hooks_registered = True
        self._store_ref = store_ref

    def _bounded_sequence_tensor(self, store: TensorStore, tensor: torch.Tensor, step: int) -> torch.Tensor:
        """Detach/copy activations while bounding long-context memory."""
        value = tensor.detach().squeeze(0).to(torch.float32).cpu()
        if self.capture_level != "full" and value.ndim >= 2 and value.shape[-2] > self.capture_limit:
            start = int(value.shape[-2] - self.capture_limit)
            store.sequence_starts[step] = start
            value = value[..., start:, :]
        return value

    def run(self, prompt: str, params: dict[str, Any], on_event, owner=None, session_id: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._cancel.clear()
            self._pause.set()
            self._owner = owner
            self._on_event = on_event
            # Hook lifetime is per generation. Re-registering here prevents
            # stale closures from retaining prior session tensors.
            self._register_hooks()
            self._current = self._new_session(prompt, params, session_id=session_id)
            try:
                self._run_session(self._current, prompt, params, on_event)
            except Exception as exc:
                self._current.status = "failed"
                self._current.errors.append(str(exc))
                try:
                    self._emit(self._current, on_event, "inference.failed", {"message": str(exc), "summary": self._current.summary()})
                except Exception:
                    pass
            finally:
                self._current.status = "cancelled" if self._cancel.is_set() else self._current.status
                self._cleanup_hooks()
                self._release_device_cache()
                self._owner = None
                self._on_event = None

    def _release_device_cache(self) -> None:
        """Release unused accelerator blocks after a completed generation.

        MPS and CUDA allocators cache freed blocks by design. BrainOS keeps the
        loaded model resident, but should not let temporary KV/activation
        allocations accumulate after each session. The calls are guarded so
        CPU fallback and older torch builds remain ordinary no-ops.
        """
        try:
            device = str(getattr(self.adapter, "device", "cpu"))
            if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Cache release is best-effort and must never turn a completed run
            # into an application failure.
            pass

    def _new_session(self, prompt: str, params: dict[str, Any], session_id: str | None = None) -> SessionRecord:
        sid = session_id or uuid.uuid4().hex[:12]
        metadata = self.adapter.metadata.to_dict() if self.adapter.is_loaded else {}
        metadata.update({"capture_level": self.capture_level, "attention_capture": self.attention_capture})
        rec = SessionRecord(sid, prompt, self.adapter.model_id, params, metadata, adapter=self.adapter)
        rec.store.session_id = sid
        self._store_ref["store"] = rec.store
        return rec

    def _emit(self, rec: SessionRecord, on_event, etype: str, data: dict[str, Any]) -> None:
        ev = Event(type=etype, data=data, session_id=rec.session_id)
        rec.add_event(ev)
        on_event(ev)

    def _run_session(self, rec: SessionRecord, prompt: str, params: dict[str, Any], on_event) -> None:
        t_start = time.time()
        self._emit(rec, on_event, "inference.started", {"prompt": prompt, "params": params, "model_id": rec.model_id})
        self._emit(rec, on_event, "dev.log", {"level": "info", "message": f"session {rec.session_id} started"})

        conversation = params.get("messages")
        if isinstance(conversation, list):
            conversation = [item for item in conversation if isinstance(item, dict) and item.get("role") in {"system", "user", "assistant"} and isinstance(item.get("content"), str)]
        try:
            tokens, input_ids, tok_time = self.adapter.tokenize(
                prompt, self.max_prompt_tokens, use_chat_template=bool(params.get("use_chat_template", True)), messages=conversation or None
            )
        except TypeError:
            # Compatibility for third-party adapters that still implement the
            # original three-argument tokenizer contract.
            tokens, input_ids, tok_time = self.adapter.tokenize(
                prompt, self.max_prompt_tokens, use_chat_template=bool(params.get("use_chat_template", True))
            )
        if hasattr(self.adapter, "chat_template_snapshot"):
            try:
                template_snapshot = self.adapter.chat_template_snapshot(prompt, use_chat_template=bool(params.get("use_chat_template", True)), messages=conversation or None)
            except TypeError:
                template_snapshot = self.adapter.chat_template_snapshot(prompt, use_chat_template=bool(params.get("use_chat_template", True)))
        else:
            template_snapshot = {
                "available": False,
                "name": "unavailable",
                "serialized": "",
                "message_count": 1,
                "has_system_message": False,
                "generation_prompt": False,
            }
        raw_count = self.adapter.user_text_token_count(prompt) if hasattr(self.adapter, "user_text_token_count") else None
        rec.metadata["token_usage"] = {"user_text_tokens": raw_count, "model_input_tokens": int(input_ids.shape[-1]),
            "template_history_overhead": int(input_ids.shape[-1]) - raw_count if raw_count is not None else None}
        rec.store.prompt_length = int(input_ids.shape[-1])
        rec.tokens = [t.to_dict() for t in tokens]
        self._emit(
            rec,
            on_event,
            "tokenization.complete",
            {
                "tokens": rec.tokens,
                "count": len(tokens),
                "time_ms": round(tok_time * 1000, 2),
                "input_shape": list(input_ids.shape),
                "context_length": self.adapter.metadata.context_length,
                "context_used": len(tokens),
                "context_remaining": max(0, self.adapter.metadata.context_length - len(tokens)),
                "chat_template": template_snapshot,
                "token_usage": rec.metadata["token_usage"],
                "source": "hf_tokenizer",
            },
        )
        rec.timings["tokenization_ms"] = tok_time * 1000

        embedding_started = time.time()
        embeddings = self.adapter.embed(input_ids)
        rec.metadata["embedding_tensor"] = {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype), "device": str(embeddings.device), "source": "forward_hook"}
        rec.store.embeddings = embeddings.squeeze(0).to(torch.float32).cpu()
        pca = pca_projection(rec.store.embeddings.numpy(), dims=3)
        rec.store.pca = pca
        token_payload = []
        for i, t in enumerate(rec.tokens):
            v = rec.store.embeddings[i]
            token_payload.append(
                {
                    "position": i,
                    "text": token_view(t["text"]),
                    "id": t["id"],
                    "norm": float(v.norm()),
                    "pca3": pca["coords"][i],
                }
            )
        self._emit(
            rec,
            on_event,
            "embeddings.complete",
            {
                "tokens": token_payload,
                "count": len(token_payload),
                "explained_variance": pca["explained_variance"],
                "pca_method": pca["method"],
                "embedding_dim": int(rec.store.embeddings.shape[1]),
                "tensor": rec.metadata["embedding_tensor"],
                "time_ms": (time.time() - embedding_started) * 1000,
            },
        )
        rec.timings["embedding_ms"] = (time.time() - embedding_started) * 1000

        requested_max_new = int(params.get("max_new_tokens", 800))
        # Output length is bounded by the model's remaining context capacity.
        # Keep the requested value visible to the caller/UI, but never ask the
        # local transformer to exceed prompt_tokens + new_tokens <= context.
        context_remaining = max(0, int(self.adapter.metadata.context_length) - len(tokens))
        max_new = min(requested_max_new, context_remaining)
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 0.9))
        top_k = int(params.get("top_k", 40))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        rec.metadata["active_sampling"] = {"temperature": temperature, "top_p": top_p, "top_k": top_k,
            "do_sample": temperature > 0 and top_k != 1, "repetition_penalty": repetition_penalty,
            "requested_max_output_tokens": requested_max_new, "max_output_tokens": max_new,
            "context_clamped": max_new != requested_max_new}
        from app.providers.telemetry import native_telemetry
        self._emit(rec, on_event, "telemetry.updated", native_telemetry(rec))
        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        past = None
        output_ids: list[int] = []
        eos_id = self.adapter._tokenizer.eos_token_id
        im_end_id = self.adapter._tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = {eos_id, im_end_id} - {-1, None}
        t_gen_start = time.time()
        first_token_at = None
        num_layers = self.adapter.metadata.num_layers

        step = 0
        pause_notified = False
        while step < max_new and not self._cancel.is_set():
            if not self._pause.is_set() and not pause_notified:
                self._emit(rec, on_event, "inference.paused", {"step": step, "reason": "between model steps"})
                pause_notified = True
            self._pause.wait()
            if pause_notified:
                self._emit(rec, on_event, "inference.resumed", {"step": step})
                pause_notified = False
            if self._cancel.is_set():
                break
            is_first = step == 0
            step_input = (
                input_ids
                if is_first
                else torch.tensor([[output_ids[-1]]], dtype=torch.long, device=self.adapter.device)
            )
            new_text = "" if is_first else self.adapter.decode([output_ids[-1]])
            active_position = rec.store.prompt_length + step - 1 if not is_first else rec.store.prompt_length - 1
            self._emit(
                rec,
                on_event,
                "step.started",
                {
                    "step": step,
                    "new_token": new_text,
                    "is_first": is_first,
                    "active_position": active_position,
                    "total_length": rec.store.prompt_length + step,
                },
            )

            fwd_start = time.time()
            # Hooks execute inside forward; publish/capture against the step
            # currently being computed rather than the previous step.
            rec.store.steps_total = step
            out = self.adapter.forward(
                step_input,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
            )
            fwd_time = time.time() - fwd_start
            past = out["past_key_values"]
            if is_first:
                rec.timings["prefill_ms"] = round(fwd_time * 1000, 2)

            if self.attention_capture and out["attentions"] is not None:
                for layer_i, attn in enumerate(out["attentions"]):
                    m = attn[0].to(torch.float32).cpu()
                    if m.shape[1] > self.capture_limit:
                        # Keep the final query row for a long prompt. The
                        # complete matrix is intentionally not retained.
                        rec.store.attention_starts[step] = int(m.shape[1] - 1)
                        m = m[:, -1:, :]
                    rec.store.attention[(layer_i, step)] = m

            if out["hidden_states"] is not None:
                for layer_i, hs in enumerate(out["hidden_states"]):
                    # Layer hooks capture the actual block output, including
                    # the pre-final-norm representation for the last layer.
                    # Some Transformers implementations expose a normalized
                    # final entry in ``hidden_states`` as well. Never let
                    # that summary overwrite the more precise hook capture.
                    if (layer_i, step) in rec.store.hidden:
                        continue
                    rec.store.hidden[(layer_i, step)] = self._bounded_sequence_tensor(rec.store, hs, step)

            layer_seq = rec.store.prompt_length + step
            for layer_i in range(num_layers):
                hs_out = rec.store.hidden.get((layer_i + 1, step))
                if hs_out is None:
                    continue
                new_row = hs_out[-1] if is_first else hs_out[0]
                self._emit(
                    rec,
                    on_event,
                    "layer.complete",
                    {
                        "layer": layer_i,
                        "step": step,
                        "is_first": is_first,
                        "input_shape": [1, layer_seq, self.adapter.metadata.hidden_size],
                        "output_shape": [1, layer_seq, self.adapter.metadata.hidden_size],
                        "hidden_norm": float(new_row.norm()),
                        "hidden_mean": float(new_row.mean()),
                        "hidden_std": float(new_row.std()),
                        "duration_ms": getattr(self, "_last_module_durations", {}).get((step, "layer", layer_i)),
                    },
                )

            # Attention links
            attn_links = []
            for layer_i in range(num_layers):
                m = rec.store.attention.get((layer_i, step))
                if m is None:
                    continue
                heads = m.shape[0]
                all_links = []
                for h in range(heads):
                    w = m[h, -1]
                    if w.sum() > 0:
                        for t in attention_topk(w, k=4)[:2]:
                            all_links.append({"head": h, "token_index": t["token_index"], "weight": t["weight"]})
                all_links.sort(key=lambda x: x["weight"], reverse=True)
                attn_links.append({"layer": layer_i, "links": all_links[:8], "active_position": active_position})
            self._emit(
                rec,
                on_event,
                "attention.captured",
                {"step": step, "available": self.attention_capture, "layers": attn_links if self.attention_capture else []},
            )

            # QKV Stats
            qkv_stats = []
            for layer_i in range(num_layers):
                for name in ("q", "k", "v"):
                    t = rec.store.qkv.get((layer_i, name, step))
                    if t is None:
                        continue
                    row = t[-1] if is_first else t[0]
                    qkv_stats.append({"layer": layer_i, "name": name, "stats": vector_stats(row)})
            self._emit(rec, on_event, "qkv.captured", {"step": step, "layers": qkv_stats})

            # MLP Top Activations (combined intermediate or gate)
            mlp_top = []
            for layer_i in range(num_layers):
                t = rec.store.mlp.get((layer_i, "intermediate", step))
                if t is None:
                    t = rec.store.mlp.get((layer_i, "gate_activation", step))
                if t is None:
                    continue
                row = t[-1] if is_first else t[0]
                mlp_top.append(
                    {
                        "layer": layer_i,
                        "top": topk_activations(row, k=8),
                        "stats": vector_stats(row),
                    }
                )
            self._emit(rec, on_event, "mlp.captured", {"step": step, "layers": mlp_top})

            # Residual Stream
            residual_data = []
            for layer_i in range(num_layers):
                x0 = rec.store.raw_residual.get((layer_i, "layer_in", step))
                attn_out = rec.store.raw_residual.get((layer_i, "attn_out", step))
                mlp_out = rec.store.raw_residual.get((layer_i, "mlp_out", step))
                x2 = rec.store.raw_residual.get((layer_i, "layer_out", step))
                if x0 is not None and attn_out is not None and mlp_out is not None and x2 is not None:
                    row_idx = -1 if is_first else 0
                    rm = residual_metrics(x0[row_idx], attn_out[row_idx], mlp_out[row_idx], x2[row_idx])
                    rec.store.residual[(layer_i, step)] = rm
                    residual_data.append({"layer": layer_i, **rm})
            if residual_data:
                self._emit(rec, on_event, "residual.captured", {"step": step, "layers": residual_data})

            # KV Cache Introspection
            if past is not None:
                kv_total_bytes = 0
                kv_layer_info = []
                try:
                    num_cached_tokens = int(past.get_seq_length(0)) if hasattr(past, "get_seq_length") else int(past[0][0].shape[-2])
                    for l_idx in range(len(past)):
                        k_t, v_t = past[l_idx]
                        k_shape = list(k_t.shape)
                        v_shape = list(v_t.shape)
                        l_bytes = (k_t.numel() * k_t.element_size()) + (v_t.numel() * v_t.element_size())
                        kv_total_bytes += l_bytes
                        kv_layer_info.append({
                            "layer": l_idx,
                            "key_shape": k_shape,
                            "value_shape": v_shape,
                            "key_dtype": str(k_t.dtype), "value_dtype": str(v_t.dtype),
                            "key_device": str(k_t.device), "value_device": str(v_t.device),
                            "bytes": l_bytes,
                            "kb": round(l_bytes / 1024, 2),
                        })
                    kv_payload = {
                        "step": step,
                        "seq_length": num_cached_tokens,
                        "total_bytes": kv_total_bytes,
                        "total_kb": round(kv_total_bytes / 1024, 2),
                        "total_mb": round(kv_total_bytes / (1024 * 1024), 4),
                        "num_layers": len(past),
                        "layers": kv_layer_info,
                    }
                    rec.store.kv_cache[step] = kv_payload
                    self._emit(rec, on_event, "kv_cache.captured", kv_payload)
                except Exception:
                    pass

            # Logit Lens for key layers
            logit_lens_data = []
            sample_layers = [0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]
            sample_layers = sorted(list(set(sample_layers)))
            for layer_i in sample_layers:
                hs_out = rec.store.hidden.get((layer_i + 1, step))
                if hs_out is not None:
                    hs_row = hs_out[-1:] if is_first else hs_out[:1]
                    try:
                        proj_logits = self.adapter.project_hidden_state_to_logits(hs_row.unsqueeze(0))[0, -1]
                        cands = self.adapter.topk_candidates(proj_logits, k=5, temperature=temperature)
                        rec.store.logit_lens[(layer_i, step)] = cands
                        logit_lens_data.append({"layer": layer_i, "candidates": cands})
                    except Exception:
                        pass
            if logit_lens_data:
                self._emit(rec, on_event, "logit_lens.captured", {"step": step, "layers": logit_lens_data})

            logits = out["logits"][0, -1].to(torch.float32).cpu()
            if repetition_penalty != 1:
                seen_ids = set(input_ids[0].tolist() + output_ids)
                for seen_id in seen_ids:
                    logits[seen_id] = logits[seen_id] * repetition_penalty if logits[seen_id] < 0 else logits[seen_id] / repetition_penalty
            rec.store.logits[step] = logits
            rec.store.logit_stats[step] = vector_stats(logits)
            rec.store.logit_vocab_sizes[step] = int(logits.numel())
            if hasattr(self.adapter, "sampling_entropy"):
                entropy = float(self.adapter.sampling_entropy(logits, temperature, top_p, top_k))
            else:
                probs = torch.softmax(logits / temperature if temperature else logits, dim=-1)
                entropy = float(-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum())
            sampling_candidates = getattr(self.adapter, "sampling_candidates", self.adapter.topk_candidates)
            candidates = sampling_candidates(
                logits,
                k=12,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            rec.store.logit_candidates[step] = sampling_candidates(
                logits,
                k=50,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            self._emit(
                rec,
                on_event,
                "logits.ready",
                {
                    "step": step,
                    "candidates": candidates,
                    "vocab_size": self.adapter.metadata.vocab_size,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "entropy": entropy,
                    "sampling_method": "greedy" if temperature == 0.0 or top_k == 1 else "temperature+top-k/top-p",
                },
            )

            token_id, prob = self.adapter.sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            text = self.adapter.decode([token_id])
            rank_for_distribution = getattr(self.adapter, "sampling_rank", None)
            rank = (
                rank_for_distribution(logits, token_id, temperature, top_p, top_k)
                if rank_for_distribution is not None
                else next((c["rank"] for c in candidates if c["token_id"] == token_id), None)
            )
            gen_time_ms = (time.time() - t_gen_start) * 1000
            if step == 0:
                rec.timings["ttft_ms"] = round((time.time() - t_start) * 1000, 2)
            self._emit(
                rec,
                on_event,
                "token.selected",
                {
                    "step": step,
                    "token_id": token_id,
                    "text": token_view(text),
                    "probability": prob,
                    "rank": rank,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "entropy": entropy,
                    "sampling_method": "greedy" if temperature == 0.0 or top_k == 1 else "temperature+top-k/top-p",
                },
            )
            emb = self.adapter.embed_token(token_id)
            emb_vec = emb.to(torch.float32).cpu()
            rec.store.generated_embeddings.append(emb_vec)
            pc = pca_project_vector(emb_vec.numpy(), rec.store.pca)
            output_ids.append(token_id)
            rec.output_tokens.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "text": token_view(text),
                    "probability": prob,
                    "rank": rank,
                    "position": rec.store.prompt_length + step,
                    "is_special": token_id in set(getattr(self.adapter._tokenizer, "all_special_ids", [])),
                    "entropy": entropy,
                }
            )
            decode_output = getattr(self.adapter, "decode_output", self.adapter.decode)
            rec.response = decode_output(output_ids)
            rec.step_stats.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "text": token_view(text),
                    "probability": prob,
                    "time_ms": round(gen_time_ms, 2),
                    "forward_ms": round(fwd_time * 1000, 2),
                    "entropy": entropy,
                }
            )
            rec.timings["current_token_latency_ms"] = round(gen_time_ms, 2)
            rec.timings["average_token_latency_ms"] = round(
                sum(float(item["time_ms"]) for item in rec.step_stats) / max(len(rec.step_stats), 1),
                2,
            )
            self._emit(
                rec,
                on_event,
                "token.generated",
                {
                    "step": step,
                    "token_id": token_id,
                    "text": token_view(text),
                    "probability": prob,
                    "rank": rank,
                    "position": rec.store.prompt_length + step,
                    "output": rec.response,
                    "time_ms": round(gen_time_ms, 2),
                    "entropy": entropy,
                    "sampling_method": "greedy" if temperature == 0.0 or top_k == 1 else "temperature+top-k/top-p",
                    "embedding": {"norm": float(emb_vec.norm()), "pca3": pc},
                },
            )
            if first_token_at is None:
                first_token_at = time.time()
                rec.timings["ttft_ms"] = (first_token_at - t_start) * 1000
            t_gen_start = time.time()
            rec.store.steps_total = step + 1
            if token_id in eos_ids:
                rec.timings["eos_reached"] = True
                break
            step += 1

        elapsed = time.time() - t_start
        rec.timings["total_ms"] = round(elapsed * 1000, 2)
        rec.timings["decode_ms"] = (time.time() - first_token_at) * 1000 if first_token_at is not None else None
        self._emit(rec, on_event, "telemetry.updated", native_telemetry(rec))
        rec.timings["tokens_per_second"] = (
            round(len(rec.output_tokens) / max(elapsed, 1e-6), 2) if rec.output_tokens else 0.0
        )
        if self._cancel.is_set():
            rec.status = "cancelled"
            self._emit(
                rec,
                on_event,
                "inference.cancelled",
                {
                    "summary": rec.summary(),
                    "response": rec.response,
                    "num_output_tokens": len(rec.output_tokens),
                    "timings": rec.timings,
                },
            )
            return
        rec.status = "complete"
        self._emit(
            rec,
            on_event,
            "inference.complete",
            {
                "summary": rec.summary(),
                "response": rec.response,
                "num_output_tokens": len(rec.output_tokens),
                "timings": rec.timings,
            },
        )

    def _project_new(self, store: TensorStore, vector: np.ndarray) -> list[float]:
        return pca_project_vector(vector, store.pca)
