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
    pca_project_vector,
    pca_projection,
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
    qkv: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    mlp: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    attention: dict[int, torch.Tensor] = field(default_factory=dict)
    logits: dict[int, torch.Tensor] = field(default_factory=dict)
    steps_total: int = 0
    full_pass_steps: set[int] = field(default_factory=set)

    def position_of(self, position: int) -> tuple[int, int]:
        if position < self.prompt_length:
            return 0, position
        return position - self.prompt_length + 1, 0

    def attention_for_position(self, layer: int, position: int) -> torch.Tensor:
        step, row = self.position_of(position)
        m = self.attention.get((layer, step))
        if m is None:
            return None
        if step == 0:
            return m[:, row : row + 1, :]
        return m

    def qkv_for_position(self, layer: int, name: str, position: int) -> torch.Tensor | None:
        step, row = self.position_of(position)
        m = self.qkv.get((layer, name, step))
        if m is None:
            return None
        return m[row : row + 1]

    def mlp_for_position(self, layer: int, name: str, position: int) -> torch.Tensor | None:
        step, row = self.position_of(position)
        m = self.mlp.get((layer, name, step))
        if m is None:
            return None
        return m[row : row + 1]

    def hidden_for_position(self, layer: int, position: int) -> torch.Tensor | None:
        step, row = self.position_of(position)
        m = self.hidden.get((layer, step))
        if m is None:
            return None
        return m[row : row + 1]


class SessionRecord:
    def __init__(self, session_id: str, prompt: str, model_id: str, params: dict[str, Any], metadata: dict[str, Any]) -> None:
        self.session_id = session_id
        self.prompt = prompt
        self.model_id = model_id
        self.params = params
        self.metadata = metadata
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
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "model_id": self.model_id,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "num_tokens": len(self.tokens),
            "num_output_tokens": len(self.output_tokens),
            "response": self.response,
            "timings": self.timings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class InferenceEngine:
    def __init__(self, adapter, bus, max_prompt_tokens: int = 1024) -> None:
        self.adapter = adapter
        self.bus = bus
        self.max_prompt_tokens = max_prompt_tokens
        self._lock = threading.Lock()
        self._current: SessionRecord | None = None
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._hooks_registered = False

    @property
    def current_session(self) -> SessionRecord | None:
        return self._current

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.set()

    def pause(self) -> None:
        self._pause.clear()

    def resume(self) -> None:
        self._pause.set()

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
            store.qkv[(layer, name, step)] = tensor.squeeze(0).to(torch.float32).cpu()

        def on_mlp(layer: int, name: str, tensor: torch.Tensor) -> None:
            store = store_ref["store"]
            if store is None:
                return
            step = store.steps_total
            store.mlp[(layer, name, step)] = tensor.squeeze(0).to(torch.float32).cpu()

        def on_layer_out(layer: int, tensor: torch.Tensor) -> None:
            store = store_ref["store"]
            if store is None:
                return
            step = store.steps_total
            store.hidden[(layer + 1, step)] = tensor.squeeze(0).to(torch.float32).cpu()

        from app.instrumentation.hooks import HookManager

        self._hook_manager = HookManager(model)
        for i, layer in enumerate(model.model.layers):
            self._hook_manager.register_layer_hooks(layer, i, on_qkv, on_mlp, on_layer_out)
        self._hooks_registered = True
        self._store_ref = store_ref

    def run(self, prompt: str, params: dict[str, Any], on_event) -> None:
        with self._lock:
            self._cancel.clear()
            self._pause.set()
            self._current = self._new_session(prompt, params)
            try:
                self._run_session(self._current, prompt, params, on_event)
            finally:
                self._current.status = "complete" if not self._cancel.is_set() else "cancelled"
                self._current.store.steps_total = self._current.store.steps_total

    def _new_session(self, prompt: str, params: dict[str, Any]) -> SessionRecord:
        sid = uuid.uuid4().hex[:12]
        metadata = self.adapter.metadata.to_dict() if self.adapter.is_loaded else {}
        rec = SessionRecord(sid, prompt, self.adapter.model_id, params, metadata)
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

        tokens, input_ids, tok_time = self.adapter.tokenize(
            prompt, self.max_prompt_tokens, use_chat_template=bool(params.get("use_chat_template", True))
        )
        rec.store.prompt_length = len(tokens)
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
            },
        )
        rec.timings["tokenization_ms"] = tok_time * 1000

        embeddings = self.adapter.embed(input_ids)
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
                "time_ms": 0.0,
            },
        )
        rec.timings["embedding_ms"] = (time.time() - t_start) * 1000

        max_new = int(params.get("max_new_tokens", 128))
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 0.9))
        top_k = int(params.get("top_k", 40))

        past = None
        output_ids: list[int] = []
        eos_id = self.adapter._tokenizer.eos_token_id
        im_end_id = self.adapter._tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = {eos_id, im_end_id} - {-1, None}
        t_gen_start = time.time()

        step = 0
        while step < max_new and not self._cancel.is_set():
            self._pause.wait()
            if self._cancel.is_set():
                break
            rec.store.steps_total = step
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

            layer_times: list[dict[str, Any]] = []
            t0 = time.time()

            fwd_start = time.time()
            out = self.adapter.forward(
                step_input,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
            )
            fwd_time = time.time() - fwd_start
            past = out["past_key_values"]

            rec.store.steps_total = step
            if out["attentions"] is not None:
                for layer_i, attn in enumerate(out["attentions"]):
                    m = attn[0].to(torch.float32).cpu()
                    rec.store.attention[(layer_i, step)] = m

            if out["hidden_states"] is not None:
                for layer_i, hs in enumerate(out["hidden_states"]):
                    v = hs[0, -1].to(torch.float32).cpu()
                    rec.store.hidden[(layer_i, step)] = hs.squeeze(0).to(torch.float32).cpu()

            layer_seq = rec.store.prompt_length + step
            for layer_i in range(self.adapter.metadata.num_layers):
                hs_out = rec.store.hidden[(layer_i + 1, step)]
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
                    },
                )

            attn_links = []
            for layer_i in range(self.adapter.metadata.num_layers):
                m = rec.store.attention[(layer_i, step)]
                heads = m.shape[0]
                all_links = []
                for h in range(heads):
                    w = m[h, -1]
                    if w.sum() > 0:
                        for t in attention_topk(w, k=4)[:2]:
                            all_links.append({"head": h, "token_index": t["token_index"], "weight": t["weight"]})
                all_links.sort(key=lambda x: x["weight"], reverse=True)
                attn_links.append({"layer": layer_i, "links": all_links[:8], "active_position": active_position})
            self._emit(rec, on_event, "attention.captured", {"step": step, "layers": attn_links})

            qkv_stats = []
            for layer_i in range(self.adapter.metadata.num_layers):
                for name in ("q", "k", "v"):
                    t = rec.store.qkv[(layer_i, name, step)]
                    row = t[-1] if is_first else t[0]
                    qkv_stats.append({"layer": layer_i, "name": name, "stats": vector_stats(row)})
            self._emit(rec, on_event, "qkv.captured", {"step": step, "layers": qkv_stats})

            mlp_top = []
            for layer_i in range(self.adapter.metadata.num_layers):
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

            logits = out["logits"][0, -1].to(torch.float32).cpu()
            rec.store.logits[step] = logits
            candidates = self.adapter.topk_candidates(logits, k=12, temperature=temperature)
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
                },
            )

            token_id, prob = self.adapter.sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            text = self.adapter.decode([token_id])
            rank = next((c["rank"] for c in candidates if c["token_id"] == token_id), None)
            gen_time_ms = (time.time() - t_gen_start) * 1000
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
                    "position": rec.store.prompt_length + step - 1,
                }
            )
            rec.response = self.adapter.decode(output_ids)
            rec.step_stats.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "text": token_view(text),
                    "probability": prob,
                    "time_ms": round(gen_time_ms, 2),
                }
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
                    "output": rec.response,
                    "time_ms": round(gen_time_ms, 2),
                    "embedding": {"norm": float(emb_vec.norm()), "pca3": pc},
                },
            )
            t_gen_start = time.time()
            if token_id in eos_ids:
                rec.timings["eos_reached"] = True
                break
            step += 1
        elapsed = time.time() - t_start
        rec.timings["total_ms"] = round(elapsed * 1000, 2)
        rec.timings["tokens_per_second"] = round(len(rec.output_tokens) / max(elapsed, 1e-6), 2) if rec.output_tokens else 0.0
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
