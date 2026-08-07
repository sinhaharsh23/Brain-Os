from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.events.types import Event
from app.instrumentation.stats import sample_activations, tensor_info, topk_activations, vector_stats
from app.models.registry import SUPPORTED_MODELS, hardware_report, recommend_model
from app.providers.registry import PROVIDERS
from app.state import state

log = logging.getLogger("brainos.api")

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_status": state.model_status,
        "model": state.adapter.model_id if state.adapter else None,
        "clients": state.manager.count,
    }


@router.get("/hardware")
async def hardware() -> dict[str, Any]:
    return hardware_report()


@router.get("/models")
async def models() -> dict[str, Any]:
    return {"supported": SUPPORTED_MODELS, "recommended": recommend_model()}


@router.get("/providers")
async def providers() -> list[dict[str, Any]]:
    return [provider.to_dict() for provider in PROVIDERS]


@router.get("/model")
async def current_model() -> dict[str, Any]:
    if state.adapter is None or not state.adapter.is_loaded:
        raise HTTPException(404, "no model loaded")
    return state.adapter.metadata.to_dict()


@router.post("/model/load")
async def load_model(payload: dict[str, Any]) -> dict[str, Any]:
    if state.model_status == "loading":
        raise HTTPException(409, "model already loading")
    model_id = payload.get("model_id", settings.default_model)
    state.model_status = "loading"

    async def do_load() -> None:
        try:
            from app.models.qwen import QwenAdapter

            state.adapter = QwenAdapter(model_id=model_id, device=settings.device, dtype=settings.dtype)
            await asyncio.to_thread(state.adapter.load)
            from app.inference.engine import InferenceEngine

            state.engine = InferenceEngine(state.adapter, state.bus, max_prompt_tokens=settings.max_prompt_tokens)
            state.engine._register_hooks()
            state.model_status = "loaded"
            await state.bus.publish(Event("model.ready", {"metadata": state.adapter.metadata.to_dict()}))
        except Exception as e:
            state.model_status = "error"
            log.exception("model load failed")
            await state.bus.publish(Event("system.error", {"stage": "model_load", "message": str(e)}))

    asyncio.create_task(do_load())
    return {"status": "loading", "model_id": model_id}


@router.get("/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return state.replay.list_sessions()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = state.replay.get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: str) -> list[dict[str, Any]]:
    session = state.replay.get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    events = state.replay.load_events(session_id)
    if not events:
        raise HTTPException(404, "no event data stored for this session (tensor data may still exist)")
    return events


def _require_session(session_id: str):
    if state.engine is None or state.engine.current_session is None or state.engine.current_session.session_id != session_id:
        raise HTTPException(404, "session not available (it may have been replaced by a newer inference)")
    return state.engine.current_session


@router.get("/sessions/{session_id}/embedding/{position}")
async def session_embedding(session_id: str, position: int) -> dict[str, Any]:
    rec = _require_session(session_id)
    store = rec.store
    if position < store.prompt_length:
        if store.embeddings is None:
            raise HTTPException(404, "embeddings not captured")
        vec = store.embeddings[position]
        token = rec.tokens[position]
        pca_coord = store.pca["coords"][position] if store.pca.get("coords") else None
    else:
        gi = position - store.prompt_length
        if gi >= len(store.generated_embeddings):
            raise HTTPException(404, "position out of range")
        vec = store.generated_embeddings[gi]
        token = rec.output_tokens[gi]
        pca_coord = pca_project_vector_for(store, vec)
    return {
        "token": token,
        "vector": [float(x) for x in vec.tolist()],
        "stats": vector_stats(vec),
        "pca3": pca_coord,
        "dimension": int(vec.numel()),
    }


def pca_project_vector_for(store, vec) -> list[float] | None:
    try:
        import numpy as np

        from app.instrumentation.stats import pca_project_vector

        return pca_project_vector(vec.numpy(), store.pca)
    except Exception:
        return None


@router.get("/sessions/{session_id}/attention")
async def session_attention(
    session_id: str,
    layer: int = Query(0, ge=0),
    head: int = Query(0, ge=0),
    position: int = Query(0, ge=0),
) -> dict[str, Any]:
    rec = _require_session(session_id)
    store = rec.store
    m = store.attention_for_position(layer, position)
    if m is None:
        raise HTTPException(404, f"attention not captured for layer {layer}")
    if head >= m.shape[0]:
        raise HTTPException(404, f"head {head} out of range ({m.shape[0]} heads)")
    row = m[head, 0]
    total = store.prompt_length + store.steps_total
    rows = []
    for i, v in enumerate(row.tolist()):
        rows.append({"token_index": i, "weight": v})
    return {
        "layer": layer,
        "head": head,
        "position": position,
        "total_tokens": total,
        "weights": rows,
        "stats": vector_stats(row),
        "is_full_matrix": bool(m.shape[1] == total),
    }


@router.get("/sessions/{session_id}/qkv")
async def session_qkv(
    session_id: str,
    layer: int = Query(0, ge=0),
    name: str = Query("q", pattern="^(q|k|v|o)$"),
    position: int = Query(0, ge=0),
    limit: int = Query(512, ge=1, le=2048),
) -> dict[str, Any]:
    rec = _require_session(session_id)
    store = rec.store
    t = store.qkv_for_position(layer, name, position)
    if t is None:
        raise HTTPException(404, f"{name.upper()} not captured for layer {layer}")
    return {
        "layer": layer,
        "name": name,
        "position": position,
        "stats": vector_stats(t[0]),
        "values": sample_activations(t[0], limit=limit),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
    }


@router.get("/sessions/{session_id}/mlp")
async def session_mlp(
    session_id: str,
    layer: int = Query(0, ge=0),
    position: int = Query(0, ge=0),
    topk: int = Query(16, ge=1, le=64),
    mode: str = Query("gate_activation", pattern="^(gate_activation|up|down_output)$"),
) -> dict[str, Any]:
    rec = _require_session(session_id)
    store = rec.store
    t = store.mlp_for_position(layer, mode, position)
    if t is None:
        raise HTTPException(404, f"mlp {mode} not captured for layer {layer}")
    return {
        "layer": layer,
        "position": position,
        "mode": mode,
        "stats": vector_stats(t[0]),
        "top": topk_activations(t[0], k=topk),
        "shape": list(t.shape),
    }


@router.get("/sessions/{session_id}/hidden")
async def session_hidden(
    session_id: str,
    layer: int = Query(0, ge=0),
    position: int = Query(0, ge=0),
    limit: int = Query(512, ge=1, le=2048),
) -> dict[str, Any]:
    rec = _require_session(session_id)
    store = rec.store
    t = store.hidden_for_position(layer, position)
    if t is None:
        raise HTTPException(404, f"hidden state not captured for layer {layer}")
    return {
        "layer": layer,
        "position": position,
        "stats": vector_stats(t[0]),
        "values": sample_activations(t[0], limit=limit),
        "shape": list(t.shape),
    }


@router.get("/sessions/{session_id}/logits")
async def session_logits(
    session_id: str,
    step: int = Query(0, ge=0),
    k: int = Query(16, ge=1, le=50),
) -> dict[str, Any]:
    rec = _require_session(session_id)
    store = rec.store
    logits = store.logits.get(step)
    if logits is None:
        raise HTTPException(404, f"logits not captured for step {step}")
    candidates = state.adapter.topk_candidates(logits, k=k)
    return {
        "step": step,
        "candidates": candidates,
        "stats": vector_stats(logits),
        "vocab_size": int(logits.numel()),
    }


@router.post("/tokenize")
async def tokenize_preview(payload: dict[str, Any]) -> dict[str, Any]:
    if state.adapter is None or not state.adapter.is_loaded:
        raise HTTPException(503, "model not loaded")
    text = str(payload.get("text", ""))
    tokens, input_ids, dt = await asyncio.to_thread(
        state.adapter.tokenize, text, settings.max_prompt_tokens, bool(payload.get("use_chat_template", True))
    )
    return {
        "tokens": [t.to_dict() for t in tokens],
        "count": len(tokens),
        "time_ms": round(dt * 1000, 2),
        "input_shape": list(input_ids.shape),
    }


@router.get("/monitoring/history")
async def monitoring_history() -> dict[str, Any]:
    return {"history": state.monitor.history()}
