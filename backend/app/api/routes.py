from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Request, status

from app.config import settings
from app.events.types import Event
from app.instrumentation.stats import sample_activations, tensor_info, topk_activations, vector_stats
from app.models.registry import SUPPORTED_MODELS, hardware_report, recommend_model
from app.providers.registry import PROVIDERS
from app.security import validate_session_id
from app.state import state
from app.inference.scheduler import InferenceScheduler

log = logging.getLogger("brainos.api")

router = APIRouter(prefix="/api")


def _owner_id(request: Request | None) -> str | None:
    if settings.auth_mode != "multi_user":
        return None
    user = getattr(request.state, "user", None)
    return str(user["id"]) if user else None


def _require_owner(request: Request) -> str:
    owner_id = _owner_id(request)
    if owner_id is None or state.persistence is None:
        raise HTTPException(400, "multi-user authentication is required")
    return owner_id


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(payload: dict[str, Any]) -> dict[str, Any]:
    if settings.auth_mode != "multi_user" or state.persistence is None:
        raise HTTPException(400, "multi-user authentication is disabled")
    try:
        user, token = state.persistence.register(
            str(payload.get("username", "")),
            str(payload.get("password", "")),
            ttl_hours=settings.auth_session_ttl_hours,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"user": user, "token": token, "expires_in": settings.auth_session_ttl_hours * 3600}


@router.post("/auth/login")
async def login(payload: dict[str, Any]) -> dict[str, Any]:
    if settings.auth_mode != "multi_user" or state.persistence is None:
        raise HTTPException(400, "multi-user authentication is disabled")
    result = state.persistence.authenticate(
        str(payload.get("username", "")),
        str(payload.get("password", "")),
        ttl_hours=settings.auth_session_ttl_hours,
    )
    if result is None:
        raise HTTPException(401, "invalid username or password")
    user, token = result
    return {"user": user, "token": token, "expires_in": settings.auth_session_ttl_hours * 3600}


@router.get("/auth/me")
async def auth_me(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None and state.persistence is not None:
        from app.main import _request_token
        user = state.persistence.user_for_token(_request_token(request))
    return {"mode": settings.auth_mode, "user": user}


@router.post("/auth/logout")
async def logout(request: Request) -> dict[str, str]:
    if state.persistence is not None:
        from app.main import _request_token
        state.persistence.revoke_token(_request_token(request))
    return {"status": "logged_out"}


@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_status": state.model_status,
        "model": state.adapter.model_id if state.adapter else None,
        "clients": state.manager.count,
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def ready() -> dict[str, Any]:
    model_ready = state.model_status == "loaded" and state.adapter is not None and state.adapter.is_loaded and state.engine is not None
    if not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "model_status": state.model_status},
        )
    return {"status": "ready", "model_status": state.model_status, "model": state.adapter.model_id}


@router.get("/hardware")
async def hardware() -> dict[str, Any]:
    return hardware_report(settings.device, state.adapter.device if state.adapter else None)


@router.get("/models")
async def models() -> dict[str, Any]:
    return {"supported": SUPPORTED_MODELS, "recommended": recommend_model()}


@router.get("/providers")
async def providers() -> list[dict[str, Any]]:
    return [provider.to_dict() for provider in PROVIDERS]


@router.get("/scheduler")
async def scheduler_status() -> dict[str, Any]:
    return state.scheduler.snapshot() if state.scheduler is not None else {"queue_size": 0, "queue_limit": 0, "active": 0, "active_limit": 0, "runs": 0}


@router.post("/workspaces", status_code=status.HTTP_201_CREATED)
async def create_workspace(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    owner_id = _require_owner(request)
    try:
        return state.persistence.create_workspace(owner_id, str(payload.get("name", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/workspaces")
async def list_workspaces(request: Request) -> list[dict[str, Any]]:
    return state.persistence.list_workspaces(_require_owner(request))


@router.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str, request: Request) -> dict[str, Any]:
    workspace = state.persistence.get_workspace(_require_owner(request), workspace_id)
    if workspace is None:
        raise HTTPException(404, "workspace not found")
    return workspace


@router.patch("/workspaces/{workspace_id}")
async def update_workspace(workspace_id: str, payload: dict[str, Any], request: Request) -> dict[str, Any]:
    try:
        workspace = state.persistence.update_workspace(_require_owner(request), workspace_id, str(payload.get("name", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if workspace is None:
        raise HTTPException(404, "workspace not found")
    return workspace


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str, request: Request) -> dict[str, bool]:
    if not state.persistence.delete_workspace(_require_owner(request), workspace_id):
        raise HTTPException(404, "workspace not found")
    return {"deleted": True}


@router.post("/projects", status_code=status.HTTP_201_CREATED)
async def create_project(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    owner_id = _require_owner(request)
    try:
        return state.persistence.create_project(owner_id, str(payload.get("workspace_id", "")), str(payload.get("name", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/projects")
async def list_projects(request: Request, workspace_id: str | None = Query(default=None)) -> list[dict[str, Any]]:
    return state.persistence.list_projects(_require_owner(request), workspace_id)


@router.get("/projects/{project_id}")
async def get_project(project_id: str, request: Request) -> dict[str, Any]:
    project = state.persistence.get_project(_require_owner(request), project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return project


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, payload: dict[str, Any], request: Request) -> dict[str, Any]:
    try:
        project = state.persistence.update_project(
            _require_owner(request), project_id,
            name=str(payload["name"]) if "name" in payload else None,
            workspace_id=str(payload["workspace_id"]) if "workspace_id" in payload else None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if project is None:
        raise HTTPException(404, "project not found")
    return project


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, request: Request) -> dict[str, bool]:
    if not state.persistence.delete_project(_require_owner(request), project_id):
        raise HTTPException(404, "project not found")
    return {"deleted": True}


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
    from app.models.registry import adapter_for_model

    try:
        adapter_class = adapter_for_model(model_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state.model_status = "loading"

    async def do_load() -> None:
        adapter = None
        try:
            from app.main import unload_current_model
            from app.inference.engine import InferenceEngine

            await unload_current_model()
            adapter = adapter_class(model_id=model_id, device=settings.device, dtype=settings.dtype)
            await asyncio.to_thread(adapter.load)
            engine = InferenceEngine(adapter, state.bus, max_prompt_tokens=settings.max_prompt_tokens)
            engine._register_hooks()
            state.adapter = adapter
            state.engine = engine
            scheduler = InferenceScheduler(
                engine,
                state.bus,
                max_queue_size=settings.scheduler_max_queue_size,
                max_active=settings.scheduler_max_active,
                timeout_s=settings.scheduler_run_timeout_s,
            )
            await scheduler.start()
            state.scheduler = scheduler
            state.model_status = "loaded"
            await state.bus.publish(Event("model.ready", {"metadata": adapter.metadata.to_dict()}))
        except Exception as e:
            if adapter is not None:
                await asyncio.to_thread(adapter.unload)
            state.adapter = None
            state.engine = None
            state.model_status = "error"
            log.exception("model load failed")
            await state.bus.publish(Event("system.error", {"stage": "model_load", "message": str(e)}))

    asyncio.create_task(do_load())
    return {"status": "loading", "model_id": model_id}


@router.get("/sessions")
async def list_sessions(request: Request) -> list[dict[str, Any]]:
    owner_id = _owner_id(request)
    if owner_id is not None and state.persistence is not None:
        return state.persistence.list_owned_sessions(owner_id)
    return state.replay.list_sessions()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request = None) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    session = state.replay.get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    owner_id = _owner_id(request)
    if owner_id is not None and state.persistence is not None:
        if not state.persistence.owns_session(session_id, owner_id):
            if not state.persistence.claim_legacy_session(session_id, owner_id, str(session.get("summary", {}).get("prompt", ""))):
                raise HTTPException(404, "session not found")
    session["summary"]["capture_available"] = state.replay.has_capture(session_id)
    return session


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: str, request: Request = None) -> list[dict[str, Any]]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    session = state.replay.get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    owner_id = _owner_id(request)
    if owner_id is not None and state.persistence is not None and not state.persistence.owns_session(session_id, owner_id):
        if not state.persistence.claim_legacy_session(session_id, owner_id, str(session.get("summary", {}).get("prompt", ""))):
            raise HTTPException(404, "session not found")
    events = state.replay.load_events(session_id)
    if not events:
        raise HTTPException(404, "no event data stored for this session (tensor data may still exist)")
    return events


def _require_session(session_id: str, request: Request = None):
    owner_id = _owner_id(request)
    if owner_id is not None and state.persistence is not None and not state.persistence.owns_session(session_id, owner_id):
        payload = state.replay.get_session(session_id)
        if payload is None or not state.persistence.claim_legacy_session(session_id, owner_id, str(payload.get("summary", {}).get("prompt", ""))):
            raise HTTPException(404, "session not found")
    if state.engine is not None:
        if state.engine.current_session is not None and state.engine.current_session.session_id == session_id:
            return state.engine.current_session
        archived = state.engine._archived.get(session_id)
        if archived is not None:
            return archived
        payload = state.replay.get_session(session_id)
        capture = state.replay.load_capture(session_id)
        if payload is not None and capture is not None:
            return state.engine.restore_session(payload["summary"], capture)
    raise HTTPException(404, "session capture not available")


@router.get("/sessions/{session_id}/embedding/{position}")
async def session_embedding(session_id: str, request: Request = None, position: int = Path(..., ge=0)) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
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
    request: Request = None,
    layer: int = Query(0, ge=0),
    head: int = Query(0, ge=0),
    position: int = Query(0, ge=0),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
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
    request: Request = None,
    layer: int = Query(0, ge=0),
    name: str = Query("q", pattern="^(q|k|v|o)$"),
    position: int = Query(0, ge=0),
    limit: int = Query(512, ge=1, le=2048),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
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
    request: Request = None,
    layer: int = Query(0, ge=0),
    position: int = Query(0, ge=0),
    topk: int = Query(16, ge=1, le=64),
    mode: str = Query("gate_activation", pattern="^(gate_activation|up|down_output)$"),
    neuron: int = Query(-1, ge=-1),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
    store = rec.store
    t = store.mlp_for_position(layer, mode, position)
    if t is None:
        raise HTTPException(404, f"mlp {mode} not captured for layer {layer}")
    response = {
        "layer": layer,
        "position": position,
        "mode": mode,
        "stats": vector_stats(t[0]),
        "top": topk_activations(t[0], k=topk),
        "shape": list(t.shape),
    }
    if neuron >= 0:
        if neuron >= t.shape[-1]:
            raise HTTPException(404, f"neuron {neuron} out of range ({t.shape[-1]} units)")
        response["neuron"] = {"index": neuron, "value": float(t[0, neuron])}
    return response


@router.get("/sessions/{session_id}/hidden")
async def session_hidden(
    session_id: str,
    request: Request = None,
    layer: int = Query(0, ge=0),
    position: int = Query(0, ge=0),
    limit: int = Query(512, ge=1, le=2048),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
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
    request: Request = None,
    step: int = Query(0, ge=0),
    k: int = Query(16, ge=1, le=50),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
    store = rec.store
    logits = store.logits.get(step)
    if logits is None:
        raise HTTPException(404, f"logits not captured for step {step}")
    candidates = store.logit_candidates.get(step)
    if candidates is None:
        adapter = getattr(rec, "adapter", None)
        if adapter is None or adapter.model_id != rec.model_id:
            raise HTTPException(409, "archived logits do not contain model-specific token candidates")
        candidates = adapter.topk_candidates(logits, k=50)
    return {
        "step": step,
        "candidates": candidates[:k],
        "stats": vector_stats(logits),
        "vocab_size": int(logits.numel()),
    }


@router.post("/tokenize")
async def tokenize_preview(payload: dict[str, Any]) -> dict[str, Any]:
    if state.adapter is None or not state.adapter.is_loaded:
        raise HTTPException(503, "model not loaded")
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise HTTPException(400, "text must be a string")
    if len(text) > settings.max_prompt_chars:
        raise HTTPException(413, f"prompt exceeds {settings.max_prompt_chars} characters")
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
