from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.events.types import Event
from app.instrumentation.stats import sample_activations, tensor_info, topk_activations, vector_stats
from app.models.registry import SUPPORTED_MODELS, hardware_report, recommend_model
from app.providers.registry import PROVIDERS, provider_registry
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
        "model_error_code": state.model_error_code,
        "model_error": state.model_error,
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
    return [provider.to_dict() for provider in provider_registry.list()]


@router.get("/provider-registry")
async def provider_registry_snapshot() -> dict[str, Any]:
    """Return the unified provider/model catalog without secret material."""
    return {
        "providers": [provider.to_dict() for provider in provider_registry.list()],
        "models": [model.to_dict() for model in provider_registry.list_models()],
    }


@router.post("/providers/{provider_id}/validate")
async def validate_provider_configuration(provider_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate configuration; only an explicit test may contact a provider."""
    try:
        return provider_registry.validate_configuration(provider_id, test_connection=bool((payload or {}).get("test_connection", False)))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/scheduler")
async def scheduler_status() -> dict[str, Any]:
    return state.scheduler.snapshot() if state.scheduler is not None else {"queue_size": 0, "queue_limit": 0, "active": 0, "active_limit": 0, "runs": 0}


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> PlainTextResponse:
    """Expose low-cardinality Prometheus metrics without prompts or tokens."""
    snapshot = state.monitor.snapshot()
    scheduler = state.scheduler.snapshot() if state.scheduler is not None else {}
    model_loaded = 1 if state.model_status == "loaded" else 0
    lines = [
        "# HELP brainos_model_loaded Whether the configured model is loaded.",
        "# TYPE brainos_model_loaded gauge",
        f"brainos_model_loaded {model_loaded}",
        "# HELP brainos_websocket_clients Current connected WebSocket clients.",
        "# TYPE brainos_websocket_clients gauge",
        f"brainos_websocket_clients {state.manager.count}",
        "# HELP brainos_cpu_percent Current host CPU utilization percentage.",
        "# TYPE brainos_cpu_percent gauge",
        f"brainos_cpu_percent {snapshot['cpu_percent']}",
        "# HELP brainos_process_ram_bytes Current BrainOS process resident memory.",
        "# TYPE brainos_process_ram_bytes gauge",
        f"brainos_process_ram_bytes {snapshot['process_ram_gb'] * 1024**3}",
    ]
    for metric in ("queue_size", "active", "runs"):
        if metric in scheduler:
            lines.extend([
                f"# HELP brainos_scheduler_{metric} Current scheduler {metric}.",
                f"# TYPE brainos_scheduler_{metric} gauge",
                f"brainos_scheduler_{metric} {scheduler[metric]}",
            ])
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


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


@router.get("/model/architecture")
async def model_architecture(max_depth: int = Query(3, ge=1, le=6)) -> dict[str, Any]:
    if state.adapter is None or not state.adapter.is_loaded:
        raise HTTPException(503, "no model loaded")
    tree = state.adapter.get_architecture_tree(max_depth=max_depth)
    return {
        "model_id": state.adapter.model_id,
        "architecture": state.adapter.metadata.architecture,
        "tree": tree,
    }


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
        try:
            from app.main import unload_current_model
            from app.inference.engine import InferenceEngine

            await unload_current_model()
            adapter = await state.model_manager.load_model(
                model_id=model_id,
                device=settings.device,
                dtype=settings.dtype,
            )
            engine = InferenceEngine(
                adapter,
                state.bus,
                max_prompt_tokens=settings.max_prompt_tokens,
                capture_limit=settings.capture_limit_tokens,
                capture_level=settings.capture_level,
                attention_capture=settings.attention_capture,
            )
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
            # ModelManager performs adapter cleanup on failed loads. Clear the
            # runtime aliases so a stale engine cannot accept a new request.
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


@router.get("/sessions/compare")
async def compare_sessions(
    session_a: str = Query(...),
    session_b: str = Query(...),
    request: Request = None,
) -> dict[str, Any]:
    for sid in (session_a, session_b):
        try:
            validate_session_id(sid)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    rec_a = _require_session(session_a, request)
    rec_b = _require_session(session_b, request)

    tokens_a = rec_a.output_tokens
    tokens_b = rec_b.output_tokens
    max_steps = max(len(tokens_a), len(tokens_b))

    steps_diff = []
    matches = 0
    for i in range(max_steps):
        ta = tokens_a[i] if i < len(tokens_a) else None
        tb = tokens_b[i] if i < len(tokens_b) else None
        match = bool(ta and tb and ta.get("token_id") == tb.get("token_id"))
        if match:
            matches += 1
        steps_diff.append({
            "step": i,
            "token_a": ta,
            "token_b": tb,
            "match": match,
            "prob_diff": round((ta["probability"] - tb["probability"]), 4) if (ta and tb) else None,
            "entropy_a": ta.get("entropy") if ta else None,
            "entropy_b": tb.get("entropy") if tb else None,
            "latency_ms_a": _step_latency(rec_a, i),
            "latency_ms_b": _step_latency(rec_b, i),
        })

    token_similarity = round(matches / max(max_steps, 1), 4)

    same_model = rec_a.model_id == rec_b.model_id
    attention_a = _attention_comparison_summary(rec_a)
    attention_b = _attention_comparison_summary(rec_b)
    lens_a = _logit_lens_comparison_summary(rec_a)
    lens_b = _logit_lens_comparison_summary(rec_b)
    common_lens = set(lens_a["entries"]) & set(lens_b["entries"])
    lens_matches = sum(
        1
        for key in common_lens
        if lens_a["top_ids"].get(key) is not None
        and lens_a["top_ids"].get(key) == lens_b["top_ids"].get(key)
    )
    kv_a = _kv_cache_peak(rec_a)
    kv_b = _kv_cache_peak(rec_b)
    return {
        "session_a": rec_a.summary(),
        "session_b": rec_b.summary(),
        "token_similarity": token_similarity,
        "steps_comparison": steps_diff,
        "configuration": {
            "a": {"model_id": rec_a.model_id, "params": rec_a.params, "metadata": rec_a.metadata},
            "b": {"model_id": rec_b.model_id, "params": rec_b.params, "metadata": rec_b.metadata},
            "same_model": same_model,
        },
        "output": {"a": rec_a.response, "b": rec_b.response},
        "entropy": {
            "average_a": _average([t.get("entropy") for t in tokens_a]),
            "average_b": _average([t.get("entropy") for t in tokens_b]),
        },
        "memory": {
            "kv_cache_peak_mb_a": kv_a,
            "kv_cache_peak_mb_b": kv_b,
            "process_memory_gb_a": None,
            "process_memory_gb_b": None,
            "process_memory_status": "Unavailable for this session: process memory is not sampled into replay records.",
        },
        "attention": {
            "compatible": same_model and attention_a["available"] and attention_b["available"],
            "a": {k: v for k, v in attention_a.items() if k != "entries"},
            "b": {k: v for k, v in attention_b.items() if k != "entries"},
        },
        "logit_lens": {
            "compatible": same_model and bool(common_lens),
            "common_entries": len(common_lens),
            "top1_similarity": round(lens_matches / max(len(common_lens), 1), 4) if common_lens else None,
            "a": {"available": bool(lens_a["entries"]), "entries": len(lens_a["entries"])},
            "b": {"available": bool(lens_b["entries"]), "entries": len(lens_b["entries"])},
        },
        "performance": {
            "ttft_ms_a": rec_a.timings.get("ttft_ms"),
            "ttft_ms_b": rec_b.timings.get("ttft_ms"),
            "prefill_ms_a": rec_a.timings.get("prefill_ms"),
            "prefill_ms_b": rec_b.timings.get("prefill_ms"),
            "average_token_latency_ms_a": rec_a.timings.get("average_token_latency_ms"),
            "average_token_latency_ms_b": rec_b.timings.get("average_token_latency_ms"),
            "current_token_latency_ms_a": rec_a.timings.get("current_token_latency_ms"),
            "current_token_latency_ms_b": rec_b.timings.get("current_token_latency_ms"),
            "tps_a": rec_a.timings.get("tokens_per_second"),
            "tps_b": rec_b.timings.get("tokens_per_second"),
            "total_ms_a": rec_a.timings.get("total_ms"),
            "total_ms_b": rec_b.timings.get("total_ms"),
        },
    }


def _average(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(numeric) / len(numeric), 6) if numeric else None


def _step_latency(record, step: int) -> float | None:
    if step < len(record.step_stats):
        value = record.step_stats[step].get("time_ms")
        return float(value) if isinstance(value, (int, float)) else None
    return None


def _kv_cache_peak(record) -> float | None:
    values = [item.get("total_mb") for item in record.store.kv_cache.values()]
    return max((float(value) for value in values if isinstance(value, (int, float))), default=None)


def _attention_comparison_summary(record) -> dict[str, Any]:
    if not record.store.attention:
        return {"available": False, "entries": [], "layers": 0, "steps": 0, "average_peak_weight": None, "average_entropy": None}
    peak_values: list[float] = []
    entropy_values: list[float] = []
    layer_ids: set[int] = set()
    step_ids: set[int] = set()
    for (layer, step), matrix in record.store.attention.items():
        layer_ids.add(layer)
        step_ids.add(step)
        if getattr(matrix, "numel", lambda: 0)() == 0:
            continue
        row = matrix[:, -1, :].reshape(-1)
        peak_values.append(float(row.max()))
        probabilities = row.clamp_min(1e-12)
        entropy_values.append(float(-(probabilities * probabilities.log()).sum() / max(matrix.shape[0], 1)))
    return {
        "available": True,
        "entries": [f"{layer}:{step}" for layer, step in record.store.attention],
        "layers": len(layer_ids),
        "steps": len(step_ids),
        "average_peak_weight": _average(peak_values),
        "average_entropy": _average(entropy_values),
    }


def _logit_lens_comparison_summary(record) -> dict[str, Any]:
    entries: list[str] = []
    top_ids: dict[str, int | None] = {}
    for (layer, step), candidates in record.store.logit_lens.items():
        key = f"{layer}:{step}"
        entries.append(key)
        top_ids[key] = int(candidates[0]["token_id"]) if candidates else None
    return {"entries": entries, "top_ids": top_ids}


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
    native_session = state.native_sessions.get(session_id)
    if native_session is not None:
        return native_session
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
    full: bool = Query(False),
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
    column_start = int(getattr(store, "attention_column_starts", {}).get(store.position_of(position)[0], 0))
    rows = []
    for i, v in enumerate(row.tolist()):
        rows.append({"token_index": column_start + i, "weight": v})
    response: dict[str, Any] = {
        "layer": layer,
        "head": head,
        "position": position,
        "total_tokens": total,
        "weights": rows,
        "stats": vector_stats(row),
        "is_full_matrix": bool(m.shape[1] == total),
    }
    if full:
        # Attention capture is intentionally bounded by the engine. Expose
        # exactly the retained rows/columns, never a reconstructed matrix.
        matrix = m[:, :, : min(int(m.shape[-1]), 256)]
        response["matrix"] = [[float(value) for value in matrix[head, row_index].tolist()] for row_index in range(matrix.shape[1])]
        response["matrix_start"] = column_start
        average = m[:, 0, :].mean(dim=0)
        response["average_weights"] = [{"token_index": column_start + i, "weight": float(value)} for i, value in enumerate(average.tolist())]
        summaries = []
        for head_index in range(m.shape[0]):
            values = m[head_index, 0]
            safe = values.clamp_min(1e-12)
            summaries.append({
                "head": head_index,
                "entropy": float(-(safe * safe.log()).sum()),
                "max_weight": float(values.max()),
                "top_position": int(values.argmax()),
            })
        response["head_summaries"] = summaries
    return response


@router.get("/sessions/{session_id}/qkv")
async def session_qkv(
    session_id: str,
    request: Request = None,
    layer: int = Query(0, ge=0),
    name: str = Query("q", pattern="^(q|k|v|o)$"),
    position: int = Query(0, ge=0),
    limit: int = Query(512, ge=1, le=2048),
    head: int = Query(-1, ge=-1),
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

    if head >= 0:
        q_heads = int(rec.metadata.get("num_attention_heads", 0) or 0)
        kv_heads = int(rec.metadata.get("num_kv_heads", q_heads) or 0)
        head_dim = int(rec.metadata.get("head_dim", 0) or 0)
        if not q_heads or not kv_heads or not head_dim:
            raise HTTPException(404, "head metadata is unavailable for this model")
        if head >= q_heads:
            raise HTTPException(404, f"query head {head} out of range ({q_heads} heads)")
        effective_head = head
        if name in ("k", "v"):
            # The UI selection is in query-head space. For grouped-query
            # attention, map that query head to the actual shared KV head.
            effective_head = (head * kv_heads) // q_heads
        head_count = q_heads if name in ("q", "o") else kv_heads
        res = store.qkv_head_for_position(layer, name, effective_head, position, head_dim, head_count)
        if res is None:
            raise HTTPException(404, f"{name.upper()} head {head} is unavailable for this state")
        response: dict[str, Any] = {
            "layer": layer,
            "name": name,
            "position": position,
            "head": effective_head,
            "requested_head": head,
            "head_space": "query",
            "head_dim": head_dim,
            "stats": res["stats"],
            "values": res["values"],
            "shape": [1, head_dim],
            "dtype": str(t.dtype),
            "gqa": {
                "query_heads": q_heads,
                "kv_heads": kv_heads,
                "requested_query_head": head,
                "mapped_kv_head": effective_head if name in ("k", "v") else (head * kv_heads) // q_heads,
                "attention_type": "Grouped-Query Attention" if q_heads != kv_heads else "Multi-Head Attention",
            },
        }
        if name == "q":
            import torch
            q_head = store.qkv_head_for_position(layer, "q", head, position, head_dim, q_heads)
            attention = store.attention_for_position(layer, position)
            if q_head is not None and attention is not None:
                q_vector = torch.tensor([item["value"] for item in q_head["values"]], dtype=torch.float32)
                kv_head = (head * kv_heads) // q_heads
                matches = []
                contributions = []
                for token_index in range(int(attention.shape[-1])):
                    key = store.qkv_head_for_position(layer, "k", kv_head, token_index, head_dim, kv_heads)
                    value = store.qkv_head_for_position(layer, "v", kv_head, token_index, head_dim, kv_heads)
                    if key is None:
                        continue
                    key_vector = torch.tensor([item["value"] for item in key["values"]], dtype=torch.float32)
                    raw_score = float(torch.dot(q_vector, key_vector))
                    scaled_score = raw_score / (head_dim ** 0.5)
                    weight = float(attention[head, 0, token_index]) if token_index < attention.shape[-1] else 0.0
                    contribution_norm = 0.0
                    if value is not None:
                        value_vector = torch.tensor([item["value"] for item in value["values"]], dtype=torch.float32)
                        contribution_norm = float((value_vector * weight).norm())
                    matches.append({"token_index": token_index, "raw_score": raw_score, "scaled_score": scaled_score, "attention_probability": weight})
                    contributions.append({"token_index": token_index, "attention_weight": weight, "contribution_norm": contribution_norm})
                response["query_key_matches"] = sorted(matches, key=lambda item: item["scaled_score"], reverse=True)[:12]
                response["value_contributions"] = sorted(contributions, key=lambda item: item["contribution_norm"], reverse=True)[:12]
        return response

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
    mode: str = Query("gate_activation", pattern="^(intermediate|gate_activation|up|down_output)$"),
    neuron: int = Query(-1, ge=-1),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
    store = rec.store
    t = store.mlp_for_position(layer, mode, position)
    if t is None and mode == "intermediate":
        t = store.mlp_for_position(layer, "gate_activation", position)
    if t is None:
        raise HTTPException(404, f"mlp {mode} not captured for layer {layer}")
    response = {
        "layer": layer,
        "position": position,
        "mode": mode,
        "stats": rec.metadata.get("native_mlp_stats", {}).get(f"{layer}:{store.position_of(position)[0]}", vector_stats(t[0])),
        "top": topk_activations(t[0], k=topk),
        "shape": list(t.shape),
    }
    if neuron >= 0:
        if neuron >= t.shape[-1]:
            raise HTTPException(404, f"neuron {neuron} out of range ({t.shape[-1]} units)")
        response["neuron"] = {"index": neuron, "value": float(t[0, neuron])}
    return response


@router.get("/sessions/{session_id}/residual")
async def session_residual(
    session_id: str,
    request: Request = None,
    layer: int = Query(0, ge=0),
    position: int = Query(0, ge=0),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
    store = rec.store
    res = store.residual_for_position(layer, position)
    if res is None:
        raise HTTPException(404, f"residual metrics not captured for layer {layer} at position {position}")

    num_layers = int(rec.metadata.get("num_layers", 0) or 0)
    if not num_layers:
        num_layers = max((layer_idx for layer_idx, _ in store.residual), default=-1) + 1
    all_layers = []
    for l_idx in range(num_layers):
        l_res = store.residual_for_position(l_idx, position)
        if l_res is not None:
            all_layers.append({"layer": l_idx, **l_res})

    return {
        "session_id": session_id,
        "layer": layer,
        "position": position,
        "metrics": res,
        "all_layers": all_layers,
    }


@router.get("/sessions/{session_id}/logit-lens")
async def session_logit_lens(
    session_id: str,
    request: Request = None,
    step: int = Query(0, ge=0),
    layer: int | None = Query(None, ge=0),
    k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
    store = rec.store
    num_layers = int(rec.metadata.get("num_layers", 0) or 0)
    if not num_layers:
        num_layers = max((layer_idx for layer_idx, _ in store.hidden), default=-1)

    if layer is not None:
        cands = store.logit_lens.get((layer, step))
        if cands is None and state.adapter is not None and state.adapter.is_loaded:
            hs = store.hidden_for_step(layer + 1, step)
            if hs is not None:
                try:
                    logits = state.adapter.project_hidden_state_to_logits(hs.unsqueeze(0))[0, -1]
                    cands = state.adapter.topk_candidates(logits, k=k)
                    store.logit_lens[(layer, step)] = cands
                except Exception:
                    pass
        if cands is None:
            raise HTTPException(404, f"logit lens not available for layer {layer} at step {step}")
        return {
            "session_id": session_id,
            "step": step,
            "layer": layer,
            "candidates": cands[:k],
        }

    layers_data = []
    for l_idx in range(num_layers):
        cands = store.logit_lens.get((l_idx, step))
        if cands is None and state.adapter is not None and state.adapter.is_loaded:
            hs = store.hidden_for_step(l_idx + 1, step)
            if hs is not None:
                try:
                    logits = state.adapter.project_hidden_state_to_logits(hs.unsqueeze(0))[0, -1]
                    cands = state.adapter.topk_candidates(logits, k=k)
                    store.logit_lens[(l_idx, step)] = cands
                except Exception:
                    pass
        if cands is not None:
            layers_data.append({"layer": l_idx, "candidates": cands[:k]})

    return {
        "session_id": session_id,
        "step": step,
        "layers": layers_data,
    }


@router.get("/sessions/{session_id}/kv-cache")
async def session_kv_cache(
    session_id: str,
    request: Request = None,
    step: int = Query(0, ge=0),
) -> dict[str, Any]:
    try:
        validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rec = _require_session(session_id, request)
    store = rec.store
    kv = store.kv_cache.get(step)
    if kv is None:
        raise HTTPException(404, f"kv cache metadata not available for step {step}")
    return {"session_id": session_id, **kv}


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
        "stats": store.logit_stats.get(step, vector_stats(logits)),
        "vocab_size": int(store.logit_vocab_sizes.get(step, logits.numel())),
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


@router.get("/providers/ollama/models")
async def ollama_models():
    import asyncio
    from app.providers.ollama import installed_models
    try:
        return {"models": await asyncio.to_thread(installed_models), "status": "READY"}
    except (OSError, ValueError):
        return {"models": [], "status": "OFFLINE"}


@router.get("/providers/{provider_id}/model-info")
async def provider_model_info(provider_id: str, model: str):
    import asyncio
    from app.providers.ollama import model_info
    from app.providers.telemetry import telemetry
    from app.providers.registry import LIMITED
    if provider_id == "ollama":
        info = await asyncio.to_thread(model_info, model)
        return telemetry("ollama", model, model=info, capabilities=LIMITED.to_dict(), sources={"model": "ollama_api"})
    raise HTTPException(404, "provider model metadata unavailable")
