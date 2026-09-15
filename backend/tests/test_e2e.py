import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BRAINOS_REPLAY_DIR", "/tmp/brainos_replay_test")
os.environ.setdefault("BRAINOS_DEFAULT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait_for_model(client, timeout_s=120):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = client.get("/api/health")
        if r.json().get("model_status") == "loaded":
            return
        time.sleep(1)
    raise AssertionError("model did not load in time")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness(client):
    wait_for_model(client)
    r = client.get("/api/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_hardware(client):
    r = client.get("/api/hardware")
    assert r.status_code == 200
    data = r.json()
    assert data["cpu"]["count"] > 0
    assert data["ram"]["total_gb"] > 0
    assert "cuda_available" in data


def test_metrics_endpoint_is_prometheus_compatible(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "brainos_model_loaded" in r.text
    assert "brainos_websocket_clients" in r.text
    assert "token" not in r.text.lower()


def test_models_listing(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["supported"]) > 0
    assert data["recommended"]["model_id"]


def test_unknown_model_is_rejected(client):
    r = client.post("/api/model/load", json={"model_id": "unknown/untrusted-model"})
    assert r.status_code == 400


def test_tokenize_endpoint(client):
    wait_for_model(client)
    r = client.post("/api/tokenize", json={"text": "Hello BrainOS"})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] > 0
    assert all("id" in t and "position" in t for t in data["tokens"])


def test_tokenize_rejects_oversized_prompt(client):
    wait_for_model(client)
    r = client.post("/api/tokenize", json={"text": "x" * 12001})
    assert r.status_code == 413


def test_websocket_rejects_invalid_generation_parameters(client):
    wait_for_model(client)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "run", "prompt": "invalid params", "params": {"top_p": 0}})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "top_p" in error["data"]["message"]


def test_tensor_endpoints_after_inference(client):
    wait_for_model(client)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "run", "prompt": "Two plus two equals", "params": {"max_new_tokens": 3}})
        types = []
        while True:
            msg = ws.receive_json()
            types.append(msg.get("type"))
            if msg.get("type") == "inference.complete":
                break

    assert "tokenization.complete" in types
    assert "token.generated" in types
    assert "inference.complete" in types

    from app.state import state

    sid = state.engine.current_session.session_id
    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    r = client.get(f"/api/sessions/{sid}/embedding/0")
    assert r.status_code == 200
    assert r.json()["dimension"] == 896
    r = client.get(f"/api/sessions/{sid}/attention?layer=0&head=0&position=0")
    assert r.status_code == 200
    assert len(r.json()["weights"]) > 0
    r = client.get(f"/api/sessions/{sid}/qkv?layer=0&name=q&position=0")
    assert r.status_code == 200
    assert r.json()["stats"]["n"] == 896
    r = client.get(f"/api/sessions/{sid}/mlp?layer=0&position=0&topk=5")
    assert r.status_code == 200
    assert len(r.json()["top"]) == 5
    r = client.get(f"/api/sessions/{sid}/mlp?layer=0&position=0&topk=5&neuron=0")
    assert r.status_code == 200
    assert r.json()["neuron"]["index"] == 0
    r = client.get(f"/api/sessions/{sid}/logits?step=0&k=5")
    assert r.status_code == 200
    assert len(r.json()["candidates"]) == 5
    r = client.get(f"/api/sessions/{sid}/events")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_websocket_controls_are_connection_scoped(client):
    wait_for_model(client)
    with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
        first.send_json({"action": "run", "prompt": "Connection one", "params": {"max_new_tokens": 2, "top_k": 1, "temperature": 0.0}})
        while True:
            event = first.receive_json()
            if event.get("type") == "inference.started":
                break

        second.send_json({"action": "cancel"})
        while True:
            error = second.receive_json()
            if error.get("type") == "error":
                break
        assert error["type"] == "error"
        assert "another connection" in error["data"]["message"]

        while True:
            event = first.receive_json()
            if event.get("type") in {"inference.complete", "inference.cancelled"}:
                assert event["type"] == "inference.complete"
                break


def test_concurrent_websocket_runs_are_serialized_and_isolated(client):
    wait_for_model(client)

    def drain(socket):
        events = []
        while True:
            event = socket.receive_json()
            events.append(event)
            if event.get("type") == "inference.complete":
                return events

    with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
        params = {"max_new_tokens": 1, "temperature": 0.0, "top_k": 1}
        first.send_json({"action": "run", "prompt": "first concurrent run", "params": params})
        second.send_json({"action": "run", "prompt": "second concurrent run", "params": params})
        first_events = drain(first)
        second_events = drain(second)

    first_sessions = {event.get("session_id") for event in first_events if event.get("session_id")}
    second_sessions = {event.get("session_id") for event in second_events if event.get("session_id")}
    assert len(first_sessions) == 1
    assert len(second_sessions) == 1
    assert first_sessions != second_sessions


def test_external_websocket_does_not_require_local_model(client, monkeypatch):
    wait_for_model(client)
    from app.state import state

    old_adapter, old_engine = state.adapter, state.engine
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state.adapter = None
    state.engine = None
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "run", "provider": "openai", "prompt": "external only", "params": {"max_new_tokens": 1}})
            types = []
            while "system.error" not in types:
                types.append(ws.receive_json()["type"])
            assert "external.started" in types
    finally:
        state.adapter, state.engine = old_adapter, old_engine


def test_replay_session_listing(client):
    wait_for_model(client)
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
