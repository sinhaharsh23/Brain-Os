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


def test_hardware(client):
    r = client.get("/api/hardware")
    assert r.status_code == 200
    data = r.json()
    assert data["cpu"]["count"] > 0
    assert data["ram"]["total_gb"] > 0
    assert "cuda_available" in data


def test_models_listing(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["supported"]) > 0
    assert data["recommended"]["model_id"]


def test_tokenize_endpoint(client):
    wait_for_model(client)
    r = client.post("/api/tokenize", json={"text": "Hello BrainOS"})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] > 0
    assert all("id" in t and "position" in t for t in data["tokens"])


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
    r = client.get(f"/api/sessions/{sid}/logits?step=0&k=5")
    assert r.status_code == 200
    assert len(r.json()["candidates"]) == 5
    r = client.get(f"/api/sessions/{sid}/events")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_replay_session_listing(client):
    wait_for_model(client)
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
