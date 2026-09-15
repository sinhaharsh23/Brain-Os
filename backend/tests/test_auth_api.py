import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from sqlalchemy import select

from app.config import settings
from app.main import app
from app.persistence import AuthSession, Persistence, migrate_database
from app.replay.store import ReplayStore
from app.state import state


def test_multi_user_auth_protects_http_websocket_and_replay(tmp_path, monkeypatch):
    old_persistence = state.persistence
    old_replay = state.replay
    monkeypatch.setattr(settings, "auth_mode", "multi_user")
    async def no_model_startup():
        state.model_status = "not_loaded"
    monkeypatch.setattr("app.main._load_model_on_startup", no_model_startup)
    replay = ReplayStore(str(tmp_path / "replays"))
    replay.save_session({"session_id": "owned-session", "prompt": "private", "created_at": 1.0})
    state.replay = replay

    with TestClient(app) as client:
        # Lifespan wires the configured default store; use an isolated store for
        # this test after startup so no real user data is touched.
        database_url = f"sqlite:///{tmp_path / 'auth.db'}"
        migrate_database(database_url)
        state.persistence = Persistence(database_url)
        state.replay = replay
        anonymous = client.get("/api/sessions")
        assert anonymous.status_code == 401

        registered_a = client.post("/api/auth/register", json={"username": "user-a", "password": "password-a"})
        registered_b = client.post("/api/auth/register", json={"username": "user-b", "password": "password-b"})
        assert registered_a.status_code == 201
        assert registered_b.status_code == 201
        token_a = registered_a.json()["token"]
        token_b = registered_b.json()["token"]

        headers_a = {"X-BrainOS-Token": token_a}
        headers_b = {"X-BrainOS-Token": token_b}
        assert client.get("/api/auth/me", headers=headers_a).json()["user"]["username"] == "user-a"
        assert client.get("/api/sessions", headers=headers_a).status_code == 200

        state.persistence.create_session("owned-session", registered_a.json()["user"]["id"], "private")
        assert client.get("/api/sessions/owned-session", headers=headers_a).status_code == 200
        assert client.get("/api/sessions/owned-session", headers=headers_b).status_code == 404
        assert client.get("/api/sessions/owned-session/events", headers=headers_b).status_code == 404

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass
        with client.websocket_connect(f"/ws?token={token_a}") as ws:
            ws.send_json({"action": "unsupported"})
            assert ws.receive_json()["type"] == "error"

        assert client.post("/api/auth/logout", headers=headers_a).status_code == 200
        assert client.get("/api/sessions", headers=headers_a).status_code == 401

    state.persistence = old_persistence
    state.replay = old_replay


def test_auth_session_ttl_configuration_applies_to_registration_and_login(tmp_path, monkeypatch):
    old_persistence = state.persistence
    old_replay = state.replay
    old_mode = settings.auth_mode
    old_ttl = settings.auth_session_ttl_hours
    monkeypatch.setattr(settings, "auth_mode", "multi_user")
    monkeypatch.setattr(settings, "auth_session_ttl_hours", 1)

    async def no_model_startup():
        state.model_status = "not_loaded"

    monkeypatch.setattr("app.main._load_model_on_startup", no_model_startup)
    database_url = f"sqlite:///{tmp_path / 'ttl-api.db'}"
    migrate_database(database_url)
    state.replay = ReplayStore(str(tmp_path / "replays"))
    try:
        with TestClient(app) as client:
            state.persistence = Persistence(database_url)
            registered = client.post("/api/auth/register", json={"username": "ttl-api-user", "password": "secure password"})
            assert registered.status_code == 201
            logged_in = client.post("/api/auth/login", json={"username": "ttl-api-user", "password": "secure password"})
            assert logged_in.status_code == 200

            with state.persistence.db() as db:
                sessions = db.scalars(select(AuthSession).order_by(AuthSession.created_at.asc())).all()
            assert len(sessions) == 2
            for session in sessions:
                effective_hours = (session.expires_at - session.created_at).total_seconds() / 3600
                assert 0.99 <= effective_hours <= 1.01
    finally:
        state.persistence = old_persistence
        state.replay = old_replay
        settings.auth_mode = old_mode
        settings.auth_session_ttl_hours = old_ttl
