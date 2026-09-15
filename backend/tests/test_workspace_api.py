from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.persistence import Persistence, migrate_database
from app.replay.store import ReplayStore
from app.security import RateLimiter
from app.state import state


def test_workspace_project_routes_are_owner_scoped(tmp_path, monkeypatch):
    old_persistence = state.persistence
    old_replay = state.replay
    old_mode = settings.auth_mode
    monkeypatch.setattr(settings, "auth_mode", "multi_user")
    monkeypatch.setattr("app.main.request_limiter", RateLimiter(1000))
    async def no_model_startup():
        state.model_status = "not_loaded"
    monkeypatch.setattr("app.main._load_model_on_startup", no_model_startup)
    database_url = f"sqlite:///{tmp_path / 'workspace-api.db'}"
    migrate_database(database_url)
    state.replay = ReplayStore(str(tmp_path / "replays"))
    try:
        with TestClient(app) as client:
            # Lifespan wires the configured store; replace it after startup so
            # this route test remains isolated from the developer database.
            state.persistence = Persistence(database_url)
            user_a = client.post("/api/auth/register", json={"username": "api-user-a", "password": "api password a"}).json()
            user_b = client.post("/api/auth/register", json={"username": "api-user-b", "password": "api password b"}).json()
            headers_a = {"X-BrainOS-Token": user_a["token"]}
            headers_b = {"X-BrainOS-Token": user_b["token"]}
            workspace = client.post("/api/workspaces", headers=headers_a, json={"name": "A workspace"})
            assert workspace.status_code == 201
            workspace_id = workspace.json()["id"]
            assert client.get("/api/workspaces", headers=headers_a).json()[-1]["id"] == workspace_id
            assert client.get(f"/api/workspaces/{workspace_id}", headers=headers_b).status_code == 404
            assert client.patch(f"/api/workspaces/{workspace_id}", headers=headers_b, json={"name": "stolen"}).status_code == 404

            project = client.post("/api/projects", headers=headers_a, json={"workspace_id": workspace_id, "name": "A project"})
            assert project.status_code == 201
            project_id = project.json()["id"]
            assert client.get(f"/api/projects/{project_id}", headers=headers_b).status_code == 404
            assert client.delete(f"/api/projects/{project_id}", headers=headers_b).status_code == 404
            assert client.patch(f"/api/workspaces/{workspace_id}", headers=headers_a, json={"name": "updated"}).json()["name"] == "updated"
            assert client.delete(f"/api/workspaces/{workspace_id}", headers=headers_a).json() == {"deleted": True}
            assert client.get(f"/api/projects/{project_id}", headers=headers_a).status_code == 404
    finally:
        state.persistence = old_persistence
        state.replay = old_replay
        settings.auth_mode = old_mode
