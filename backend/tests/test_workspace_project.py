import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.persistence import BrainSession, Persistence, Project, User, migrate_database


@pytest.fixture()
def persistence(tmp_path):
    url = f"sqlite:///{tmp_path / 'ownership.db'}"
    migrate_database(url)
    return Persistence(url)


def test_workspace_project_crud_and_ownership(persistence):
    user_a, _ = persistence.register("workspace-a", "workspace password a")
    user_b, _ = persistence.register("workspace-b", "workspace password b")
    workspace = persistence.create_workspace(user_a["id"], "Research")
    project = persistence.create_project(user_a["id"], workspace["id"], "Trace run")

    assert persistence.get_workspace(user_a["id"], workspace["id"])["name"] == "Research"
    assert persistence.get_workspace(user_b["id"], workspace["id"]) is None
    assert persistence.get_project(user_b["id"], project["id"]) is None
    assert persistence.update_project(user_b["id"], project["id"], name="stolen") is None
    assert not persistence.delete_project(user_b["id"], project["id"])
    assert persistence.update_workspace(user_a["id"], workspace["id"], "Renamed")["name"] == "Renamed"

    persistence.create_session("project-session", user_a["id"], "prompt", project["id"])
    assert persistence.list_projects(user_a["id"], workspace["id"])[0]["id"] == project["id"]
    assert persistence.delete_project(user_a["id"], project["id"])
    with persistence.db() as db:
        assert db.get(BrainSession, "project-session").project_id is None

    project_two = persistence.create_project(user_a["id"], workspace["id"], "Second")
    assert persistence.delete_workspace(user_a["id"], workspace["id"])
    assert persistence.get_project(user_a["id"], project_two["id"]) is None
    assert all(item["id"] != workspace["id"] for item in persistence.list_workspaces(user_a["id"]))


def test_integrity_auth_expiry_and_foreign_keys(persistence):
    user, token = persistence.register("integrity-user", "integrity password")
    with pytest.raises(ValueError, match="already registered"):
        persistence.register("integrity-user", "another password")
    with pytest.raises(ValueError, match="does not exist"):
        persistence.create_token("missing-user")
    with pytest.raises(ValueError, match="does not exist"):
        persistence.create_session("bad-owner-session", "missing-user", "bad")

    from app.persistence import AuthSession, utcnow
    with persistence.db() as db:
        row = db.query(AuthSession).filter(AuthSession.token_hash == persistence.hash_token(token)).one()
        row.expires_at = utcnow()
        db.commit()
    assert persistence.user_for_token(token) is None

    with persistence.db() as db:
        with pytest.raises(IntegrityError):
            db.add(Project(id="orphan", owner_id=user["id"], workspace_id="missing-workspace", name="orphan"))
            db.commit()
        db.rollback()
        assert db.scalar(select(User).where(User.id == user["id"])) is not None


def test_transaction_rollback_leaves_database_usable(persistence):
    persistence.register("rollback-user", "rollback password")
    with pytest.raises(ValueError):
        persistence.register("rollback-user", "short")
    other, _ = persistence.register("rollback-other", "rollback password two")
    assert other["username"] == "rollback-other"
