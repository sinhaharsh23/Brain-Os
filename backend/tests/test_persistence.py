from sqlalchemy import select

from app.persistence import AuthSession, InferenceRun, Persistence, User, migrate_database
from app.replay.store import ReplayStore


def test_sqlite_auth_ownership_and_restart_persistence(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'brainos.db'}"
    migrate_database(database_url)
    persistence = Persistence(database_url)
    user_a, token_a = persistence.register("operator-a", "correct horse battery")
    user_b, token_b = persistence.register("operator-b", "another secure password")

    assert persistence.user_for_token(token_a)["id"] == user_a["id"]
    assert persistence.user_for_token(token_b)["id"] == user_b["id"]
    assert persistence.authenticate("operator-a", "wrong password") is None
    assert persistence.authenticate("operator-a", "correct horse battery") is not None

    persistence.create_session("session-a", user_a["id"], "private prompt A")
    persistence.create_session("session-b", user_b["id"], "private prompt B")
    persistence.create_run("run-a", "request-a", "session-a", user_a["id"], "connection-a", "private prompt A", "qwen")
    persistence.update_run("run-a", "COMPLETED")

    assert persistence.owns_session("session-a", user_a["id"])
    assert not persistence.owns_session("session-a", user_b["id"])
    assert persistence.list_owned_sessions(user_a["id"])[0]["session_id"] == "session-a"
    with persistence.db() as db:
        assert db.get(InferenceRun, "run-a").status == "COMPLETED"
        assert db.scalar(select(User).where(User.username == "operator-a")) is not None

    restarted = Persistence(database_url)
    assert restarted.user_for_token(token_a)["username"] == "operator-a"
    assert restarted.owns_session("session-a", user_a["id"])
    assert not restarted.owns_session("session-b", user_a["id"])
    assert restarted.list_owned_sessions(user_b["id"])[0]["session_id"] == "session-b"


def test_legacy_json_replay_remains_readable_and_claimable(tmp_path):
    replay = ReplayStore(str(tmp_path / "replays"))
    replay.save_session({"session_id": "legacy-session", "prompt": "legacy", "created_at": 1.0})
    replay.append_events("legacy-session", [{"type": "dev.log", "data": {"message": "kept"}, "ts": 1.0, "session_id": "legacy-session"}])

    database_url = f"sqlite:///{tmp_path / 'brainos.db'}"
    migrate_database(database_url)
    persistence = Persistence(database_url)
    user, _ = persistence.register("legacy-owner", "legacy secure password")
    payload = replay.get_session("legacy-session")
    assert payload is not None
    assert replay.load_events("legacy-session")[0]["data"]["message"] == "kept"
    assert persistence.claim_legacy_session("legacy-session", user["id"], payload["summary"]["prompt"])
    assert persistence.owns_session("legacy-session", user["id"])


def test_default_auth_session_ttl_remains_one_week(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'default-ttl.db'}"
    migrate_database(database_url)
    persistence = Persistence(database_url)
    user, _ = persistence.register("default-ttl-user", "secure password")

    with persistence.db() as db:
        session = db.scalar(select(AuthSession).where(AuthSession.user_id == user["id"]))
    effective_hours = (session.expires_at - session.created_at).total_seconds() / 3600
    assert 167.99 <= effective_hours <= 168.01
