from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, event, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


EXPECTED_TABLES = {
    "users",
    "auth_sessions",
    "workspaces",
    "projects",
    "brain_sessions",
    "inference_runs",
    "replay_metadata",
    "model_metadata",
    "provider_metadata",
}


def migrate_database(url: str) -> None:
    """Apply Alembic migrations, bootstrapping legacy create_all databases safely.

    Older BrainOS versions created the current tables directly. If such a
    database is complete but has no Alembic version row, stamping the initial
    revision preserves its data and establishes migration ownership. Partial
    schemas are intentionally not guessed at and fail through Alembic.
    """
    from alembic import command
    from alembic.config import Config

    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    config = Config(str(config_path))
    config.set_main_option("script_location", str(config_path.parent / "backend" / "alembic"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}, future=True)
    try:
        tables = set(inspect(engine).get_table_names())
        if tables and "alembic_version" not in tables and EXPECTED_TABLES.issubset(tables):
            command.stamp(config, "head")
        else:
            command.upgrade(config, "head")
    finally:
        engine.dispose()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BrainSession(Base):
    __tablename__ = "brain_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    project_id: Mapped[Optional[str]] = mapped_column(ForeignKey("projects.id"), index=True, nullable=True)
    prompt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("brain_sessions.id"), index=True)
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    connection_id: Mapped[str] = mapped_column(String(128), default="")
    model_id: Mapped[str] = mapped_column(String(240), default="")
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", index=True)
    prompt: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class ReplayMetadata(Base):
    __tablename__ = "replay_metadata"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("brain_sessions.id"), unique=True, index=True)
    bundle_path: Mapped[str] = mapped_column(String(500), default="")
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelMetadataRow(Base):
    __tablename__ = "model_metadata"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(240), unique=True, index=True)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProviderMetadataRow(Base):
    __tablename__ = "provider_metadata"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider_id: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Persistence:
    """Relational metadata store; tensor payloads remain file/object references."""

    def __init__(self, url: str) -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        engine_options: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if not url.startswith("sqlite"):
            engine_options["pool_recycle"] = 1800
        self.engine = create_engine(url, connect_args=connect_args, **engine_options)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=Session)
        self.password_hasher = PasswordHasher()

    @staticmethod
    def _workspace_dict(workspace: Workspace) -> dict[str, Any]:
        return {
            "id": workspace.id,
            "owner_id": workspace.owner_id,
            "name": workspace.name,
            "created_at": aware(workspace.created_at).isoformat(),
        }

    @staticmethod
    def _project_dict(project: Project) -> dict[str, Any]:
        return {
            "id": project.id,
            "owner_id": project.owner_id,
            "workspace_id": project.workspace_id,
            "name": project.name,
            "created_at": aware(project.created_at).isoformat(),
        }

    def db(self) -> Session:
        return self.session_factory()

    @staticmethod
    def _user_dict(user: User) -> dict[str, Any]:
        return {"id": user.id, "username": user.username, "created_at": user.created_at.isoformat()}

    def register(self, username: str, password: str, ttl_hours: int = 24 * 7) -> tuple[dict[str, Any], str]:
        username = username.strip().lower()
        if len(username) < 3 or len(username) > 120:
            raise ValueError("username must be between 3 and 120 characters")
        if len(password) < 8 or len(password) > 1024:
            raise ValueError("password must be between 8 and 1024 characters")
        user = User(id=uuid.uuid4().hex, username=username, password_hash=self.password_hasher.hash(password))
        with self.db() as db:
            try:
                db.add(user)
                db.flush()
                workspace = Workspace(id=uuid.uuid4().hex, owner_id=user.id, name="Personal workspace")
                db.add(workspace)
                db.commit()
            except IntegrityError as exc:
                db.rollback()
                raise ValueError("username is already registered") from exc
        return self._user_dict(user), self.create_token(user.id, ttl_hours=ttl_hours)

    def authenticate(self, username: str, password: str, ttl_hours: int = 24 * 7) -> tuple[dict[str, Any], str] | None:
        with self.db() as db:
            user = db.scalar(select(User).where(User.username == username.strip().lower()))
            if user is None:
                return None
            try:
                valid = self.password_hasher.verify(user.password_hash, password)
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                valid = False
            if not valid:
                return None
            return self._user_dict(user), self.create_token(user.id, ttl_hours=ttl_hours)

    def create_token(self, user_id: str, ttl_hours: int = 24 * 7) -> str:
        raw = secrets.token_urlsafe(48)
        with self.db() as db:
            if db.get(User, user_id) is None:
                raise ValueError("user does not exist")
            db.add(AuthSession(
                id=uuid.uuid4().hex,
                user_id=user_id,
                token_hash=self.hash_token(raw),
                expires_at=utcnow() + timedelta(hours=ttl_hours),
            ))
            db.commit()
        return raw

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def user_for_token(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self.db() as db:
            row = db.scalar(select(AuthSession).where(AuthSession.token_hash == self.hash_token(token), AuthSession.revoked.is_(False)))
            if row is None or aware(row.expires_at) <= utcnow():
                return None
            user = db.get(User, row.user_id)
            return self._user_dict(user) if user else None

    def revoke_token(self, token: str | None) -> None:
        if not token:
            return
        with self.db() as db:
            row = db.scalar(select(AuthSession).where(AuthSession.token_hash == self.hash_token(token)))
            if row:
                row.revoked = True
                db.commit()

    def create_session(self, session_id: str, owner_id: str | None, prompt: str, project_id: str | None = None) -> None:
        with self.db() as db:
            if owner_id is not None and db.get(User, owner_id) is None:
                raise ValueError("session owner does not exist")
            if project_id is not None:
                project = db.get(Project, project_id)
                if project is None or project.owner_id != owner_id:
                    raise ValueError("project is not owned by this user")
            if db.get(BrainSession, session_id) is None:
                db.add(BrainSession(id=session_id, owner_id=owner_id, prompt=prompt, project_id=project_id))
                db.commit()

    def create_run(self, run_id: str, request_id: str, session_id: str, owner_id: str | None, connection_id: str, prompt: str, model_id: str) -> None:
        with self.db() as db:
            db.add(InferenceRun(
                id=run_id, request_id=request_id, session_id=session_id, owner_id=owner_id,
                connection_id=connection_id, prompt=prompt, model_id=model_id, status="QUEUED",
            ))
            db.commit()

    def update_run(self, run_id: str, status: str, error: str = "") -> None:
        with self.db() as db:
            row = db.get(InferenceRun, run_id)
            if row is None:
                return
            row.status = status
            if status in {"STARTING", "RUNNING"} and row.started_at is None:
                row.started_at = utcnow()
            if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                row.finished_at = utcnow()
            if error:
                row.error = error[:4000]
            db.commit()

    def upsert_replay(self, session_id: str, owner_id: str | None, bundle_path: str, event_count: int) -> None:
        with self.db() as db:
            row = db.scalar(select(ReplayMetadata).where(ReplayMetadata.session_id == session_id))
            if row is None:
                row = ReplayMetadata(id=uuid.uuid4().hex, session_id=session_id, owner_id=owner_id)
                db.add(row)
            row.bundle_path = bundle_path
            row.event_count = event_count
            db.commit()

    def create_workspace(self, owner_id: str, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 160:
            raise ValueError("workspace name must be between 1 and 160 characters")
        with self.db() as db:
            if db.get(User, owner_id) is None:
                raise ValueError("user does not exist")
            workspace = Workspace(id=uuid.uuid4().hex, owner_id=owner_id, name=name)
            db.add(workspace)
            db.commit()
            return self._workspace_dict(workspace)

    def list_workspaces(self, owner_id: str) -> list[dict[str, Any]]:
        with self.db() as db:
            rows = db.scalars(select(Workspace).where(Workspace.owner_id == owner_id).order_by(Workspace.created_at.asc())).all()
            return [self._workspace_dict(row) for row in rows]

    def get_workspace(self, owner_id: str, workspace_id: str) -> dict[str, Any] | None:
        with self.db() as db:
            row = db.scalar(select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == owner_id))
            return self._workspace_dict(row) if row else None

    def update_workspace(self, owner_id: str, workspace_id: str, name: str) -> dict[str, Any] | None:
        name = name.strip()
        if not name or len(name) > 160:
            raise ValueError("workspace name must be between 1 and 160 characters")
        with self.db() as db:
            row = db.scalar(select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == owner_id))
            if row is None:
                return None
            row.name = name
            db.commit()
            return self._workspace_dict(row)

    def delete_workspace(self, owner_id: str, workspace_id: str) -> bool:
        with self.db() as db:
            row = db.scalar(select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == owner_id))
            if row is None:
                return False
            projects = db.scalars(select(Project).where(Project.workspace_id == workspace_id, Project.owner_id == owner_id)).all()
            project_ids = [project.id for project in projects]
            if project_ids:
                db.query(BrainSession).filter(BrainSession.project_id.in_(project_ids)).update({BrainSession.project_id: None}, synchronize_session=False)
                for project in projects:
                    db.delete(project)
            db.delete(row)
            db.commit()
            return True

    def create_project(self, owner_id: str, workspace_id: str, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 160:
            raise ValueError("project name must be between 1 and 160 characters")
        with self.db() as db:
            workspace = db.scalar(select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == owner_id))
            if workspace is None:
                raise ValueError("workspace is not owned by this user")
            project = Project(id=uuid.uuid4().hex, owner_id=owner_id, workspace_id=workspace_id, name=name)
            db.add(project)
            db.commit()
            return self._project_dict(project)

    def list_projects(self, owner_id: str, workspace_id: str | None = None) -> list[dict[str, Any]]:
        with self.db() as db:
            query = select(Project).where(Project.owner_id == owner_id)
            if workspace_id is not None:
                query = query.where(Project.workspace_id == workspace_id)
            rows = db.scalars(query.order_by(Project.created_at.asc())).all()
            return [self._project_dict(row) for row in rows]

    def get_project(self, owner_id: str, project_id: str) -> dict[str, Any] | None:
        with self.db() as db:
            row = db.scalar(select(Project).where(Project.id == project_id, Project.owner_id == owner_id))
            return self._project_dict(row) if row else None

    def update_project(self, owner_id: str, project_id: str, name: str | None = None, workspace_id: str | None = None) -> dict[str, Any] | None:
        if name is not None:
            name = name.strip()
            if not name or len(name) > 160:
                raise ValueError("project name must be between 1 and 160 characters")
        with self.db() as db:
            row = db.scalar(select(Project).where(Project.id == project_id, Project.owner_id == owner_id))
            if row is None:
                return None
            if workspace_id is not None:
                workspace = db.scalar(select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == owner_id))
                if workspace is None:
                    raise ValueError("workspace is not owned by this user")
                row.workspace_id = workspace_id
            if name is not None:
                row.name = name
            db.commit()
            return self._project_dict(row)

    def delete_project(self, owner_id: str, project_id: str) -> bool:
        with self.db() as db:
            row = db.scalar(select(Project).where(Project.id == project_id, Project.owner_id == owner_id))
            if row is None:
                return False
            db.query(BrainSession).filter(BrainSession.project_id == project_id).update({BrainSession.project_id: None}, synchronize_session=False)
            db.delete(row)
            db.commit()
            return True

    def owns_session(self, session_id: str, owner_id: str | None) -> bool:
        if owner_id is None:
            return True
        with self.db() as db:
            row = db.get(BrainSession, session_id)
            return row is not None and row.owner_id == owner_id

    def owns_run(self, run_id: str, owner_id: str) -> bool:
        with self.db() as db:
            row = db.get(InferenceRun, run_id)
            return row is not None and row.owner_id == owner_id

    def owns_replay(self, session_id: str, owner_id: str) -> bool:
        with self.db() as db:
            row = db.scalar(select(ReplayMetadata).where(ReplayMetadata.session_id == session_id))
            return row is not None and row.owner_id == owner_id

    def claim_legacy_session(self, session_id: str, owner_id: str, prompt: str = "") -> bool:
        with self.db() as db:
            row = db.get(BrainSession, session_id)
            if row is not None:
                return row.owner_id == owner_id
            db.add(BrainSession(id=session_id, owner_id=owner_id, prompt=prompt))
            db.add(ReplayMetadata(id=uuid.uuid4().hex, session_id=session_id, owner_id=owner_id))
            db.commit()
            return True

    def list_owned_sessions(self, owner_id: str) -> list[dict[str, Any]]:
        with self.db() as db:
            rows = db.scalars(select(BrainSession).where(BrainSession.owner_id == owner_id).order_by(BrainSession.created_at.desc())).all()
            return [{"session_id": row.id, "prompt": row.prompt, "created_at": row.created_at.timestamp(), "owner_id": owner_id} for row in rows]
