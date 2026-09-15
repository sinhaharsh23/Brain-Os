"""Create the BrainOS relational metadata schema.

This revision is additive. Destructive downgrades are disabled; data deletion
must happen through explicit application operations or a separately reviewed
migration.
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def _indexed(table: str, column: str) -> None:
    op.create_index(f"ix_{table}_{column}", table, [column])


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("username", sa.String(120), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    _indexed("auth_sessions", "user_id")
    op.create_index("ix_auth_sessions_token_hash", "auth_sessions", ["token_hash"], unique=True)
    _indexed("auth_sessions", "expires_at")

    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    _indexed("workspaces", "owner_id")

    op.create_table(
        "projects",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("workspace_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    _indexed("projects", "owner_id")
    _indexed("projects", "workspace_id")

    op.create_table(
        "brain_sessions",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=True),
        sa.Column("project_id", sa.String(64), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    _indexed("brain_sessions", "owner_id")
    _indexed("brain_sessions", "project_id")

    op.create_table(
        "inference_runs",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=True),
        sa.Column("connection_id", sa.String(128), nullable=False),
        sa.Column("model_id", sa.String(240), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["brain_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ("request_id", "session_id", "owner_id", "status"):
        _indexed("inference_runs", column)

    op.create_table(
        "replay_metadata",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("bundle_path", sa.String(500), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["brain_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
    )
    _indexed("replay_metadata", "owner_id")
    op.create_index("ix_replay_metadata_session_id", "replay_metadata", ["session_id"], unique=True)

    op.create_table(
        "model_metadata",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("model_id", sa.String(240), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id"),
    )
    op.create_index("ix_model_metadata_model_id", "model_metadata", ["model_id"], unique=True)

    op.create_table(
        "provider_metadata",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("provider_id", sa.String(120), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id"),
    )
    op.create_index("ix_provider_metadata_provider_id", "provider_metadata", ["provider_id"], unique=True)


def downgrade() -> None:
    raise RuntimeError("Destructive Alembic downgrades are disabled for BrainOS metadata")
