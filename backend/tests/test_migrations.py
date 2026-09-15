from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from app.persistence import EXPECTED_TABLES, Persistence, migrate_database


def test_initial_migration_is_reproducible_and_preserves_data(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'migrated.db'}"
    migrate_database(database_url)
    persistence = Persistence(database_url)
    user, token = persistence.register("migration-owner", "migration password")

    # A second application of the same migration is a no-op and data remains.
    migrate_database(database_url)
    restarted = Persistence(database_url)
    assert restarted.user_for_token(token)["id"] == user["id"]
    assert EXPECTED_TABLES.issubset(set(inspect(restarted.engine).get_table_names()))

    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.check(config)


def test_legacy_schema_is_stamped_without_data_loss(tmp_path):
    from app.persistence import Base
    from sqlalchemy import create_engine

    database_url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO users (id, username, password_hash, created_at) VALUES ('legacy', 'legacy-owner', 'argon2-hash', CURRENT_TIMESTAMP)"))
    engine.dispose()

    migrate_database(database_url)
    persistence = Persistence(database_url)
    assert persistence.db().execute(text("SELECT username FROM users WHERE id = 'legacy'")).scalar_one() == "legacy-owner"
    assert "alembic_version" in inspect(persistence.engine).get_table_names()
