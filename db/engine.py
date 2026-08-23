from pathlib import Path

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db.base import Base
from db.config import SCHEMA_VERSION, resolve_db_path
from db.models import SchemaMigration, utc_now_iso


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


def create_engine_for_path(path: Path | str | None = None) -> Engine:
    db_path = resolve_db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        _sqlite_url(db_path),
        connect_args={"check_same_thread": False, "timeout": 30},
    )


def _ensure_columns(engine: Engine) -> None:
    """Lightweight additive migrations for columns added after initial release."""
    from sqlalchemy import text

    required = {"index_runs": {"phase": "TEXT NOT NULL DEFAULT 'pending'"}}
    with engine.begin() as conn:
        for table, columns in required.items():
            existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def init_database(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _ensure_columns(engine)

    with Session(engine) as session:
        current_version = session.scalar(select(func.max(SchemaMigration.version))) or 0
        if current_version >= SCHEMA_VERSION:
            return
        session.add(
            SchemaMigration(version=SCHEMA_VERSION, applied_at=utc_now_iso())
        )
        session.commit()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autocommit=False, autoflush=True, expire_on_commit=False)
