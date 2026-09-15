"""SQLite database setup (SQLAlchemy ORM, synchronous, no migrations).

The database file lives at the project root (documents.db by default)
and is git-ignored. Tests override get_db with an isolated tmp database.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create tables if they do not exist yet."""
    from backend import models  # noqa: F401  (register models on Base)

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator:
    """FastAPI dependency yielding a session, always closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
