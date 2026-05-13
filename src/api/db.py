"""
Database session factory.

Reads DATABASE_URL from the environment; defaults to the local SQLite file
used in development without Docker.

Usage in FastAPI dependencies:
    session: Session = Depends(get_session)
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ..echo_ridge_scoring.persistence import Base  # noqa: F401 — registers ORM models
import src.echo_ridge_scoring.db_models  # noqa: F401 — registers User + ApiKey

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "sqlite:///echo_ridge_scoring.db"
)

# SQLite needs connect_args for thread safety in dev; Postgres does not.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session; close it after the request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def create_tables() -> None:
    """Create all tables (used in tests and seed script; Alembic handles prod)."""
    Base.metadata.create_all(bind=engine)
