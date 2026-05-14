"""
Database session factory.

Reads DATABASE_URL from the environment; defaults to SQLite for local dev
without Docker.  In Docker, DATABASE_URL is set to the Postgres DSN.

Usage in FastAPI dependencies:
    session: Session = Depends(get_session)
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..echo_ridge_scoring.persistence import Base  # noqa: F401 — registers ORM models
import src.echo_ridge_scoring.db_models  # noqa: F401 — registers User + ApiKey

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "sqlite:///echo_ridge_scoring.db"
)

# SQLite needs check_same_thread=False; Postgres does not.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# pool_pre_ping keeps Postgres connections healthy after idle time.
engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session; close it after the request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def create_tables() -> None:
    """Create all tables from ORM metadata.

    Used by the seed script and tests; Alembic handles production migrations.
    """
    Base.metadata.create_all(bind=engine)
