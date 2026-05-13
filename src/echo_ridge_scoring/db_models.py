"""
SQLAlchemy ORM models for users and API keys.

Users authenticate with JWT tokens obtained via POST /auth/token.
Per-user API keys are hashed with SHA-256 before storage; only the
prefix (first 8 chars) is stored in plaintext for lookup.
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from .persistence import Base


class User(Base):
    """Registered user."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email!r})>"


class ApiKey(Base):
    """Per-user API key.

    The full key is generated once and returned to the user at creation time.
    Only the SHA-256 hash and an 8-char prefix (for O(1) lookup) are stored.
    """

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_prefix = Column(String(8), nullable=False, index=True)
    key_hash = Column(String(64), nullable=False)  # SHA-256 hex digest
    rate_limit_rpm = Column(Integer, nullable=False, default=60)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<ApiKey(id={self.id}, prefix={self.key_prefix!r}, user_id={self.user_id})>"


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        (full_key, prefix, sha256_hex) — full_key is shown to the user once;
        prefix and sha256_hex are stored in the database.
    """
    raw = secrets.token_urlsafe(32)
    prefix = raw[:8]
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return raw, prefix, digest


def hash_api_key(raw_key: str) -> str:
    """Return SHA-256 hex digest of a raw API key string."""
    return hashlib.sha256(raw_key.encode()).hexdigest()
