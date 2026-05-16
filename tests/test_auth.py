"""
Unit tests for src/api/auth.py.

Tests JWT creation/decoding, password hashing, and the require_auth
dependency in isolation (no HTTP layer).
"""

import time

import pytest
from fastapi import HTTPException
from jose import jwt

from src.api.auth import (
    _ALGORITHM,
    _SECRET_KEY,
    create_access_token,
    hash_password,
    verify_password,
)


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

class TestPasswordHelpers:
    def test_hash_is_not_plain(self):
        hashed = hash_password("secret")
        assert hashed != "secret"

    def test_verify_correct_password(self):
        hashed = hash_password("correct")
        assert verify_password("correct", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_two_hashes_differ(self):
        # bcrypt uses a random salt
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

class TestJWT:
    def test_token_contains_expected_claims(self):
        token = create_access_token(user_id=42, email="u@example.com")
        claims = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        assert claims["sub"] == "42"
        assert claims["email"] == "u@example.com"
        assert "exp" in claims

    def test_token_expires_in_future(self):
        token = create_access_token(user_id=1, email="u@example.com")
        claims = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        assert claims["exp"] > time.time()

    def test_tampered_token_raises_401(self):
        token = create_access_token(user_id=1, email="u@example.com")
        bad_token = token[:-4] + "XXXX"
        from src.api.auth import _decode_token
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(bad_token)
        assert exc_info.value.status_code == 401

    def test_wrong_secret_raises_401(self):
        token = jwt.encode({"sub": "1", "email": "u@x.com"}, "wrong-secret", algorithm=_ALGORITHM)
        from src.api.auth import _decode_token
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(token)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# require_auth dependency
# ---------------------------------------------------------------------------

class TestRequireAuthDependency:
    def test_no_credentials_raises_401(self, db_session):
        """require_auth with no token and no API key must return 401."""
        from unittest.mock import MagicMock
        from src.api.auth import require_auth

        with pytest.raises(HTTPException) as exc_info:
            require_auth(bearer=None, x_api_key=None, session=db_session)
        assert exc_info.value.status_code == 401

    def test_valid_bearer_returns_user(self, db_session, test_user):
        from src.api.auth import require_auth

        token = create_access_token(user_id=test_user.id, email=test_user.email)
        user = require_auth(bearer=token, x_api_key=None, session=db_session)
        assert user.id == test_user.id

    def test_valid_api_key_returns_user(self, db_session, test_user, test_api_key):
        from src.api.auth import require_auth

        _, raw = test_api_key
        user = require_auth(bearer=None, x_api_key=raw, session=db_session)
        assert user.id == test_user.id

    def test_invalid_api_key_raises_401(self, db_session):
        from src.api.auth import require_auth

        with pytest.raises(HTTPException) as exc_info:
            require_auth(bearer=None, x_api_key="invalid-key-value", session=db_session)
        assert exc_info.value.status_code == 401
