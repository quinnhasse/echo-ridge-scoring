"""
Integration tests for auth endpoints and rate limiting.

Tests run against a TestClient backed by an in-memory SQLite database.
"""

import pytest

from tests.conftest import VALID_COMPANY


class TestRegistration:
    def test_register_new_user(self, client):
        resp = client.post("/auth/register", json={"email": "new@test.com", "password": "password123"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "new@test.com"
        assert "id" in data

    def test_register_duplicate_email(self, client, test_user):
        resp = client.post("/auth/register", json={"email": test_user.email, "password": "password123"})
        assert resp.status_code == 409

    def test_register_short_password(self, client):
        resp = client.post("/auth/register", json={"email": "short@test.com", "password": "abc"})
        assert resp.status_code == 422


class TestLogin:
    def test_login_returns_token(self, client, test_user):
        resp = client.post(
            "/auth/token",
            data={"username": test_user.email, "password": "testpass123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, test_user):
        resp = client.post(
            "/auth/token",
            data={"username": test_user.email, "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        resp = client.post(
            "/auth/token",
            data={"username": "nobody@example.com", "password": "whatever"},
        )
        assert resp.status_code == 401


class TestApiKeyManagement:
    def test_create_api_key(self, client, auth_headers):
        resp = client.post(
            "/auth/keys",
            json={"name": "my-key", "rate_limit_rpm": 30},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my-key"
        assert "key" in data  # raw key shown once
        assert len(data["key"]) > 20

    def test_create_key_requires_auth(self, client):
        resp = client.post("/auth/keys", json={"name": "k", "rate_limit_rpm": 10})
        assert resp.status_code == 401

    def test_list_api_keys(self, client, auth_headers, test_api_key):
        resp = client.get("/auth/keys", headers=auth_headers)
        assert resp.status_code == 200
        keys = resp.json()
        assert isinstance(keys, list)
        assert len(keys) >= 1

    def test_revoke_api_key(self, client, auth_headers, test_api_key):
        api_key, _ = test_api_key
        resp = client.delete(f"/auth/keys/{api_key.id}", headers=auth_headers)
        assert resp.status_code == 204

    def test_revoke_other_users_key_returns_404(self, client, auth_headers, db_session):
        from src.api.auth import hash_password
        from src.echo_ridge_scoring.db_models import ApiKey, User, generate_api_key

        other = User(email="other@test.com", hashed_password=hash_password("pass1234"))
        db_session.add(other)
        db_session.flush()
        raw, prefix, key_hash = generate_api_key()
        other_key = ApiKey(
            user_id=other.id, name="other-key", key_prefix=prefix,
            key_hash=key_hash, rate_limit_rpm=60,
        )
        db_session.add(other_key)
        db_session.commit()

        resp = client.delete(f"/auth/keys/{other_key.id}", headers=auth_headers)
        assert resp.status_code == 404


class TestScoringAuth:
    def test_score_without_auth_returns_401(self, client):
        resp = client.post("/score", json=VALID_COMPANY)
        assert resp.status_code == 401

    def test_score_with_bearer_token(self, client, auth_headers):
        resp = client.post("/score", json=VALID_COMPANY, headers=auth_headers)
        # 200 or 503 (if persistence not initialized) — not 401
        assert resp.status_code != 401

    def test_score_with_api_key(self, client, api_key_headers):
        resp = client.post("/score", json=VALID_COMPANY, headers=api_key_headers)
        assert resp.status_code != 401

    def test_healthz_no_auth_required(self, client):
        resp = client.get("/healthz")
        # Should not return 401 regardless of auth state
        assert resp.status_code != 401
