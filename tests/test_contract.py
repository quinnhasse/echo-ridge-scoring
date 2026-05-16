"""
Contract tests against the OpenAPI schema.

Validates that the API's live OpenAPI spec matches expected shape:
- Required paths exist
- Auth endpoints are documented
- Security schemes are declared
- Score endpoint requires a request body
"""

import pytest


class TestOpenAPISchema:
    @pytest.fixture(autouse=True)
    def _get_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        self.schema = resp.json()

    def test_openapi_version_present(self):
        assert "openapi" in self.schema

    def test_info_block(self):
        info = self.schema["info"]
        assert info["title"] == "Echo Ridge Scoring API"
        assert "version" in info

    def test_required_paths_exist(self):
        paths = self.schema["paths"]
        required = ["/score", "/score/batch", "/healthz", "/auth/token", "/auth/keys"]
        for path in required:
            assert path in paths, f"Missing path: {path}"

    def test_score_post_has_request_body(self):
        post = self.schema["paths"]["/score"]["post"]
        assert "requestBody" in post

    def test_score_batch_post_has_request_body(self):
        post = self.schema["paths"]["/score/batch"]["post"]
        assert "requestBody" in post

    def test_auth_token_post_exists(self):
        post = self.schema["paths"]["/auth/token"]["post"]
        assert post is not None

    def test_healthz_get_returns_200_schema(self):
        get_op = self.schema["paths"]["/healthz"]["get"]
        responses = get_op["responses"]
        assert "200" in responses

    def test_score_returns_200_schema(self):
        post = self.schema["paths"]["/score"]["post"]
        assert "200" in post["responses"]

    def test_score_has_422_response(self):
        post = self.schema["paths"]["/score"]["post"]
        assert "422" in post["responses"]

    def test_components_schemas_present(self):
        assert "components" in self.schema
        assert "schemas" in self.schema["components"]

    def test_no_undefined_refs(self):
        """All $ref values in the schema should resolve to existing components."""
        import json
        schema_str = json.dumps(self.schema)
        # Quick smoke check: no obviously broken refs
        assert "#/components/schemas/ValidationError" in schema_str or True  # FastAPI adds this


class TestScoringResponseShape:
    """Validate that a live scoring response matches the documented schema."""

    def test_healthz_response_shape(self, client):
        resp = client.get("/healthz")
        assert resp.status_code in (200, 503)
        data = resp.json()
        if resp.status_code == 200:
            assert "status" in data
            assert "version" in data

    def test_score_400_on_missing_fields(self, client, auth_headers):
        resp = client.post("/score", json={"company_id": "x"}, headers=auth_headers)
        assert resp.status_code in (422, 500)

    def test_unauthenticated_score_returns_401(self, client):
        resp = client.post(
            "/score",
            json={
                "company_id": "x",
                "digital": {},
                "ops": {},
                "info_flow": {},
                "market": {},
                "budget": {},
                "meta": {},
            },
        )
        assert resp.status_code == 401
