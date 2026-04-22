import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.auth import Auth


def _mock_http_client(status: int, body: dict):
    """Build a mock httpx.AsyncClient that returns a fixed response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = body

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=mock_resp)
    return mock_http


class TestRequireApiKey:
    """Auth.require_api_key: constant-time comparison of the app-level API key header."""

    def setup_method(self):
        self.auth = Auth(api_key="test-app-api-key", allowed_emails=set())

    def test_correct_key_passes(self):
        # Matching key must not raise.
        self.auth.require_api_key("test-app-api-key")

    def test_wrong_key_raises_401(self):
        # Non-matching key must raise HTTP 401.
        with pytest.raises(HTTPException) as exc:
            self.auth.require_api_key("wrong-key")
        assert exc.value.status_code == 401, (
            f"Expected HTTP 401 for a wrong API key, got {exc.value.status_code}: {exc.value.detail!r}"
        )

    def test_none_key_raises_401(self):
        # A missing header (None) is treated the same as a wrong key: 401.
        with pytest.raises(HTTPException) as exc:
            self.auth.require_api_key(None)
        assert exc.value.status_code == 401, (
            f"Expected HTTP 401 for a missing API key, got {exc.value.status_code}: {exc.value.detail!r}"
        )


class TestRequireAuthorizedUser:
    """Auth.require_authorized_user: validates a Google OAuth token and enforces the email allowlist."""

    def setup_method(self):
        self.auth = Auth(api_key="", allowed_emails={"allowed@example.com"})

    async def test_valid_token_allowed_email_returns_email(self):
        # Happy path: Google returns 200 + allowed email → the email is returned verbatim.
        mock_http = _mock_http_client(200, {"email": "allowed@example.com"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            result = await self.auth.require_authorized_user("ya29.valid")
        assert result == "allowed@example.com", (
            f"Expected 'allowed@example.com', got {result!r}"
        )

    async def test_email_comparison_is_case_insensitive(self):
        # Google may return the address in mixed case — the allowlist check must be case-insensitive,
        # and the returned value must be normalised to lowercase.
        mock_http = _mock_http_client(200, {"email": "ALLOWED@EXAMPLE.COM"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            result = await self.auth.require_authorized_user("ya29.valid")
        assert result == "allowed@example.com", (
            f"Expected lowercased 'allowed@example.com', got {result!r}"
        )

    async def test_invalid_token_google_returns_400_raises_401(self):
        # Google returns 400 invalid_token → map to HTTP 401 and include Google's error_description
        # in the detail so clients can show it.
        mock_http = _mock_http_client(
            400, {"error": "invalid_token", "error_description": "Token has expired"}
        )
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await self.auth.require_authorized_user("expired-token")
        assert exc.value.status_code == 401, (
            f"Expected HTTP 401 when Google returns 400 invalid_token, "
            f"got {exc.value.status_code}: {exc.value.detail!r}"
        )
        assert "Token has expired" in exc.value.detail, (
            f"Expected Google's error_description in detail, got {exc.value.detail!r}"
        )

    async def test_token_without_error_description_still_raises_401(self):
        # Google's error body is missing error_description → still 401, just without the message.
        mock_http = _mock_http_client(401, {"error": "unauthorized"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await self.auth.require_authorized_user("bad-token")
        assert exc.value.status_code == 401, (
            f"Expected HTTP 401 even without error_description, "
            f"got {exc.value.status_code}: {exc.value.detail!r}"
        )

    async def test_valid_token_email_not_in_allowlist_raises_403(self):
        # Token is valid (Google 200) but the email is not allowlisted → HTTP 403, and the
        # detail must include the rejected email for debugging.
        mock_http = _mock_http_client(200, {"email": "stranger@example.com"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await self.auth.require_authorized_user("ya29.valid")
        assert exc.value.status_code == 403, (
            f"Expected HTTP 403 for a non-allowlisted email, "
            f"got {exc.value.status_code}: {exc.value.detail!r}"
        )
        assert "stranger@example.com" in exc.value.detail, (
            f"Expected rejected email in detail, got {exc.value.detail!r}"
        )

    async def test_token_response_missing_email_raises_401(self):
        # Token response contains no 'email' claim → 401 with a detail that mentions 'email'.
        mock_http = _mock_http_client(200, {"sub": "12345"})  # no email field
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await self.auth.require_authorized_user("ya29.valid")
        assert exc.value.status_code == 401, (
            f"Expected HTTP 401 when tokeninfo response has no 'email' field, "
            f"got {exc.value.status_code}: {exc.value.detail!r}"
        )
        assert "email" in exc.value.detail, (
            f"Expected 'email' mentioned in detail, got {exc.value.detail!r}"
        )
