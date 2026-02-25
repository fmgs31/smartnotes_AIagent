import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.auth import require_api_key, require_authorized_user


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
    def test_correct_key_passes(self):
        # Should not raise
        require_api_key("test-app-api-key")

    def test_wrong_key_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            require_api_key("wrong-key")
        assert exc.value.status_code == 401

    def test_none_key_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            require_api_key(None)
        assert exc.value.status_code == 401


class TestRequireAuthorizedUser:
    @pytest.fixture(autouse=True)
    def patch_allowed_emails(self, monkeypatch):
        monkeypatch.setattr("app.auth._allowed_emails", {"allowed@example.com"})

    async def test_valid_token_allowed_email_returns_email(self):
        mock_http = _mock_http_client(200, {"email": "allowed@example.com"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            result = await require_authorized_user("ya29.valid")
        assert result == "allowed@example.com"

    async def test_email_comparison_is_case_insensitive(self):
        mock_http = _mock_http_client(200, {"email": "ALLOWED@EXAMPLE.COM"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            result = await require_authorized_user("ya29.valid")
        assert result == "allowed@example.com"

    async def test_invalid_token_google_returns_400_raises_401(self):
        mock_http = _mock_http_client(
            400, {"error": "invalid_token", "error_description": "Token has expired"}
        )
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await require_authorized_user("expired-token")
        assert exc.value.status_code == 401
        assert "Token has expired" in exc.value.detail

    async def test_token_without_error_description_still_raises_401(self):
        mock_http = _mock_http_client(401, {"error": "unauthorized"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await require_authorized_user("bad-token")
        assert exc.value.status_code == 401

    async def test_valid_token_email_not_in_allowlist_raises_403(self):
        mock_http = _mock_http_client(200, {"email": "stranger@example.com"})
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await require_authorized_user("ya29.valid")
        assert exc.value.status_code == 403
        assert "stranger@example.com" in exc.value.detail

    async def test_token_response_missing_email_raises_401(self):
        mock_http = _mock_http_client(200, {"sub": "12345"})  # no email field
        with patch("app.auth.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(HTTPException) as exc:
                await require_authorized_user("ya29.valid")
        assert exc.value.status_code == 401
        assert "email" in exc.value.detail
