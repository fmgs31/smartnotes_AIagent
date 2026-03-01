import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.main import app
from app.schemas import ExecutedToolStep, ProcedureSummary, DriveItem

VALID_API_KEY = "test-app-api-key"
VALID_TOKEN = "ya29.valid-token"


@pytest.fixture
def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _success_response():
    step = ExecutedToolStep(
        step=1,
        server_id="test-server",
        tool_name="test-server.search",
        arguments={"q": "test"},
        mcp_result="3 documents found",
    )
    procedure = ProcedureSummary(
        text="Found 3 documents.",
        items=[DriveItem(name="My Doc", id="abc123", type="document", location="/")],
    )
    return procedure, [step]


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    async def test_returns_200_ok(self, client):
        async with client as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


# ── /run ──────────────────────────────────────────────────────────────────────

class TestRunEndpoint:
    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.svc")
    async def test_valid_request_returns_200(self, mock_svc, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"
        mock_svc.handle_text = AsyncMock(return_value=_success_response())

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find my docs", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == "Found 3 documents."
        assert data["items"][0]["id"] == "abc123"

    async def test_missing_api_key_returns_401(self, client):
        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": VALID_TOKEN},
            )
        assert resp.status_code == 401

    async def test_wrong_api_key_returns_401(self, client):
        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": "not-the-right-key"},
            )
        assert resp.status_code == 401

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    async def test_unauthorized_user_returns_403(self, mock_auth, client):
        mock_auth.side_effect = HTTPException(status_code=403, detail="Not in allowlist")

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 403

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    async def test_expired_token_returns_401(self, mock_auth, client):
        mock_auth.side_effect = HTTPException(
            status_code=401, detail="Invalid or expired GCP token"
        )

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": "expired"},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 401

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.svc")
    async def test_service_value_error_returns_400(self, mock_svc, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"
        mock_svc.handle_text = AsyncMock(side_effect=ValueError("planner produced no steps"))

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "do nothing", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 400

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.svc")
    async def test_service_runtime_error_returns_500(self, mock_svc, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"
        mock_svc.handle_text = AsyncMock(side_effect=RuntimeError("unexpected crash"))

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "crash me", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 500

    async def test_missing_text_field_returns_422(self, client):
        async with client as c:
            resp = await c.post(
                "/run",
                json={"mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 422

    async def test_empty_text_returns_422(self, client):
        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 422


# ── /run-audio ────────────────────────────────────────────────────────────────

class TestRunAudioEndpoint:
    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.svc")
    @patch("app.main.openai_client")
    async def test_valid_audio_transcribed_and_processed(
        self, mock_openai, mock_svc, mock_auth, client
    ):
        mock_auth.return_value = "allowed@example.com"
        mock_svc.handle_text = AsyncMock(return_value=_success_response())
        mock_transcription = MagicMock()
        mock_transcription.text = "find my documents"
        mock_openai.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        async with client as c:
            resp = await c.post(
                "/run-audio",
                data={"mcp_token": VALID_TOKEN},
                files={"audio": ("test.webm", b"fake audio bytes", "audio/webm")},
                headers={"X-API-Key": VALID_API_KEY},
            )

        assert resp.status_code == 200
        mock_svc.handle_text.assert_called_once()
        call_kwargs = mock_svc.handle_text.call_args.kwargs
        assert call_kwargs["text"] == "find my documents"

    async def test_missing_api_key_returns_401(self, client):
        async with client as c:
            resp = await c.post(
                "/run-audio",
                data={"mcp_token": VALID_TOKEN},
                files={"audio": ("test.webm", b"audio", "audio/webm")},
            )
        assert resp.status_code == 401

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    async def test_empty_audio_file_returns_400(self, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"

        async with client as c:
            resp = await c.post(
                "/run-audio",
                data={"mcp_token": VALID_TOKEN},
                files={"audio": ("test.webm", b"", "audio/webm")},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 400

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.openai_client")
    async def test_empty_transcription_returns_400(self, mock_openai, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"
        mock_transcription = MagicMock()
        mock_transcription.text = "   "  # whitespace-only
        mock_openai.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        async with client as c:
            resp = await c.post(
                "/run-audio",
                data={"mcp_token": VALID_TOKEN},
                files={"audio": ("test.webm", b"some audio", "audio/webm")},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 400

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.openai_client")
    async def test_transcription_failure_returns_500(self, mock_openai, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"
        mock_openai.audio.transcriptions.create = AsyncMock(
            side_effect=RuntimeError("Whisper API down")
        )

        async with client as c:
            resp = await c.post(
                "/run-audio",
                data={"mcp_token": VALID_TOKEN},
                files={"audio": ("test.webm", b"some audio", "audio/webm")},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 500
