import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.main import app
from app.schemas import DriveItem

VALID_API_KEY = "test-app-api-key"
VALID_TOKEN = "ya29.valid-token"


@pytest.fixture
def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _success_stream():
    async def _gen(**kwargs):
        yield {"type": "step", "step": 1, "tool": "test-server.search"}
        yield {"type": "result", "summary": "Found 3 documents.", "items": [
            {"name": "My Doc", "id": "abc123", "type": "document", "location": "/"}
        ]}
    return _gen


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
        mock_svc.stream = _success_stream()

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find my docs", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        result = next(e for e in events if e["type"] == "result")
        assert result["summary"] == "Found 3 documents."
        assert result["items"][0]["id"] == "abc123"

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
    async def test_service_error_yields_error_event(self, mock_svc, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"

        async def _error_stream(**kwargs):
            yield {"type": "error", "detail": "planner produced no steps"}

        mock_svc.stream = _error_stream

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "do nothing", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert any(e["type"] == "error" for e in events)

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

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.svc")
    async def test_known_drive_items_as_list_accepted(self, mock_svc, mock_auth, client):
        mock_auth.return_value = "allowed@example.com"
        captured = {}

        async def _gen(**kwargs):
            captured.update(kwargs)
            yield {"type": "result", "summary": "", "items": []}

        mock_svc.stream = _gen

        async with client as c:
            resp = await c.post(
                "/run",
                json={
                    "text": "find docs",
                    "mcp_token": VALID_TOKEN,
                    "known_drive_items": [{"name": "Ideas", "id": "abc", "type": "folder", "location": "/"}],
                },
                headers={"X-API-Key": VALID_API_KEY},
            )

        assert resp.status_code == 200
        assert isinstance(captured["known_drive_items"], list)
        assert captured["known_drive_items"][0]["id"] == "abc"

    async def test_known_drive_items_as_string_returns_422(self, client):
        async with client as c:
            resp = await c.post(
                "/run",
                json={
                    "text": "find docs",
                    "mcp_token": VALID_TOKEN,
                    "known_drive_items": "not a list",
                },
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
        captured = {}

        async def _gen(**kwargs):
            captured.update(kwargs)
            yield {"type": "result", "summary": "Found 3 documents.", "items": []}

        mock_svc.stream = _gen
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
        events = _parse_sse(resp.text)
        assert any(e["type"] == "transcript" and e["text"] == "find my documents" for e in events)
        assert captured["text"] == "find my documents"

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
        mock_transcription.text = "   "
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
