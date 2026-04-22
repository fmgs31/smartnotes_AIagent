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
        yield {"type": "step_planned", "tool_name": "test-server.search", "next_tool_explanation": None, "arguments": {}}
        yield {"type": "result", "text": "Found 3 documents.", "items": [
            {"name": "My Doc", "id": "abc123", "type": "document", "location": "/"}
        ]}
    return _gen


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    """/health: simple liveness probe — always returns {"ok": true}."""

    async def test_returns_200_ok(self, client):
        async with client as c:
            resp = await c.get("/health")
        assert resp.status_code == 200, (
            f"Expected /health → 200, got {resp.status_code}: {resp.text!r}"
        )
        body = resp.json()
        assert body == {"ok": True}, f"Expected body {{'ok': True}}, got {body!r}"


# ── /run ──────────────────────────────────────────────────────────────────────

class TestRunEndpoint:
    """/run: authenticated streaming endpoint that runs the agent on a text request."""

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.agent")
    async def test_valid_request_returns_200(self, mock_agent, mock_auth, client):
        # Happy path: valid API key + authorized user → 200 + SSE stream with agent events.
        mock_auth.return_value = "allowed@example.com"
        mock_agent.stream = _success_stream()

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find my docs", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )

        assert resp.status_code == 200, (
            f"Expected /run → 200, got {resp.status_code}: {resp.text!r}"
        )
        events = _parse_sse(resp.text)
        result = next(e for e in events if e["type"] == "result")
        assert result["text"] == "Found 3 documents.", (
            f"Expected result.text='Found 3 documents.', got {result.get('text')!r}"
        )
        assert result["items"][0]["id"] == "abc123", (
            f"Expected first item id='abc123', got {result['items'][0].get('id')!r}"
        )

    async def test_missing_api_key_returns_401(self, client):
        # No X-API-Key header → 401.
        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": VALID_TOKEN},
            )
        assert resp.status_code == 401, (
            f"Expected 401 with no X-API-Key header, got {resp.status_code}: {resp.text!r}"
        )

    async def test_wrong_api_key_returns_401(self, client):
        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": "not-the-right-key"},
            )
        assert resp.status_code == 401, (
            f"Expected 401 for an invalid X-API-Key, got {resp.status_code}: {resp.text!r}"
        )

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    async def test_unauthorized_user_returns_403(self, mock_auth, client):
        # Auth raises 403 → endpoint propagates as 403 (user is not in the allowlist).
        mock_auth.side_effect = HTTPException(status_code=403, detail="Not in allowlist")

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 403, (
            f"Expected 403 when the user is not in the allowlist, got {resp.status_code}: {resp.text!r}"
        )

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    async def test_expired_token_returns_401(self, mock_auth, client):
        # Auth raises 401 (expired/invalid Google token) → endpoint propagates as 401.
        mock_auth.side_effect = HTTPException(
            status_code=401, detail="Invalid or expired GCP token"
        )

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "find docs", "mcp_token": "expired"},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 401, (
            f"Expected 401 for an expired GCP token, got {resp.status_code}: {resp.text!r}"
        )

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.agent")
    async def test_service_error_yields_error_event(self, mock_agent, mock_auth, client):
        # Agent errors are surfaced as SSE 'error' events, not HTTP errors (the stream itself still 200s).
        mock_auth.return_value = "allowed@example.com"

        async def _error_stream(**kwargs):
            yield {"type": "error", "detail": "planner produced no steps"}

        mock_agent.stream = _error_stream

        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "do nothing", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )

        assert resp.status_code == 200, (
            f"Expected 200 (error is inside the SSE stream), got {resp.status_code}: {resp.text!r}"
        )
        events = _parse_sse(resp.text)
        assert any(e["type"] == "error" for e in events), (
            f"Expected at least one 'error' SSE event, got events={events!r}"
        )

    async def test_missing_text_field_returns_422(self, client):
        # Pydantic 422 for missing required field.
        async with client as c:
            resp = await c.post(
                "/run",
                json={"mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for missing 'text' field, got {resp.status_code}: {resp.text!r}"
        )

    async def test_empty_text_returns_422(self, client):
        # Empty 'text' fails validation (min length check), not reaching the agent.
        async with client as c:
            resp = await c.post(
                "/run",
                json={"text": "", "mcp_token": VALID_TOKEN},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for empty 'text' (min_length=1), got {resp.status_code}: {resp.text!r}"
        )


# ── /run-audio ────────────────────────────────────────────────────────────────

class TestRunAudioEndpoint:
    """/run-audio: same as /run but transcribes the uploaded audio with Whisper first."""

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.agent")
    @patch("app.main.openai_client")
    async def test_valid_audio_transcribed_and_processed(
        self, mock_openai, mock_agent, mock_auth, client
    ):
        # Whisper returns 'find my documents' → a 'transcript' event is emitted AND
        # the transcribed text is handed to agent.stream as 'text'.
        mock_auth.return_value = "allowed@example.com"
        captured = {}

        async def _gen(**kwargs):
            captured.update(kwargs)
            yield {"type": "result", "summary": "Found 3 documents.", "items": []}

        mock_agent.stream = _gen
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

        assert resp.status_code == 200, (
            f"Expected /run-audio → 200, got {resp.status_code}: {resp.text!r}"
        )
        events = _parse_sse(resp.text)
        assert any(e["type"] == "transcript" and e["text"] == "find my documents" for e in events), (
            f"Expected a 'transcript' event with text='find my documents'. Events={events!r}"
        )
        assert captured["text"] == "find my documents", (
            f"Expected agent.stream to receive text='find my documents', got {captured.get('text')!r}"
        )

    async def test_missing_api_key_returns_401(self, client):
        async with client as c:
            resp = await c.post(
                "/run-audio",
                data={"mcp_token": VALID_TOKEN},
                files={"audio": ("test.webm", b"audio", "audio/webm")},
            )
        assert resp.status_code == 401, (
            f"Expected 401 with no X-API-Key header, got {resp.status_code}: {resp.text!r}"
        )

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    async def test_empty_audio_file_returns_400(self, mock_auth, client):
        # Zero-byte audio upload → 400 before we even call Whisper.
        mock_auth.return_value = "allowed@example.com"

        async with client as c:
            resp = await c.post(
                "/run-audio",
                data={"mcp_token": VALID_TOKEN},
                files={"audio": ("test.webm", b"", "audio/webm")},
                headers={"X-API-Key": VALID_API_KEY},
            )
        assert resp.status_code == 400, (
            f"Expected 400 for an empty audio upload, got {resp.status_code}: {resp.text!r}"
        )

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.openai_client")
    async def test_empty_transcription_returns_400(self, mock_openai, mock_auth, client):
        # Whisper returned only whitespace (silent recording) → 400 because there's nothing to run.
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
        assert resp.status_code == 400, (
            f"Expected 400 when Whisper returns a whitespace-only transcription, "
            f"got {resp.status_code}: {resp.text!r}"
        )

    @patch("app.main.auth.require_authorized_user", new_callable=AsyncMock)
    @patch("app.main.openai_client")
    async def test_transcription_failure_returns_500(self, mock_openai, mock_auth, client):
        # Whisper itself errors out → upstream failure → 500.
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
        assert resp.status_code == 500, (
            f"Expected 500 when Whisper raises, got {resp.status_code}: {resp.text!r}"
        )
