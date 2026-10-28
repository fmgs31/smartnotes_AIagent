import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.mcp_client import MCPClient


@pytest.fixture(autouse=True)
def fast_retry(monkeypatch):
    """Zero wait times so retry tests complete instantly."""
    monkeypatch.setattr("app.mcp_client.settings.mcp_max_retries", 2)
    monkeypatch.setattr("app.mcp_client.settings.mcp_retry_initial_wait_s", 0.0)
    monkeypatch.setattr("app.mcp_client.settings.mcp_retry_max_wait_s", 0.0)
    monkeypatch.setattr("app.mcp_client.settings.mcp_timeout_s", 5.0)


def _mock_fastmcp_client(list_tools_result=None, call_tool_result="ok"):
    """Create a mock fastmcp Client that behaves as an async context manager."""
    mock = MagicMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.list_tools = AsyncMock(return_value=list_tools_result or [])
    mock.call_tool = AsyncMock(return_value=call_tool_result)
    return mock


class TestBuildTransport:
    """MCPClient._build_transport: wires StreamableHttpTransport with/without auth headers."""

    @patch("app.mcp_client.StreamableHttpTransport")
    def test_with_token_passes_auth_header(self, mock_transport):
        # A token must be passed as Bearer in the Authorization header.
        MCPClient("http://test/mcp/")._build_transport("ya29.abc")
        mock_transport.assert_called_once_with(
            url="http://test/mcp",
            headers={"Authorization": "Bearer ya29.abc"},
        )

    @patch("app.mcp_client.StreamableHttpTransport")
    def test_without_token_passes_no_headers(self, mock_transport):
        # No token → headers must be None, not {}, not an empty dict.
        MCPClient("http://test/mcp/")._build_transport(None)
        mock_transport.assert_called_once_with(url="http://test/mcp", headers=None)

    def test_trailing_slash_stripped_from_url(self):
        # fastmcp 0.x chokes on trailing slashes — MCPClient normalises them away.
        url = MCPClient("http://test/mcp/").server_url
        assert url == "http://test/mcp", f"Expected trailing '/' to be stripped, got {url!r}"

    def test_url_without_trailing_slash_unchanged(self):
        url = MCPClient("http://test/mcp").server_url
        assert url == "http://test/mcp", f"URL without trailing slash should be unchanged, got {url!r}"


class TestOpenSession:
    """MCPClient._open_session: session lifecycle + token-binding + retry on connection errors."""

    @patch("app.mcp_client.Client")
    async def test_success_sets_client_and_token(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")

        await client._open_session("tok")

        assert client._client is not None, "Expected _client to be set after successful open"
        assert client._session_token == "tok", (
            f"Expected _session_token='tok', got {client._session_token!r}"
        )
        assert client._exit_stack is not None, "Expected _exit_stack to be set after successful open"

    @patch("app.mcp_client.Client")
    async def test_same_token_is_idempotent(self, mock_cls):
        # Opening an already-open session with the same token is a no-op.
        instance = _mock_fastmcp_client()
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        await client._open_session("tok")
        await client._open_session("tok")  # second call should be no-op

        assert instance.__aenter__.call_count == 1, (
            f"Expected __aenter__ called once (idempotent), got {instance.__aenter__.call_count}"
        )

    @patch("app.mcp_client.Client")
    async def test_different_token_when_open_raises(self, mock_cls):
        # Token-binding guard: trying to swap tokens mid-session must fail loudly.
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")
        await client._open_session("token-a")

        with pytest.raises(ValueError, match="already bound"):
            await client._open_session("token-b")

    @patch("app.mcp_client.Client")
    async def test_retries_on_connection_error(self, mock_cls):
        # First __aenter__ fails with ConnectionError, second succeeds → retry works.
        instance = _mock_fastmcp_client()
        instance.__aenter__ = AsyncMock(
            side_effect=[ConnectionError("cold start"), instance]
        )
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        await client._open_session("tok")

        assert instance.__aenter__.call_count == 2, (
            f"Expected 2 __aenter__ calls (1 fail + 1 success), got {instance.__aenter__.call_count}"
        )
        assert client._client is not None, "Expected _client to be set after successful retry"

    @patch("app.mcp_client.Client")
    async def test_raises_underlying_error_when_all_attempts_fail(self, mock_cls):
        # All attempts fail → tenacity is configured with reraise=True so the
        # original exception surfaces unchanged (no RetryError wrapper).
        instance = _mock_fastmcp_client()
        instance.__aenter__ = AsyncMock(side_effect=ConnectionError("always down"))
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        with pytest.raises(ConnectionError, match="always down"):
            await client._open_session("tok")

        # mcp_max_retries=2 → 3 total attempts
        assert instance.__aenter__.call_count == 3, (
            f"Expected 3 __aenter__ attempts (mcp_max_retries=2), got {instance.__aenter__.call_count}"
        )


class TestCloseSession:
    """MCPClient._close_session: releases all session state."""

    @patch("app.mcp_client.Client")
    async def test_close_clears_all_state(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")
        await client._open_session("tok")

        await client._close_session()

        assert client._client is None, f"Expected _client=None after close, got {client._client!r}"
        assert client._session_token is None, (
            f"Expected _session_token=None after close, got {client._session_token!r}"
        )
        assert client._exit_stack is None, (
            f"Expected _exit_stack=None after close, got {client._exit_stack!r}"
        )

    async def test_close_when_not_open_is_noop(self):
        # Calling close on a never-opened client must not raise.
        client = MCPClient("http://test/mcp/")
        await client._close_session()  # should not raise


class TestSessionContextManager:
    """MCPClient.session: async context manager that opens on enter and always closes on exit."""

    @patch("app.mcp_client.Client")
    async def test_opens_and_closes_around_block(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            assert client._client is not None, "Expected _client to be set inside session block"

        assert client._client is None, (
            f"Expected _client=None after session block exits cleanly, got {client._client!r}"
        )

    @patch("app.mcp_client.Client")
    async def test_closes_even_when_body_raises(self, mock_cls):
        # Exception inside the block must still close the session — no leaked state.
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")

        with pytest.raises(RuntimeError):
            async with client.session(token="tok"):
                raise RuntimeError("boom")

        assert client._client is None, (
            f"Expected _client=None after session block raised, got {client._client!r}"
        )


class TestRequireClient:
    """MCPClient._require_client: enforces 'must be in an open session with matching token'."""

    async def test_raises_when_no_session_open(self):
        client = MCPClient("http://test/mcp/")
        with pytest.raises(RuntimeError, match="No MCP session"):
            client._require_client("tok")

    @patch("app.mcp_client.Client")
    async def test_raises_on_token_mismatch(self, mock_cls):
        # Session bound to token-a; asking for token-b must raise.
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")
        await client._open_session("token-a")

        with pytest.raises(ValueError, match="mismatch"):
            client._require_client("token-b")

    @patch("app.mcp_client.Client")
    async def test_returns_client_when_session_open(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")
        await client._open_session("tok")

        result = client._require_client("tok")
        assert result is client._client, (
            f"Expected _require_client to return the same object as self._client. "
            f"Got {result!r} vs {client._client!r}"
        )


class TestListTools:
    """MCPClient.list_tools: delegates to the active fastmcp session with token validation."""

    @patch("app.mcp_client.Client")
    async def test_returns_tools_from_active_session(self, mock_cls):
        fake_tools = [MagicMock()]
        instance = _mock_fastmcp_client(list_tools_result=fake_tools)
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.list_tools(token="tok")

        assert result is fake_tools, (
            f"Expected the exact tools object from list_tools(). Got {result!r} vs {fake_tools!r}"
        )

    @patch("app.mcp_client.Client")
    async def test_token_mismatch_raises(self, mock_cls):
        # Session open with token-a; list_tools(token-b) must raise — never leak tools across tokens.
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")
        await client._open_session("token-a")

        with pytest.raises(ValueError, match="mismatch"):
            await client.list_tools(token="token-b")

    async def test_raises_without_session(self):
        client = MCPClient("http://test/mcp/")
        with pytest.raises(RuntimeError, match="No MCP session"):
            await client.list_tools(token="tok")


class TestCallTool:
    """MCPClient.call_tool: invokes a tool within the active session, retrying on transient errors."""

    @patch("app.mcp_client.Client")
    async def test_returns_result_from_active_session(self, mock_cls):
        instance = _mock_fastmcp_client(call_tool_result="search results")
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.call_tool(
                token="tok", tool_name="search", arguments={"q": "hello"}
            )

        assert result == "search results", (
            f"Expected tool result 'search results', got {result!r}"
        )

    async def test_raises_without_session(self):
        client = MCPClient("http://test/mcp/")
        with pytest.raises(RuntimeError, match="No MCP session"):
            await client.call_tool(token="tok", tool_name="search", arguments={})

    @patch("app.mcp_client.Client")
    async def test_passes_tool_name_and_arguments(self, mock_cls):
        # The raw tool name + arguments dict must be forwarded to fastmcp unchanged.
        instance = _mock_fastmcp_client()
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            await client.call_tool(token="tok", tool_name="docs_create", arguments={"x": 1})

        instance.call_tool.assert_called_once_with(name="docs_create", arguments={"x": 1})

    @patch("app.mcp_client.Client")
    async def test_retries_on_transient_error_and_succeeds(self, mock_cls):
        # First call raises ConnectionError, second succeeds → retry logic recovers.
        instance = _mock_fastmcp_client(call_tool_result="ok")
        instance.call_tool = AsyncMock(side_effect=[ConnectionError("blip"), "ok"])
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.call_tool(token="tok", tool_name="search", arguments={})

        assert result == "ok", f"Expected 'ok' after retry, got {result!r}"
        assert instance.call_tool.call_count == 2, (
            f"Expected 2 call_tool invocations (1 fail + 1 success), got {instance.call_tool.call_count}"
        )

    @patch("app.mcp_client.Client")
    async def test_raises_underlying_error_when_all_attempts_fail(self, mock_cls):
        instance = _mock_fastmcp_client()
        instance.call_tool = AsyncMock(side_effect=ConnectionError("always down"))
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            with pytest.raises(ConnectionError, match="always down"):
                await client.call_tool(token="tok", tool_name="search", arguments={})

        # mcp_max_retries=2 → 3 total attempts
        assert instance.call_tool.call_count == 3, (
            f"Expected 3 call_tool attempts (mcp_max_retries=2), got {instance.call_tool.call_count}"
        )

    @patch("app.mcp_client.Client")
    async def test_tool_error_result_returned_without_retry(self, mock_cls):
        """isError=True is a successful HTTP response — call_tool returns it as-is, no retry."""
        tool_error_result = MagicMock()
        tool_error_result.isError = True

        instance = _mock_fastmcp_client(call_tool_result=tool_error_result)
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.call_tool(token="tok", tool_name="insert_doc_elements", arguments={})

        assert result is tool_error_result, (
            f"Expected the isError=True result passed through unchanged. Got {result!r}"
        )
        assert instance.call_tool.call_count == 1, (
            f"Expected no retry for isError=True result (it's a 200 OK response), "
            f"got {instance.call_tool.call_count} attempts"
        )


# google_workspace_mcp raises on API errors, so an upstream HttpError reaches the client
# as an exception carrying the stringified error rather than as a string result.
DOCS_400 = (
    "Error in on_call_tool middleware: <HttpError 400 when requesting "
    'https://docs.googleapis.com/v1/documents/abc:batchUpdate returned "Invalid '
    "requests[0].insertText: Index 14272 must be less than the end index of the "
    'referenced segment, 14272.">'
)


class TestRetryClassification:
    """Only failures that can succeed on a second attempt are retried."""

    @patch("app.mcp_client.Client")
    async def test_permanent_4xx_is_not_retried(self, mock_cls):
        # A 4xx from the upstream Google API is deterministic: the same arguments always
        # produce the same rejection, so backoff only delays the refiner that repairs it.
        instance = _mock_fastmcp_client()
        instance.call_tool = AsyncMock(side_effect=RuntimeError(DOCS_400))
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            with pytest.raises(RuntimeError, match="must be less than the end index"):
                await client.call_tool(
                    token="tok", tool_name="modify_doc_text", arguments={"start_index": 14272}
                )

        assert instance.call_tool.call_count == 1, (
            f"Expected a deterministic 400 to fail on the first attempt, got "
            f"{instance.call_tool.call_count} attempts"
        )

    @patch("app.mcp_client.Client")
    async def test_400_mentioning_429_is_not_retried(self, mock_cls):
        # Only the status code counts as a rate limit, not any "429" in the message.
        instance = _mock_fastmcp_client()
        instance.call_tool = AsyncMock(
            side_effect=RuntimeError("<HttpError 400 when requesting ... returned Index 14290 is invalid>")
        )
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            with pytest.raises(RuntimeError):
                await client.call_tool(token="tok", tool_name="modify_doc_text", arguments={})

        assert instance.call_tool.call_count == 1, (
            f"Expected a 400 whose message contains '429' to fail on the first attempt, got "
            f"{instance.call_tool.call_count} attempts"
        )

    @pytest.mark.parametrize(
        "label,message",
        [
            ("rate limit", "<HttpError 429 when requesting ... returned rateLimitExceeded>"),
            ("backend error", "<HttpError 503 when requesting ... returned backendError>"),
            # Drive signals per-user quota with 403, not 429
            ("drive 403 rate limit", "<HttpError 403 when requesting ... returned userRateLimitExceeded>"),
        ],
    )
    @patch("app.mcp_client.Client")
    async def test_retryable_status_codes_still_retry(self, mock_cls, label, message):
        instance = _mock_fastmcp_client()
        instance.call_tool = AsyncMock(side_effect=RuntimeError(message))
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            with pytest.raises(RuntimeError):
                await client.call_tool(token="tok", tool_name="search", arguments={})

        # mcp_max_retries=2 → 3 total attempts
        assert instance.call_tool.call_count == 3, (
            f"Expected a {label} to keep its 3 attempts, got {instance.call_tool.call_count}"
        )

    @patch("app.mcp_client.Client")
    async def test_permanent_4xx_recovers_via_retry_when_transient_first(self, mock_cls):
        # A transient failure followed by success must still recover — the predicate
        # narrows what is retried, it does not disable retrying.
        instance = _mock_fastmcp_client()
        instance.call_tool = AsyncMock(side_effect=[TimeoutError("cold start"), "ok"])
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.call_tool(token="tok", tool_name="search", arguments={})

        assert result == "ok", f"Expected 'ok' after a transient failure, got {result!r}"
