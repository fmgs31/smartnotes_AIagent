import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tenacity import RetryError

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
    @patch("app.mcp_client.StreamableHttpTransport")
    def test_with_token_passes_auth_header(self, mock_transport):
        MCPClient("http://test/mcp/")._build_transport("ya29.abc")
        mock_transport.assert_called_once_with(
            url="http://test/mcp",
            headers={"Authorization": "Bearer ya29.abc"},
        )

    @patch("app.mcp_client.StreamableHttpTransport")
    def test_without_token_passes_no_headers(self, mock_transport):
        MCPClient("http://test/mcp/")._build_transport(None)
        mock_transport.assert_called_once_with(url="http://test/mcp", headers=None)

    def test_trailing_slash_stripped_from_url(self):
        assert MCPClient("http://test/mcp/").server_url == "http://test/mcp"

    def test_url_without_trailing_slash_unchanged(self):
        assert MCPClient("http://test/mcp").server_url == "http://test/mcp"


class TestOpenSession:
    @patch("app.mcp_client.Client")
    async def test_success_sets_client_and_token(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")

        await client._open_session("tok")

        assert client._client is not None
        assert client._session_token == "tok"
        assert client._exit_stack is not None

    @patch("app.mcp_client.Client")
    async def test_same_token_is_idempotent(self, mock_cls):
        instance = _mock_fastmcp_client()
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        await client._open_session("tok")
        await client._open_session("tok")  # second call should be no-op

        assert instance.__aenter__.call_count == 1

    @patch("app.mcp_client.Client")
    async def test_different_token_when_open_raises(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")
        await client._open_session("token-a")

        with pytest.raises(ValueError, match="already bound"):
            await client._open_session("token-b")

    @patch("app.mcp_client.Client")
    async def test_retries_on_connection_error(self, mock_cls):
        instance = _mock_fastmcp_client()
        instance.__aenter__ = AsyncMock(
            side_effect=[ConnectionError("cold start"), instance]
        )
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        await client._open_session("tok")

        assert instance.__aenter__.call_count == 2
        assert client._client is not None

    @patch("app.mcp_client.Client")
    async def test_raises_retry_error_when_all_attempts_fail(self, mock_cls):
        instance = _mock_fastmcp_client()
        instance.__aenter__ = AsyncMock(side_effect=ConnectionError("always down"))
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        with pytest.raises(RetryError):
            await client._open_session("tok")

        # mcp_max_retries=2 → 3 total attempts
        assert instance.__aenter__.call_count == 3


class TestCloseSession:
    @patch("app.mcp_client.Client")
    async def test_close_clears_all_state(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")
        await client._open_session("tok")

        await client._close_session()

        assert client._client is None
        assert client._session_token is None
        assert client._exit_stack is None

    async def test_close_when_not_open_is_noop(self):
        client = MCPClient("http://test/mcp/")
        await client._close_session()  # should not raise


class TestSessionContextManager:
    @patch("app.mcp_client.Client")
    async def test_opens_and_closes_around_block(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            assert client._client is not None

        assert client._client is None  # cleaned up

    @patch("app.mcp_client.Client")
    async def test_closes_even_when_body_raises(self, mock_cls):
        mock_cls.return_value = _mock_fastmcp_client()
        client = MCPClient("http://test/mcp/")

        with pytest.raises(RuntimeError):
            async with client.session(token="tok"):
                raise RuntimeError("boom")

        assert client._client is None  # still cleaned up


class TestRequireClient:
    async def test_raises_when_no_session_open(self):
        client = MCPClient("http://test/mcp/")
        with pytest.raises(RuntimeError, match="No MCP session"):
            client._require_client("tok")

    @patch("app.mcp_client.Client")
    async def test_raises_on_token_mismatch(self, mock_cls):
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
        assert result is client._client


class TestListTools:
    @patch("app.mcp_client.Client")
    async def test_returns_tools_from_active_session(self, mock_cls):
        fake_tools = [MagicMock()]
        instance = _mock_fastmcp_client(list_tools_result=fake_tools)
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.list_tools(token="tok")

        assert result is fake_tools

    @patch("app.mcp_client.Client")
    async def test_token_mismatch_raises(self, mock_cls):
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
    @patch("app.mcp_client.Client")
    async def test_returns_result_from_active_session(self, mock_cls):
        instance = _mock_fastmcp_client(call_tool_result="search results")
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.call_tool(
                token="tok", tool_name="search", arguments={"q": "hello"}
            )

        assert result == "search results"

    async def test_raises_without_session(self):
        client = MCPClient("http://test/mcp/")
        with pytest.raises(RuntimeError, match="No MCP session"):
            await client.call_tool(token="tok", tool_name="search", arguments={})

    @patch("app.mcp_client.Client")
    async def test_passes_tool_name_and_arguments(self, mock_cls):
        instance = _mock_fastmcp_client()
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            await client.call_tool(token="tok", tool_name="docs_create", arguments={"x": 1})

        instance.call_tool.assert_called_once_with(name="docs_create", arguments={"x": 1})

    @patch("app.mcp_client.Client")
    async def test_retries_on_transient_error_and_succeeds(self, mock_cls):
        instance = _mock_fastmcp_client(call_tool_result="ok")
        instance.call_tool = AsyncMock(side_effect=[ConnectionError("blip"), "ok"])
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            result = await client.call_tool(token="tok", tool_name="search", arguments={})

        assert result == "ok"
        assert instance.call_tool.call_count == 2

    @patch("app.mcp_client.Client")
    async def test_raises_retry_error_when_all_attempts_fail(self, mock_cls):
        instance = _mock_fastmcp_client()
        instance.call_tool = AsyncMock(side_effect=ConnectionError("always down"))
        mock_cls.return_value = instance
        client = MCPClient("http://test/mcp/")

        async with client.session(token="tok"):
            with pytest.raises(RetryError):
                await client.call_tool(token="tok", tool_name="search", arguments={})

        # mcp_max_retries=2 → 3 total attempts
        assert instance.call_tool.call_count == 3

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

        assert result is tool_error_result
        assert instance.call_tool.call_count == 1
