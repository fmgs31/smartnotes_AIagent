import pytest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent import Agent
from app.server_registry import ServerConfig, ServerRegistry
from app.schemas import NextStepPlan


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_registry(*server_ids: str):
    servers = [ServerConfig(id=sid, url=f"http://{sid}/mcp/") for sid in server_ids]
    registry = MagicMock(spec=ServerRegistry)
    registry.all.return_value = servers
    return registry, servers


def _make_service(*server_ids: str):
    registry, servers = _make_registry(*server_ids or ("test-server",))
    svc = Agent(registry=registry, model="test-model", max_plan_steps=5)
    return svc, servers


def _mcp_mock(tools=(), call_result="tool output"):
    """Mock MCPClient whose session() is a no-op async context manager."""
    mock = MagicMock()

    @asynccontextmanager
    async def fake_session(**kwargs):
        yield

    mock.session = fake_session
    mock.list_tools = AsyncMock(return_value=list(tools))
    mock.call_tool = AsyncMock(return_value=call_result)
    return mock


def _tool(name: str):
    return SimpleNamespace(name=name, inputSchema={"type": "object"})


# ── handle_text ───────────────────────────────────────────────────────────────

class TestHandleText:
    @patch("app.agent.MCPClient")
    @patch("app.agent.plan_next_step")
    async def test_single_step_returns_result(self, mock_plan, mock_mcp_cls):
        svc, servers = _make_service()
        server = servers[0]
        mock_mcp_cls.return_value = _mcp_mock(call_result="3 documents found")

        svc._tool_id_to_tool = {"test-server.search": (server, "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}

        mock_plan.side_effect = [
            NextStepPlan(done=False, tool_name="test-server.search", arguments={"q": "test"}),
            NextStepPlan(done=True, tool_name=""),
        ]

        sid, tool_call, result, steps = await svc.handle_text(text="find docs", token="tok")

        assert sid == "test-server"
        assert tool_call.tool_name == "test-server.search"
        assert result == "3 documents found"
        assert len(steps) == 1

    @patch("app.agent.MCPClient")
    @patch("app.agent.plan_next_step")
    async def test_multi_step_returns_last_result(self, mock_plan, mock_mcp_cls):
        svc, servers = _make_service()
        server = servers[0]
        mock_mcp_cls.return_value = _mcp_mock(call_result="second result")

        svc._tool_id_to_tool = {
            "test-server.search": (server, "search"),
            "test-server.create": (server, "create"),
        }
        svc._tool_id_to_schema = {"test-server.search": {}, "test-server.create": {}}

        mock_plan.side_effect = [
            NextStepPlan(done=False, tool_name="test-server.search", arguments={"q": "a"}),
            NextStepPlan(done=False, tool_name="test-server.create", arguments={"title": "b"}),
            NextStepPlan(done=True, tool_name=""),
        ]

        _, _, result, steps = await svc.handle_text(text="search and create", token="tok")

        assert len(steps) == 2
        assert result == "second result"

    @patch("app.agent.MCPClient")
    @patch("app.agent.plan_next_step")
    async def test_planner_immediately_done_raises(self, mock_plan, mock_mcp_cls):
        svc, servers = _make_service()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.return_value = NextStepPlan(done=True, tool_name="")
        mock_mcp_cls.return_value = _mcp_mock()

        with pytest.raises(ValueError, match="did not produce"):
            await svc.handle_text(text="do something", token="tok")

    @patch("app.agent.MCPClient")
    @patch("app.agent.plan_next_step")
    async def test_loop_detection_raises(self, mock_plan, mock_mcp_cls):
        svc, servers = _make_service()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        same = NextStepPlan(done=False, tool_name="test-server.search", arguments={"q": "x"})
        mock_plan.side_effect = [same, same]
        mock_mcp_cls.return_value = _mcp_mock()

        with pytest.raises(ValueError, match="loop detected"):
            await svc.handle_text(text="loop me", token="tok")

    @patch("app.agent.MCPClient")
    @patch("app.agent.plan_next_step")
    async def test_unknown_tool_name_raises(self, mock_plan, mock_mcp_cls):
        svc, servers = _make_service()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.return_value = NextStepPlan(
            done=False, tool_name="test-server.nonexistent", arguments={}
        )
        mock_mcp_cls.return_value = _mcp_mock()

        with pytest.raises(ValueError, match="not available"):
            await svc.handle_text(text="call missing tool", token="tok")


# ── _ensure_tools_discovered ──────────────────────────────────────────────────

class TestEnsureToolsDiscovered:
    async def test_discovers_and_caches_tools(self):
        svc, servers = _make_service()
        mock_client = _mcp_mock(tools=[_tool("search"), _tool("create")])
        preopened = {servers[0].id: mock_client}

        await svc._ensure_tools_discovered("tok", preopened_clients=preopened)

        assert "test-server.search" in svc._tool_id_to_tool
        assert "test-server.create" in svc._tool_id_to_tool

    async def test_second_call_uses_cache(self):
        svc, servers = _make_service()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_client = _mcp_mock()

        await svc._ensure_tools_discovered("tok", preopened_clients={"test-server": mock_client})

        mock_client.list_tools.assert_not_called()

    async def test_no_tools_after_discovery_raises(self):
        svc, servers = _make_service()
        mock_client = _mcp_mock(tools=[])
        preopened = {servers[0].id: mock_client}

        with pytest.raises(ValueError, match="No tools available"):
            await svc._ensure_tools_discovered("tok", preopened_clients=preopened)

    async def test_allowed_tools_filter_is_applied(self):
        server = ServerConfig(id="s", url="http://s/mcp/", allowed_tools=["search"])
        registry = MagicMock(spec=ServerRegistry)
        registry.all.return_value = [server]
        svc = Agent(registry=registry, model="m", max_plan_steps=5)
        mock_client = _mcp_mock(tools=[_tool("search"), _tool("create")])  # create should be filtered out
        preopened = {"s": mock_client}

        await svc._ensure_tools_discovered("tok", preopened_clients=preopened)

        assert "s.search" in svc._tool_id_to_tool
        assert "s.create" not in svc._tool_id_to_tool



# ── Static helpers ────────────────────────────────────────────────────────────

def _make_call_result(*texts: str):
    """Minimal CallToolResult-like object with TextContent items."""
    parts = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(content=parts)


LIMIT = Agent._SUMMARY_LIMIT
MARKER = Agent._NEXT_PAGE_TOKEN_MARKER


class TestSummarizeMcpResult:
    # ── basic cases ────────────────────────────────────────────────────────────

    def test_none_returns_empty(self):
        assert Agent._summarize_mcp_result(None) == ""

    def test_short_string_returned_unchanged(self):
        assert Agent._summarize_mcp_result("hello") == "hello"

    def test_string_exactly_at_limit_returned_unchanged(self):
        text = "x" * LIMIT
        assert Agent._summarize_mcp_result(text) == text

    def test_long_string_truncated(self):
        text = "x" * (LIMIT + 100)
        result = Agent._summarize_mcp_result(text)
        assert result == "x" * LIMIT + "...(truncated)"

    def test_non_string_is_converted_via_str(self):
        result = Agent._summarize_mcp_result({"key": "val"})
        assert isinstance(result, str)
        assert "key" in result

    # ── CallToolResult extraction ──────────────────────────────────────────────

    def test_calltoolresult_text_is_extracted(self):
        obj = _make_call_result("item 1\nitem 2")
        assert Agent._summarize_mcp_result(obj) == "item 1\nitem 2"

    def test_calltoolresult_multiple_parts_are_joined(self):
        obj = _make_call_result("part A", "part B")
        assert Agent._summarize_mcp_result(obj) == "part A\npart B"

    def test_calltoolresult_empty_content_falls_back_to_str(self):
        obj = SimpleNamespace(content=[])
        result = Agent._summarize_mcp_result(obj)
        assert isinstance(result, str)

    # ── nextPageToken preservation ─────────────────────────────────────────────

    def test_token_at_end_appended_after_truncation(self):
        # Token is beyond the limit — must be preserved.
        token_line = MARKER + "~!!~MYTOKEN"
        text = "a" * (LIMIT + 100) + "\n" + token_line
        result = Agent._summarize_mcp_result(text)
        assert result.endswith("...(truncated)\n" + token_line)

    def test_token_at_beginning_already_included_no_duplicate(self):
        # Token is within the first LIMIT chars — no need to append.
        token_line = MARKER + "~!!~MYTOKEN"
        text = token_line + "\n" + "a" * (LIMIT + 100)
        result = Agent._summarize_mcp_result(text)
        assert token_line in result[:LIMIT]
        assert result.count(token_line) == 1  # not duplicated

    def test_token_near_boundary_full_line_appended(self):
        # Token starts just inside the limit but the value extends beyond it.
        long_token = "~!!~" + "T" * 200
        token_line = MARKER + long_token
        # Place the marker so it begins 10 chars before the limit.
        padding = "a" * (LIMIT - len(MARKER) - 10)
        text = padding + token_line + "\nextra stuff"
        result = Agent._summarize_mcp_result(text)
        assert result.endswith("...(truncated)\n" + token_line)

    def test_no_token_no_suffix(self):
        text = "a" * (LIMIT + 100)
        result = Agent._summarize_mcp_result(text)
        assert result == "a" * LIMIT + "...(truncated)"

    def test_calltoolresult_long_text_token_at_end_preserved(self):
        token_line = MARKER + "~!!~BIGTOKEN"
        long_text = "item\n" * 1100 + token_line  # >5000 chars
        obj = _make_call_result(long_text)
        result = Agent._summarize_mcp_result(obj)
        assert result.endswith("...(truncated)\n" + token_line)


class TestExtractToolsWithSchema:
    def test_list_of_objects_with_input_schema(self):
        tools = [SimpleNamespace(name="search", inputSchema={"type": "object"})]
        result = Agent._extract_tools_with_schema(tools)
        assert "search" in result

    def test_dict_format_with_tools_key(self):
        result = Agent._extract_tools_with_schema(
            {"tools": [{"name": "search", "inputSchema": {"type": "object"}}]}
        )
        assert "search" in result

    def test_none_input_returns_empty_dict(self):
        assert Agent._extract_tools_with_schema(None) == {}

    def test_tool_without_name_is_skipped(self):
        tools = [SimpleNamespace(name=None, inputSchema={})]
        assert Agent._extract_tools_with_schema(tools) == {}

    def test_tool_with_empty_name_is_skipped(self):
        tools = [SimpleNamespace(name="", inputSchema={})]
        assert Agent._extract_tools_with_schema(tools) == {}

    def test_schema_falls_back_to_input_schema_variants(self):
        tool = SimpleNamespace(name="t", input_schema={"type": "object"})
        result = Agent._extract_tools_with_schema([tool])
        assert result["t"] == {"type": "object"}
