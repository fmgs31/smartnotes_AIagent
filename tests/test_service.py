import pytest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.service import OrchestratorService
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
    svc = OrchestratorService(registry=registry, model="test-model", max_plan_steps=5)
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
    @patch("app.service.MCPClient")
    @patch("app.service.plan_next_step")
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

    @patch("app.service.MCPClient")
    @patch("app.service.plan_next_step")
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

    @patch("app.service.MCPClient")
    @patch("app.service.plan_next_step")
    async def test_planner_immediately_done_raises(self, mock_plan, mock_mcp_cls):
        svc, servers = _make_service()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.return_value = NextStepPlan(done=True, tool_name="")
        mock_mcp_cls.return_value = _mcp_mock()

        with pytest.raises(ValueError, match="did not produce"):
            await svc.handle_text(text="do something", token="tok")

    @patch("app.service.MCPClient")
    @patch("app.service.plan_next_step")
    async def test_loop_detection_raises(self, mock_plan, mock_mcp_cls):
        svc, servers = _make_service()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        same = NextStepPlan(done=False, tool_name="test-server.search", arguments={"q": "x"})
        mock_plan.side_effect = [same, same]
        mock_mcp_cls.return_value = _mcp_mock()

        with pytest.raises(ValueError, match="loop detected"):
            await svc.handle_text(text="loop me", token="tok")

    @patch("app.service.MCPClient")
    @patch("app.service.plan_next_step")
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
    @patch("app.service.MCPClient")
    async def test_discovers_and_caches_tools(self, mock_mcp_cls):
        svc, _ = _make_service()
        mock_mcp_cls.return_value = _mcp_mock(tools=[_tool("search"), _tool("create")])

        await svc._ensure_tools_discovered("tok")

        assert "test-server.search" in svc._tool_id_to_tool
        assert "test-server.create" in svc._tool_id_to_tool

    @patch("app.service.MCPClient")
    async def test_second_call_uses_cache(self, mock_mcp_cls):
        svc, servers = _make_service()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}

        await svc._ensure_tools_discovered("tok")

        mock_mcp_cls.assert_not_called()

    @patch("app.service.MCPClient")
    async def test_no_tools_after_discovery_raises(self, mock_mcp_cls):
        svc, _ = _make_service()
        mock_mcp_cls.return_value = _mcp_mock(tools=[])

        with pytest.raises(ValueError, match="No tools available"):
            await svc._ensure_tools_discovered("tok")

    @patch("app.service.MCPClient")
    async def test_allowed_tools_filter_is_applied(self, mock_mcp_cls):
        server = ServerConfig(id="s", url="http://s/mcp/", allowed_tools=["search"])
        registry = MagicMock(spec=ServerRegistry)
        registry.all.return_value = [server]
        svc = OrchestratorService(registry=registry, model="m", max_plan_steps=5)
        mock_mcp_cls.return_value = _mcp_mock(
            tools=[_tool("search"), _tool("create")]  # create should be filtered out
        )

        await svc._ensure_tools_discovered("tok")

        assert "s.search" in svc._tool_id_to_tool
        assert "s.create" not in svc._tool_id_to_tool



# ── Static helpers ────────────────────────────────────────────────────────────

class TestSummarizeMcpResult:
    def test_short_string_unchanged(self):
        assert OrchestratorService._summarize_mcp_result("hello") == "hello"

    def test_long_string_truncated_with_suffix(self):
        result = OrchestratorService._summarize_mcp_result("x" * 2000)
        assert result.endswith("...(truncated)")
        assert len(result) < 2000

    def test_none_returns_empty_string(self):
        assert OrchestratorService._summarize_mcp_result(None) == ""

    def test_non_string_is_converted(self):
        result = OrchestratorService._summarize_mcp_result({"key": "val"})
        assert isinstance(result, str)
        assert "key" in result


class TestExtractToolsWithSchema:
    def test_list_of_objects_with_input_schema(self):
        tools = [SimpleNamespace(name="search", inputSchema={"type": "object"})]
        result = OrchestratorService._extract_tools_with_schema(tools)
        assert "search" in result

    def test_dict_format_with_tools_key(self):
        result = OrchestratorService._extract_tools_with_schema(
            {"tools": [{"name": "search", "inputSchema": {"type": "object"}}]}
        )
        assert "search" in result

    def test_none_input_returns_empty_dict(self):
        assert OrchestratorService._extract_tools_with_schema(None) == {}

    def test_tool_without_name_is_skipped(self):
        tools = [SimpleNamespace(name=None, inputSchema={})]
        assert OrchestratorService._extract_tools_with_schema(tools) == {}

    def test_tool_with_empty_name_is_skipped(self):
        tools = [SimpleNamespace(name="", inputSchema={})]
        assert OrchestratorService._extract_tools_with_schema(tools) == {}

    def test_schema_falls_back_to_input_schema_variants(self):
        tool = SimpleNamespace(name="t", input_schema={"type": "object"})
        result = OrchestratorService._extract_tools_with_schema([tool])
        assert result["t"] == {"type": "object"}
