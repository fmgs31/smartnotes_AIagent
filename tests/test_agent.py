import pytest
from contextlib import asynccontextmanager, AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent import Agent, _RequestContext
from app.server_registry import ServerConfig, ServerRegistry
from app.schemas import EventType, NextStepPlan, FullTaskSummary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_registry(*server_ids: str):
    servers = [ServerConfig(id=sid, url=f"http://{sid}/mcp/") for sid in server_ids]
    registry = MagicMock(spec=ServerRegistry)
    registry.all.return_value = servers
    return registry, servers


async def _echo_summary(*, done_plan, **_kw):
    """Mirror Planner.summarize's no-summarizer-prompt branch."""
    return FullTaskSummary(text=done_plan.full_task_summary, items=done_plan.items)


def _make_agent(*server_ids: str, planner=None):
    registry, servers = _make_registry(*server_ids or ("test-server",))
    if planner is None:
        planner = AsyncMock()
    # Default summarize behavior so tests that only mock plan_next_step keep working.
    # Tests that need v4-style summarization can override planner.summarize themselves.
    planner.summarize.side_effect = _echo_summary
    svc = Agent(registry=registry, planner=planner, max_plan_steps=5)
    return svc, servers


def _mcp_mock(tools=(), call_result="tool output"):
    """Mock MCPClient whose session() is a no-op async context manager."""
    mock = MagicMock()

    @asynccontextmanager
    async def fake_session(**kwargs):
        yield mock

    mock.session = fake_session
    mock.list_tools = AsyncMock(return_value=list(tools))
    mock.call_tool = AsyncMock(return_value=call_result)
    return mock


def _tool(name: str):
    return SimpleNamespace(name=name, inputSchema={"type": "object"})


# ── execute ───────────────────────────────────────────────────────────────

class TestHandleText:
    @patch("app.agent.MCPClient")
    async def test_single_step_returns_result(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        server = servers[0]
        mock_mcp_cls.return_value = _mcp_mock(call_result="3 documents found")

        svc._tool_id_to_tool = {"test-server.search": (server, "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}

        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "test"}),
            NextStepPlan(done=True, full_task_summary="Found 3 documents."),
        ]

        procedure, steps = await svc.execute(text="find docs", token="tok")

        assert steps[-1].tool_name == "test-server.search"
        assert isinstance(procedure, FullTaskSummary)
        assert procedure.text == "Found 3 documents."
        assert len(steps) == 1

    @patch("app.agent.MCPClient")
    async def test_multi_step_returns_last_result(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        server = servers[0]
        mock_mcp_cls.return_value = _mcp_mock(call_result="second result")

        svc._tool_id_to_tool = {
            "test-server.search": (server, "search"),
            "test-server.create": (server, "create"),
        }
        svc._tool_id_to_schema = {"test-server.search": {}, "test-server.create": {}}

        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "a"}),
            NextStepPlan(done=False, next_tool="test-server.create", arguments={"title": "b"}),
            NextStepPlan(done=True, full_task_summary="Searched and created."),
        ]

        procedure, steps = await svc.execute(text="search and create", token="tok")

        assert len(steps) == 2
        assert isinstance(procedure, FullTaskSummary)
        assert procedure.text == "Searched and created."

    @patch("app.agent.MCPClient")
    async def test_planner_immediately_done_returns_summary(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.return_value = NextStepPlan(done=True, full_task_summary="Nothing to do.")
        mock_mcp_cls.return_value = _mcp_mock()

        result, steps = await svc.execute(text="do something", token="tok")
        assert result.text == "Nothing to do."
        assert steps == []

    @patch("app.agent.MCPClient")
    async def test_loop_detection_records_error_and_continues(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        same = NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"})
        mock_plan.plan_next_step.side_effect = [same, same, NextStepPlan(done=True, full_task_summary="partial")]
        mock_mcp_cls.return_value = _mcp_mock()

        procedure, steps = await svc.execute(text="loop me", token="tok")

        assert len(steps) == 2
        assert "[Agent error]" in steps[1].mcp_result

    @patch("app.agent.MCPClient")
    async def test_unknown_tool_records_error_and_continues(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.nonexistent", arguments={}),
            NextStepPlan(done=True, full_task_summary="failed"),
        ]
        mock_mcp_cls.return_value = _mcp_mock()

        procedure, steps = await svc.execute(text="call missing tool", token="tok")

        assert len(steps) == 1
        assert "[Agent error]" in steps[0].mcp_result

    @patch("app.agent.MCPClient")
    async def test_tool_call_failure_records_error_and_continues(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {
            "test-server.search": (servers[0], "search"),
            "test-server.create": (servers[0], "create"),
        }
        svc._tool_id_to_schema = {"test-server.search": {}, "test-server.create": {}}
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"}),
            NextStepPlan(done=False, next_tool="test-server.create", arguments={"title": "y"}),
            NextStepPlan(done=True, full_task_summary="partial"),
        ]
        failing_mock = _mcp_mock()
        failing_mock.call_tool = AsyncMock(side_effect=[RuntimeError("MCP server error"), "created"])
        mock_mcp_cls.return_value = failing_mock

        procedure, steps = await svc.execute(text="search then create", token="tok")

        assert len(steps) == 2
        assert "[Tool error]" in steps[0].mcp_result
        assert steps[1].mcp_result == "created"

    @patch("app.agent.MCPClient")
    async def test_tools_are_passed_to_planner(self, mock_mcp_cls):
        """load_tools must populate allowed_tools and tool_schemas before plan_next_step is called."""
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        server = servers[0]
        mock_mcp_cls.return_value = _mcp_mock(call_result="ok")

        svc._tool_id_to_tool = {"test-server.search": (server, "search")}
        svc._tool_id_to_schema = {"test-server.search": {"type": "object", "properties": {}}}

        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={}),
            NextStepPlan(done=True, full_task_summary="done"),
        ]

        await svc.execute(text="find docs", token="tok")

        call_kwargs = mock_plan.plan_next_step.call_args_list[0][1]
        assert call_kwargs["allowed_tools"] == ["test-server.search"], \
            "allowed_tools was empty — load_tools may not have been called"
        assert call_kwargs["tool_schemas"] == {"test-server.search": {"type": "object", "properties": {}}}, \
            "tool_schemas was empty — load_tools may not have been called"

    @patch("app.agent.MCPClient")
    async def test_summarize_called_with_done_plan_and_executed_steps(self, mock_mcp_cls):
        """The agent must hand done_plan plus the full execution context to planner.summarize."""
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        done = NextStepPlan(done=True, full_task_summary="done")
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "abc"}),
            done,
        ]
        mock_mcp_cls.return_value = _mcp_mock(call_result="abc result")

        await svc.execute(
            text="find abc",
            token="tok",
            user_guidelines="be terse",
            known_drive_items=[{"name": "X", "id": "id1", "type": "document", "location": "/"}],
            planner_model="some-model",
        )

        mock_plan.summarize.assert_awaited_once()
        kwargs = mock_plan.summarize.await_args.kwargs
        assert kwargs["text"] == "find abc"
        assert kwargs["done_plan"] is done
        assert kwargs["user_guidelines"] == "be terse"
        assert kwargs["planner_model"] == "some-model"
        assert kwargs["known_drive_items"] == [
            {"name": "X", "id": "id1", "type": "document", "location": "/"}
        ]
        # executed_steps reflects the one tool call before done=true
        executed = kwargs["executed_steps"]
        assert len(executed) == 1
        assert executed[0]["next_tool"] == "test-server.search"
        assert executed[0]["arguments"] == {"q": "abc"}
        assert "abc result" in executed[0]["result_summary"]

    @patch("app.agent.MCPClient")
    async def test_split_summarizer_overrides_done_plan_summary(self, mock_mcp_cls):
        """v4 path: when summarize() returns its own FullTaskSummary, the agent uses it
        instead of the (empty) summary fields the lean-schema planner emits."""
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        # Lean-schema planner: done=true with empty summary fields (defaults).
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"}),
            NextStepPlan(done=True),
        ]
        # Summarizer call produces the real summary from a separate LLM call.
        from app.schemas import DriveItem
        summarizer_output = FullTaskSummary(
            text="Found 1 result for 'x'.",
            items=[DriveItem(name="result.doc", id="real-id-42", type="document", location="/")],
        )
        mock_plan.summarize.side_effect = None
        mock_plan.summarize.return_value = summarizer_output
        mock_mcp_cls.return_value = _mcp_mock(call_result="x result")

        result, _steps = await svc.execute(text="search x", token="tok")

        assert result is summarizer_output
        assert result.text == "Found 1 result for 'x'."
        assert result.items[0].id == "real-id-42"

    @patch("app.agent.MCPClient")
    async def test_summarize_not_called_when_max_steps_exhausted(self, mock_mcp_cls):
        """No done_plan → no summarize call; agent falls back to an empty summary."""
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": str(i)})
            for i in range(svc.max_plan_steps)
        ]
        mock_mcp_cls.return_value = _mcp_mock(call_result="result")

        result, _steps = await svc.execute(text="loop", token="tok")

        mock_plan.summarize.assert_not_awaited()
        assert result.text == ""
        assert result.items == []

    @patch("app.agent.MCPClient")
    async def test_max_steps_exhausted_returns_all_steps(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        # Always returns a different tool call (different args) — never done
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": str(i)})
            for i in range(svc.max_plan_steps)
        ]
        mock_mcp_cls.return_value = _mcp_mock(call_result="result")

        procedure, steps = await svc.execute(text="keep going", token="tok")

        assert len(steps) == svc.max_plan_steps
        assert procedure.text == ""


# ── _ensure_tools_discovered ──────────────────────────────────────────────────

def _make_request_ctxt(clients: dict, token: str = "tok", text: str = "") -> _RequestContext:
    return _RequestContext(stack=AsyncExitStack(), token=token, text=text, clients=clients)


class TestEnsureToolsDiscovered:
    async def test_discovers_and_caches_tools(self):
        svc, servers = _make_agent()
        mock_client = _mcp_mock(tools=[_tool("search"), _tool("create")])

        await svc._ensure_tools_discovered(_make_request_ctxt({servers[0].id: mock_client}))

        assert "test-server.search" in svc._tool_id_to_tool
        assert "test-server.create" in svc._tool_id_to_tool

    async def test_second_call_uses_cache(self):
        svc, servers = _make_agent()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_client = _mcp_mock()

        await svc._ensure_tools_discovered(_make_request_ctxt({"test-server": mock_client}))

        mock_client.list_tools.assert_not_called()

    async def test_no_tools_after_discovery_raises(self):
        svc, servers = _make_agent()
        mock_client = _mcp_mock(tools=[])

        with pytest.raises(ValueError, match="No tools available"):
            await svc._ensure_tools_discovered(_make_request_ctxt({servers[0].id: mock_client}))

    async def test_allowed_tools_filter_is_applied(self):
        server = ServerConfig(id="s", url="http://s/mcp/", allowed_tools=["search"])
        registry = MagicMock(spec=ServerRegistry)
        registry.all.return_value = [server]
        svc = Agent(registry=registry, planner=AsyncMock(), max_plan_steps=5)
        mock_client = _mcp_mock(tools=[_tool("search"), _tool("create")])  # create should be filtered out

        await svc._ensure_tools_discovered(_make_request_ctxt({"s": mock_client}))

        assert "s.search" in svc._tool_id_to_tool
        assert "s.create" not in svc._tool_id_to_tool



# ── Static helpers ────────────────────────────────────────────────────────────

def _make_call_result(*texts: str):
    """Minimal CallToolResult-like object with TextContent items."""
    parts = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(content=parts)


LIMIT = Agent._SUMMARY_LIMIT
# MARKER = Agent._NEXT_PAGE_TOKEN_MARKER


class TestExtractResultText:
    def test_none_returns_empty(self):
        assert Agent._extract_result_text(None) == ""

    def test_string_passthrough(self):
        assert Agent._extract_result_text("hello") == "hello"

    def test_non_string_falls_back_to_str(self):
        result = Agent._extract_result_text({"key": "val"})
        assert isinstance(result, str)
        assert "key" in result

    def test_structured_content_preferred(self):
        obj = SimpleNamespace(
            structured_content={"result": "from structured"},
            content=[SimpleNamespace(text="from legacy")],
        )
        assert Agent._extract_result_text(obj) == "from structured"

    def test_structured_content_non_string_result_falls_through_to_content(self):
        obj = SimpleNamespace(
            structured_content={"result": 42},
            content=[SimpleNamespace(text="from legacy")],
        )
        assert Agent._extract_result_text(obj) == "from legacy"

    def test_structured_content_missing_result_key_falls_through_to_content(self):
        obj = SimpleNamespace(
            structured_content={"other": "data"},
            content=[SimpleNamespace(text="from legacy")],
        )
        assert Agent._extract_result_text(obj) == "from legacy"

    def test_calltoolresult_text_is_extracted(self):
        obj = _make_call_result("item 1\nitem 2")
        assert Agent._extract_result_text(obj) == "item 1\nitem 2"

    def test_calltoolresult_multiple_parts_are_joined(self):
        obj = _make_call_result("part A", "part B")
        assert Agent._extract_result_text(obj) == "part A\npart B"

    def test_calltoolresult_empty_content_falls_back_to_str(self):
        obj = SimpleNamespace(content=[])
        result = Agent._extract_result_text(obj)
        assert isinstance(result, str)

    def test_tool_error_prefixed(self):
        obj = SimpleNamespace(isError=True, content=[SimpleNamespace(text="Unsupported element type")])
        assert Agent._extract_result_text(obj) == "[Tool error] Unsupported element type"

    def test_tool_error_empty_content_uses_fallback(self):
        obj = SimpleNamespace(isError=True, content=[])
        assert Agent._extract_result_text(obj) == "[Tool error] MCP server returned a ToolError"


class TestCapResult:
    def test_empty_string_returned_unchanged(self):
        assert Agent._cap_result("") == ""

    def test_short_string_returned_unchanged(self):
        assert Agent._cap_result("hello") == "hello"

    def test_string_exactly_at_limit_returned_unchanged(self):
        text = "x" * LIMIT
        assert Agent._cap_result(text) == text

    def test_long_string_truncated(self):
        text = "x" * (LIMIT + 100)
        result = Agent._cap_result(text)
        assert result == "x" * LIMIT + "...(truncated)"

    def test_calltoolresult_long_text_truncated(self):
        long_text = "item\n" * 1100  # >5000 chars
        obj = _make_call_result(long_text)
        result = Agent._cap_result(Agent._extract_result_text(obj))
        assert result.endswith("...(truncated)")


# ── nextPageToken preservation ─────────────────────────────────────────────

    # def test_token_at_end_appended_after_truncation(self):
    #     # Token is beyond the limit — must be preserved.
    #     token_line = MARKER + "~!!~MYTOKEN"
    #     text = "a" * (LIMIT + 100) + "\n" + token_line
    #     result = Agent._summarize_mcp_result(text)
    #     assert result.endswith("...(truncated)\n" + token_line)

    # def test_token_at_beginning_already_included_no_duplicate(self):
    #     # Token is within the first LIMIT chars — no need to append.
    #     token_line = MARKER + "~!!~MYTOKEN"
    #     text = token_line + "\n" + "a" * (LIMIT + 100)
    #     result = Agent._summarize_mcp_result(text)
    #     assert token_line in result[:LIMIT]
    #     assert result.count(token_line) == 1  # not duplicated

    # def test_token_near_boundary_full_line_appended(self):
    #     # Token starts just inside the limit but the value extends beyond it.
    #     long_token = "~!!~" + "T" * 200
    #     token_line = MARKER + long_token
    #     # Place the marker so it begins 10 chars before the limit.
    #     padding = "a" * (LIMIT - len(MARKER) - 10)
    #     text = padding + token_line + "\nextra stuff"
    #     result = Agent._summarize_mcp_result(text)
    #     assert result.endswith("...(truncated)\n" + token_line)

    # def test_no_token_no_suffix(self):
    #     text = "a" * (LIMIT + 100)
    #     result = Agent._summarize_mcp_result(text)
    #     assert result == "a" * LIMIT + "...(truncated)"

    # def test_calltoolresult_long_text_token_at_end_preserved(self):
    #     token_line = MARKER + "~!!~BIGTOKEN"
    #     long_text = "item\n" * 1100 + token_line  # >5000 chars
    #     obj = _make_call_result(long_text)
    #     result = Agent._summarize_mcp_result(obj)
    #     assert result.endswith("...(truncated)\n" + token_line)


class TestStream:
    @patch("app.agent.MCPClient")
    async def test_planner_exception_yields_error_event(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.side_effect = RuntimeError("planner exploded")
        mock_mcp_cls.return_value = _mcp_mock()

        events = [e async for e in svc.stream(text="do something", token="tok")]

        assert len(events) == 1
        assert events[0]["type"] == EventType.ERROR
        assert "planner exploded" in events[0]["detail"]

    @patch("app.agent.MCPClient")
    async def test_done_plan_summary_in_result_event(self, mock_mcp_cls):
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"}),
            NextStepPlan(done=True, full_task_summary="Found results."),
        ]
        mock_mcp_cls.return_value = _mcp_mock()

        events = [e async for e in svc.stream(text="do something", token="tok")]

        result_events = [e for e in events if e["type"] == EventType.RESULT]
        assert len(result_events) == 1
        assert result_events[0]["text"] == "Found results."


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
