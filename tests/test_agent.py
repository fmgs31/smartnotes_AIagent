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
    """Agent.execute: planner + MCP loop behaviour from a single call to done=true."""

    @patch("app.agent.MCPClient")
    async def test_single_step_returns_result(self, mock_mcp_cls):
        # One plan step that invokes a tool, then done=true → exactly one step executed
        # and the FullTaskSummary reflects the done_plan text.
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

        assert steps[-1].tool_name == "test-server.search", (
            f"Expected last step to invoke 'test-server.search', got {steps[-1].tool_name!r}"
        )
        assert isinstance(procedure, FullTaskSummary), (
            f"Expected FullTaskSummary, got {type(procedure).__name__}"
        )
        assert procedure.text == "Found 3 documents.", (
            f"Summary text mismatch: {procedure.text!r}"
        )
        assert len(steps) == 1, f"Expected 1 executed step, got {len(steps)}: {steps}"

    @patch("app.agent.MCPClient")
    async def test_multi_step_returns_last_result(self, mock_mcp_cls):
        # Two plan steps (search then create) followed by done=true → both recorded
        # as executed steps and the summary comes from the done_plan.
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

        assert len(steps) == 2, f"Expected 2 executed steps, got {len(steps)}: {steps}"
        assert isinstance(procedure, FullTaskSummary), (
            f"Expected FullTaskSummary, got {type(procedure).__name__}"
        )
        assert procedure.text == "Searched and created.", (
            f"Summary text mismatch: {procedure.text!r}"
        )

    @patch("app.agent.MCPClient")
    async def test_planner_immediately_done_returns_summary(self, mock_mcp_cls):
        # First plan response is done=true → no tool is invoked, summary carried through.
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.return_value = NextStepPlan(done=True, full_task_summary="Nothing to do.")
        mock_mcp_cls.return_value = _mcp_mock()

        result, steps = await svc.execute(text="do something", token="tok")
        assert result.text == "Nothing to do.", f"Summary text mismatch: {result.text!r}"
        assert steps == [], f"Expected no steps when planner is immediately done, got {steps}"

    @patch("app.agent.MCPClient")
    async def test_loop_detection_records_error_and_continues(self, mock_mcp_cls):
        # Planner emits the same (tool, args) twice in a row → the second call must
        # be short-circuited with an [Agent error] message instead of re-invoking the tool.
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        same = NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"})
        mock_plan.plan_next_step.side_effect = [same, same, NextStepPlan(done=True, full_task_summary="partial")]
        mock_mcp_cls.return_value = _mcp_mock()

        procedure, steps = await svc.execute(text="loop me", token="tok")

        assert len(steps) == 2, f"Expected 2 recorded steps (1 exec + 1 loop-detected), got {len(steps)}: {steps}"
        assert "[Agent error]" in steps[1].mcp_result, (
            f"Expected '[Agent error]' marker in second step's result (loop detection), "
            f"got {steps[1].mcp_result!r}"
        )

    @patch("app.agent.MCPClient")
    async def test_unknown_tool_records_error_and_continues(self, mock_mcp_cls):
        # Planner picks a tool id that was never discovered → step recorded with
        # [Agent error] rather than raising.
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

        assert len(steps) == 1, f"Expected 1 recorded step for the unknown tool, got {len(steps)}: {steps}"
        assert "[Agent error]" in steps[0].mcp_result, (
            f"Expected '[Agent error]' marker in result for unknown tool, got {steps[0].mcp_result!r}"
        )

    @patch("app.agent.MCPClient")
    async def test_tool_call_failure_records_error_and_continues(self, mock_mcp_cls):
        # First tool call raises; second succeeds. The agent must record a [Tool error]
        # for the first step then continue executing — not abort the whole run.
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

        assert len(steps) == 2, f"Expected 2 executed steps, got {len(steps)}: {steps}"
        assert "[Tool error]" in steps[0].mcp_result, (
            f"Expected '[Tool error]' marker in first step (tool raised RuntimeError), "
            f"got {steps[0].mcp_result!r}"
        )
        assert steps[1].mcp_result == "created", (
            f"Expected second step to carry the successful tool output 'created', got {steps[1].mcp_result!r}"
        )

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
        assert call_kwargs["allowed_tools"] == ["test-server.search"], (
            f"allowed_tools was not propagated to planner (load_tools may not have run). "
            f"Got: {call_kwargs.get('allowed_tools')!r}"
        )
        assert call_kwargs["tool_schemas"] == {"test-server.search": {"type": "object", "properties": {}}}, (
            f"tool_schemas was not propagated to planner (load_tools may not have run). "
            f"Got: {call_kwargs.get('tool_schemas')!r}"
        )

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
        assert kwargs["text"] == "find abc", f"summarize 'text' kwarg mismatch: {kwargs.get('text')!r}"
        assert kwargs["done_plan"] is done, (
            f"summarize did not receive the same done_plan object. Got: {kwargs.get('done_plan')!r}"
        )
        assert kwargs["user_guidelines"] == "be terse", (
            f"summarize 'user_guidelines' kwarg mismatch: {kwargs.get('user_guidelines')!r}"
        )
        assert kwargs["planner_model"] == "some-model", (
            f"summarize 'planner_model' kwarg mismatch: {kwargs.get('planner_model')!r}"
        )
        assert kwargs["known_drive_items"] == [
            {"name": "X", "id": "id1", "type": "document", "location": "/"}
        ], f"summarize 'known_drive_items' kwarg mismatch: {kwargs.get('known_drive_items')!r}"
        # executed_steps reflects the one tool call before done=true
        executed = kwargs["executed_steps"]
        assert len(executed) == 1, f"Expected 1 executed step, got {len(executed)}: {executed}"
        assert executed[0]["next_tool"] == "test-server.search", (
            f"Executed step tool mismatch: {executed[0]['next_tool']!r}"
        )
        assert executed[0]["arguments"] == {"q": "abc"}, (
            f"Executed step args mismatch: {executed[0]['arguments']!r}"
        )
        assert "abc result" in executed[0]["result_summary"], (
            f"Expected 'abc result' in step result_summary, got {executed[0]['result_summary']!r}"
        )

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

        assert result is summarizer_output, (
            f"Agent did not return the exact summarizer output object. Got: {result!r}"
        )
        assert result.text == "Found 1 result for 'x'.", f"Summary text mismatch: {result.text!r}"
        assert result.items[0].id == "real-id-42", (
            f"Expected first item id 'real-id-42', got {result.items[0].id!r}"
        )

    @patch("app.agent.MCPClient")
    async def test_max_steps_exhausted_raises(self, mock_mcp_cls):
        """No done_plan → raise rather than return a blank summary that looks successful."""
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": str(i)})
            for i in range(svc.max_plan_steps)
        ]
        mock_mcp_cls.return_value = _mcp_mock(call_result="result")

        with pytest.raises(RuntimeError, match="Max steps"):
            await svc.execute(text="loop", token="tok")

        mock_plan.summarize.assert_not_awaited()

    @patch("app.agent.MCPClient")
    async def test_max_steps_exhausted_runs_the_whole_budget(self, mock_mcp_cls):
        # Planner never emits done=true → agent uses every step of the budget before
        # giving up, and does not stop early.
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        # Always returns a different tool call (different args) — never done
        mock_plan.plan_next_step.side_effect = [
            NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": str(i)})
            for i in range(svc.max_plan_steps)
        ]
        mock_client = _mcp_mock(call_result="result")
        mock_mcp_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Max steps"):
            await svc.execute(text="keep going", token="tok")

        assert mock_plan.plan_next_step.await_count == svc.max_plan_steps, (
            f"Expected exactly max_plan_steps ({svc.max_plan_steps}) planner calls, "
            f"got {mock_plan.plan_next_step.await_count}"
        )
        assert mock_client.call_tool.await_count == svc.max_plan_steps, (
            f"Expected exactly max_plan_steps ({svc.max_plan_steps}) tool calls, "
            f"got {mock_client.call_tool.await_count}"
        )


# ── _ensure_tools_discovered ──────────────────────────────────────────────────

def _make_request_ctxt(clients: dict, token: str = "tok", text: str = "") -> _RequestContext:
    return _RequestContext(stack=AsyncExitStack(), token=token, text=text, clients=clients)


class TestEnsureToolsDiscovered:
    """Agent._ensure_tools_discovered: caching, validation, and per-server allowed-tools filter."""

    async def test_discovers_and_caches_tools(self):
        # First discovery pass builds the tool-id map from the MCP client's list_tools().
        svc, servers = _make_agent()
        mock_client = _mcp_mock(tools=[_tool("search"), _tool("create")])

        await svc._ensure_tools_discovered(_make_request_ctxt({servers[0].id: mock_client}))

        assert "test-server.search" in svc._tool_id_to_tool, (
            f"'test-server.search' not discovered. Tool map: {list(svc._tool_id_to_tool)}"
        )
        assert "test-server.create" in svc._tool_id_to_tool, (
            f"'test-server.create' not discovered. Tool map: {list(svc._tool_id_to_tool)}"
        )

    async def test_second_call_uses_cache(self):
        # Discovery is a one-time cost per Agent instance: a pre-populated cache must
        # prevent list_tools() from being called again.
        svc, servers = _make_agent()
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_client = _mcp_mock()

        await svc._ensure_tools_discovered(_make_request_ctxt({"test-server": mock_client}))

        mock_client.list_tools.assert_not_called()

    async def test_no_tools_after_discovery_raises(self):
        # If the MCP server reports zero tools, the agent can't do anything — fail fast.
        svc, servers = _make_agent()
        mock_client = _mcp_mock(tools=[])

        with pytest.raises(ValueError, match="No tools available"):
            await svc._ensure_tools_discovered(_make_request_ctxt({servers[0].id: mock_client}))

    async def test_allowed_tools_filter_is_applied(self):
        # A server configured with allowed_tools=["search"] must expose only 'search',
        # even if the MCP server advertises more tools.
        server = ServerConfig(id="s", url="http://s/mcp/", allowed_tools=["search"])
        registry = MagicMock(spec=ServerRegistry)
        registry.all.return_value = [server]
        svc = Agent(registry=registry, planner=AsyncMock(), max_plan_steps=5)
        mock_client = _mcp_mock(tools=[_tool("search"), _tool("create")])  # create should be filtered out

        await svc._ensure_tools_discovered(_make_request_ctxt({"s": mock_client}))

        assert "s.search" in svc._tool_id_to_tool, (
            f"Allowed tool 's.search' missing after discovery. Map: {list(svc._tool_id_to_tool)}"
        )
        assert "s.create" not in svc._tool_id_to_tool, (
            f"'s.create' was not filtered out by allowed_tools. Map: {list(svc._tool_id_to_tool)}"
        )



# ── Static helpers ────────────────────────────────────────────────────────────

def _make_call_result(*texts: str):
    """Minimal CallToolResult-like object with TextContent items."""
    parts = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(content=parts)


LIMIT = Agent._SUMMARY_LIMIT
# MARKER = Agent._NEXT_PAGE_TOKEN_MARKER


class TestExtractResultText:
    """Agent._extract_result_text: normalises many shapes of MCP result into a single string."""

    def test_none_returns_empty(self):
        result = Agent._extract_result_text(None)
        assert result == "", f"Expected '' for None input, got {result!r}"

    def test_string_passthrough(self):
        result = Agent._extract_result_text("hello")
        assert result == "hello", f"String input should pass through unchanged, got {result!r}"

    def test_non_string_falls_back_to_str(self):
        # Unknown objects go through str(); sanity check that the key shows up.
        result = Agent._extract_result_text({"key": "val"})
        assert isinstance(result, str), f"Expected str fallback, got {type(result).__name__}"
        assert "key" in result, f"Expected 'key' in str() fallback output, got {result!r}"

    def test_structured_content_preferred(self):
        # When structured_content.result is a string, it wins over the legacy content list.
        obj = SimpleNamespace(
            structured_content={"result": "from structured"},
            content=[SimpleNamespace(text="from legacy")],
        )
        result = Agent._extract_result_text(obj)
        assert result == "from structured", (
            f"Expected structured_content.result to win, got {result!r}"
        )

    def test_structured_content_non_string_result_falls_through_to_content(self):
        # structured_content.result is not a string → fall back to legacy content list.
        obj = SimpleNamespace(
            structured_content={"result": 42},
            content=[SimpleNamespace(text="from legacy")],
        )
        result = Agent._extract_result_text(obj)
        assert result == "from legacy", (
            f"Non-string structured result should fall through to content, got {result!r}"
        )

    def test_structured_content_missing_result_key_falls_through_to_content(self):
        # No 'result' key in structured_content → legacy content wins.
        obj = SimpleNamespace(
            structured_content={"other": "data"},
            content=[SimpleNamespace(text="from legacy")],
        )
        result = Agent._extract_result_text(obj)
        assert result == "from legacy", (
            f"Missing 'result' key should fall through to content, got {result!r}"
        )

    def test_calltoolresult_text_is_extracted(self):
        obj = _make_call_result("item 1\nitem 2")
        result = Agent._extract_result_text(obj)
        assert result == "item 1\nitem 2", f"Extracted text mismatch: {result!r}"

    def test_calltoolresult_multiple_parts_are_joined(self):
        # Multiple TextContent parts are joined by newlines into a single string.
        obj = _make_call_result("part A", "part B")
        result = Agent._extract_result_text(obj)
        assert result == "part A\npart B", f"Expected joined-by-newline output, got {result!r}"

    def test_calltoolresult_empty_content_falls_back_to_str(self):
        obj = SimpleNamespace(content=[])
        result = Agent._extract_result_text(obj)
        assert isinstance(result, str), (
            f"Expected str fallback for empty content, got {type(result).__name__}"
        )

    def test_tool_error_prefixed(self):
        # isError=True → result is prefixed with [Tool error] so downstream can detect failure.
        obj = SimpleNamespace(isError=True, content=[SimpleNamespace(text="Unsupported element type")])
        result = Agent._extract_result_text(obj)
        assert result == "[Tool error] Unsupported element type", (
            f"Expected '[Tool error]' prefix, got {result!r}"
        )

    def test_tool_error_empty_content_uses_fallback(self):
        # isError=True with no content parts → a generic fallback message is used.
        obj = SimpleNamespace(isError=True, content=[])
        result = Agent._extract_result_text(obj)
        assert result == "[Tool error] MCP server returned a ToolError", (
            f"Expected generic '[Tool error]' fallback, got {result!r}"
        )


class TestCapResult:
    """Agent._cap_result: enforces the per-step result length limit with a '(truncated)' marker."""

    def test_empty_string_returned_unchanged(self):
        result = Agent._cap_result("")
        assert result == "", f"Empty input should return empty, got {result!r}"

    def test_short_string_returned_unchanged(self):
        result = Agent._cap_result("hello")
        assert result == "hello", f"Short input should pass through, got {result!r}"

    def test_string_exactly_at_limit_returned_unchanged(self):
        # A string whose length exactly equals the limit must not be truncated.
        text = "x" * LIMIT
        result = Agent._cap_result(text)
        assert result == text, (
            f"String at exact limit ({LIMIT}) should not be truncated. "
            f"Got length {len(result)}, endswith truncated marker: {result.endswith('...(truncated)')}"
        )

    def test_long_string_truncated(self):
        # Over the limit → truncated to exactly LIMIT chars + '(truncated)' suffix.
        text = "x" * (LIMIT + 100)
        result = Agent._cap_result(text)
        expected = "x" * LIMIT + "...(truncated)"
        assert result == expected, (
            f"Long string not truncated correctly. "
            f"Expected length {len(expected)}, got {len(result)}. "
            f"Last 20 chars: {result[-20:]!r}"
        )

    def test_calltoolresult_long_text_truncated(self):
        # Realistic path: long CallToolResult → extract + cap must end with the marker.
        long_text = "item\n" * 1100  # >5000 chars
        obj = _make_call_result(long_text)
        result = Agent._cap_result(Agent._extract_result_text(obj))
        assert result.endswith("...(truncated)"), (
            f"Expected result to end with '...(truncated)'. Last 30 chars: {result[-30:]!r}"
        )


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
    """Agent.stream: SSE-event-emitting variant of execute()."""

    @patch("app.agent.MCPClient")
    async def test_planner_exception_yields_error_event(self, mock_mcp_cls):
        # A planner exception must be surfaced as a single ERROR event, not re-raised.
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        mock_plan.plan_next_step.side_effect = RuntimeError("planner exploded")
        mock_mcp_cls.return_value = _mcp_mock()

        events = [e async for e in svc.stream(text="do something", token="tok")]

        assert len(events) == 1, (
            f"Expected exactly 1 error event when planner raises, got {len(events)}: {events}"
        )
        assert events[0]["type"] == EventType.ERROR, (
            f"Expected ERROR event type, got {events[0].get('type')!r}. Event: {events[0]}"
        )
        assert "planner exploded" in events[0]["detail"], (
            f"Expected 'planner exploded' in error detail, got {events[0].get('detail')!r}"
        )

    @patch("app.agent.MCPClient")
    async def test_done_plan_summary_in_result_event(self, mock_mcp_cls):
        # On normal completion, stream() must emit exactly one RESULT event carrying
        # the done_plan's summary text.
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
        assert len(result_events) == 1, (
            f"Expected exactly 1 RESULT event, got {len(result_events)}. All events: {events}"
        )
        assert result_events[0]["text"] == "Found results.", (
            f"RESULT event text mismatch: {result_events[0].get('text')!r}"
        )


class TestExtractToolsWithSchema:
    """Agent._extract_tools_with_schema: parses assorted MCP tool-list shapes into {name: schema}."""

    def test_list_of_objects_with_input_schema(self):
        # SimpleNamespace(name=..., inputSchema=...) list → keyed by name.
        tools = [SimpleNamespace(name="search", inputSchema={"type": "object"})]
        result = Agent._extract_tools_with_schema(tools)
        assert "search" in result, f"Expected 'search' key in extracted tools, got {list(result)}"

    def test_dict_format_with_tools_key(self):
        # Some servers return {"tools": [...]} instead of a bare list — both must work.
        result = Agent._extract_tools_with_schema(
            {"tools": [{"name": "search", "inputSchema": {"type": "object"}}]}
        )
        assert "search" in result, (
            f"Expected 'search' key when input is a dict with 'tools' list, got {list(result)}"
        )

    def test_none_input_returns_empty_dict(self):
        result = Agent._extract_tools_with_schema(None)
        assert result == {}, f"None input should return empty dict, got {result!r}"

    def test_tool_without_name_is_skipped(self):
        # name=None is invalid — the tool is silently dropped, not added with a None key.
        tools = [SimpleNamespace(name=None, inputSchema={})]
        result = Agent._extract_tools_with_schema(tools)
        assert result == {}, f"Tool with name=None should be skipped, got {result!r}"

    def test_tool_with_empty_name_is_skipped(self):
        # Empty string name is also invalid.
        tools = [SimpleNamespace(name="", inputSchema={})]
        result = Agent._extract_tools_with_schema(tools)
        assert result == {}, f"Tool with name='' should be skipped, got {result!r}"

    def test_schema_falls_back_to_input_schema_variants(self):
        # Some servers use `input_schema` (snake_case) instead of `inputSchema` — support both.
        tool = SimpleNamespace(name="t", input_schema={"type": "object"})
        result = Agent._extract_tools_with_schema([tool])
        assert result["t"] == {"type": "object"}, (
            f"Expected snake_case 'input_schema' to be picked up. Got result['t']={result.get('t')!r}"
        )


class TestExtractReadOnlyFlags:
    """Agent._extract_read_only_flags: reads the server's readOnlyHint annotation, failing closed."""

    def test_annotation_object_true(self):
        tool = SimpleNamespace(name="search", annotations=SimpleNamespace(readOnlyHint=True))
        result = Agent._extract_read_only_flags([tool])
        assert result == {"search": True}, f"readOnlyHint=True should map to True, got {result!r}"

    def test_annotation_dict_true(self):
        result = Agent._extract_read_only_flags(
            {"tools": [{"name": "search", "annotations": {"readOnlyHint": True}}]}
        )
        assert result == {"search": True}, f"Dict-shaped annotations should be read, got {result!r}"

    def test_annotation_false(self):
        tool = SimpleNamespace(name="create", annotations=SimpleNamespace(readOnlyHint=False))
        result = Agent._extract_read_only_flags([tool])
        assert result == {"create": False}, f"readOnlyHint=False should map to False, got {result!r}"

    def test_missing_annotations_fail_closed(self):
        # No annotations at all → treated as mutating, so the strict session-wide guard applies.
        tool = SimpleNamespace(name="mystery", inputSchema={})
        result = Agent._extract_read_only_flags([tool])
        assert result == {"mystery": False}, f"Missing annotations must fail closed, got {result!r}"

    def test_non_boolean_hint_fails_closed(self):
        # A truthy non-bool ("yes", 1) must not be mistaken for a read-only declaration.
        tool = SimpleNamespace(name="odd", annotations={"readOnlyHint": "yes"})
        result = Agent._extract_read_only_flags([tool])
        assert result == {"odd": False}, f"Non-boolean hint must fail closed, got {result!r}"

    def test_unnamed_tool_is_skipped(self):
        tool = SimpleNamespace(name=None, annotations=SimpleNamespace(readOnlyHint=True))
        result = Agent._extract_read_only_flags([tool])
        assert result == {}, f"Tool with name=None should be skipped, got {result!r}"

    def test_none_input_returns_empty_dict(self):
        assert Agent._extract_read_only_flags(None) == {}


class TestReadOnlyLoopGuard:
    """Read-only tools block only on a back-to-back repeat; mutating tools block on any repeat."""

    @patch("app.agent.MCPClient")
    async def test_read_only_repeat_allowed_after_intervening_step(self, mock_mcp_cls):
        # read → write → same read again. The re-read returns fresh state, so it must run.
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {
            "test-server.search": (servers[0], "search"),
            "test-server.create": (servers[0], "create"),
        }
        svc._tool_id_to_schema = {"test-server.search": {}, "test-server.create": {}}
        svc._tool_id_to_read_only = {"test-server.search": True, "test-server.create": False}
        read = NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"})
        write = NextStepPlan(done=False, next_tool="test-server.create", arguments={"t": "y"})
        mock_plan.plan_next_step.side_effect = [
            read, write, read, NextStepPlan(done=True, full_task_summary="ok"),
        ]
        mock_mcp_cls.return_value = _mcp_mock()

        _result, steps = await svc.execute(text="read write read", token="tok")

        assert len(steps) == 3, f"Expected 3 executed steps, got {len(steps)}: {steps}"
        assert "[Agent error]" not in steps[2].mcp_result, (
            f"Re-read after an intervening write must not be blocked, got {steps[2].mcp_result!r}"
        )

    @patch("app.agent.MCPClient")
    async def test_read_only_consecutive_repeat_blocked(self, mock_mcp_cls):
        # Same read twice in a row → nothing changed in between, so it is a genuine spin.
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {"test-server.search": (servers[0], "search")}
        svc._tool_id_to_schema = {"test-server.search": {}}
        svc._tool_id_to_read_only = {"test-server.search": True}
        same = NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"})
        mock_plan.plan_next_step.side_effect = [same, same, NextStepPlan(done=True, full_task_summary="partial")]
        mock_mcp_cls.return_value = _mcp_mock()

        _result, steps = await svc.execute(text="spin", token="tok")

        assert "[Agent error]" in steps[1].mcp_result, (
            f"Consecutive identical read must be blocked, got {steps[1].mcp_result!r}"
        )

    @patch("app.agent.MCPClient")
    async def test_mutating_repeat_blocked_across_session(self, mock_mcp_cls):
        # write → read → same write again. The duplicate write must still be blocked.
        mock_plan = AsyncMock()
        svc, servers = _make_agent(planner=mock_plan)
        svc._tool_id_to_tool = {
            "test-server.search": (servers[0], "search"),
            "test-server.create": (servers[0], "create"),
        }
        svc._tool_id_to_schema = {"test-server.search": {}, "test-server.create": {}}
        svc._tool_id_to_read_only = {"test-server.search": True, "test-server.create": False}
        read = NextStepPlan(done=False, next_tool="test-server.search", arguments={"q": "x"})
        write = NextStepPlan(done=False, next_tool="test-server.create", arguments={"t": "y"})
        mock_plan.plan_next_step.side_effect = [
            write, read, write, NextStepPlan(done=True, full_task_summary="partial"),
        ]
        mock_mcp_cls.return_value = _mcp_mock()

        _result, steps = await svc.execute(text="write read write", token="tok")

        assert "[Agent error]" in steps[2].mcp_result, (
            f"Repeated write must be blocked even with an intervening read, got {steps[2].mcp_result!r}"
        )
