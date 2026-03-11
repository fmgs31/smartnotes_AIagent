import asyncio
import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from app.llm_router import Planner
from app.mcp_client import MCPClient
from app.schemas import ErrorEvent, EventType, ExecutedToolStep, NextStepPlan, ProcedureSummary
from app.server_registry import ServerConfig, ServerRegistry
from app.utils import stable_json

log = logging.getLogger("agent")


@dataclass
class _RequestContext:
    stack: AsyncExitStack
    token: str
    text: str
    user_email: str = ""
    known_drive_items: list = field(default_factory=list)
    user_guidelines: str | None = None
    planner_model: str | None = None
    clients: dict[str, MCPClient] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    tool_schemas: dict[str, Any] = field(default_factory=dict)

    def load_tools(
        self,
        tool_id_to_tool: dict[str, tuple],
        tool_id_to_schema: dict[str, Any],
    ) -> None:
        """Populate allowed_tools and tool_schemas from the agent's discovered tool cache."""
        self.allowed_tools = list(tool_id_to_tool.keys())
        self.tool_schemas = {tool_id: tool_id_to_schema.get(tool_id) for tool_id in self.allowed_tools}


class Agent:
    def __init__(
        self,
        registry: ServerRegistry,
        planner: Planner,
        max_plan_steps: int,
    ):
        self.registry = registry
        self._planner = planner
        self.max_plan_steps = max_plan_steps
        # Maps fully-qualified tool id -> (server_config, real_tool_name)
        self._tool_id_to_tool: dict[str, tuple[ServerConfig, str]] = {}
        # Maps fully-qualified tool id -> input schema (as returned by list_tools)
        self._tool_id_to_schema: dict[str, Any] = {}
        self._discovery_lock = asyncio.Lock()

    @asynccontextmanager
    async def _open_context(
        self,
        *,
        text: str,
        token: str,
        user_email: str = "",
        known_drive_items: list = [],
        user_guidelines: str | None = None,
        planner_model: str | None = None,
        requested_tool_sets: list[str] | None = None,
    ):
        """Set up MCP clients and yield a ready _RequestContext."""
        async with AsyncExitStack() as stack:
            request_ctxt = _RequestContext(
                stack=stack,
                token=token,
                text=text,
                user_email=user_email,
                known_drive_items=known_drive_items,
                user_guidelines=user_guidelines,
                planner_model=planner_model,
            )
            if not self._tool_id_to_tool:
                for server in self.registry.all():
                    await self._get_or_open_client(server, request_ctxt)
                await self._ensure_tools_discovered(request_ctxt)
            # request_ctxt.load_tools(self._tool_id_to_tool, self._tool_id_to_schema, self._tool_sets_config)
            yield request_ctxt

    _SSE_EVENT_TYPE: dict[type, EventType] = {
        NextStepPlan: EventType.STEP_PLANNED,
        ProcedureSummary: EventType.RESULT,
    }

    async def _handle(self, request_ctxt: _RequestContext):
        """Core logic: yields NextStepPlan and ExecutedToolStep per step, then the final ProcedureSummary."""
        planning_context: list[dict[str, Any]] = []
        seen_signatures: set[tuple[str, str]] = set()
        steps: list[ExecutedToolStep] = []

        for step_idx in range(1, self.max_plan_steps + 1):
            had_plan = False
            async for obj in self._run_step(
                step_idx=step_idx,
                request_ctxt=request_ctxt,
                planning_context=planning_context,
                seen_signatures=seen_signatures,
            ):
                yield obj
                if isinstance(obj, NextStepPlan):
                    had_plan = True
                elif isinstance(obj, ExecutedToolStep):
                    steps.append(obj)
            if not had_plan:
                break

        if not steps:
            raise ValueError("Planner did not produce any executable step for this request")

        try:
            result = await self._planner.summarize_procedure(
                text=request_ctxt.text,
                executed_steps=planning_context,
                planner_model=request_ctxt.planner_model,
            )
        except Exception:
            log.exception("Summarization failed; cannot preserve Drive items, aborting")
            raise

        yield result

    async def execute(
        self,
        *,
        text: str,
        token: str,
        user_email: str = "",
        known_drive_items: list = [],
        user_guidelines: str | None = None,
        planner_model: str | None = None,
    ) -> tuple:
        """Run the pipeline and return (ProcedureSummary, list[ExecutedToolStep])."""
        async with self._open_context(
            text=text,
            token=token,
            user_email=user_email,
            known_drive_items=known_drive_items,
            user_guidelines=user_guidelines,
            planner_model=planner_model,
        ) as request_ctxt:
            steps: list[ExecutedToolStep] = []
            result = None
            async for obj in self._handle(request_ctxt):
                if isinstance(obj, ExecutedToolStep):
                    steps.append(obj)
                elif isinstance(obj, ProcedureSummary):
                    result = obj
            return result, steps

    async def stream(
        self,
        *,
        text: str,
        token: str,
        user_email: str = "",
        known_drive_items: list = [],
        user_guidelines: str | None = None,
        planner_model: str | None = None,
    ):
        """Async generator yielding typed SSE event dicts for each step and the final result."""
        async with self._open_context(
            text=text,
            token=token,
            user_email=user_email,
            known_drive_items=known_drive_items,
            user_guidelines=user_guidelines,
            planner_model=planner_model,
        ) as request_ctxt:
            try:
                async for obj in self._handle(request_ctxt):
                    if isinstance(obj, ExecutedToolStep):  # Not sending these anymore.
                        continue
                    d = obj.model_dump()
                    d["type"] = self._SSE_EVENT_TYPE[type(obj)]
                    if isinstance(obj, NextStepPlan):
                        d.pop("done", None)
                    yield d
            except Exception as exc:
                log.exception("Unhandled error in stream")
                yield ErrorEvent(detail=str(exc)).model_dump()

    async def _get_or_open_client(self, server: ServerConfig, request_ctxt: _RequestContext) -> MCPClient:
        """Return the already-open client for this server, or open a new one and register it with the request stack."""
        existing = request_ctxt.clients.get(server.id)
        if existing is not None:
            return existing
        mcp_client = MCPClient(server_url=server.url)
        await request_ctxt.stack.enter_async_context(mcp_client.session(token=request_ctxt.token))
        request_ctxt.clients[server.id] = mcp_client
        return mcp_client

    async def _run_step(
        self,
        step_idx: int,
        request_ctxt: _RequestContext,
        planning_context: list[dict[str, Any]],
        seen_signatures: set[tuple[str, str]],
    ):
        """
        Yields NextStepPlan when the planner chooses a tool, then ExecutedToolStep with the result.
        Yields nothing if the planner signals done.
        """
        next_step: NextStepPlan = await self._planner.plan_next_step(
            text=request_ctxt.text,
            known_drive_items=request_ctxt.known_drive_items,
            user_guidelines=request_ctxt.user_guidelines,
            planner_model=request_ctxt.planner_model,
            allowed_tools=request_ctxt.allowed_tools,
            tool_schemas=request_ctxt.tool_schemas,
            executed_steps=planning_context,
            remaining_steps=self.max_plan_steps - step_idx + 1,
        )

        if next_step.done:
            return
        else:
            yield next_step
        yield await self._execute_step(next_step, step_idx, request_ctxt, planning_context, seen_signatures)

    async def _execute_step(
        self,
        next_step: NextStepPlan,
        step_idx: int,
        request_ctxt: _RequestContext,
        planning_context: list[dict[str, Any]],
        seen_signatures: set[tuple[str, str]],
    ) -> ExecutedToolStep:
        """Resolve and call the tool chosen by the planner. Returns the executed step."""
        tool_name = next_step.tool_name
        arguments = next_step.arguments or {}
        try:
            server, real_tool_name = self._resolve_tool(tool_name, arguments, seen_signatures)
        except ValueError as exc:
            log.warning("Step=%s resolve failed: %s", step_idx, exc)
            result = f"[Agent error] {exc}"
            server_id = ""
        else:
            server_id = server.id
            log.info(
                "Step=%s Server=%s ToolId=%s Tool=%s",
                step_idx, server.id, tool_name, real_tool_name,
            )
            if next_step.explanation:
                log.info("Planner rationale: %s", next_step.explanation)
            try:
                result = await self._call_tool(step_idx, server, real_tool_name, arguments, request_ctxt)
            except Exception as exc:
                log.warning("Step=%s tool call failed: %s", step_idx, exc)
                result = f"[Tool error] {exc}"

        result_summary = self._cap_result(result)
        log.debug("Step=%s result_summary: %s", step_idx, result_summary)

        planning_context.append({
            "step": step_idx,
            "server_id": server_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "result_summary": result_summary,
        })
        return ExecutedToolStep(
            step=step_idx,
            server_id=server_id,
            tool_name=tool_name,
            arguments=arguments,
            mcp_result=result,
        )

    def _resolve_tool(
        self,
        tool_name: str,
        arguments: dict,
        seen_signatures: set[tuple[str, str]],
    ) -> tuple[ServerConfig, str]:
        """Look up the planned tool by id, verify it exists, and guard against repeated identical calls."""
        selected = self._tool_id_to_tool.get(tool_name)
        if selected is None:
            raise ValueError(f"Planned tool '{tool_name}' is not available")
        server, real_tool_name = selected

        signature = (tool_name, stable_json(arguments))
        if signature in seen_signatures:
            raise ValueError(
                f"Planner loop detected: repeated step '{tool_name}' with identical arguments"
            )
        seen_signatures.add(signature)

        return server, real_tool_name

    async def _call_tool(
        self,
        step_idx: int,
        server: ServerConfig,
        real_tool_name: str,
        arguments: dict,
        request_ctxt: _RequestContext,
    ) -> str:
        """Invoke the tool via MCP. Returns the result as a plain string. Raises on failure."""
        mcp_client = await self._get_or_open_client(server, request_ctxt)
        t0 = time.perf_counter()
        raw = await mcp_client.call_tool(
            token=request_ctxt.token,
            tool_name=real_tool_name,
            arguments=arguments,
        )
        result = self._extract_result_text(raw)
        log.info("Step=%s tool call took %.2fs result_len=%s", step_idx, time.perf_counter() - t0, len(result))
        log.debug(f"\n -------------\nMCP SERVER RESULT: {result}\n -------------")
        return result

    async def _ensure_tools_discovered(self, request_ctxt: _RequestContext) -> None:
        """
        Query each registered MCP server for its tool list and populate the agent-level
        tool cache. Uses double-checked locking so only the first concurrent caller does
        the work; subsequent callers return immediately once the cache is filled.
        """
        if self._tool_id_to_tool:
            return

        async with self._discovery_lock:
            if self._tool_id_to_tool:
                return

            tool_id_to_tool: dict[str, tuple[ServerConfig, str]] = {}
            tool_id_to_schema: dict[str, Any] = {}

            for server in self.registry.all():
                mcp_client = request_ctxt.clients[server.id]
                raw_tools = await mcp_client.list_tools(token=request_ctxt.token)

                tools_with_schema = self._extract_tools_with_schema(raw_tools)
                discovered_tools = list(tools_with_schema.keys())

                configured_allowed = set(server.allowed_tools) if server.allowed_tools is not None else None
                effective_tools = [
                    tool_name
                    for tool_name in discovered_tools
                    if configured_allowed is None or tool_name in configured_allowed
                ]

                for tool_name in effective_tools:
                    tool_id = self._build_tool_id(server.id, tool_name)
                    if tool_id in tool_id_to_tool:
                        raise ValueError(f"Duplicate tool id discovered: '{tool_id}'")
                    tool_id_to_tool[tool_id] = (server, tool_name)
                    tool_id_to_schema[tool_id] = tools_with_schema.get(tool_name)

            if not tool_id_to_tool:
                raise ValueError("No tools available after token-based discovery and server allowed_tools filtering")

            self._tool_id_to_tool = tool_id_to_tool
            self._tool_id_to_schema = tool_id_to_schema

    @staticmethod
    def _build_tool_id(server_id: str, tool_name: str) -> str:
        """Build the fully-qualified tool id used as the unique key across all servers."""
        return f"{server_id}.{tool_name}"

    @staticmethod
    def _extract_tools_with_schema(raw_tools: Any) -> dict[str, Any]:
        """
        Normalize the raw tool list returned by list_tools (dict, list, or object)
        into a mapping of tool name -> input schema.
        """
        if raw_tools is None:
            return {}

        if isinstance(raw_tools, dict):
            candidates = raw_tools.get("tools", [])
        elif isinstance(raw_tools, list):
            candidates = raw_tools
        else:
            candidates = getattr(raw_tools, "tools", raw_tools)

        tools: dict[str, Any] = {}
        for item in candidates:
            if isinstance(item, dict):
                name = item.get("name")
                schema = item.get("inputSchema") or item.get("input_schema") or item.get("schema")
            else:
                name = getattr(item, "name", None)
                schema = (
                    getattr(item, "inputSchema", None)
                    or getattr(item, "input_schema", None)
                    or getattr(item, "schema", None)
                )

            if isinstance(name, str) and name:
                tools[name] = schema

        return tools

    @staticmethod
    def _extract_result_text(result: Any) -> str:
        """Extract plain text from an MCP CallToolResult, or pass through strings as-is.
        Prefers structured_content (newer MCP format) over the legacy content array.
        ToolError responses (isError=True) are returned with a [Tool error] prefix so the
        planner can distinguish a failed step from a tool that returned an error-like string."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if getattr(result, "isError", False):
            parts = [
                item.text
                for item in (getattr(result, "content", None) or [])
                if hasattr(item, "text") and isinstance(item.text, str)
            ]
            return "[Tool error] " + (parts[0] if parts else "MCP server returned a ToolError")
        sc = getattr(result, "structured_content", None)
        if isinstance(sc, dict):
            val = sc.get("result")
            if isinstance(val, str):
                return val
        if hasattr(result, "content"):
            parts = [
                item.text
                for item in (result.content or [])
                if hasattr(item, "text") and isinstance(item.text, str)
            ]
            if parts:
                return "\n".join(parts)
        return str(result)

    _SUMMARY_LIMIT = 5000

    @staticmethod
    def _cap_result(text: str) -> str:
        """Cap result text at _SUMMARY_LIMIT characters for the planner context."""
        if len(text) <= Agent._SUMMARY_LIMIT:
            return text
        return text[: Agent._SUMMARY_LIMIT] + "...(truncated)"
