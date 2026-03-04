import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from app.llm_router import Planner
from app.mcp_client import MCPClient
from app.schemas import ExecutedToolStep, NextStepPlan, ProcedureSummary
from app.server_registry import ServerConfig, ServerRegistry
from app.utils import stable_json

log = logging.getLogger("agent")


@dataclass
class _RequestContext:
    stack: AsyncExitStack
    token: str
    text: str
    known_drive_items: list | str | None = None
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

    async def handle_text(self, *, text: str, token: str, known_drive_items: list | str | None = None, user_guidelines: str | None = None, planner_model: str | None = None):
        """
        Main entry point. Runs the planner loop for the given user text and returns
        the server id, tool call, MCP result, and full list of executed steps.
        Opens MCP sessions for the duration of the request via AsyncExitStack.
        """
        async with AsyncExitStack() as stack:
            
            request_ctxt = _RequestContext(stack=stack, token=token, text=text, known_drive_items=known_drive_items, user_guidelines=user_guidelines, planner_model=planner_model)

            if not self._tool_id_to_tool:
                for server in self.registry.all():
                    await self._get_or_open_client(server, request_ctxt)
                await self._ensure_tools_discovered(request_ctxt)

            request_ctxt.load_tools(self._tool_id_to_tool, self._tool_id_to_schema)

            steps: list[ExecutedToolStep] = []
            planning_context: list[dict[str, Any]] = []
            seen_signatures: set[tuple[str, str]] = set()

            for step_idx in range(1, self.max_plan_steps + 1):
                executed = await self._run_step(
                    step_idx=step_idx,
                    request_ctxt=request_ctxt,
                    planning_context=planning_context,
                    seen_signatures=seen_signatures,
                )
                if executed is None:
                    break
                steps.append(executed)

            if not steps:
                raise ValueError("Planner did not produce any executable step for this request")

            try:
                result = await self._planner.summarize_procedure(
                    text=text,
                    executed_steps=planning_context,
                    planner_model=request_ctxt.planner_model,
                )
            except Exception as exc:
                log.warning("Summarization failed, falling back to last step result: %s", exc)
                result = ProcedureSummary(text=steps[-1].mcp_result or "")

            return result, steps

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
    ) -> ExecutedToolStep | None:
        """
        Execute one planning step. Calls the LLM planner, validates and deduplicates
        the chosen tool, invokes it via MCP, and returns the result.
        Returns None when the planner signals it is done.
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
            return None

        tool_name = next_step.tool_name
        arguments = next_step.arguments or {}
        server, real_tool_name = self._resolve_tool(tool_name, arguments, seen_signatures)

        log.info(
            "Step=%s Server=%s ToolId=%s Tool=%s",
            step_idx, server.id, tool_name, real_tool_name,
        )
        if next_step.explanation:
            log.info("Planner rationale: %s", next_step.explanation)

        result = await self._call_tool(step_idx, server, real_tool_name, arguments, request_ctxt)
        result_summary = self._summarize_mcp_result(result)
        log.debug("Step=%s result_summary: %s", step_idx, result_summary)

        planning_context.append({
            "step": step_idx,
            "server_id": server.id,
            "tool_name": tool_name,
            "arguments": arguments,
            "result_summary": result_summary,
        })
        return ExecutedToolStep(
            step=step_idx,
            server_id=server.id,
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
        """Invoke the tool via MCP. Returns the result as a plain string.
        Converts exceptions into error-string results so the planner can adapt."""
        mcp_client = await self._get_or_open_client(server, request_ctxt)
        try:
            raw = await mcp_client.call_tool(
                token=request_ctxt.token,
                tool_name=real_tool_name,
                arguments=arguments,
            )
            result = self._extract_result_text(raw)
        except Exception as exc:
            log.warning("Step=%s tool call failed: %s", step_idx, exc)
            result = f"[Tool error] {exc}"
        log.debug(f"\n -------------\nMCP SERVER RESULT: {result}\n -------------")
        log.info(f"Step={step_idx} result_len={len(result)}")
        log.info("\n")
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

    # _NEXT_PAGE_TOKEN_MARKER = "nextPageToken: "
    _SUMMARY_LIMIT = 5000

    @staticmethod
    def _extract_result_text(result: Any) -> str:
        """Extract plain text from an MCP CallToolResult, or pass through strings as-is.
        Prefers structured_content (newer MCP format) over the legacy content array."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Prefer structured_content (newer MCP format): typically {"result": "..."}
        sc = getattr(result, "structured_content", None)
        if isinstance(sc, dict):
            val = sc.get("result")
            if isinstance(val, str):
                return val
        # Fall back to legacy content array
        if hasattr(result, "content"):
            parts = [
                item.text
                for item in (result.content or [])
                if hasattr(item, "text") and isinstance(item.text, str)
            ]
            if parts:
                return "\n".join(parts)
        return str(result)

    @staticmethod
    def _summarize_mcp_result(result: Any) -> str:
        """Truncate a result string to _SUMMARY_LIMIT characters for the planner context."""
        text = result if isinstance(result, str) else Agent._extract_result_text(result)
        if len(text) <= Agent._SUMMARY_LIMIT:
            return text

        # TODO: Use proper summary maybe.

        # # Always preserve the nextPageToken even after truncation so the planner
        # # can use it for cursor-based pagination.
        # # Only append if the token falls beyond the truncation boundary; if it's
        # # at the beginning (or anywhere within the limit) it is already included.
        # marker = Agent._NEXT_PAGE_TOKEN_MARKER
        # token_idx = text.find(marker)
        # token_suffix = ""
        # if token_idx != -1:
        #     line_end = text.find("\n", token_idx)
        #     token_line = text[token_idx:] if line_end == -1 else text[token_idx:line_end]
        #     # Append the full token line only when it isn't already fully contained
        #     # within the truncated portion (covers: token at end, and the edge case
        #     # where it starts just before the boundary but extends beyond it).
        #     if token_idx + len(token_line) > Agent._SUMMARY_LIMIT:
        #         token_suffix = f"\n{token_line}"

        # return text[: Agent._SUMMARY_LIMIT] + "...(truncated)" + token_suffix

        return text[: Agent._SUMMARY_LIMIT] + "...(truncated)"

