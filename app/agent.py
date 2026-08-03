import asyncio
import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from app.agent_memory import AgentMemory
from app.planner import Planner
from app.mcp_client import MCPClient
from app.schemas import ErrorEvent, EventType, ExecutedToolStep, NextStepPlan, FullTaskSummary
from app.server_registry import ServerConfig, ServerRegistry
from app.tool_refiners import DEFAULT_REFINERS, ToolResultRefiner
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
        agent_memory: AgentMemory | None = None,
        tool_refiners: tuple[ToolResultRefiner, ...] = DEFAULT_REFINERS,
    ):
        self.registry = registry
        self._planner = planner
        self.max_plan_steps = max_plan_steps
        self._agent_memory = agent_memory
        self._tool_refiners = tool_refiners
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
        known_drive_items: list | None = None,
        user_guidelines: str | None = None,
        planner_model: str | None = None,
        requested_tool_sets: list[str] | None = None,
    ):
        """Set up MCP clients and yield a ready _RequestContext."""
        async with AsyncExitStack() as stack:
            if self._agent_memory and user_email:
                context_items = await self._agent_memory.load_context_items(user_email)
            else:
                context_items = list(known_drive_items or [])

            request_ctxt = _RequestContext(
                stack=stack,
                token=token,
                text=text,
                user_email=user_email,
                known_drive_items=context_items,
                user_guidelines=user_guidelines,
                planner_model=planner_model,
            )
            if not self._tool_id_to_tool:
                for server in self.registry.all():
                    await self._get_or_open_client(server, request_ctxt)
                await self._ensure_tools_discovered(request_ctxt)
            request_ctxt.load_tools(self._tool_id_to_tool, self._tool_id_to_schema)
            log.info("Request context ready: %d tools available", len(request_ctxt.allowed_tools))
            yield request_ctxt

    async def _handle(self, request_ctxt: _RequestContext):
        """Core logic: yields (NextStepPlan, ExecutedToolStep) per step, then the final FullTaskSummary."""
        planning_context: list[dict[str, Any]] = []
        seen_signatures: set[tuple[str, str]] = set()
        step_count = 0
        done_plan: NextStepPlan | None = None

        for step_idx in range(1, self.max_plan_steps + 1):
            plan, executed = await self._run_step(
                step_idx=step_idx,
                request_ctxt=request_ctxt,
                planning_context=planning_context,
                seen_signatures=seen_signatures,
            )
            if plan.done:
                done_plan = plan
                break
            step_count += 1
            yield plan, executed

        if not step_count and not done_plan:
            raise ValueError("Planner did not produce any executable step for this request")

        if not done_plan:
            # Ran out of step budget before the planner signaled done. Surface this as a
            # failure (→ ErrorEvent in stream / raise in execute) instead of returning an
            # empty summary that looks like a successful but blank result.
            raise RuntimeError(
                f"Max steps ({self.max_plan_steps}) exhausted without the planner signaling "
                "done; request did not complete"
            )

        result = await self._planner.summarize(
            text=request_ctxt.text,
            done_plan=done_plan,
            executed_steps=planning_context,
            known_drive_items=request_ctxt.known_drive_items,
            user_guidelines=request_ctxt.user_guidelines,
            planner_model=request_ctxt.planner_model,
        )

        # Save before yielding the result: if the client disconnects right after the
        # final event, a post-yield write would be skipped (generator never resumed).
        if self._agent_memory and request_ctxt.user_email and result.items:
            await self._agent_memory.update(request_ctxt.user_email, result.items)

        yield result


    async def execute(
        self,
        *,
        text: str,
        token: str,
        user_email: str = "",
        known_drive_items: list | None = None,
        user_guidelines: str | None = None,
        planner_model: str | None = None,
    ) -> tuple:
        """Run the pipeline and return (FullTaskSummary, list[ExecutedToolStep])."""
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
                if isinstance(obj, tuple):
                    _plan, executed = obj
                    steps.append(executed)
                elif isinstance(obj, FullTaskSummary):
                    result = obj
            return result, steps

    async def stream(
        self,
        *,
        text: str,
        token: str,
        user_email: str = "",
        known_drive_items: list | None = None,
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
                    if isinstance(obj, tuple):
                        plan, _executed = obj
                        d = plan.model_dump(include={"next_tool", "arguments", "next_tool_explanation"})
                        d["type"] = EventType.STEP_PLANNED
                    elif isinstance(obj, FullTaskSummary):
                        d = obj.model_dump()
                        d["type"] = EventType.RESULT
                    else:
                        continue
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
    ) -> tuple[NextStepPlan, ExecutedToolStep | None]:
        """
        Ask the planner for the next step and execute it.
        Returns (plan, executed_step) when done=false, or (plan, None) when done=true.
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
            log.info("Step=%s done=true  rationale: %s", step_idx, next_step.next_tool_explanation or "")
            return next_step, None

        if not next_step.next_tool and not next_step.pending_actions:
            log.warning("Step=%s model returned done=false with no next_tool and empty pending_actions — treating as done", step_idx)
            next_step.done = True
            return next_step, None

        executed = await self._execute_step(next_step, step_idx, request_ctxt, planning_context, seen_signatures)
        return next_step, executed

    async def _execute_step(
        self,
        next_step: NextStepPlan,
        step_idx: int,
        request_ctxt: _RequestContext,
        planning_context: list[dict[str, Any]],
        seen_signatures: set[tuple[str, str]],
    ) -> ExecutedToolStep:
        """Resolve and call the tool chosen by the planner. Returns the executed step."""
        tool_name = next_step.next_tool
        arguments = next_step.arguments or {}
        rationale = next_step.next_tool_explanation or ""
        last_signature = None
        if planning_context:
            prev = planning_context[-1]
            last_signature = (prev["next_tool"], stable_json(prev["arguments"]))
        try:
            server, real_tool_name = self._resolve_tool(tool_name, arguments, seen_signatures, last_signature)
        except ValueError as exc:
            log.warning(
                "Step=%s BLOCKED — %s\n  rationale: %s\n  tool: %s\n  args: %s",
                step_idx, exc, rationale, tool_name, arguments,
            )
            result = f"[Agent error] {exc}"
            server_id = ""
        else:
            server_id = server.id
            log.info(
                "Step=%s → %s  args=%s\n  rationale: %s",
                step_idx, tool_name, arguments, rationale,
            )
            try:
                result = await self._call_tool(step_idx, server, real_tool_name, arguments, request_ctxt)
            except Exception as exc:
                log.warning("Step=%s tool call failed: %s", step_idx, exc)
                result = f"[Tool error] {exc}"

        result_summary = self._cap_result(result)
        log.info("Result step=%s ↳ %s", step_idx, result_summary)

        planning_context.append({
            "step": step_idx,
            "server_id": server_id,
            "next_tool": tool_name,
            "arguments": arguments,
            "locked_facts": next_step.locked_facts,
            "result_summary": result_summary,
        })
        return ExecutedToolStep(
            step=step_idx,
            server_id=server_id,
            tool_name=tool_name,
            arguments=arguments,
            mcp_result=result,
        )

    _READ_ONLY_TOOLS = frozenset({
        "inspect_doc_structure",
        "get_doc_content",
        "get_drive_file_content",
        "list_drive_items",
        "search_drive_files",
    })

    def _resolve_tool(
        self,
        tool_name: str,
        arguments: dict,
        seen_signatures: set[tuple[str, str]],
        last_signature: tuple[str, str] | None = None,
    ) -> tuple[ServerConfig, str]:
        """Look up the planned tool by id, verify it exists, and guard against repeated identical calls.

        Read-only tools are only blocked if the immediately preceding step was identical (consecutive loop).
        Mutating tools are blocked on any repetition across the entire session.
        """
        selected = self._tool_id_to_tool.get(tool_name)
        if selected is None:
            raise ValueError(f"Planned tool '{tool_name}' is not available")
        server, real_tool_name = selected

        signature = (tool_name, stable_json(arguments))
        if real_tool_name in self._READ_ONLY_TOOLS:
            if signature == last_signature:
                raise ValueError(
                    f"Planner loop detected: consecutive repeated step '{tool_name}' with identical arguments"
                )
        else:
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
        try:
            raw = await mcp_client.call_tool(
                token=request_ctxt.token,
                tool_name=real_tool_name,
                arguments=arguments,
            )
        except Exception as exc:
            # Give refiners a shot at repairing a failed call (e.g. an index
            # rejected by the Docs API). Unrefined failures propagate as before.
            refined = await self._refine_result(
                step_idx, mcp_client, request_ctxt, real_tool_name, arguments, f"[Tool error] {exc}"
            )
            if refined is None:
                raise
            return refined
        result = self._extract_result_text(raw)
        refined = await self._refine_result(
            step_idx, mcp_client, request_ctxt, real_tool_name, arguments, result
        )
        return result if refined is None else refined

    async def _refine_result(
        self,
        step_idx: int,
        mcp_client: MCPClient,
        request_ctxt: _RequestContext,
        real_tool_name: str,
        arguments: dict,
        result: str,
    ) -> str | None:
        """
        Single refinement pass (see tool_refiners): the first applicable refiner
        gets one follow-up call to the same tool and its result replaces the
        original. Returns None when no refiner applied.
        """
        for refiner in self._tool_refiners:
            if not refiner.applies(real_tool_name, arguments, result):
                continue
            refined_args = refiner.refined_arguments(arguments, result)
            if not refined_args:
                continue
            log.info(
                "Step=%s %s result refined by %s → follow-up args=%s",
                step_idx, real_tool_name, type(refiner).__name__, refined_args,
            )
            raw = await mcp_client.call_tool(
                token=request_ctxt.token,
                tool_name=real_tool_name,
                arguments=refined_args,
            )
            return self._extract_result_text(raw)
        return None

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
                blocked = set(server.blocked_tools)
                effective_tools = [
                    tool_name
                    for tool_name in discovered_tools
                    if (configured_allowed is None or tool_name in configured_allowed)
                    and tool_name not in blocked
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
            error_text = " ".join(parts) if parts else "MCP server returned a ToolError"
            log.warning("MCP ToolError: %s", error_text)
            return "[Tool error] " + error_text
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
