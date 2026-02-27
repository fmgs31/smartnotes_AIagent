import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from app.llm_router import plan_next_step
from app.mcp_client import MCPClient
from app.schemas import ExecutedToolStep, NextStepPlan, ToolCallPlan
from app.server_registry import ServerConfig, ServerRegistry

log = logging.getLogger("service")


class OrchestratorService:
    def __init__(
        self,
        registry: ServerRegistry,
        model: str,
        max_plan_steps: int,
    ):
        self.registry = registry
        self.model = model
        self.max_plan_steps = max_plan_steps
        # Maps fully-qualified tool id -> (server_config, real_tool_name)
        self._tool_id_to_tool: dict[str, tuple[ServerConfig, str]] = {}
        # Maps fully-qualified tool id -> input schema (as returned by list_tools)
        self._tool_id_to_schema: dict[str, Any] = {}
        self._discovery_lock = asyncio.Lock()

    async def handle_text(self, *, text: str, token: str):
        async with AsyncExitStack() as session_stack:
            server_clients: dict[str, MCPClient] = {}

            async def _get_or_open_client(server: ServerConfig) -> MCPClient:
                existing = server_clients.get(server.id)
                if existing is not None:
                    return existing

                mcp_client = MCPClient(server_url=server.url)
                await session_stack.enter_async_context(mcp_client.session(token=token))
                server_clients[server.id] = mcp_client
                return mcp_client

            if not self._tool_id_to_tool:
                for server in self.registry.all():
                    await _get_or_open_client(server)

            await self._ensure_tools_discovered(token, preopened_clients=server_clients)

            allowed_tools = list(self._tool_id_to_tool.keys())
            tool_schemas = {tool_id: self._tool_id_to_schema.get(tool_id) for tool_id in allowed_tools}

            steps: list[ExecutedToolStep] = []
            planning_context: list[dict[str, Any]] = []
            seen_signatures: set[tuple[str, str]] = set()

            for step_idx in range(1, self.max_plan_steps + 1):
                next_step: NextStepPlan = await plan_next_step(
                    model=self.model,
                    text=text,
                    allowed_tools=allowed_tools,
                    tool_schemas=tool_schemas,
                    executed_steps=planning_context,
                    remaining_steps=self.max_plan_steps - step_idx + 1,
                )

                if next_step.done:
                    break

                tool_plan = ToolCallPlan(
                    tool_name=next_step.tool_name,
                    arguments=next_step.arguments or {},
                )
                selected = self._tool_id_to_tool.get(tool_plan.tool_name)
                if selected is None:
                    raise ValueError(f"Planned tool '{tool_plan.tool_name}' is not available")
                server, real_tool_name = selected

                signature = (tool_plan.tool_name, self._stable_json(tool_plan.arguments))
                if signature in seen_signatures:
                    raise ValueError(
                        f"Planner loop detected: repeated step '{tool_plan.tool_name}' with identical arguments"
                    )
                seen_signatures.add(signature)

                log.info(
                    "Step=%s Server=%s ToolId=%s Tool=%s",
                    step_idx,
                    server.id,
                    tool_plan.tool_name,
                    real_tool_name,
                )
                if next_step.explanation:
                    log.info("Planner rationale: %s", next_step.explanation)

                mcp_client = await _get_or_open_client(server)
                result = await mcp_client.call_tool(
                    token=token,
                    tool_name=real_tool_name,
                    arguments=tool_plan.arguments,
                )
                result_type, result_len = self._result_metrics(result)
                log.debug(f"\n -------------\nMCP SERVER RESULT: {result}\n")
                log.info(f"Step={step_idx} result_type={result_type} result_len={result_len}")

                steps.append(
                    ExecutedToolStep(
                        step=step_idx,
                        server_id=server.id,
                        tool_call=tool_plan,
                        mcp_result=result,
                    )
                )
                planning_context.append(
                    {
                        "step": step_idx,
                        "server_id": server.id,
                        "tool_name": tool_plan.tool_name,
                        "arguments": tool_plan.arguments,
                        "result_summary": self._summarize_mcp_result(result),
                    }
                )

            if not steps:
                raise ValueError("Planner did not produce any executable step for this request")

            last_step = steps[-1]
            return last_step.server_id, last_step.tool_call, last_step.mcp_result, steps

    async def _ensure_tools_discovered(
        self,
        token: str,
        preopened_clients: dict[str, MCPClient] | None = None,
    ) -> None:
        if self._tool_id_to_tool:
            return

        async with self._discovery_lock:
            if self._tool_id_to_tool:
                return

            tool_id_to_tool: dict[str, tuple[ServerConfig, str]] = {}
            tool_id_to_schema: dict[str, Any] = {}

            for server in self.registry.all():
                mcp_client = preopened_clients.get(server.id) if preopened_clients is not None else None
                if mcp_client is not None:
                    # Session is already open and managed externally — call directly;
                    # wrapping in session() would close it on exit, wasting the pre-open.
                    raw_tools = await mcp_client.list_tools(token=token)
                else:
                    mcp_client = MCPClient(server_url=server.url)
                    async with mcp_client.session(token=token):
                        raw_tools = await mcp_client.list_tools(token=token)

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
        return f"{server_id}.{tool_name}"

    @staticmethod
    def _extract_tools_with_schema(raw_tools: Any) -> dict[str, Any]:
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

    _NEXT_PAGE_TOKEN_MARKER = "nextPageToken: "
    _SUMMARY_LIMIT = 5000

    @staticmethod
    def _summarize_mcp_result(result: Any) -> str:
        if result is None:
            return ""

        # Extract plain text from CallToolResult content items; fall back to str().
        if hasattr(result, "content"):
            parts = [
                item.text
                for item in (result.content or [])
                if hasattr(item, "text") and isinstance(item.text, str)
            ]
            text = "\n".join(parts) if parts else str(result)
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)

        if len(text) <= OrchestratorService._SUMMARY_LIMIT:
            return text

        # Always preserve the nextPageToken even after truncation so the planner
        # can use it for cursor-based pagination.
        # Only append if the token falls beyond the truncation boundary; if it's
        # at the beginning (or anywhere within the limit) it is already included.
        marker = OrchestratorService._NEXT_PAGE_TOKEN_MARKER
        token_idx = text.find(marker)
        token_suffix = ""
        if token_idx != -1:
            line_end = text.find("\n", token_idx)
            token_line = text[token_idx:] if line_end == -1 else text[token_idx:line_end]
            # Append the full token line only when it isn't already fully contained
            # within the truncated portion (covers: token at end, and the edge case
            # where it starts just before the boundary but extends beyond it).
            if token_idx + len(token_line) > OrchestratorService._SUMMARY_LIMIT:
                token_suffix = f"\n{token_line}"

        return text[: OrchestratorService._SUMMARY_LIMIT] + "...(truncated)" + token_suffix

    @staticmethod
    def _stable_json(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(value)

    @staticmethod
    def _result_metrics(result: Any) -> tuple[str, int]:
        result_type = type(result).__name__
        try:
            if isinstance(result, (str, bytes, bytearray)):
                return result_type, len(result)
            encoded = json.dumps(result, ensure_ascii=False, default=str)
            return result_type, len(encoded)
        except Exception:
            return result_type, len(str(result))
