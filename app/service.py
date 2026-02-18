import asyncio
import json
import logging
from typing import Any

from app.llm_router import plan_next_step
from app.mcp_client import MCPClient
from app.schemas import ExecutedToolStep, NextStepPlan, ToolCallPlan
from app.server_registry import ServerConfig, ServerRegistry

log = logging.getLogger("service")
log.setLevel(logging.INFO)


class OrchestratorService:
    def __init__(
        self,
        registry: ServerRegistry,
        model: str,
        mcp_timeout_s: float,
        max_retries: int,
        max_plan_steps: int,
    ):
        self.registry = registry
        self.model = model
        self.mcp_timeout_s = mcp_timeout_s
        self.max_retries = max_retries
        self.max_plan_steps = max_plan_steps
        # Maps fully-qualified tool id -> (server_config, real_tool_name)
        self._tool_id_to_tool: dict[str, tuple[ServerConfig, str]] = {}
        # Maps fully-qualified tool id -> input schema (as returned by list_tools)
        self._tool_id_to_schema: dict[str, Any] = {}
        self._discovery_lock = asyncio.Lock()

    async def handle_text(self, *, text: str, token: str):
        await self._ensure_tools_discovered(token)

        allowed_tools = list(self._tool_id_to_tool.keys())
        tool_schemas = {tool_id: self._tool_id_to_schema.get(tool_id) for tool_id in allowed_tools}

        steps: list[ExecutedToolStep] = []
        planning_context: list[dict[str, Any]] = []
        previous_signature: tuple[str, str] | None = None

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
            if signature == previous_signature:
                raise ValueError(
                    f"Planner loop detected: repeated step '{tool_plan.tool_name}' with identical arguments"
                )
            previous_signature = signature

            log.info(
                "Step=%s Server=%s ToolId=%s Tool=%s",
                step_idx,
                server.id,
                tool_plan.tool_name,
                real_tool_name,
            )
            if next_step.explanation:
                log.info("Planner rationale: %s", next_step.explanation)
            log.info("LLM tool plan: %s", tool_plan.model_dump())

            mcp_client = MCPClient(
                server_url=server.url,
                timeout_s=self.mcp_timeout_s,
                max_retries=self.max_retries,
            )

            result = await mcp_client.call_tool(
                token=token,
                tool_name=real_tool_name,
                arguments=tool_plan.arguments,
            )
            result_type, result_len = self._result_metrics(result)
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

    async def _ensure_tools_discovered(self, token: str) -> None:
        if self._tool_id_to_tool:
            return

        async with self._discovery_lock:
            if self._tool_id_to_tool:
                return

            tool_id_to_tool: dict[str, tuple[ServerConfig, str]] = {}
            tool_id_to_schema: dict[str, Any] = {}

            for server in self.registry.all():
                mcp_client = MCPClient(
                    server_url=server.url,
                    timeout_s=self.mcp_timeout_s,
                    max_retries=self.max_retries,
                )

                raw_tools = await mcp_client.list_tools(token=token)
                tools_with_schema = self._extract_tools_with_schema(raw_tools)
                discovered_tools = list(tools_with_schema.keys())

                configured_allowed = set(server.allowed_tools) if server.allowed_tools is not None else None
                effective_tools = [
                    tool_name
                    for tool_name in discovered_tools
                    if configured_allowed is None or tool_name in configured_allowed
                ]

                log.info(
                    "Server=%s discovered_tools=%s allowed_tools=%s effective_tools=%s",
                    server.id,
                    discovered_tools,
                    server.allowed_tools,
                    effective_tools,
                )

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

    @staticmethod
    def _summarize_mcp_result(result: Any) -> str:
        if result is None:
            return ""
        text = result if isinstance(result, str) else str(result)
        if len(text) > 1200:
            return text[:1200] + "...(truncated)"
        return text

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
