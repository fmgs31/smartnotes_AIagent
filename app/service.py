import logging
from app.schemas import ToolCallPlan
from app.llm_router import plan_tool_call
from app.mcp_client import MCPClient
from app.server_registry import ServerRegistry

log = logging.getLogger("service")


class OrchestratorService:
    def __init__(self, registry: ServerRegistry, model: str, mcp_timeout_s: float, max_retries: int):
        self.registry = registry
        self.model = model
        self.mcp_timeout_s = mcp_timeout_s
        self.max_retries = max_retries

    async def handle_text(self, *, text: str, token: str, server_id: str | None, allowed_tools: list[str] | None):
        server = self.registry.get(server_id)

        tool_plan: ToolCallPlan = await plan_tool_call(
            model=self.model,
            text=text,
            allowed_tools=allowed_tools,
        )

        log.info("Server=%s Tool=%s", server.id, tool_plan.tool_name)
        log.info("LLM tool plan: %s", tool_plan.model_dump())

        mcp_client = MCPClient(
            server_url=server.url,
            timeout_s=self.mcp_timeout_s,
            max_retries=self.max_retries,
        )

        result = await mcp_client.call_tool(
            token=token,
            tool_name=tool_plan.tool_name,
            arguments=tool_plan.arguments,
        )

        return server.id, tool_plan, result
