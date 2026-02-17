import logging

from fastapi import FastAPI

from app.config import settings
from app.error_mapping import map_exception_to_http
from app.logging_config import configure_app_logging
from app.schemas import InboundRequest, OutboundResponse
from app.server_registry import ServerRegistry
from app.service import OrchestratorService

configure_app_logging()
log = logging.getLogger("api")

app = FastAPI(title="MCP + LLM Gateway", version="1.0.0")

registry = ServerRegistry.load_from_file(settings.servers_config_path)
svc = OrchestratorService(
    registry=registry,
    model=settings.openai_model,
    mcp_timeout_s=settings.mcp_timeout_s,
    max_retries=settings.max_retries,
    max_plan_steps=settings.max_plan_steps,
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/run", response_model=OutboundResponse)
async def run(req: InboundRequest):
    try:
        sid, tool_call, mcp_result, steps = await svc.handle_text(
            text=req.text,
            token=req.mcp_token,
        )
        return OutboundResponse(
            server_id=sid,
            tool_call=tool_call,
            mcp_result=mcp_result,
            steps=steps,
        )
    except Exception as e:
        log.exception("Request failed")
        raise map_exception_to_http(e)
