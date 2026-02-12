# from fastapi import FastAPI, HTTPException
# import logging

# from app.config import settings
# from app.schemas import InboundRequest, OutboundResponse
# from app.server_registry import ServerRegistry
# from app.service import OrchestratorService

# log = logging.getLogger("api")

# app = FastAPI(title="MCP + LLM Gateway", version="1.0.0")

# registry = ServerRegistry.load_from_file(settings.servers_config_path)
# svc = OrchestratorService(
#     registry=registry,
#     model=settings.openai_model,
#     mcp_timeout_s=settings.mcp_timeout_s,
#     max_retries=settings.max_retries,
# )


# @app.get("/health")
# def health():
#     return {"ok": True}


# @app.post("/run", response_model=OutboundResponse)
# async def run(req: InboundRequest):
#     try:
#         sid, tool_call, mcp_result = await svc.handle_text(
#             text=req.text,
#             token=req.mcp_token,
#             server_id=req.server_id,
#             allowed_tools=req.allowed_tools,
#         )
#         return OutboundResponse(server_id=sid, tool_call=tool_call, mcp_result=mcp_result)
#     except KeyError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         log.exception("Request failed")
#         raise HTTPException(status_code=500, detail=str(e))


# from fastapi import Request


# # @app.post("/run-debug")
# # async def run_debug(request: Request):
# #     raw = await request.body()
# #     return {
# #         "headers": request.headers,
# #         "content_type": request.headers.get("content-type"),
# #         "raw_text_prefix": raw[:60].decode("utf-8", "replace"),
# #         "raw_text_suffix": raw[-60:].decode("utf-8", "replace"),
# #         "raw_len": len(raw),
# #     }


from fastapi import FastAPI, HTTPException, Request
import logging

from app.schemas import InboundRequest, OutboundResponse
from app.server_registry import ServerRegistry
from app.config import settings
from app.service import OrchestratorService

logging.basicConfig(level=logging.INFO)

log = logging.getLogger("api")

app = FastAPI(title="MCP + LLM Gateway", version="1.0.0")

registry = ServerRegistry.load_from_file(settings.servers_config_path)
svc = OrchestratorService(
    registry=registry,
    model=settings.openai_model,
    mcp_timeout_s=settings.mcp_timeout_s,
    max_retries=settings.max_retries,
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/run", response_model=OutboundResponse)
async def run(req: InboundRequest):
    """
    FastAPI parses JSON body into InboundRequest automatically.
    No manual body parsing needed.
    """
    try:
        sid, tool_call, mcp_result = await svc.handle_text(
            text=req.text,
            token=req.mcp_token,
            server_id=req.server_id,
            allowed_tools=req.allowed_tools,
        )
        return OutboundResponse(server_id=sid, tool_call=tool_call, mcp_result=mcp_result)
    except KeyError as e:
        # unknown server_id
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=500, detail=str(e))


# from pydantic import BaseModel
# from typing import List
# class DebugRequest(BaseModel):
#     text: str
#     mcp_token: str
#     server_id: str
#     allowed_tools: List[str]

# @app.post("/run-debug")
# async def run_debug(payload: DebugRequest):
#     # payload is now a Python object with the data correctly typed
#     return {
#         "status": "success",
#         "server_received": payload.server_id,
#         "tools_count": len(payload.allowed_tools)
#     }

from fastapi import Request

@app.post("/run-debug")
async def run_debug(request: Request):
    """
    Shows the parsed JSON object, not raw bytes.
    """
    data = await request.json()
    return {
        "content_type": request.headers.get("content-type"),
        "received_data": data,
        "types": {k: type(v).__name__ for k, v in data.items()} if isinstance(data, dict) else type(data).__name__,
    }

