from fastapi import FastAPI, HTTPException
import logging
import httpx
from tenacity import RetryError

from app.schemas import InboundRequest, OutboundResponse
from app.server_registry import ServerRegistry
from app.config import settings
from app.service import OrchestratorService

def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.WARNING)
    root.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    for logger_name in ("service", "llm_router"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)


_configure_logging()

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


@app.on_event("startup")
async def startup_event():
    await svc.initialize()


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
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else 502
        raise HTTPException(status_code=status, detail=str(e))
    except RetryError as e:
        status_error = _find_http_status_error(e)
        if status_error is not None:
            status = status_error.response.status_code if status_error.response is not None else 502
            raise HTTPException(
                status_code=status,
                detail=str(status_error),
            )
        raise HTTPException(status_code=503, detail="Upstream request retries exhausted")
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=500, detail=str(e))


def _find_http_status_error(exc: BaseException) -> httpx.HTTPStatusError | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc

    if isinstance(exc, RetryError):
        last = exc.last_attempt.exception()
        if isinstance(last, BaseException):
            nested = _find_http_status_error(last)
            if nested is not None:
                return nested

    for candidate in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if isinstance(candidate, BaseException):
            nested = _find_http_status_error(candidate)
            if nested is not None:
                return nested

    return None
