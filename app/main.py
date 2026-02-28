import logging
from io import BytesIO

from fastapi import Depends, FastAPI, File, Form, UploadFile
from openai import AsyncOpenAI

from app.auth import Auth, load_allowed_emails
from app.config import settings
from app.error_mapping import map_exception_to_http
from app.llm_router import Planner
from app.logging_config import configure_app_logging
from app.schemas import InboundRequest, OutboundResponse
from app.server_registry import ServerRegistry
from app.agent import Agent

configure_app_logging()
log = logging.getLogger("api")

app = FastAPI(title="MCP + LLM Gateway", version="1.0.0")
openai_client = AsyncOpenAI()

registry = ServerRegistry.load_from_file(settings.servers_config_path)
for _srv in registry.all():
    log.info("Registered MCP server: id=%s url=%s", _srv.id, _srv.url)

planner = Planner(client=openai_client, model=settings.openai_model)
auth = Auth(api_key=settings.app_api_key, allowed_emails=load_allowed_emails())
svc = Agent(
    registry=registry,
    planner=planner,
    max_plan_steps=settings.max_plan_steps,
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/run", response_model=OutboundResponse, dependencies=[Depends(auth.require_api_key)])
async def run(req: InboundRequest):
    await auth.require_authorized_user(req.mcp_token)
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


@app.post("/run-audio", response_model=OutboundResponse, dependencies=[Depends(auth.require_api_key)])
async def run_audio(audio: UploadFile = File(...), mcp_token: str = Form(...)):
    await auth.require_authorized_user(mcp_token)
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise ValueError("Audio file is empty")

        audio_stream = BytesIO(audio_bytes)
        audio_stream.name = audio.filename or "audio.webm"

        transcription = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_stream,
        )
        transcript_text = (getattr(transcription, "text", "") or "").strip()
        log.info(f"Transcription:{transcript_text}")
        if not transcript_text:
            raise ValueError("Whisper transcription returned empty text")
    except Exception as e:
        log.exception("Audio transcription failed")
        raise map_exception_to_http(e)
    finally:
        await audio.close()

    return await run(InboundRequest(text=transcript_text, mcp_token=mcp_token))
