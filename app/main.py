import json
import logging
from io import BytesIO

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.auth import Auth, load_allowed_emails
from app.config import settings
from app.error_mapping import map_exception_to_http
from app.llm_router import Planner
from app.logging_config import configure_app_logging
from app.schemas import EventType, InboundRequest
from app.server_registry import ServerRegistry
from app.agent import Agent

configure_app_logging()
log = logging.getLogger("api")


class AudioRequest:
    def __init__(
        self,
        mcp_token: str = Form(...),
        known_drive_items: str | None = Form(None),
        user_guidelines: str | None = Form(None),
        planner_model: str | None = Form(None),
    ):
        self.mcp_token = mcp_token
        self.known_drive_items = known_drive_items
        self.user_guidelines = user_guidelines
        self.planner_model = planner_model

app = FastAPI(title="MCP + LLM Gateway", version="1.0.0")
openai_client = AsyncOpenAI()

registry = ServerRegistry.load_from_file(settings.servers_config_path)
for _srv in registry.all():
    log.info("Registered MCP server: id=%s url=%s", _srv.id, _srv.url)

planner = Planner(client=openai_client, model=settings.openai_model)
auth = Auth(api_key=settings.app_api_key, allowed_emails=load_allowed_emails())
agent = Agent(
    registry=registry,
    planner=planner,
    max_plan_steps=settings.max_plan_steps,
)


@app.get("/health")
def health():
    return {"ok": True}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/run", dependencies=[Depends(auth.require_api_key)])
async def run(req: InboundRequest):
    await auth.require_authorized_user(req.mcp_token)

    async def _stream():
        async for event in agent.stream(
            text=req.text,
            token=req.mcp_token,
            known_drive_items=req.known_drive_items,
            user_guidelines=req.user_guidelines,
            planner_model=req.planner_model,
        ):
            yield _sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/run-audio", dependencies=[Depends(auth.require_api_key)])
async def run_audio(audio: UploadFile = File(...), req: AudioRequest = Depends()):
    await auth.require_authorized_user(req.mcp_token)
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

    async def _stream():
        yield _sse({"type": EventType.TRANSCRIPT, "text": transcript_text})
        async for event in agent.stream(
            text=transcript_text,
            token=req.mcp_token,
            known_drive_items=req.known_drive_items,
            user_guidelines=req.user_guidelines,
            planner_model=req.planner_model,
        ):
            yield _sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream")
