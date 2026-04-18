import json
import logging
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.auth import Auth, load_allowed_emails
from app.config import settings
from app.agent_memory_firestore import FirestoreAgentMemory
from google.cloud.firestore_v1.async_client import AsyncClient
from app.error_mapping import map_exception_to_http
from app.model_registry import ModelRegistry
from app.planner import Planner
from app.logging_config import configure_app_logging
from app.schemas import EventType, InboundRequest
from app.server_registry import ServerRegistry
from app.agent import Agent
from app.utils import load_json

configure_app_logging()
log = logging.getLogger("api")


class AudioRequest:
    def __init__(
        self,
        mcp_token: str = Form(...),
        user_guidelines: str | None = Form(None),
        planner_model: str | None = Form(None),
    ):
        self.mcp_token = mcp_token
        self.user_guidelines = user_guidelines
        self.planner_model = planner_model

app = FastAPI(title="MCP + LLM Gateway", version="1.0.0")
openai_client = AsyncOpenAI()

registry = ServerRegistry.load_from_file(settings.servers_config_path)
for _srv in registry.all():
    log.info("Registered MCP server: id=%s url=%s", _srv.id, _srv.url)

model_registry = ModelRegistry(
    map_path=settings.model_prompt_map_path,
    prompts_folder=Path(__file__).resolve().parent / "prompts",
)
planner = Planner(model_registry=model_registry)
auth = Auth(api_key=settings.app_api_key, allowed_emails=load_allowed_emails())
agent_memory = FirestoreAgentMemory(AsyncClient(project=settings.firestore_project)) if settings.firestore_enabled else None
agent = Agent(
    registry=registry,
    planner=planner,
    max_plan_steps=settings.max_plan_steps,
    agent_memory=agent_memory,
)


@app.get("/health")
def health():
    return {"ok": True}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/run", dependencies=[Depends(auth.require_api_key)])
async def run(req: InboundRequest):
    email = await auth.require_authorized_user(req.mcp_token)

    async def _stream():
        async for event in agent.stream(
            text=req.text,
            token=req.mcp_token,
            user_email=email,
            user_guidelines=req.user_guidelines,
            planner_model=req.planner_model,
        ):
            yield _sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/run-audio", dependencies=[Depends(auth.require_api_key)])
async def run_audio(audio: UploadFile = File(...), req: AudioRequest = Depends()):
    email = await auth.require_authorized_user(req.mcp_token)
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
            user_email=email,
            user_guidelines=req.user_guidelines,
            planner_model=req.planner_model,
        ):
            yield _sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream")
