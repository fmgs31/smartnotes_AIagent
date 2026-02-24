import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from app.schemas import NextStepPlan

log = logging.getLogger("llm_router")
log.setLevel(logging.INFO)

client = AsyncOpenAI()
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "next_step_system_prompt.txt"

NEXT_STEP_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "done": {"type": "boolean"},
        "tool_name": {"type": "string"},
        "arguments": {"type": "object", "additionalProperties": True},
        "explanation": {"type": "string"},
    },
    "required": ["done", "tool_name", "arguments"],
}

def _extract_output_text(resp) -> str:
    # SDK convenience property (common)
    if getattr(resp, "output_text", None):
        return resp.output_text
    # fallback (best-effort)
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                return c.text
    raise RuntimeError("Could not extract output_text from response")

def _summarize_executed_steps(executed_steps: list[dict]) -> str:
    if not executed_steps:
        return "None"
    return json.dumps(executed_steps, ensure_ascii=False)


def _load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


async def plan_next_step(
    *,
    model: str,
    text: str,
    allowed_tools: list[str] | None,
    tool_schemas: dict[str, dict] | None = None,
    executed_steps: list[dict] | None = None,
    remaining_steps: int = 1,
) -> NextStepPlan:
    """
    Ask the LLM for the next step in a multi-step plan.
    """
    system = _load_system_prompt()

    schemas_json = json.dumps(tool_schemas or {}, ensure_ascii=False)
    executed_json = _summarize_executed_steps(executed_steps or [])
    user = (
        f"User text:\n{text}\n\n"
        f"Remaining execution budget (steps): {remaining_steps}\n\n"
        f"Executed steps and outputs so far:\n{executed_json}\n\n"
        f"Allowed tools (IDs):\n{allowed_tools if allowed_tools else 'ANY'}\n\n"
        "Tool input schemas (mapping from tool_id to JSON Schema):\n"
        f"{schemas_json}"
    )
    
    log.info( f"Planning next step: remaining_steps={remaining_steps}, executed_steps={len(executed_steps or [])}" )

    resp = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "next_step_plan",
                "strict": False,
                "schema": NEXT_STEP_PLAN_SCHEMA,
            }
        },
    )

    data = json.loads(_extract_output_text(resp))
    plan = NextStepPlan(**data)
    log.info(f"Data: {data}")
    log.info(
        f"Planner output: done={plan.done} tool_name={plan.tool_name} has_arguments={bool(plan.arguments)} \n"
    )
    return plan
