import json
from openai import AsyncOpenAI
from app.schemas import ToolCallPlan

client = AsyncOpenAI()

TOOL_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tool_name": {"type": "string"},
        "arguments": {"type": "object", "additionalProperties": True},
    },
    "required": ["tool_name", "arguments"],
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

async def plan_tool_call(*, model: str, text: str, allowed_tools: list[str] | None) -> ToolCallPlan:
    system = (
        "You are a router. Pick ONE tool and arguments.\n"
        "If allowed_tools is provided, tool_name must be one of them.\n"
        "Return only JSON matching the schema."
    )

    user = f"User text:\n{text}\nAllowed tools:\n{allowed_tools if allowed_tools else 'ANY'}"

    resp = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "tool_call_plan",
                "strict": False,
                "schema": TOOL_PLAN_SCHEMA,
            }
        },
    )

    data = json.loads(_extract_output_text(resp))
    return ToolCallPlan(**data)
