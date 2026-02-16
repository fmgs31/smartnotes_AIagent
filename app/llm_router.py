import json
from openai import AsyncOpenAI
from app.schemas import NextStepPlan

client = AsyncOpenAI()

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
    system = (
        "You are a router that can execute tasks through MCP tools one step at a time.\n"
        "Tool names may be fully-qualified IDs in the form 'server_id.tool_name'.\n"
        "If allowed_tools is provided, tool_name must be one of them.\n"
        "You receive previously executed steps and their outputs.\n"
        "Use those outputs to fill required IDs/arguments for subsequent steps.\n"
        "If the goal is already completed, set done=true.\n"
        "If more work is needed, set done=false and provide the NEXT single tool call only.\n"
        "Do not return more than one step.\n"
        "You are also given JSON Schemas for each allowed tool's input.\n"
        "Use those schemas to choose valid argument names and value shapes.\n"
        "Return only JSON matching the schema."
    )

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
    return NextStepPlan(**data)
