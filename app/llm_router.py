import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from app.schemas import NextStepPlan

log = logging.getLogger("llm_router")

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "next_step_system_prompt.txt"


class Planner:
    """Wraps the OpenAI Responses API to produce single-step plans for the agent."""

    _NEXT_STEP_PLAN_SCHEMA = {
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

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model
        self._system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    async def plan_next_step(
        self,
        *,
        text: str,
        allowed_tools: list[str] | None,
        tool_schemas: dict[str, dict] | None = None,
        executed_steps: list[dict] | None = None,
        remaining_steps: int = 1,
    ) -> NextStepPlan:
        """Ask the LLM for the next step in a multi-step plan."""
        schemas_json = json.dumps(tool_schemas or {}, ensure_ascii=False)
        executed_json = self._summarize_executed_steps(executed_steps or [])
        user = (
            f"User text:\n{text}\n\n"
            f"Remaining execution budget (steps): {remaining_steps}\n\n"
            f"Executed steps and outputs so far:\n{executed_json}\n\n"
            f"Allowed tools (IDs):\n{allowed_tools if allowed_tools else 'ANY'}\n\n"
            "Tool input schemas (mapping from tool_id to JSON Schema):\n"
            f"{schemas_json}"
        )

        log.info(f"Planning next step: remaining_steps={remaining_steps}, executed_steps={len(executed_steps or [])}")

        resp = await self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "next_step_plan",
                    "strict": False,
                    "schema": self._NEXT_STEP_PLAN_SCHEMA,
                }
            },
        )

        data = json.loads(self._extract_output_text(resp))
        plan = NextStepPlan(**data)
        log.info(f"Data: {data}")
        log.info(
            f"Planner output: done={plan.done} tool_name={plan.tool_name} has_arguments={bool(plan.arguments)}"
        )
        return plan

    @staticmethod
    def _extract_output_text(resp) -> str:
        """Extract the text output from an OpenAI Responses API response object."""
        if getattr(resp, "output_text", None):
            return resp.output_text
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    return c.text
        raise RuntimeError("Could not extract output_text from response")

    @staticmethod
    def _summarize_executed_steps(executed_steps: list[dict]) -> str:
        """Serialize executed steps to JSON for inclusion in the planner prompt."""
        if not executed_steps:
            return "None"
        return json.dumps(executed_steps, ensure_ascii=False)
