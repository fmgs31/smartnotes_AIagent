import json
import logging
import time
from datetime import date
from pathlib import Path

from openai import AsyncOpenAI

from app.schemas import NextStepPlan, ProcedureSummary

log = logging.getLogger("llm_router")

_PROMPTS = Path(__file__).resolve().parent / "prompts"


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
        "required": ["done"],
    }

    _SUMMARIZE_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "location": {"type": "string"},
                    },
                    "required": ["name", "id", "type", "location"],
                },
            },
        },
        "required": ["text", "items"],
    }

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model
        self._system_prompt = (_PROMPTS / "next_step_system_prompt.txt").read_text(encoding="utf-8").strip()
        self._summarize_prompt = (_PROMPTS / "summarize_system_prompt.txt").read_text(encoding="utf-8").strip()

    async def plan_next_step(
        self,
        *,
        text: str,
        known_drive_items: list = [],
        user_guidelines: str | None = None,
        planner_model: str | None = None,
        allowed_tools: list[str] | None,
        tool_schemas: dict[str, dict] | None = None,
        executed_steps: list[dict] | None = None,
        remaining_steps: int = 1,
    ) -> NextStepPlan:
        """Ask the LLM for the next step in a multi-step plan."""
        schemas_json = json.dumps(tool_schemas or {}, ensure_ascii=False)
        executed_json = self._summarize_executed_steps(executed_steps or [])
        items_str = json.dumps(known_drive_items, ensure_ascii=False) if known_drive_items else None
        drive_items_section = f"Known Drive items:\n{items_str}\n\n" if items_str else ""
        guidelines_section = f"User guidelines:\n{user_guidelines}\n\n" if user_guidelines else ""
        user = (
            f"Current date: {date.today().isoformat()}\n\n"
            f"User request:\n{text}\n\n"
            f"{drive_items_section}"
            f"{guidelines_section}"
            f"Remaining execution budget (steps): {remaining_steps}\n\n"
            f"Executed steps and outputs so far:\n{executed_json}\n\n"
            f"Allowed tools (IDs):\n{allowed_tools if allowed_tools else 'ANY'}\n\n"
            "Tool input schemas (mapping from tool_id to JSON Schema):\n"
            f"{schemas_json}"
        )

        log.info(f"Calling LLM now: Planning next step: remaining_steps={remaining_steps}, executed_steps={len(executed_steps or [])}")

        t0 = time.perf_counter()
        resp = await self._client.responses.create(
            model=planner_model or self._model,
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
        log.info("plan_next_step took %.2fs", time.perf_counter() - t0)
        self._log_usage(resp, "plan_next_step")
        data = json.loads(self._extract_output_text(resp))
        plan = NextStepPlan(**data)
        log.info(f"Data: {data}")
        log.info(
            f"Planner output: done={plan.done} tool_name={plan.tool_name} has_arguments={bool(plan.arguments)}"
        )
        return plan

    async def summarize_procedure(
        self,
        *,
        text: str,
        executed_steps: list[dict],
        planner_model: str | None = None,
    ) -> ProcedureSummary:
        """Ask the LLM to summarize the completed procedure and extract Drive items."""
        user = (
            f"User request:\n{text}\n\n"
            f"Executed steps:\n{self._summarize_executed_steps(executed_steps)}"
        )

        log.info("Summarizing procedure (%d steps)", len(executed_steps))

        t0 = time.perf_counter()
        resp = await self._client.responses.create(
            model=planner_model or self._model,
            input=[
                {"role": "system", "content": self._summarize_prompt},
                {"role": "user", "content": user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "procedure_summary",
                    "strict": False,
                    "schema": self._SUMMARIZE_SCHEMA,
                }
            },
        )
        log.info("summarize_procedure took %.2fs", time.perf_counter() - t0)
        self._log_usage(resp, "summarize_procedure")
        data = json.loads(self._extract_output_text(resp))
        log.info("Summary: %s", data.get("text"))
        return ProcedureSummary(**data)

    @staticmethod
    def _log_usage(resp, call_name: str) -> None:
        usage = getattr(resp, "usage", None)
        if usage:
            log.info(
                "%s tokens — input: %s, output: %s, total: %s",
                call_name,
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )

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
