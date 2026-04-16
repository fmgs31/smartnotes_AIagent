import json
import logging
import time
from datetime import date
from pathlib import Path

from openai import AsyncOpenAI

from app.config import settings
from app.schemas import NextStepPlan

log = logging.getLogger("llm_router")

_PROMPTS_FOLDER = Path(__file__).resolve().parent / "prompts"


class Planner:
    """Wraps the OpenAI Responses API to produce single-step plans for the agent."""

    _NEXT_STEP_PLAN_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "done": {"type": "boolean"},
            "last_step_check": {"type": "string"},
            "tool_name": {"type": "string"},
            "arguments": {"type": "object", "additionalProperties": True},
            "explanation": {"type": "string"},
            "summary": {"type": "string"},
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
        "required": ["done"],
    }

    def __init__(self, client: AsyncOpenAI):
        self._client = client
        prompt_map = json.loads(Path(settings.model_prompt_map_path).read_text(encoding="utf-8"))
        if "default_model" not in prompt_map or "models" not in prompt_map:
            raise ValueError("model_prompt_map.json must define 'default_model' and 'models'")
        self._models_mapping: dict[str, dict] = prompt_map["models"]
        for model_name, attrs in self._models_mapping.items():
            missing = {"full_name", "vendor", "prompt_file"} - attrs.keys()
            if missing:
                raise ValueError(f"Model '{model_name}' missing keys: {sorted(missing)}")
        self._default_model: str = prompt_map["default_model"]
        self._model_prompt_map: dict[str, str] = {
            model_name: (_PROMPTS_FOLDER / attrs["prompt_file"]).read_text(encoding="utf-8").strip()
            for model_name, attrs in self._models_mapping.items()
        }

    def _system_prompt_from_model_name(self, model: str) -> str:
        for candidate in (model, self._default_model):
            for prefix, prompt in self._model_prompt_map.items():
                if candidate.startswith(prefix):
                    return prompt
        raise RuntimeError(f"No prompt found for model {model!r} or default {self._default_model!r}")

    async def plan_next_step(
        self,
        *,
        text: str,
        known_drive_items: list | None = None,
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
        core_context = (
            f"User request:\n{text}\n\n"
            f"{drive_items_section}"
            f"Remaining execution budget (steps): {remaining_steps}\n\n"
            f"Executed steps and outputs so far:\n{executed_json}\n\n"
        )
        user = (
            f"Current date: {date.today().isoformat()}\n\n"
            f"{core_context}"
            f"{guidelines_section}"
            f"Allowed tools (IDs):\n{allowed_tools if allowed_tools else 'ANY'}\n\n"
            "Tool input schemas (mapping from tool_id to JSON Schema):\n"
            f"{schemas_json}"
        )

        log.info(f"Calling LLM now: Planning next step: remaining_steps={remaining_steps}, executed_steps={len(executed_steps or [])}")
        log.debug("plan_next_step user prompt:\n%s", core_context)

        t0 = time.perf_counter()
        resolved_model = planner_model or self._default_model
        resp = await self._client.responses.create(
            model=resolved_model,
            input=[
                {"role": "system", "content": self._system_prompt_from_model_name(resolved_model)},
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
            f"Planner output: done={plan.done} tool_name={plan.tool_name} "
            f"has_arguments={bool(plan.arguments)} last_step_check={plan.last_step_check!r}"
        )
        return plan

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
