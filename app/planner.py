import json
import logging
import time
from datetime import date

from app.model_registry import ModelRegistry
from app.schemas import NextStepPlan

log = logging.getLogger("planner")


class Planner:
    """Wraps the OpenAI Responses API to produce single-step plans for the agent."""

    _NEXT_STEP_PLAN_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "last_step_check": {"type": "string"},
            "pending_actions": {"type": "string"},
            "next_tool_explanation": {"type": "string"},
            "next_tool": {"type": "string"},
            "arguments": {"type": "object", "additionalProperties": True},
            "done": {"type": "boolean"},
            "full_task_summary": {"type": "string"},
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

    def __init__(self, model_registry: ModelRegistry):
        self._model_registry = model_registry

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

        # history = steps[:-1] if steps else []
        history_section = (
            f"Earlier steps:\n{json.dumps(executed_steps, ensure_ascii=False)}\n\n"
        ) if executed_steps else ""
        last_step = executed_steps[-1] if executed_steps else None
        last_step_section = (
            f"Most recent step (step {last_step['step']}):\n"
            f"tool={last_step['next_tool']}  result={last_step['result_summary']!r}\n\n"
        ) if last_step else ""

        items_str = json.dumps(known_drive_items, ensure_ascii=False) if known_drive_items else None
        drive_items_section = f"Known Drive items:\n{items_str}\n\n" if items_str else ""
        guidelines_section = f"User guidelines:\n{user_guidelines}\n\n" if user_guidelines else ""
        core_context = (
            f"User request:\n{text}\n\n"
            f"{drive_items_section}"
            # f"Remaining execution budget (steps): {remaining_steps}\n\n"
            f"{history_section}"
            f"{last_step_section}"
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
        model = self._model_registry.resolve(planner_model)
        resp_text = await model.complete(
            input=[
                {"role": "system", "content": model.prompt},
                {"role": "user", "content": user},
            ],
            schema=self._NEXT_STEP_PLAN_SCHEMA,
        )
        log.info("plan_next_step took %.2fs", time.perf_counter() - t0)
        data = json.loads(resp_text)
        plan = NextStepPlan(**data)
        log.info(f"Data: {data}")
        log.info(
            f"Planner output: done={plan.done} tool_name={plan.next_tool} "
            f"has_arguments={bool(plan.arguments)} last_step_check={plan.last_step_check!r} "
            f"pending_actions={plan.pending_actions!r}"
        )
        return plan

    @staticmethod
    def _summarize_executed_steps(executed_steps: list[dict]) -> str:
        """Serialize executed steps to JSON for inclusion in the planner prompt."""
        if not executed_steps:
            return "None"
        return json.dumps(executed_steps, ensure_ascii=False)
