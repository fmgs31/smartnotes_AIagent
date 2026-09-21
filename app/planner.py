import json
import logging
import time
from datetime import datetime

from app.model_registry import ModelRegistry
from app.schemas import DriveItem, NextStepPlan, FullTaskSummary
from app.utils import now_in_timezone
from pydantic import ValidationError
log = logging.getLogger("planner")


def _now_line(now: datetime | None) -> str:
    """Render the request timestamp with its offset and zone so the model can resolve
    relative dates ("yesterday", "last Monday") for a caller in any timezone."""
    now = now or now_in_timezone(None)
    return f"Current date and time: {now.isoformat(timespec='minutes')} ({now:%A}, {now.tzinfo})\n\n"


_DRIVE_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "id": {"type": "string"},
        "type": {"type": "string"},
        "location": {"type": "string"},
    },
    "required": ["name", "id", "type", "location"],
}


class Planner:
    """Wraps the OpenAI Responses API to produce single-step plans for the agent."""

    # Planner emits the next step AND the final summary fields when done.
    # Used for models without a separate summarizer prompt.
    _STEP_AND_SUMMARY_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "last_step_check": {"type": "string"},
            "pending_actions": {"type": "string"},
            "locked_facts": {"type": "string"},
            "next_tool_explanation": {"type": "string"},
            "next_tool": {"type": "string"},
            "arguments": {"type": "object", "additionalProperties": True},
            "done": {"type": "boolean"},
            "full_task_summary": {"type": "string"},
            "items": {"type": "array", "items": _DRIVE_ITEM_SCHEMA},
        },
        "required": ["done"],
    }

    # Planner only chooses the next step or signals done.
    # Used when the model has a summarizer_prompt — a separate call produces the summary.
    _STEP_ONLY_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "last_step_check": {"type": "string"},
            "pending_actions": {"type": "string"},
            "locked_facts": {"type": "string"},
            "next_tool_explanation": {"type": "string"},
            "next_tool": {"type": "string"},
            "arguments": {"type": "object", "additionalProperties": True},
            "done": {"type": "boolean"},
        },
        "required": ["done"],
    }

    _SUMMARY_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "full_task_summary": {"type": "string"},
            "items": {"type": "array", "items": _DRIVE_ITEM_SCHEMA},
        },
        "required": ["full_task_summary", "items"],
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
        now: datetime | None = None,
    ) -> NextStepPlan:
        """Ask the LLM for the next step in a multi-step plan."""
        schemas_json = json.dumps(tool_schemas or {}, ensure_ascii=False)

        history_section = (
            f"Earlier steps:\n{json.dumps(executed_steps, ensure_ascii=False)}\n\n"
        ) if executed_steps else ""
        last_step = executed_steps[-1] if executed_steps else None
        last_step_section = (
            f"Most recent step (step {last_step['step']}):\n"
            f"tool={last_step['next_tool']}  "
            f"arguments={json.dumps(last_step['arguments'], ensure_ascii=False)}  "
            f"result={last_step['result_summary']!r}\n\n"
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
            f"{core_context}"
            f"{guidelines_section}"
        )

        model = self._model_registry.resolve(planner_model)
        log.info(
            f"Calling LLM now: Planning next step: model={model.full_name} prompt={model.prompt_file} "
            f"remaining_steps={remaining_steps}, executed_steps={len(executed_steps or [])}"
        )
        log.debug("plan_next_step user prompt:\n%s", core_context)

        # Everything the app asserts — capabilities and the clock — is system context;
        # the user turn holds only the request and what the tools returned for it.
        # The tool list and schemas are identical for every request and every step
        # (Agent caches tool discovery process-wide), so they stay inside the cached
        # prompt prefix. The timestamp goes LAST, after them: it is the one part of
        # the system message that varies, so anything before it stays cacheable.
        system = (
            f"{model.prompt}\n\n"
            f"Allowed tools (IDs):\n{allowed_tools if allowed_tools else 'ANY'}\n\n"
            "Tool input schemas (mapping from tool_id to JSON Schema):\n"
            f"{schemas_json}\n\n"
            f"{_now_line(now)}"
        )
        schema = self._STEP_ONLY_SCHEMA if model.summarizer_prompt else self._STEP_AND_SUMMARY_SCHEMA
        resp_text = await model.generate(
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            schema=schema,
        )
        data = json.loads(resp_text)
        log.info(f"LLM output: {data}")
        try:
            plan = NextStepPlan(**data)
        except ValidationError:
            log.error(f"LLM output failed NextStepPlan validation: {data}")  # Log for error level
            raise
        log.info(
            f"Planner output: done={plan.done} tool_name={plan.next_tool} "
            f"has_arguments={bool(plan.arguments)} last_step_check={plan.last_step_check!r} "
            f"pending_actions={plan.pending_actions!r}"
        )
        return plan

    async def summarize(
        self,
        *,
        text: str,
        done_plan: NextStepPlan,
        executed_steps: list[dict] | None = None,
        known_drive_items: list | None = None,
        user_guidelines: str | None = None,
        planner_model: str | None = None,
        now: datetime | None = None,
    ) -> FullTaskSummary:
        """Produce the final FullTaskSummary.

        If the resolved model has a summarizer_prompt, call the LLM with the executed
        steps to produce {full_task_summary, items}. Otherwise return the values the
        planner already emitted in done_plan.
        """
        model = self._model_registry.resolve(planner_model)
        if not model.summarizer_prompt:
            return FullTaskSummary(
                text=done_plan.full_task_summary,
                items=done_plan.items,
            )

        items_str = json.dumps(known_drive_items, ensure_ascii=False) if known_drive_items else "[]"
        steps_str = json.dumps(executed_steps or [], ensure_ascii=False)
        guidelines_section = f"User guidelines:\n{user_guidelines}\n\n" if user_guidelines else ""
        system = f"{model.summarizer_prompt}\n\n{_now_line(now)}"
        user = (
            f"User request:\n{text}\n\n"
            f"Known Drive items:\n{items_str}\n\n"
            f"{guidelines_section}"
            f"Executed steps:\n{steps_str}\n"
        )

        log.info(
            "Calling LLM now: Summarizing task: model=%s executed_steps=%d",
            model.full_name, len(executed_steps or []),
        )
        t0 = time.perf_counter()
        resp_text = await model.generate(
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            schema=self._SUMMARY_SCHEMA,
        )
        log.info("summarize took %.2fs", time.perf_counter() - t0)
        data = json.loads(resp_text)
        items = [DriveItem(**raw) for raw in data.get("items", [])]
        summary = FullTaskSummary(text=data.get("full_task_summary", ""), items=items)
        log.info(
            "Summarizer output: text_len=%d items=%d",
            len(summary.text), len(summary.items),
        )
        return summary

