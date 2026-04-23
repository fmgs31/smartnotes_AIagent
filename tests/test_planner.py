import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.model_registry import ModelRegistry
from app.planner import Planner


@pytest.fixture
def build_registry(tmp_path: Path, monkeypatch):
    """Build a ModelRegistry against tmp prompts, with each model's complete_fn replaced by an AsyncMock."""
    def _build(default_model: str, models: dict[str, dict[str, str]]):
        for attrs in models.values():
            (tmp_path / attrs["prompt_file"]).write_text(f"PROMPT:{attrs['prompt_file']}", encoding="utf-8")
        map_path = tmp_path / "model_prompt_map.json"
        map_path.write_text(json.dumps({"default_model": default_model, "models": models}), encoding="utf-8")

        # Track which complete_fn was built for each full_name so tests can assert on them.
        complete_mocks: dict[str, AsyncMock] = {}

        def mock_factory(full_name):
            mock = AsyncMock(return_value='{"done": true}')
            complete_mocks[full_name] = mock
            return mock

        vendors = {attrs["vendor"] for attrs in models.values()}
        monkeypatch.setattr(
            "app.model_registry._CLIENT_FACTORIES",
            {v: mock_factory for v in vendors},
        )
        registry = ModelRegistry(map_path=map_path, prompts_folder=tmp_path)
        return registry, complete_mocks
    return _build


class TestPlanNextStepModelResolution:
    """Verify the complete_fn called is the one built for the correct full_name."""

    def test_request_model_key_resolves_to_full_name(self, build_registry):
        # Passing planner_model="gpt-5.4-nano" must route to the gpt-5.4-nano full_name's
        # complete_fn and leave the other model untouched.
        registry, complete_mocks = build_registry(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        planner = Planner(model_registry=registry)
        asyncio.run(planner.plan_next_step(text="hi", allowed_tools=None, planner_model="gpt-5.4-nano"))
        assert complete_mocks["gpt-5.4-nano-2025-08-07"].called, (
            "Expected 'gpt-5.4-nano-2025-08-07' complete_fn to be invoked for planner_model='gpt-5.4-nano', "
            "but it was never called."
        )
        assert not complete_mocks["gpt-5-nano-2025-08-07"].called, (
            "Expected 'gpt-5-nano-2025-08-07' complete_fn NOT to be invoked when a different "
            "planner_model was requested, but it was called."
        )

    def test_no_request_model_uses_default_full_name(self, build_registry):
        # Omitting planner_model → the default_model's complete_fn handles the request.
        registry, complete_mocks = build_registry(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        planner = Planner(model_registry=registry)
        asyncio.run(planner.plan_next_step(text="hi", allowed_tools=None))
        assert complete_mocks["gpt-5-nano-2025-08-07"].called, (
            "Expected default model 'gpt-5-nano-2025-08-07' complete_fn to be invoked when "
            "planner_model is omitted, but it was never called."
        )

    def test_invalid_llm_output_raises_validation_error(self, build_registry, monkeypatch):
        # When the LLM returns JSON that doesn't satisfy NextStepPlan (e.g. missing the
        # required `done` field), the planner must re-raise — not swallow and fall through
        # into `plan.done`, which would surface as an UnboundLocalError and hide the root cause.
        registry, complete_mocks = build_registry(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        complete_mocks["gpt-5-nano-2025-08-07"].return_value = '{"next_tool": "list_drive_items"}'
        planner = Planner(model_registry=registry)
        with pytest.raises(ValidationError):
            asyncio.run(planner.plan_next_step(text="hi", allowed_tools=None))

    def test_system_prompt_from_resolved_model_is_sent(self, build_registry):
        # Each model has its own prompt file; the system message sent to the LLM must reflect
        # the prompt of the resolved model (not the default).
        registry, complete_mocks = build_registry(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        planner = Planner(model_registry=registry)
        asyncio.run(planner.plan_next_step(text="hi", allowed_tools=None, planner_model="gpt-5.4-nano"))
        call_args = complete_mocks["gpt-5.4-nano-2025-08-07"].call_args
        messages = call_args.kwargs["input"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert system_msg["content"] == "PROMPT:p_54_nano.txt", (
            f"Expected system prompt 'PROMPT:p_54_nano.txt' (from the requested model's prompt file), "
            f"got {system_msg['content']!r}"
        )
