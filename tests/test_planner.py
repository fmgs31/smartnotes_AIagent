import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

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
        # The system message is the resolved model's prompt followed by the static
        # tool list and schemas, so assert on the prefix.
        assert system_msg["content"].startswith("PROMPT:p_54_nano.txt"), (
            f"Expected system prompt to start with 'PROMPT:p_54_nano.txt' (from the requested "
            f"model's prompt file), got {system_msg['content']!r}"
        )


class TestPlanNextStepPromptLayout:
    """The static half of the prompt must sit in the system message, the per-request half in the user message."""

    @staticmethod
    def _messages(build_registry, **kwargs) -> tuple[str, str]:
        registry, complete_mocks = build_registry(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        planner = Planner(model_registry=registry)
        asyncio.run(planner.plan_next_step(**kwargs))
        messages = complete_mocks["gpt-5-nano-2025-08-07"].call_args.kwargs["input"]
        system = next(m for m in messages if m["role"] == "system")["content"]
        user = next(m for m in messages if m["role"] == "user")["content"]
        return system, user

    def test_tools_and_schemas_go_in_the_system_message(self, build_registry):
        # They are identical for every request and every step, so keeping them in the
        # system message keeps them inside the cached prompt prefix — and out of the
        # same turn as user-controlled text.
        system, user = self._messages(
            build_registry,
            text="hi",
            allowed_tools=["gw.list_drive_items"],
            tool_schemas={"gw.list_drive_items": {"type": "object"}},
        )
        assert "gw.list_drive_items" in system, (
            f"Expected the allowed tool id and its schema in the system message, got {system!r}"
        )
        assert "gw.list_drive_items" not in user, (
            f"Tool ids/schemas must not be re-sent in the user message, got {user!r}"
        )

    def test_system_message_carries_the_caller_local_time(self, build_registry):
        # The clock is app-asserted context, not something the caller typed, so it goes in
        # the system message — out of reach of the user text. It must show the caller's own
        # offset, not the server's UTC clock, since the planner resolves "yesterday" against it.
        now = datetime(2026, 8, 3, 0, 30, tzinfo=ZoneInfo("America/New_York"))
        system, user = self._messages(build_registry, text="hi", allowed_tools=None, now=now)
        assert "2026-08-03T00:30-04:00" in system, (
            f"Expected the caller-local timestamp with offset in the system message, got {system!r}"
        )
        assert "2026-08-03" not in user, (
            f"The timestamp must not also appear in the user message, got {user!r}"
        )

    def test_timestamp_is_the_last_thing_in_the_system_message(self, build_registry):
        # It is the only part of the system message that varies, so everything cacheable
        # (prompt, tool list, schemas) must precede it.
        now = datetime(2026, 8, 3, 0, 30, tzinfo=ZoneInfo("America/New_York"))
        system, _user = self._messages(
            build_registry,
            text="hi",
            allowed_tools=["gw.list_drive_items"],
            tool_schemas={"gw.list_drive_items": {"type": "object"}},
            now=now,
        )
        assert system.index("gw.list_drive_items") < system.index("2026-08-03"), (
            f"Tool ids/schemas must precede the timestamp so they stay in the cached prefix, got {system!r}"
        )

    def test_missing_now_falls_back_to_utc(self, build_registry):
        # Callers that omit `now` (e.g. direct planner use) still get a usable timestamp.
        system, _user = self._messages(build_registry, text="hi", allowed_tools=None)
        assert datetime.now(timezone.utc).strftime("%Y-%m-%d") in system, (
            f"Expected a UTC date in the system message when `now` is omitted, got {system!r}"
        )
