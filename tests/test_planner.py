import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.planner import Planner


@pytest.fixture
def write_prompt_map(tmp_path: Path, monkeypatch):
    """Write prompt files + model_prompt_map.json under tmp_path and point settings at it."""
    def _write(default_model: str, models: dict[str, dict[str, str]]) -> Path:
        for attrs in models.values():
            (tmp_path / attrs["prompt_file"]).write_text(f"PROMPT:{attrs['prompt_file']}", encoding="utf-8")
        map_path = tmp_path / "model_prompt_map.json"
        map_path.write_text(json.dumps({"default_model": default_model, "models": models}), encoding="utf-8")
        monkeypatch.setattr("app.planner.settings.model_prompt_map_path", str(map_path))
        monkeypatch.setattr("app.planner._PROMPTS_FOLDER", tmp_path)
        return map_path
    return _write


class TestSystemPromptFromModelName:
    def test_exact_prefix_match_returns_matching_prompt(self, write_prompt_map):
        write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        planner = Planner(client=None)
        assert planner._system_prompt_from_model_name("gpt-5-nano") == "PROMPT:p_nano.txt"
        assert planner._system_prompt_from_model_name("gpt-5.4-nano") == "PROMPT:p_54_nano.txt"

    def test_full_versioned_model_resolves_via_prefix(self, write_prompt_map):
        write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        planner = Planner(client=None)
        assert planner._system_prompt_from_model_name("gpt-5-nano-2025-08-07") == "PROMPT:p_nano.txt"
        assert planner._system_prompt_from_model_name("gpt-5.4-nano-2025-08-07") == "PROMPT:p_54_nano.txt"

    def test_unknown_model_falls_back_to_default(self, write_prompt_map):
        write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        planner = Planner(client=None)
        assert planner._system_prompt_from_model_name("claude-opus-4-6") == "PROMPT:p_nano.txt"

    def test_raises_when_neither_requested_nor_default_matches(self, write_prompt_map):
        write_prompt_map(
            default_model="does-not-match-anything",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        planner = Planner(client=None)
        with pytest.raises(RuntimeError, match="No prompt found"):
            planner._system_prompt_from_model_name("claude-opus-4-6")

    def test_prompt_files_loaded_once_at_init(self, write_prompt_map, tmp_path: Path):
        write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        planner = Planner(client=None)
        (tmp_path / "p_nano.txt").write_text("MUTATED", encoding="utf-8")
        assert planner._system_prompt_from_model_name("gpt-5-nano") == "PROMPT:p_nano.txt"


def _mock_openai_client(output_text: str = '{"done": true}'):
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=MagicMock(output_text=output_text, usage=None))
    return client


class TestPlanNextStepModelResolution:
    """Verify the model ID sent to OpenAI is the 'full_name' from the map, not the prefix."""

    @pytest.mark.xfail(reason="problem 1: full_name from the map is not yet used when calling OpenAI")
    async def test_request_model_prefix_resolves_to_full_name(self, write_prompt_map):
        write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        client = _mock_openai_client()
        planner = Planner(client=client)
        await planner.plan_next_step(text="hi", allowed_tools=None, planner_model="gpt-5.4-nano")
        assert client.responses.create.call_args.kwargs["model"] == "gpt-5.4-nano-2025-08-07"

    @pytest.mark.xfail(reason="problem 1: full_name from the map is not yet used when calling OpenAI")
    async def test_no_request_model_uses_default_full_name(self, write_prompt_map):
        write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        client = _mock_openai_client()
        planner = Planner(client=client)
        await planner.plan_next_step(text="hi", allowed_tools=None)
        assert client.responses.create.call_args.kwargs["model"] == "gpt-5-nano-2025-08-07"
