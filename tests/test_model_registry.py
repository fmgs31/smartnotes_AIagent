import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.model_registry import ModelRegistry


@pytest.fixture
def write_prompt_map(tmp_path: Path, monkeypatch):
    """Write prompt files + model_prompt_map.json under tmp_path; patch client factories with mocks.

    Returns a callable that writes the JSON and returns (map_path, client_mocks_by_vendor).
    """
    def _write(default_model: str, models: dict[str, dict[str, str]]):
        for attrs in models.values():
            (tmp_path / attrs["prompt_file"]).write_text(f"PROMPT:{attrs['prompt_file']}", encoding="utf-8")
        map_path = tmp_path / "model_prompt_map.json"
        map_path.write_text(json.dumps({"default_model": default_model, "models": models}), encoding="utf-8")

        vendors = {attrs["vendor"] for attrs in models.values()}
        complete_mocks = {v: AsyncMock(return_value='{"done": true}') for v in vendors}
        monkeypatch.setattr(
            "app.model_registry._CLIENT_FACTORIES",
            {v: (lambda full_name, reasoning_effort=None, v=v: complete_mocks[v]) for v in vendors},
        )
        return map_path, complete_mocks
    return _write


def _build(map_path: Path, prompts_folder: Path) -> ModelRegistry:
    return ModelRegistry(map_path=map_path, prompts_folder=prompts_folder)


class TestResolve:
    """ModelRegistry.resolve: maps a short model key (or None) to a ModelEntry."""

    def test_exact_prefix_match(self, write_prompt_map, tmp_path):
        # Each registered short key resolves to its declared full_name — no fuzzy matching.
        map_path, _ = write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        registry = _build(map_path, tmp_path)
        nano = registry.resolve("gpt-5-nano").full_name
        assert nano == "gpt-5-nano-2025-08-07", (
            f"Expected 'gpt-5-nano' → 'gpt-5-nano-2025-08-07', got {nano!r}"
        )
        nano_54 = registry.resolve("gpt-5.4-nano").full_name
        assert nano_54 == "gpt-5.4-nano-2025-08-07", (
            f"Expected 'gpt-5.4-nano' → 'gpt-5.4-nano-2025-08-07', got {nano_54!r}"
        )

    def test_unknown_model_raises(self, write_prompt_map, tmp_path):
        # Any unregistered key (typo, client-requested model that isn't configured) must raise.
        map_path, _ = write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        registry = _build(map_path, tmp_path)
        with pytest.raises(ValueError, match="not registered"):
            registry.resolve("claude-opus-4-6")

    def test_none_model_uses_default(self, write_prompt_map, tmp_path):
        # resolve(None) falls back to the default_model key.
        map_path, _ = write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        registry = _build(map_path, tmp_path)
        resolved = registry.resolve(None).full_name
        assert resolved == "gpt-5-nano-2025-08-07", (
            f"Expected None to resolve to the default 'gpt-5-nano-2025-08-07', got {resolved!r}"
        )

    def test_uses_correct_factory_per_vendor(self, write_prompt_map, tmp_path):
        # The ModelEntry's _generate_fn must come from the vendor-specific factory.
        map_path, generate_mocks = write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        registry = _build(map_path, tmp_path)
        fn = registry.resolve("gpt-5-nano")._generate_fn
        assert fn is generate_mocks["open_ai"], (
            f"Expected _generate_fn to be the open_ai vendor mock, got {fn!r}"
        )


class TestInitValidation:
    """ModelRegistry.__init__: strict validation of model_prompt_map.json structure."""

    def test_missing_top_level_keys_raises(self, tmp_path, monkeypatch):
        # The file must declare both 'default_model' and 'models'.
        monkeypatch.setattr("app.model_registry._CLIENT_FACTORIES", {"open_ai": lambda full_name, reasoning_effort=None: AsyncMock(return_value='{"done": true}')})
        map_path = tmp_path / "m.json"
        map_path.write_text(json.dumps({"models": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="default_model.*models"):
            _build(map_path, tmp_path)

    def test_missing_per_model_keys_raises(self, tmp_path, monkeypatch):
        # Each model entry must declare full_name + vendor + prompt_file.
        monkeypatch.setattr("app.model_registry._CLIENT_FACTORIES", {"open_ai": lambda full_name, reasoning_effort=None: AsyncMock(return_value='{"done": true}')})
        map_path = tmp_path / "m.json"
        map_path.write_text(json.dumps({
            "default_model": "x",
            "models": {"x": {"full_name": "x-1", "vendor": "open_ai"}},
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="missing keys.*prompt_file"):
            _build(map_path, tmp_path)

    def test_unknown_vendor_raises(self, write_prompt_map, tmp_path):
        # vendor must be one of the registered _CLIENT_FACTORIES keys.
        map_path, _ = write_prompt_map(
            default_model="m",
            models={"m": {"full_name": "m-1", "vendor": "open_ai", "prompt_file": "p.txt"}},
        )
        # write_prompt_map only registers vendors actually used; rewrite the JSON to use an unknown one.
        map_path.write_text(json.dumps({
            "default_model": "m",
            "models": {"m": {"full_name": "m-1", "vendor": "telepathy", "prompt_file": "p.txt"}},
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown vendor 'telepathy'"):
            _build(map_path, tmp_path)

    def test_default_model_not_registered_raises(self, write_prompt_map, tmp_path):
        # default_model must point at one of the keys in the 'models' dict.
        map_path, _ = write_prompt_map(
            default_model="does-not-exist",
            models={"gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p.txt"}},
        )
        with pytest.raises(ValueError, match="default_model.*not a registered"):
            _build(map_path, tmp_path)


class TestReasoningEffort:
    """Optional per-model reasoning_effort: passed to the vendor factory and sent to the API only when set."""

    def test_effort_passed_to_factory_and_exposed(self, tmp_path, monkeypatch):
        # The registry hands each model's reasoning_effort (or None) to its factory.
        seen: dict[str, str | None] = {}

        def factory(full_name, reasoning_effort):
            seen[full_name] = reasoning_effort
            return AsyncMock(return_value='{"done": true}')

        monkeypatch.setattr("app.model_registry._CLIENT_FACTORIES", {"open_ai": factory})
        (tmp_path / "p.txt").write_text("PROMPT", encoding="utf-8")
        map_path = tmp_path / "m.json"
        map_path.write_text(json.dumps({
            "default_model": "a",
            "models": {
                "a": {"full_name": "a-1", "vendor": "open_ai", "prompt_file": "p.txt", "reasoning_effort": "low"},
                "b": {"full_name": "b-1", "vendor": "open_ai", "prompt_file": "p.txt"},
            },
        }), encoding="utf-8")
        registry = _build(map_path, tmp_path)
        assert seen == {"a-1": "low", "b-1": None}, f"Unexpected efforts passed to factory: {seen!r}"
        assert registry.resolve("a").reasoning_effort == "low"
        assert registry.resolve("b").reasoning_effort is None

    @pytest.mark.parametrize("effort, expected", [("low", {"effort": "low"}), (None, None)])
    def test_openai_fn_sends_reasoning_only_when_set(self, monkeypatch, effort, expected):
        # Unset effort must not send a `reasoning` field, so the model keeps its API default.
        import asyncio
        from app import model_registry

        client = MagicMock()
        client.responses.create = AsyncMock(return_value=MagicMock(output_text='{"done": true}'))
        monkeypatch.setattr(model_registry, "get_openai_client", lambda: client)
        fn = model_registry._make_openai_generate_fn("m-1", effort)
        asyncio.run(fn(input=[], schema={}))
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs.get("reasoning") == expected, f"Expected reasoning={expected!r}, got {kwargs.get('reasoning')!r}"
