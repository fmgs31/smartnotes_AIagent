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
            {v: (lambda full_name, v=v: complete_mocks[v]) for v in vendors},
        )
        return map_path, complete_mocks
    return _write


def _build(map_path: Path, prompts_folder: Path) -> ModelRegistry:
    return ModelRegistry(map_path=map_path, prompts_folder=prompts_folder)


class TestResolve:
    def test_exact_prefix_match(self, write_prompt_map, tmp_path):
        map_path, _ = write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
                "gpt-5.4-nano": {"full_name": "gpt-5.4-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_54_nano.txt"},
            },
        )
        registry = _build(map_path, tmp_path)
        assert registry.resolve("gpt-5-nano").full_name == "gpt-5-nano-2025-08-07"
        assert registry.resolve("gpt-5.4-nano").full_name == "gpt-5.4-nano-2025-08-07"

    def test_unknown_model_raises(self, write_prompt_map, tmp_path):
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
        map_path, _ = write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        registry = _build(map_path, tmp_path)
        assert registry.resolve(None).full_name == "gpt-5-nano-2025-08-07"

    def test_uses_correct_factory_per_vendor(self, write_prompt_map, tmp_path):
        map_path, complete_mocks = write_prompt_map(
            default_model="gpt-5-nano",
            models={
                "gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p_nano.txt"},
            },
        )
        registry = _build(map_path, tmp_path)
        assert registry.resolve("gpt-5-nano")._complete_fn is complete_mocks["open_ai"]


class TestInitValidation:
    def test_missing_top_level_keys_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.model_registry._CLIENT_FACTORIES", {"open_ai": lambda full_name: AsyncMock(return_value='{"done": true}')})
        map_path = tmp_path / "m.json"
        map_path.write_text(json.dumps({"models": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="default_model.*models"):
            _build(map_path, tmp_path)

    def test_missing_per_model_keys_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.model_registry._CLIENT_FACTORIES", {"open_ai": lambda full_name: AsyncMock(return_value='{"done": true}')})
        map_path = tmp_path / "m.json"
        map_path.write_text(json.dumps({
            "default_model": "x",
            "models": {"x": {"full_name": "x-1", "vendor": "open_ai"}},
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="missing keys.*prompt_file"):
            _build(map_path, tmp_path)

    def test_unknown_vendor_raises(self, write_prompt_map, tmp_path):
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
        map_path, _ = write_prompt_map(
            default_model="does-not-exist",
            models={"gpt-5-nano": {"full_name": "gpt-5-nano-2025-08-07", "vendor": "open_ai", "prompt_file": "p.txt"}},
        )
        with pytest.raises(ValueError, match="default_model.*not a registered"):
            _build(map_path, tmp_path)
