import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI, RateLimitError


def _extract_openai_output_text(resp) -> str:
    if getattr(resp, "output_text", None):
        return resp.output_text
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                return c.text
    raise RuntimeError("Could not extract output_text from response")


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)(ms|s)", re.IGNORECASE)


def _retry_after_seconds(err: RateLimitError) -> float:
    msg = str(err)
    m = _RETRY_AFTER_RE.search(msg)
    if not m:
        return 1.0
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "ms" else value


_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return the process-wide AsyncOpenAI client, creating it on first use.

    Shared by every model's generate callable and by the audio transcription
    path in main.py so we keep a single connection pool instead of one per model.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


def _make_openai_generate_fn(full_name: str, reasoning_effort: str | None) -> Callable:
    """Build the async generate callable for one OpenAI model (binds the model name and effort)."""
    client = get_openai_client()
    # Unset = the model's API default, which differs per model (e.g. gpt-5.4-nano
    # defaults to "none", gpt-5.6-luna to "medium").
    reasoning = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}

    async def _generate(*, input: list[dict], schema: dict) -> str:
        # Retry on OpenAI RateLimitError (HTTP 429). TPM is a rolling 1-minute
        # window, so chatty bursts (looping planner, functional tests) briefly
        # starve the next call. OpenAI returns the exact wait in the error body
        # ("try again in Xms/Xs" — observed 128ms–1.86s on the 200k TPM tier).
        # We honor that hint but floor at 2^attempt s so repeated failures back
        # off instead of tight-looping against a still-full window.
        attempt = 0
        while True:
            try:
                resp = await client.responses.create(
                    model=full_name,
                    input=input,
                    **reasoning,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "response_schema",
                            "strict": False,
                            "schema": schema,
                        }
                    },
                )
                return _extract_openai_output_text(resp)
            except RateLimitError as err:
                attempt += 1
                if attempt > 5:
                    raise
                delay = max(_retry_after_seconds(err), 2.0 ** attempt)
                await asyncio.sleep(delay)

    return _generate


# Vendor name -> factory: (full_name: str, reasoning_effort: str | None) -> async callable(*, input, schema) -> str
# Add an entry here when introducing a new vendor.
_CLIENT_FACTORIES: dict[str, Callable[[str, str | None], Callable]] = {
    "open_ai": _make_openai_generate_fn,
}


class ResolvedModel:
    """Everything needed to make one LLM call: name, prompt, and a vendor-agnostic generate callable."""

    def __init__(
        self,
        *,
        full_name: str,
        prompt: str,
        vendor: str,
        generate_fn: Callable,
        summarizer_prompt: str | None = None,
        prompt_file: str = "",
        reasoning_effort: str | None = None,
    ):
        self.full_name = full_name
        self.prompt = prompt
        self.vendor = vendor
        # Kept for logging only: which prompt version produced a given plan is the
        # first thing needed when diagnosing a failed request from its logs.
        self.prompt_file = prompt_file
        self.reasoning_effort = reasoning_effort
        self._generate_fn = generate_fn
        # When set, the planner emits a lean next-step output (no summary fields)
        # and a separate LLM call with this prompt produces full_task_summary + items.
        self.summarizer_prompt = summarizer_prompt

    async def generate(self, *, input: list[dict], schema: dict) -> str:
        return await self._generate_fn(input=input, schema=schema)


class ModelRegistry:
    """Resolves a model prefix to its vendor API ID, system prompt, and complete callable."""

    def __init__(self, *, map_path: str | Path, prompts_folder: str | Path):
        prompts_folder = Path(prompts_folder)
        prompt_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
        if "default_model" not in prompt_map or "models" not in prompt_map:
            raise ValueError("model prompt map must define 'default_model' and 'models'")
        models: dict[str, dict] = prompt_map["models"]
        for name, attrs in models.items():
            missing = {"full_name", "vendor", "prompt_file"} - attrs.keys()
            if missing:
                raise ValueError(f"Model '{name}' missing keys: {sorted(missing)}")
            if attrs["vendor"] not in _CLIENT_FACTORIES:
                raise ValueError(
                    f"Model '{name}' uses unknown vendor '{attrs['vendor']}'. "
                    f"Known vendors: {sorted(_CLIENT_FACTORIES)}"
                )

        def _load_summarizer_prompt(attrs: dict) -> str | None:
            path = attrs.get("summarizer_prompt_file")
            if not path:
                return None
            return (prompts_folder / path).read_text(encoding="utf-8").strip()

        self._default_model: str = prompt_map["default_model"]
        self._models: dict[str, ResolvedModel] = {
            name: ResolvedModel(
                full_name=attrs["full_name"],
                prompt=(prompts_folder / attrs["prompt_file"]).read_text(encoding="utf-8").strip(),
                vendor=attrs["vendor"],
                generate_fn=_CLIENT_FACTORIES[attrs["vendor"]](attrs["full_name"], attrs.get("reasoning_effort")),
                summarizer_prompt=_load_summarizer_prompt(attrs),
                prompt_file=attrs["prompt_file"],
                reasoning_effort=attrs.get("reasoning_effort"),
            )
            for name, attrs in models.items()
        }

        if self._default_model not in self._models:
            raise ValueError(
                f"default_model {self._default_model!r} is not a registered model key"
            )

    def resolve(self, model_key: str | None) -> ResolvedModel:
        key = model_key if model_key is not None else self._default_model
        if key not in self._models:
            raise ValueError(
                f"Model {key!r} is not registered. Available: {list(self._models)}"
            )
        return self._models[key]
