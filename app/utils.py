import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def stable_json(value: Any) -> str:
    # Serializes value to JSON with sorted keys so that two dicts with the same
    # content always produce identical strings — used for loop-detection signatures.
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)
