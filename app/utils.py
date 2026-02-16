import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)
