import json
import logging
from datetime import datetime, timezone as _timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger("api")


def now_in_timezone(tz_name: str | None) -> datetime:
    # The client tells us its IANA zone (e.g. "Europe/Madrid"); the server clock is
    # UTC, so "today" would be wrong near midnight for anyone east/west of it.
    # An unknown or missing zone degrades to UTC rather than failing the request.
    if not tz_name:
        return datetime.now(_timezone.utc)
    try:
        return datetime.now(ZoneInfo(tz_name))
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("Unknown timezone %r from client; falling back to UTC", tz_name)
        return datetime.now(_timezone.utc)


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
