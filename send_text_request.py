import json
from pathlib import Path

import requests

RUN_URL = "http://localhost:8001/run"
TOKEN_FILE = Path("gcp_token.json")

# Edit this variable with the text you want to send.
TEXT = "list the last 10 spreadsheets. Then tell me what's on the first written cell of the first one"


def load_token(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    token = payload.get("token", "").strip()
    if not token:
        raise ValueError(f"Missing 'token' in {path}")
    return token


def main() -> None:
    token = load_token(TOKEN_FILE)
    body = {
        "text": TEXT,
        "mcp_token": token,
    }

    response = requests.post(RUN_URL, json=body, timeout=120)
    print(f"Status: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
