from pathlib import Path

import requests
from token_utils import get_valid_token

RUN_URL = "http://localhost:8001/run"
TOKEN_FILE = Path("gcp_token.json")
APP_API_KEY = ""

# Edit this variable with the text you want to send.
TEXT = "¿Qué archivos hay en la carpeta a la pingüina?"

def main() -> None:
    token = get_valid_token(TOKEN_FILE)
    body = {
        "text": TEXT,
        "mcp_token": token,
    }
    headers = {"X-API-Key": APP_API_KEY}

    response = requests.post(RUN_URL, json=body, headers=headers, timeout=120)
    print(f"Status: {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
