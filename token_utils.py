from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


def get_valid_token(token_file: Path) -> str:
    if not token_file.exists():
        raise FileNotFoundError(
            f"Token file not found: {token_file}. Run access_token_getter.py first."
        )

    creds = Credentials.from_authorized_user_file(str(token_file))
    should_persist = False

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            should_persist = True
        else:
            raise ValueError(
                "Token is invalid and cannot be refreshed. Run access_token_getter.py again."
            )

    if should_persist:
        token_file.write_text(creds.to_json(), encoding="utf-8")

    token = (creds.token or "").strip()
    if not token:
        raise ValueError(f"Missing token value in {token_file}")

    return token
