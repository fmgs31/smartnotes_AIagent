import httpx
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings
from app.utils import load_json

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def _load_allowed_emails() -> set[str]:
    # Prefer the env var (Cloud Run); fall back to the JSON file (local dev).
    if settings.allowed_emails:
        return {e.strip().lower() for e in settings.allowed_emails.split(",") if e.strip()}
    data = load_json(settings.allowed_users_path)
    return {email.lower() for email in data.get("allowed_emails", [])}


# Loaded once at startup.
_allowed_emails: set[str] = _load_allowed_emails()


def require_api_key(x_api_key: str | None = Security(_api_key_header)) -> None:
    if x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def require_authorized_user(token: str) -> str:
    """Validate the GCP token with Google and verify the user is in the allowlist.

    Returns the user's email on success.
    Raises HTTPException with a specific message on every failure mode.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(_GOOGLE_TOKENINFO_URL, params={"access_token": token})

    if resp.status_code != 200:
        error_data = resp.json()
        error_desc = (
            error_data.get("error_description")
            or error_data.get("error")
            or "Unknown error"
        )
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or expired GCP token: {error_desc}",
        )

    data = resp.json()
    email = (data.get("email") or "").lower()

    if not email:
        raise HTTPException(
            status_code=401,
            detail="GCP token does not include an email address",
        )

    if email not in _allowed_emails:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: {email} is not authorized to use this service",
        )

    return email
