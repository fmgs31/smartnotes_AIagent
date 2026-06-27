import logging

import httpx
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
import secrets
from app.config import settings
from app.utils import load_json

log = logging.getLogger("auth")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def load_allowed_emails() -> set[str]:
    """Load the set of allowed emails from env var or JSON file (fallback)."""
    if settings.allowed_emails:
        return {e.strip().lower() for e in settings.allowed_emails.split(",") if e.strip()}
    data = load_json(settings.allowed_users_path)
    return {email.lower() for email in data.get("allowed_emails", [])}


class Auth:
    """Handles API key validation and GCP token-based user authorization."""

    def __init__(self, api_key: str, allowed_emails: set[str]):
        self._api_key = api_key
        self._allowed_emails = allowed_emails

    def require_api_key(self, x_api_key: str | None = Security(_api_key_header)) -> None:
        """FastAPI dependency: reject requests with a missing or wrong API key."""
        if not secrets.compare_digest(x_api_key or "", self._api_key):
            received = f"{len(x_api_key)} chars" if x_api_key else "none"
            log.debug(f"API key mismatch: received={received}, expected={len(self._api_key)} chars")
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    async def require_authorized_user(self, token: str) -> str:
        """
        Validate the GCP token with Google and verify the user is in the allowlist.
        Returns the user's email on success.
        Raises HTTPException with a specific message on every failure mode.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(_GOOGLE_TOKENINFO_URL, params={"access_token": token})

        if resp.status_code != 200:
            try:
                error_data = resp.json()
                error_desc = (
                    error_data.get("error_description")
                    or error_data.get("error")
                    or "Unknown error"
                )
            except ValueError:
                error_desc = resp.text or "Unknown error"
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

        if email not in self._allowed_emails:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: {email} is not authorized to use this service",
            )

        return email
