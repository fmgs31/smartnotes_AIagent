"""
Conftest: env vars must be set before any app module is imported,
because pydantic-settings reads them at Settings() instantiation time.
"""
import os
from pathlib import Path

# ── Required env vars ─────────────────────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("APP_API_KEY", "test-app-api-key")
os.environ.setdefault("ALLOWED_EMAILS", "allowed@example.com")
os.environ.setdefault(
    "SERVERS_CONFIG_PATH",
    str(Path(__file__).parent / "servers.json"),
)
# ─────────────────────────────────────────────────────────────────────────────
