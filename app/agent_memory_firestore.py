import logging

from google.cloud.firestore_v1.async_client import AsyncClient

from app.agent_memory import AgentMemory
from app.schemas import DriveItem

log = logging.getLogger("agent_memory_firestore")

_COLLECTION = "agent_memory"


class FirestoreAgentMemory(AgentMemory):
    def __init__(self, client: AsyncClient):
        self._db = client

    async def load(self, user_email: str) -> list[dict]:
        """Return stored Drive items for a user. Returns [] on any failure."""
        try:
            doc = await self._db.collection(_COLLECTION).document(user_email).get()
            if doc.exists:
                return doc.to_dict().get("items", [])
        except Exception:
            log.exception("Failed to load agent memory for %s", user_email)
        return []

    async def save(self, user_email: str, items: list[DriveItem]) -> None:
        """Persist Drive items for a user. Silently logs failures."""
        try:
            await self._db.collection(_COLLECTION).document(user_email).set(
                {"items": [item.model_dump() for item in items]}
            )
        except Exception:
            log.exception("Failed to save agent memory for %s", user_email)
