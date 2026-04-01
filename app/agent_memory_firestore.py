import logging

from google.cloud.firestore_v1.async_client import AsyncClient

from app.agent_memory import AgentMemory

log = logging.getLogger("agent_memory_firestore")

_COLLECTION = "agent_memory"


class FirestoreAgentMemory(AgentMemory):
    def __init__(self, client: AsyncClient):
        self._db = client

    async def _load_all(self, user_email: str) -> list[dict]:
        try:
            doc = await self._db.collection(_COLLECTION).document(user_email).get()
            if doc.exists:
                return doc.to_dict().get("items", [])
        except Exception:
            log.exception("Failed to load agent memory for %s", user_email)
        return []

    async def _save_all(self, user_email: str, items: list[dict]) -> None:
        try:
            await self._db.collection(_COLLECTION).document(user_email).set({"items": items})
        except Exception:
            log.exception("Failed to save agent memory for %s", user_email)
