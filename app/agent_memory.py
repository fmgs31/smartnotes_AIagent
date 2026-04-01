import logging
from abc import ABC, abstractmethod

from app.schemas import DriveItem

log = logging.getLogger("agent_memory")


class AgentMemory(ABC):

    async def load_context_items(self, user_email: str) -> list[dict]:
        """Return items for the planner context: folders (access_freq >= 1) and frequent files (access_freq > 1)."""
        all_items = await self._load_all(user_email)
        context = self._filter_context_items(all_items)
        log.info("Loaded %d context items (of %d stored) from memory for %s", len(context), len(all_items), user_email)
        return context

    async def update(self, user_email: str, new_items: list[DriveItem]) -> None:
        """Merge new_items into stored memory, incrementing access_freq for each."""
        all_items = await self._load_all(user_email)
        merged = {i["id"]: i for i in all_items if i.get("id")}
        for item in new_items:
            d = item.model_dump()
            existing = merged.get(item.id)
            d["access_freq"] = (existing.get("access_freq", 0) if existing else 0) + 1
            merged[item.id] = d
        await self._save_all(user_email, list(merged.values()))
        log.info("Saved %d Drive items to memory for %s", len(merged), user_email)

    @staticmethod
    def _filter_context_items(items: list[dict]) -> list[dict]:
        result = []
        for item in items:
            freq = item.get("access_freq", 0)
            if item.get("type") == "folder":
                if freq >= 1:
                    result.append(item)
            else:
                if freq > 1:
                    result.append(item)
        return result

    @abstractmethod
    async def _load_all(self, user_email: str) -> list[dict]: ...

    @abstractmethod
    async def _save_all(self, user_email: str, items: list[dict]) -> None: ...
