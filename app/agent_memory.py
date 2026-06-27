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
        merged: dict[str, DriveItem] = {i.id: i for i in all_items if i.id}
        for item in new_items:
            existing = merged.get(item.id)
            freq = (existing.access_freq if existing else 0) + 1
            merged[item.id] = item.model_copy(update={"access_freq": freq})
        # access_freq is exclude=True, so re-add it explicitly for persistence.
        to_save = [{**i.model_dump(), "access_freq": i.access_freq} for i in merged.values()]
        await self._save_all(user_email, to_save)
        log.info("Saved %d Drive items to memory for %s", len(merged), user_email)

    @staticmethod
    def _filter_context_items(items: list[DriveItem]) -> list[dict]:
        result = []
        for item in items:
            if item.type == "folder":
                if item.access_freq >= 1:
                    result.append(item.model_dump())
            else:
                if item.access_freq > 1:
                    result.append(item.model_dump())
        return result

    @abstractmethod
    async def _load_all(self, user_email: str) -> list[DriveItem]: ...

    @abstractmethod
    async def _save_all(self, user_email: str, items: list[dict]) -> None: ...
