from abc import ABC, abstractmethod

from app.schemas import DriveItem


class AgentMemory(ABC):
    @abstractmethod
    async def load(self, user_email: str) -> list[dict]: ...

    @abstractmethod
    async def save(self, user_email: str, items: list[DriveItem]) -> None: ...
