import pytest
from unittest.mock import AsyncMock

from app.agent_memory import AgentMemory
from app.schemas import DriveItem


# ── In-memory stub ────────────────────────────────────────────────────────────

class StubMemory(AgentMemory):
    """Minimal concrete implementation for testing the base class behaviour."""

    def __init__(self, stored: list[dict] | None = None):
        self._store: list[dict] = stored or []
        self.save_calls: list[tuple[str, list[dict]]] = []

    async def _load_all(self, user_email: str) -> list[dict]:
        return list(self._store)

    async def _save_all(self, user_email: str, items: list[dict]) -> None:
        self._store = items
        self.save_calls.append((user_email, items))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _item(id: str, name: str, type: str = "document", freq: int = 0) -> dict:
    return {"id": id, "name": name, "type": type, "location": "/", "access_freq": freq}


def _drive_item(id: str, name: str, type: str = "document") -> DriveItem:
    return DriveItem(id=id, name=name, type=type)


# ── _filter_context_items ─────────────────────────────────────────────────────

class TestFilterContextItems:
    def test_folder_with_freq_1_is_included(self):
        items = [_item("f1", "Docs", type="folder", freq=1)]
        assert AgentMemory._filter_context_items(items) == items

    def test_folder_with_freq_0_is_excluded(self):
        items = [_item("f1", "Docs", type="folder", freq=0)]
        assert AgentMemory._filter_context_items(items) == []

    def test_folder_missing_freq_is_excluded(self):
        items = [{"id": "f1", "name": "Docs", "type": "folder"}]
        assert AgentMemory._filter_context_items(items) == []

    def test_file_with_freq_2_is_included(self):
        items = [_item("d1", "Notes", freq=2)]
        assert AgentMemory._filter_context_items(items) == items

    def test_file_with_freq_1_is_excluded(self):
        items = [_item("d1", "Notes", freq=1)]
        assert AgentMemory._filter_context_items(items) == []

    def test_file_with_freq_0_is_excluded(self):
        items = [_item("d1", "Notes", freq=0)]
        assert AgentMemory._filter_context_items(items) == []

    def test_mixed_items_filtered_correctly(self):
        items = [
            _item("f1", "Folder A", type="folder", freq=3),
            _item("f2", "Folder B", type="folder", freq=0),
            _item("d1", "Doc A", freq=2),
            _item("d2", "Doc B", freq=1),
            _item("d3", "Doc C", freq=0),
        ]
        result = AgentMemory._filter_context_items(items)
        ids = [i["id"] for i in result]
        assert ids == ["f1", "d1"]

    def test_empty_list_returns_empty(self):
        assert AgentMemory._filter_context_items([]) == []


# ── load_context_items ────────────────────────────────────────────────────────

class TestLoadContextItems:
    async def test_returns_only_filtered_items(self):
        stored = [
            _item("f1", "Folder", type="folder", freq=1),
            _item("d1", "Doc", freq=2),
            _item("d2", "Rare", freq=0),
        ]
        mem = StubMemory(stored)
        result = await mem.load_context_items("user@example.com")
        ids = [i["id"] for i in result]
        assert ids == ["f1", "d1"]

    async def test_empty_store_returns_empty(self):
        mem = StubMemory([])
        result = await mem.load_context_items("user@example.com")
        assert result == []

    async def test_all_below_threshold_returns_empty(self):
        stored = [_item("d1", "Doc", freq=1), _item("f1", "Folder", type="folder", freq=0)]
        mem = StubMemory(stored)
        result = await mem.load_context_items("user@example.com")
        assert result == []


# ── update ────────────────────────────────────────────────────────────────────

class TestUpdate:
    async def test_new_item_gets_freq_1(self):
        mem = StubMemory([])
        await mem.update("u@x.com", [_drive_item("d1", "Doc")])
        saved = mem._store
        assert len(saved) == 1
        assert saved[0]["id"] == "d1"
        assert saved[0]["access_freq"] == 1

    async def test_existing_item_freq_incremented(self):
        mem = StubMemory([_item("d1", "Doc", freq=2)])
        await mem.update("u@x.com", [_drive_item("d1", "Doc")])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d1"]["access_freq"] == 3

    async def test_untouched_items_preserved(self):
        mem = StubMemory([
            _item("d1", "Doc A", freq=5),
            _item("d2", "Doc B", freq=1),
        ])
        await mem.update("u@x.com", [_drive_item("d1", "Doc A")])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d2"]["access_freq"] == 1  # unchanged

    async def test_multiple_new_items(self):
        mem = StubMemory([_item("d1", "Doc A", freq=1)])
        await mem.update("u@x.com", [
            _drive_item("d1", "Doc A"),
            _drive_item("d2", "Doc B"),
        ])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d1"]["access_freq"] == 2
        assert saved["d2"]["access_freq"] == 1

    async def test_update_calls_save_all(self):
        mem = StubMemory([])
        await mem.update("u@x.com", [_drive_item("d1", "Doc")])
        assert len(mem.save_calls) == 1
        assert mem.save_calls[0][0] == "u@x.com"

    async def test_empty_new_items_still_saves(self):
        mem = StubMemory([_item("d1", "Doc", freq=3)])
        await mem.update("u@x.com", [])
        assert len(mem.save_calls) == 1
        assert len(mem._store) == 1  # existing item preserved

    async def test_item_name_updated_on_merge(self):
        mem = StubMemory([_item("d1", "Old Name", freq=1)])
        await mem.update("u@x.com", [_drive_item("d1", "New Name")])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d1"]["name"] == "New Name"
