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

    async def _load_all(self, user_email: str) -> list[DriveItem]:
        # Mirror the Firestore impl: stored dicts are deserialized into DriveItems.
        return [DriveItem(**i) for i in self._store]

    async def _save_all(self, user_email: str, items: list[dict]) -> None:
        self._store = items
        self.save_calls.append((user_email, items))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _item(id: str, name: str, type: str = "document", freq: int = 0) -> dict:
    return {"id": id, "name": name, "type": type, "location": "/", "access_freq": freq}


def _di(id: str, name: str, type: str = "document", freq: int = 0) -> DriveItem:
    return DriveItem(id=id, name=name, type=type, access_freq=freq)


def _drive_item(id: str, name: str, type: str = "document") -> DriveItem:
    return DriveItem(id=id, name=name, type=type)


# ── _filter_context_items ─────────────────────────────────────────────────────

class TestFilterContextItems:
    """AgentMemory._filter_context_items: per-type access_freq thresholds (folders: ≥1, files: ≥2).

    Input is DriveItems; output is plain dicts via model_dump (access_freq is exclude=True,
    so it never reaches the planner context).
    """

    def test_folder_with_freq_1_is_included(self):
        # Folders are cheap context → the threshold is 1.
        items = [_di("f1", "Docs", type="folder", freq=1)]
        result = AgentMemory._filter_context_items(items)
        ids = [i["id"] for i in result]
        assert ids == ["f1"], f"Folder with freq=1 must be included, got {ids!r}"
        assert "access_freq" not in result[0], (
            f"access_freq must not leak into context output, got {result[0]!r}"
        )

    def test_folder_with_freq_0_is_excluded(self):
        items = [_di("f1", "Docs", type="folder", freq=0)]
        result = AgentMemory._filter_context_items(items)
        assert result == [], (
            f"Folder with freq=0 must be excluded, got {result!r}"
        )

    def test_folder_missing_freq_is_excluded(self):
        # access_freq defaults to 0 → excluded.
        items = [_di("f1", "Docs", type="folder")]
        result = AgentMemory._filter_context_items(items)
        assert result == [], (
            f"Folder with default access_freq must be excluded, got {result!r}"
        )

    def test_file_with_freq_2_is_included(self):
        # Files need a higher threshold (2) to be worth the context budget.
        items = [_di("d1", "Notes", freq=2)]
        result = AgentMemory._filter_context_items(items)
        ids = [i["id"] for i in result]
        assert ids == ["d1"], f"File with freq=2 must be included, got {ids!r}"

    def test_file_with_freq_1_is_excluded(self):
        items = [_di("d1", "Notes", freq=1)]
        result = AgentMemory._filter_context_items(items)
        assert result == [], f"File with freq=1 must be excluded, got {result!r}"

    def test_file_with_freq_0_is_excluded(self):
        items = [_di("d1", "Notes", freq=0)]
        result = AgentMemory._filter_context_items(items)
        assert result == [], f"File with freq=0 must be excluded, got {result!r}"

    def test_mixed_items_filtered_correctly(self):
        # One mixed list exercises both thresholds + input order preservation.
        items = [
            _di("f1", "Folder A", type="folder", freq=3),
            _di("f2", "Folder B", type="folder", freq=0),
            _di("d1", "Doc A", freq=2),
            _di("d2", "Doc B", freq=1),
            _di("d3", "Doc C", freq=0),
        ]
        result = AgentMemory._filter_context_items(items)
        ids = [i["id"] for i in result]
        assert ids == ["f1", "d1"], (
            f"Expected only f1 (folder, freq=3) and d1 (file, freq=2), got {ids!r}"
        )

    def test_empty_list_returns_empty(self):
        result = AgentMemory._filter_context_items([])
        assert result == [], f"Empty input must return empty list, got {result!r}"


# ── load_context_items ────────────────────────────────────────────────────────

class TestLoadContextItems:
    """AgentMemory.load_context_items: fetches stored items then applies the filter."""

    async def test_returns_only_filtered_items(self):
        # Low-freq items must not leak to the caller.
        stored = [
            _item("f1", "Folder", type="folder", freq=1),
            _item("d1", "Doc", freq=2),
            _item("d2", "Rare", freq=0),
        ]
        mem = StubMemory(stored)
        result = await mem.load_context_items("user@example.com")
        ids = [i["id"] for i in result]
        assert ids == ["f1", "d1"], (
            f"Expected only above-threshold items (f1 folder, d1 file), got {ids!r}"
        )

    async def test_empty_store_returns_empty(self):
        mem = StubMemory([])
        result = await mem.load_context_items("user@example.com")
        assert result == [], f"Expected empty list from empty store, got {result!r}"

    async def test_all_below_threshold_returns_empty(self):
        # Items that exist but are all below-threshold → empty result.
        stored = [_item("d1", "Doc", freq=1), _item("f1", "Folder", type="folder", freq=0)]
        mem = StubMemory(stored)
        result = await mem.load_context_items("user@example.com")
        assert result == [], (
            f"Expected empty list when all items are below threshold, got {result!r}"
        )


# ── update ────────────────────────────────────────────────────────────────────

class TestUpdate:
    """AgentMemory.update: merges freshly-seen DriveItems into the stored freq map."""

    async def test_new_item_gets_freq_1(self):
        # A brand-new item starts at freq=1.
        mem = StubMemory([])
        await mem.update("u@x.com", [_drive_item("d1", "Doc")])
        saved = mem._store
        assert len(saved) == 1, f"Expected 1 saved item, got {len(saved)}: {saved!r}"
        assert saved[0]["id"] == "d1", f"Expected id='d1', got {saved[0]['id']!r}"
        assert saved[0]["access_freq"] == 1, (
            f"New item must start with access_freq=1, got {saved[0]['access_freq']}"
        )

    async def test_existing_item_freq_incremented(self):
        # Re-accessing an existing item increments its freq by 1.
        mem = StubMemory([_item("d1", "Doc", freq=2)])
        await mem.update("u@x.com", [_drive_item("d1", "Doc")])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d1"]["access_freq"] == 3, (
            f"Expected freq incremented 2→3, got {saved['d1']['access_freq']}"
        )

    async def test_untouched_items_preserved(self):
        # Items not in the new DriveItem list keep their existing freq.
        mem = StubMemory([
            _item("d1", "Doc A", freq=5),
            _item("d2", "Doc B", freq=1),
        ])
        await mem.update("u@x.com", [_drive_item("d1", "Doc A")])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d2"]["access_freq"] == 1, (
            f"Untouched item d2 must keep freq=1, got {saved['d2']['access_freq']}"
        )

    async def test_multiple_new_items(self):
        # Mixed merge: one existing (increment) + one new (freq=1).
        mem = StubMemory([_item("d1", "Doc A", freq=1)])
        await mem.update("u@x.com", [
            _drive_item("d1", "Doc A"),
            _drive_item("d2", "Doc B"),
        ])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d1"]["access_freq"] == 2, (
            f"Existing d1 must be incremented 1→2, got {saved['d1']['access_freq']}"
        )
        assert saved["d2"]["access_freq"] == 1, (
            f"New d2 must start at 1, got {saved['d2']['access_freq']}"
        )

    async def test_update_calls_save_all(self):
        # update() must persist exactly once, scoped to the user_email argument.
        mem = StubMemory([])
        await mem.update("u@x.com", [_drive_item("d1", "Doc")])
        assert len(mem.save_calls) == 1, (
            f"Expected exactly 1 _save_all call, got {len(mem.save_calls)}: {mem.save_calls!r}"
        )
        assert mem.save_calls[0][0] == "u@x.com", (
            f"Expected save scoped to 'u@x.com', got {mem.save_calls[0][0]!r}"
        )

    async def test_empty_new_items_still_saves(self):
        # Empty new-items list still writes (so the store is rewritten with existing items).
        mem = StubMemory([_item("d1", "Doc", freq=3)])
        await mem.update("u@x.com", [])
        assert len(mem.save_calls) == 1, (
            f"Expected save to run even with empty new items, got {len(mem.save_calls)} calls"
        )
        assert len(mem._store) == 1, (
            f"Expected existing item preserved after empty update, got {mem._store!r}"
        )

    async def test_item_name_updated_on_merge(self):
        # If Drive returns a renamed item, the stored name must follow the new one.
        mem = StubMemory([_item("d1", "Old Name", freq=1)])
        await mem.update("u@x.com", [_drive_item("d1", "New Name")])
        saved = {i["id"]: i for i in mem._store}
        assert saved["d1"]["name"] == "New Name", (
            f"Expected name updated to 'New Name', got {saved['d1']['name']!r}"
        )
