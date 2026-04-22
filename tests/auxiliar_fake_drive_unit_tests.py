"""
Unit tests for the fake MCP infrastructure (FakeDrive + FakeMCPClient).
These run via pytest and never call the real LLM or any external service.
"""
import pytest

from functional_tests.fake_drive import (
    DRIVE_ROOT,
    FakeDrive,
    FakeMCPClient,
    MIME_DOC,
    MIME_FOLDER,
    MIME_SHEET,
    _parse_range,
)
from functional_tests.test_config import TERAPIA_INITIAL_CONTENT


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def drive():
    # Each test gets a fresh in-memory copy of the template — no cleanup needed.
    return FakeDrive()


@pytest.fixture
def client(drive):
    return FakeMCPClient(drive)


# ─── FakeDrive: loading from disk ─────────────────────────────────────────────

class TestFakeDriveLoading:
    """FakeDrive initial state: template is loaded from disk with the expected items/content."""

    def test_root_item_exists(self, drive):
        # The Drive virtual root must always exist so children can use parent_id='root'.
        assert "root" in drive.items, (
            f"'root' item missing from drive.items. Keys: {list(drive.items)[:10]}..."
        )

    def test_top_level_folders_loaded(self, drive):
        # All template top-level folders must be present after a fresh load.
        names = {v["name"] for v in drive.items.values()}
        for expected in (
            "Ideas Sueltas", "Terapia-Autoanálisis", "Cuentos", "PrototipoJuego",
            "Trabajo", "Finanzas", "Viajes", "Libros", "Recetas",
            "Fotografías", "Proyectos Personales", "Salud", "Música", "Cursos",
        ):
            assert expected in names, (
                f"Expected top-level folder {expected!r} missing. "
                f"Present names (sample): {sorted(names)[:20]}"
            )

    def test_consciencia_subfolder_loaded(self, drive):
        # Nested folder 'consciencia' under Ideas Sueltas must be loaded as a folder.
        item = drive.find_by_name("consciencia")
        assert item is not None, "Subfolder 'consciencia' was not loaded from the drive template"
        assert item["mimeType"] == MIME_FOLDER, (
            f"'consciencia' must be a folder, got mimeType={item['mimeType']!r}"
        )

    def test_consciencia_parent_is_ideas_sueltas(self, drive):
        # Parent-child linkage survives the template load.
        consciencia = drive.find_by_name("consciencia")
        ideas_sueltas = drive.find_by_name("Ideas Sueltas")
        assert consciencia["parent_id"] == ideas_sueltas["id"], (
            f"Expected consciencia.parent_id == ideas_sueltas.id={ideas_sueltas['id']!r}, "
            f"got {consciencia['parent_id']!r}"
        )

    def test_terapia_doc_loaded(self, drive):
        # The Terapia document must be loaded as a Drive document (not a folder or sheet).
        item = drive.find_by_name("Terapia")
        assert item is not None, "'Terapia' document was not loaded from the drive template"
        assert item["mimeType"] == MIME_DOC, (
            f"'Terapia' must be a document, got mimeType={item['mimeType']!r}"
        )

    def test_terapia_content_matches_file(self, drive):
        # Content round-trip: what's on disk is what's in drive.contents.
        item = drive.find_by_name("Terapia")
        content = drive.contents[item["id"]]
        assert content == TERAPIA_INITIAL_CONTENT, (
            f"Terapia content mismatch.\nExpected: {TERAPIA_INITIAL_CONTENT!r}\nGot: {content!r}"
        )

    def test_spreadsheet_loaded(self, drive):
        item = drive.find_by_name("Prototipo Juego")
        assert item is not None, "'Prototipo Juego' spreadsheet was not loaded"
        assert item["mimeType"] == MIME_SHEET, (
            f"'Prototipo Juego' must be a spreadsheet, got mimeType={item['mimeType']!r}"
        )

    def test_spreadsheet_tab_loaded(self, drive):
        # Tab contents are loaded verbatim from the template JSON/tsv.
        item = drive.find_by_name("Prototipo Juego")
        rows = drive.sheets[item["id"]]["Ideas suelta 1"]
        assert rows == [["idea anterior 1"], ["idea anterior 2"]], (
            f"Initial rows of 'Ideas suelta 1' tab mismatch. Got: {rows!r}"
        )

    def test_gitkeep_files_ignored(self, drive):
        # .gitkeep is a filesystem-only marker — it must NOT become a drive item.
        names = {v["name"] for v in drive.items.values()}
        assert ".gitkeep" not in names, (
            f"'.gitkeep' must not be loaded as a drive item. Present names: {sorted(names)}"
        )

    def test_dotfiles_ignored(self, drive):
        # Generic version of the rule above: no dotfile should show up as a drive item.
        for key in drive.items:
            assert not key.startswith("."), (
                f"Dotfile leaked as drive item id: {key!r}"
            )


# ─── FakeDrive: instance isolation ────────────────────────────────────────────
#
# Replaces the old snapshot/restore tests. Each FakeDrive() is a fresh
# deep-copy of the cached template, so instances are fully independent.

class TestInstanceIsolation:
    """Each FakeDrive() must be a fresh, isolated deep-copy of the cached template."""

    def test_fresh_instance_matches_template(self):
        # A brand-new instance has the same Terapia content as the template.
        d = FakeDrive()
        terapia = d.find_by_name("Terapia")
        content = d.contents[terapia["id"]]
        assert content == TERAPIA_INITIAL_CONTENT, (
            f"Fresh instance's Terapia content differs from template.\n"
            f"Expected: {TERAPIA_INITIAL_CONTENT!r}\nGot: {content!r}"
        )

    def test_two_instances_have_equal_items(self):
        # Value equality between two fresh instances.
        d1, d2 = FakeDrive(), FakeDrive()
        assert set(d1.items.keys()) == set(d2.items.keys()), (
            f"Two fresh instances have different item id sets. "
            f"d1\\d2={set(d1.items)-set(d2.items)}, d2\\d1={set(d2.items)-set(d1.items)}"
        )
        assert d1.contents == d2.contents, "Two fresh instances have different contents dicts"
        assert d1.sheets == d2.sheets, "Two fresh instances have different sheets dicts"

    def test_items_dict_is_not_shared(self):
        # Identity check: the dicts must be distinct objects — not the same reference.
        d1, d2 = FakeDrive(), FakeDrive()
        assert d1.items is not d2.items, "d1.items and d2.items share the same dict (bad — mutations leak)"
        assert d1.contents is not d2.contents, "d1.contents and d2.contents share the same dict"
        assert d1.sheets is not d2.sheets, "d1.sheets and d2.sheets share the same dict"

    def test_new_doc_does_not_leak(self):
        # A doc created on d1 must not appear in an independently-created d2.
        d1 = FakeDrive()
        d1.create_doc("temp_doc", "root", "hello")
        assert d1.find_by_name("temp_doc") is not None, (
            "create_doc on d1 did not register 'temp_doc' — fixture broken"
        )
        d2 = FakeDrive()
        assert d2.find_by_name("temp_doc") is None, (
            "'temp_doc' created on d1 leaked into d2 — instances are not isolated"
        )

    def test_content_change_does_not_leak(self):
        # Mutating doc contents on d1 must not affect d2.
        d1 = FakeDrive()
        terapia = d1.find_by_name("Terapia")
        d1.write_doc(terapia["id"], "replaced content")
        d2 = FakeDrive()
        terapia2 = d2.find_by_name("Terapia")
        content = d2.contents[terapia2["id"]]
        assert content == TERAPIA_INITIAL_CONTENT, (
            f"Terapia content on d1 leaked into d2.\n"
            f"Expected (template): {TERAPIA_INITIAL_CONTENT!r}\nGot: {content!r}"
        )

    def test_new_folder_does_not_leak(self):
        # Folder created on d1 must not appear on d2.
        d1 = FakeDrive()
        ideas = d1.find_by_name("Ideas Sueltas")
        d1.create_folder("new_topic", ideas["id"])
        d2 = FakeDrive()
        assert d2.find_by_name("new_topic") is None, (
            "'new_topic' folder created on d1 leaked into d2"
        )

    def test_sheet_mutation_deep_copied(self):
        # Mutating a row list in place on d1 must not affect d2 — verifies the
        # nested sheets dict-of-list-of-list is deep-copied, not shallow-copied.
        d1 = FakeDrive()
        sheet = d1.find_by_name("Prototipo Juego")
        d1.sheets[sheet["id"]]["Ideas suelta 1"][0].append("injected")

        d2 = FakeDrive()
        sheet2 = d2.find_by_name("Prototipo Juego")
        rows = d2.sheets[sheet2["id"]]["Ideas suelta 1"]
        assert rows == [["idea anterior 1"], ["idea anterior 2"]], (
            f"Row mutation on d1 leaked into d2 (shallow-copy bug). Got d2 rows: {rows!r}"
        )

    def test_item_dict_mutation_deep_copied(self):
        # Mutating an item dict in place on d1 must not affect d2.
        d1 = FakeDrive()
        terapia_id = d1.find_by_name("Terapia")["id"]
        d1.items[terapia_id]["name"] = "Hacked"

        d2 = FakeDrive()
        name = d2.items[terapia_id]["name"]
        assert name == "Terapia", (
            f"Item dict mutation on d1 leaked into d2 (shallow-copy bug). Got d2 name: {name!r}"
        )

    def test_call_log_is_per_instance(self):
        # Each FakeDrive has its own call log.
        d1, d2 = FakeDrive(), FakeDrive()
        d1.log_call("t", {"x": 1})
        assert len(d1.get_calls()) == 1, f"Expected 1 call on d1, got {len(d1.get_calls())}"
        assert len(d2.get_calls()) == 0, (
            f"Expected 0 calls on d2 (separate instance), got {len(d2.get_calls())}"
        )

    def test_new_ids_do_not_collide_with_template(self):
        # The _counter in a fresh instance starts above the template's max id,
        # so newly created items don't clash with pre-existing template items.
        d = FakeDrive()
        template_ids = set(d.items.keys())
        new = d.create_doc("brand_new", "root")
        assert new["id"] not in template_ids, (
            f"Newly created id {new['id']!r} collided with an existing template id"
        )


class TestTemplateCaching:
    """FakeDrive._TEMPLATE is loaded from disk once and reused across instances."""

    def test_template_loaded_once(self):
        # Template object identity should be stable across instances.
        FakeDrive()
        first = FakeDrive._TEMPLATE
        FakeDrive()
        second = FakeDrive._TEMPLATE
        assert first is second, (
            "FakeDrive._TEMPLATE object identity changed between instantiations — "
            "template is being rebuilt from disk each time instead of being cached."
        )

    def test_mutating_instance_does_not_corrupt_template(self):
        # Mutations on a FakeDrive instance must NEVER write through to the cached template.
        d = FakeDrive()
        terapia = d.find_by_name("Terapia")
        d.write_doc(terapia["id"], "CORRUPTED")
        d.create_doc("junk", "root")

        tmpl = FakeDrive._TEMPLATE
        tmpl_content = tmpl["contents"][terapia["id"]]
        assert tmpl_content == TERAPIA_INITIAL_CONTENT, (
            f"Instance write_doc leaked into the cached template.\n"
            f"Expected template content: {TERAPIA_INITIAL_CONTENT!r}\nGot: {tmpl_content!r}"
        )
        tmpl_names = {i["name"] for i in tmpl["items"].values()}
        assert "junk" not in tmpl_names, (
            "Instance create_doc('junk') leaked into the cached template"
        )

    def test_template_counter_matches_max_template_id(self):
        # Every item id emitted by the loader is "item_<n>[_slug]"; the cached
        # counter must be ≥ the highest <n> so future allocations don't collide.
        FakeDrive()  # ensure template built
        tmpl = FakeDrive._TEMPLATE
        max_n = max(
            int(iid.split("_")[1])
            for iid in tmpl["items"]
            if iid.startswith("item_")
        )
        assert tmpl["counter"] >= max_n, (
            f"Template counter ({tmpl['counter']}) is below the max existing item id ({max_n}). "
            f"New allocations will collide with template items."
        )


class TestNoDiskWrites:
    """Mutations must not touch the drive template directory on disk."""

    def _snapshot_dir(self):
        return {p.relative_to(DRIVE_ROOT): p.stat().st_mtime_ns
                for p in DRIVE_ROOT.rglob("*") if p.is_file()}

    def test_create_doc_does_not_write_disk(self):
        # create_doc is an in-memory operation — no new file should appear on disk.
        before = self._snapshot_dir()
        d = FakeDrive()
        d.create_doc("should_not_appear", "root", "x")
        after = self._snapshot_dir()
        assert before == after, (
            f"create_doc modified the drive template directory on disk.\n"
            f"Diff: {set(after.items()) ^ set(before.items())}"
        )
        assert not (DRIVE_ROOT / "should_not_appear.txt").exists(), (
            "create_doc wrote 'should_not_appear.txt' to disk — it must stay in-memory only"
        )

    def test_write_doc_does_not_write_disk(self):
        # write_doc must mutate only the in-memory contents, never the file on disk.
        before = self._snapshot_dir()
        d = FakeDrive()
        terapia = d.find_by_name("Terapia")
        d.write_doc(terapia["id"], "CHANGED ON DISK?")
        after = self._snapshot_dir()
        assert before == after, (
            f"write_doc mutated the drive template directory on disk.\n"
            f"Diff: {set(after.items()) ^ set(before.items())}"
        )
        # Confirm the real template file is untouched
        path = DRIVE_ROOT / "Terapia-Autoanálisis" / "Terapia.txt"
        on_disk = path.read_text(encoding="utf-8")
        assert on_disk == TERAPIA_INITIAL_CONTENT, (
            f"write_doc corrupted the real template file at {path}.\n"
            f"Expected: {TERAPIA_INITIAL_CONTENT!r}\nGot: {on_disk!r}"
        )

    def test_create_folder_does_not_write_disk(self):
        # create_folder must not mkdir anything on disk.
        before = self._snapshot_dir()
        d = FakeDrive()
        d.create_folder("phantom_folder", "root")
        after = self._snapshot_dir()
        assert before == after, (
            f"create_folder modified the drive template directory on disk.\n"
            f"Diff: {set(after.items()) ^ set(before.items())}"
        )
        assert not (DRIVE_ROOT / "phantom_folder").exists(), (
            "create_folder created 'phantom_folder/' on disk — it must stay in-memory only"
        )

    def test_write_sheet_tab_does_not_write_disk(self):
        # write_sheet_tab updates FakeDrive.sheets but must not touch any file.
        before = self._snapshot_dir()
        d = FakeDrive()
        sheet = d.find_by_name("Prototipo Juego")
        d.write_sheet_tab(sheet["id"], "Ideas suelta 1", [["x", "y"], ["z"]])
        after = self._snapshot_dir()
        assert before == after, (
            f"write_sheet_tab modified the drive template directory on disk.\n"
            f"Diff: {set(after.items()) ^ set(before.items())}"
        )


# ─── FakeDrive: mutations ─────────────────────────────────────────────────────

class TestMutations:
    """FakeDrive mutation helpers (create_doc/folder/sheet, write_doc, write_sheet_tab)."""

    def test_create_doc(self, drive):
        # create_doc registers a new doc in items and stores its content.
        item = drive.create_doc("my note", "root", "content here")
        assert drive.find_by_name("my note") is not None, (
            "find_by_name('my note') returned None after create_doc — item not registered"
        )
        content = drive.contents[item["id"]]
        assert content == "content here", (
            f"Expected content 'content here' stored for the new doc, got {content!r}"
        )

    def test_create_folder(self, drive):
        # create_folder registers a folder-typed item.
        item = drive.create_folder("new_folder", "root")
        assert drive.find_by_name("new_folder") is not None, (
            "find_by_name('new_folder') returned None after create_folder — item not registered"
        )
        assert item["mimeType"] == MIME_FOLDER, (
            f"Expected mimeType={MIME_FOLDER!r} for a folder, got {item['mimeType']!r}"
        )

    def test_create_sheet(self, drive):
        # create_sheet registers a sheet-typed item with an empty tabs dict.
        item = drive.create_sheet("my_sheet", "root")
        assert drive.find_by_name("my_sheet") is not None, (
            "find_by_name('my_sheet') returned None after create_sheet — item not registered"
        )
        assert item["mimeType"] == MIME_SHEET, (
            f"Expected mimeType={MIME_SHEET!r} for a spreadsheet, got {item['mimeType']!r}"
        )
        tabs = drive.sheets[item["id"]]
        assert tabs == {}, f"Expected a freshly-created sheet to have no tabs, got {tabs!r}"

    def test_write_doc_updates_content(self, drive):
        # write_doc fully replaces the contents of the target doc.
        terapia = drive.find_by_name("Terapia")
        drive.write_doc(terapia["id"], "new content")
        content = drive.contents[terapia["id"]]
        assert content == "new content", (
            f"Expected write_doc to replace Terapia content with 'new content', got {content!r}"
        )

    def test_write_sheet_tab(self, drive):
        # write_sheet_tab replaces the rows of a named tab.
        sheet = drive.find_by_name("Prototipo Juego")
        drive.write_sheet_tab(sheet["id"], "Ideas suelta 1", [["a"], ["b"], ["c"]])
        rows = drive.sheets[sheet["id"]]["Ideas suelta 1"]
        assert rows == [["a"], ["b"], ["c"]], (
            f"Expected tab 'Ideas suelta 1' rows to be [['a'],['b'],['c']], got {rows!r}"
        )


# ─── Range parser ─────────────────────────────────────────────────────────────

class TestParseRange:
    """_parse_range: converts A1-notation into (sheet_name, start_row, start_col, end_row, end_col)."""

    def test_simple_cell(self):
        # A single cell: zero-based coords, no end range, no sheet prefix.
        sn, sr, sc, er, ec = _parse_range("A1")
        assert sn is None and sr == 0 and sc == 0 and er is None and ec is None, (
            f"Expected 'A1' → (None, 0, 0, None, None), got ({sn!r}, {sr}, {sc}, {er}, {ec})"
        )

    def test_range(self):
        # A two-cell range: both endpoints set, zero-based.
        sn, sr, sc, er, ec = _parse_range("B2:C5")
        assert sc == 1 and sr == 1 and ec == 2 and er == 4, (
            f"Expected 'B2:C5' → start=(1,1), end=(4,2); got start=({sr},{sc}), end=({er},{ec})"
        )

    def test_sheet_prefix(self):
        # Sheet prefix before '!' is captured as the sheet name.
        sn, sr, sc, er, ec = _parse_range("Sheet1!A1:Z10")
        assert sn == "Sheet1" and sr == 0 and sc == 0 and er == 9, (
            f"Expected 'Sheet1!A1:Z10' → sheet='Sheet1', start=(0,0), end_row=9; "
            f"got sheet={sn!r}, start=({sr},{sc}), end_row={er}"
        )


# ─── FakeMCPClient: tool listing ──────────────────────────────────────────────

class TestFakeMCPClientTools:
    """FakeMCPClient.list_tools: exposes the same tool catalog shape as the real MCP server."""

    async def test_list_tools_returns_tools(self, client):
        # Non-empty catalog — an empty list would mean the fake never registered anything.
        tools = await client.list_tools()
        assert len(tools) > 0, f"Expected list_tools() to return at least one tool, got {tools!r}"

    async def test_list_tools_have_name_and_schema(self, client):
        # Each tool descriptor must include 'name' and 'inputSchema', matching the real MCP format.
        tools = await client.list_tools()
        for t in tools:
            assert "name" in t, f"Tool descriptor missing 'name': {t!r}"
            assert "inputSchema" in t, f"Tool descriptor missing 'inputSchema': {t!r}"

    async def test_expected_tools_present(self, client):
        # Guard against silent removal of tools the agent/prompt depends on.
        tools = await client.list_tools()
        names = {t["name"] for t in tools}
        for expected in (
            "search_drive_files", "list_drive_items", "create_drive_file",
            "create_drive_folder", "update_drive_file",
            "get_doc_content", "modify_doc_text", "find_and_replace_doc",
            "read_sheet_values", "modify_sheet_values",
        ):
            assert expected in names, (
                f"Tool {expected!r} missing from FakeMCPClient. Present: {sorted(names)}"
            )


# ─── FakeMCPClient: tool dispatch ─────────────────────────────────────────────

class TestFakeMCPClientDispatch:
    """FakeMCPClient.call_tool: dispatching to each tool mutates FakeDrive as expected."""

    async def _call(self, client, tool, **kwargs):
        return await client.call_tool(token="t", tool_name=tool, arguments=kwargs)

    async def test_search_drive_files_finds_terapia(self, client, drive):
        # A name-contains query should surface the Terapia doc in the result string.
        result = await self._call(client, "search_drive_files", query="name contains 'Terapia'")
        assert "Terapia" in result, (
            f"Expected 'Terapia' in search result, got: {result!r}"
        )

    async def test_search_drive_files_no_match(self, client, drive):
        # A deliberately-unmatched query returns the "No files found" sentinel.
        result = await self._call(client, "search_drive_files", query="name contains 'nonexistent_xyz'")
        assert "No files found" in result, (
            f"Expected 'No files found' sentinel for unmatched query, got: {result!r}"
        )

    async def test_list_drive_items_root(self, client, drive):
        # Listing root must surface at least one known top-level item.
        result = await self._call(client, "list_drive_items", folder_id="root")
        assert "Ideas Sueltas" in result or "PrototipoJuego" in result, (
            f"Expected at least one known top-level item in root listing, got: {result!r}"
        )

    async def test_get_doc_content(self, client, drive):
        # get_doc_content returns the raw doc contents — must include a known substring.
        terapia = drive.find_by_name("Terapia")
        result = await self._call(client, "get_doc_content", document_id=terapia["id"])
        assert "infancia" in result, (
            f"Expected 'infancia' (from Terapia template) in get_doc_content result, got: {result!r}"
        )

    async def test_create_drive_file_creates_doc(self, client, drive):
        # create_drive_file with document MIME must register the item AND store its content.
        ideas = drive.find_by_name("Ideas Sueltas")
        result = await self._call(
            client, "create_drive_file",
            file_name="Nueva nota", folder_id=ideas["id"],
            mime_type="application/vnd.google-apps.document",
            content="idea aquí",
        )
        assert "Nueva nota" in result, (
            f"Expected the new file name in the result string, got: {result!r}"
        )
        item = drive.find_by_name("Nueva nota")
        assert item is not None, "find_by_name('Nueva nota') returned None — item not registered"
        stored = drive.contents[item["id"]]
        assert stored == "idea aquí", (
            f"Expected stored content 'idea aquí', got {stored!r}"
        )

    async def test_create_drive_folder(self, client, drive):
        # create_drive_folder registers the folder under the given parent.
        ideas = drive.find_by_name("Ideas Sueltas")
        result = await self._call(
            client, "create_drive_folder",
            folder_name="nuevo tema", parent_folder_id=ideas["id"],
        )
        assert "nuevo tema" in result, (
            f"Expected 'nuevo tema' in create_drive_folder result, got: {result!r}"
        )
        assert drive.find_by_name("nuevo tema") is not None, (
            "'nuevo tema' folder was not registered after create_drive_folder"
        )

    async def test_update_drive_file_trashed(self, client, drive):
        # update_drive_file(trashed=True) flips the item's trashed flag.
        terapia = drive.find_by_name("Terapia")
        result = await self._call(client, "update_drive_file", file_id=terapia["id"], trashed=True)
        assert "Updated" in result, f"Expected 'Updated' in result, got: {result!r}"
        assert terapia.get("trashed") is True, (
            f"Expected Terapia.trashed=True after update, got {terapia.get('trashed')!r}"
        )

    async def test_modify_doc_text_appends(self, client, drive):
        # start_index at EOF + new text → content is appended, original preserved.
        terapia = drive.find_by_name("Terapia")
        original = drive.contents[terapia["id"]]
        result = await self._call(
            client, "modify_doc_text",
            document_id=terapia["id"],
            start_index=len(original),
            text="\nappended text",
        )
        assert "Inserted" in result, f"Expected 'Inserted' in result, got: {result!r}"
        new = drive.contents[terapia["id"]]
        assert "appended text" in new, f"Appended text not found in new content: {new!r}"
        assert original in new, f"Original content was not preserved after append; got: {new!r}"

    async def test_modify_doc_text_range_replace(self, client, drive):
        # start_index=1, end_index=len-1 → replace the interior, keep first and last chars.
        terapia = drive.find_by_name("Terapia")
        content = drive.contents[terapia["id"]]
        result = await self._call(
            client, "modify_doc_text",
            document_id=terapia["id"],
            start_index=1,
            end_index=len(content) - 1,
            text="hello",
        )
        assert "Inserted" in result, f"Expected 'Inserted' in result, got: {result!r}"
        new_content = drive.contents[terapia["id"]]
        expected = content[0] + "hello" + content[-1]
        assert new_content == expected, (
            f"Range replace result mismatch.\nExpected: {expected!r}\nGot: {new_content!r}"
        )

    async def test_find_and_replace_doc(self, client, drive):
        # find_and_replace_doc must report the replacement count AND actually replace.
        terapia = drive.find_by_name("Terapia")
        result = await self._call(
            client, "find_and_replace_doc",
            document_id=terapia["id"],
            find_text="infancia",
            replace_text="niñez",
        )
        assert "1" in result, (
            f"Expected '1' (count of replacements) in result, got: {result!r}"
        )
        new = drive.contents[terapia["id"]]
        assert "niñez" in new, f"Replacement text 'niñez' not present in new content: {new!r}"
        assert "infancia" not in new, (
            f"Original text 'infancia' still present after find_and_replace_doc: {new!r}"
        )

    async def test_read_sheet_values(self, client, drive):
        # A range covering the initial rows must return both values.
        sheet = drive.find_by_name("Prototipo Juego")
        result = await self._call(
            client, "read_sheet_values",
            spreadsheet_id=sheet["id"], range_name="A1:A10",
        )
        assert "idea anterior 1" in result, f"'idea anterior 1' missing from read result: {result!r}"
        assert "idea anterior 2" in result, f"'idea anterior 2' missing from read result: {result!r}"

    async def test_modify_sheet_values_appends_row(self, client, drive):
        # Writing to A3 populates row index 2 of the target tab.
        sheet = drive.find_by_name("Prototipo Juego")
        result = await self._call(
            client, "modify_sheet_values",
            spreadsheet_id=sheet["id"],
            range_name="A3",
            values=[["nueva idea"]],
        )
        assert "updated" in result.lower(), (
            f"Expected 'updated' in modify_sheet_values result, got: {result!r}"
        )
        rows = drive.sheets[sheet["id"]]["Ideas suelta 1"]
        assert rows[2] == ["nueva idea"], (
            f"Expected row[2] == ['nueva idea'] after writing A3, got {rows[2]!r} (full rows: {rows!r})"
        )

    async def test_search_drive_files_rejects_invalid_field_name(self, client, drive):
        # mime_type (underscore) is invalid — Drive uses mimeType (camelCase)
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia' and mime_type = 'application/vnd.google-apps.document'",
        )
        assert "[Tool error]" in result, (
            f"Expected '[Tool error]' for invalid field 'mime_type', got: {result!r}"
        )
        assert "mime_type" in result, (
            f"Expected the offending field name 'mime_type' echoed in the error, got: {result!r}"
        )

    async def test_search_drive_files_accepts_valid_mimeType(self, client, drive):
        # mimeType (camelCase) is the valid Drive query field — should succeed.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia' and mimeType = 'application/vnd.google-apps.document'",
        )
        assert "[Tool error]" not in result, (
            f"Valid camelCase 'mimeType' query was rejected: {result!r}"
        )
        assert "Terapia" in result, (
            f"Expected 'Terapia' in the filtered result, got: {result!r}"
        )

    async def test_unknown_tool_returns_message(self, client, drive):
        # Unknown tool names return a graceful "not implemented" message (no exception).
        result = await self._call(client, "nonexistent_tool")
        assert "not implemented" in result.lower(), (
            f"Expected 'not implemented' message for unknown tool, got: {result!r}"
        )

    async def test_call_log_records_calls(self, client, drive):
        # Every call_tool invocation is recorded in the FakeDrive call log.
        terapia = drive.find_by_name("Terapia")
        await self._call(client, "get_doc_content", document_id=terapia["id"])
        calls = drive.get_calls("get_doc_content")
        assert len(calls) == 1, f"Expected exactly 1 recorded get_doc_content call, got {len(calls)}"
        got_id = calls[0]["args"]["document_id"]
        assert got_id == terapia["id"], (
            f"Expected recorded document_id={terapia['id']!r}, got {got_id!r}"
        )


# ─── Parameter validation (mirrors real API / server failures) ────────────────
#
# Each test documents what the equivalent real failure looks like:
#   - "query field" errors → Google API returns HTTP 400 Invalid Value
#     wrapped by @handle_http_errors as: Exception("API error in search_drive_files: ...")
#   - "file_type" errors → server's resolve_file_type_mime() raises ValueError
#     before the API is ever called
#   - "mime_type" errors on create → same ValueError path in resolve_file_type_mime()
#
# The fake returns [Tool error] strings instead of raising, so the agent can
# read the error and recover — same observable behaviour as a non-fatal MCP error.

class TestParameterValidation:
    """The fake mirrors the real API/server validation so the agent trains on the same error shapes."""

    async def _call(self, client, tool, **kwargs):
        return await client.call_tool(token="t", tool_name=tool, arguments=kwargs)

    # ── search_drive_files: query field validation ────────────────────────────

    async def test_query_mime_type_underscore_rejected(self, client, drive):
        # Real failure: Google API 400 "Invalid Value" — mime_type is not a Drive query field.
        # The correct field is mimeType (camelCase), but the prompt forbids it in queries
        # entirely — use file_type parameter instead.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia' and mime_type = 'application/vnd.google-apps.document'",
        )
        assert "[Tool error]" in result, (
            f"Expected '[Tool error]' for invalid field 'mime_type' in query, got: {result!r}"
        )
        assert "mime_type" in result, (
            f"Expected the offending field name 'mime_type' echoed in the error, got: {result!r}"
        )

    async def test_query_unknown_field_rejected(self, client, drive):
        # Real failure: Google API 400 "Invalid Value" for any unrecognised field name.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia' and type = 'document'",
        )
        assert "[Tool error]" in result, (
            f"Expected '[Tool error]' for unknown query field 'type', got: {result!r}"
        )

    async def test_query_valid_fields_accepted(self, client, drive):
        # Sanity check: a well-formed query must not be rejected.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia'",
        )
        assert "[Tool error]" not in result, (
            f"Valid query was rejected: {result!r}"
        )
        assert "Terapia" in result, (
            f"Expected 'Terapia' in the result of a valid name-contains query, got: {result!r}"
        )

    # ── search_drive_files: file_type validation ──────────────────────────────

    async def test_search_invalid_file_type_rejected(self, client, drive):
        # Real failure: server's resolve_file_type_mime() raises ValueError before
        # the Google API is called.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia'",
            file_type="texto",  # not a valid friendly name or MIME type
        )
        assert "[Tool error]" in result, (
            f"Expected '[Tool error]' for invalid file_type='texto', got: {result!r}"
        )
        assert "texto" in result, (
            f"Expected the offending file_type 'texto' echoed in the error, got: {result!r}"
        )

    async def test_search_valid_file_type_accepted(self, client, drive):
        # 'document' is a known friendly name — must not be rejected.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia'",
            file_type="document",
        )
        assert "[Tool error]" not in result, (
            f"Valid file_type='document' was rejected: {result!r}"
        )

    # ── list_drive_items: file_type validation ────────────────────────────────

    async def test_list_invalid_file_type_rejected(self, client, drive):
        # Real failure: same resolve_file_type_mime() ValueError path.
        result = await self._call(
            client, "list_drive_items",
            folder_id="root",
            file_type="archivo",  # Spanish, not a valid value
        )
        assert "[Tool error]" in result, (
            f"Expected '[Tool error]' for invalid file_type='archivo', got: {result!r}"
        )
        assert "archivo" in result, (
            f"Expected the offending file_type 'archivo' echoed in the error, got: {result!r}"
        )

    async def test_list_valid_file_type_accepted(self, client, drive):
        # 'folder' is a known friendly name — must not be rejected.
        result = await self._call(
            client, "list_drive_items",
            folder_id="root",
            file_type="folder",
        )
        assert "[Tool error]" not in result, (
            f"Valid file_type='folder' was rejected: {result!r}"
        )

    # ── create_drive_file: mime_type validation ───────────────────────────────

    async def test_create_invalid_mime_type_rejected(self, client, drive):
        # Real failure: resolve_file_type_mime() rejects a value that is neither
        # a known friendly name nor a valid MIME type string (contains '/').
        result = await self._call(
            client, "create_drive_file",
            file_name="test",
            folder_id="root",
            mime_type="documento",  # not valid
        )
        assert "[Tool error]" in result, (
            f"Expected '[Tool error]' for invalid mime_type='documento', got: {result!r}"
        )
        assert "documento" in result, (
            f"Expected the offending mime_type 'documento' echoed in the error, got: {result!r}"
        )

    async def test_create_valid_mime_type_accepted(self, client, drive):
        # Real Drive document MIME — must be accepted.
        result = await self._call(
            client, "create_drive_file",
            file_name="test doc",
            folder_id="root",
            mime_type="application/vnd.google-apps.document",
        )
        assert "[Tool error]" not in result, (
            f"Valid mime_type='application/vnd.google-apps.document' was rejected: {result!r}"
        )
