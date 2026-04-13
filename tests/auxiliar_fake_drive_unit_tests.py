"""
Unit tests for the fake MCP infrastructure (FakeDrive + FakeMCPClient).
These run via pytest and never call the real LLM or any external service.
"""
import pytest
from pathlib import Path

from functional_tests.fake_drive import (
    FakeDrive,
    FakeMCPClient,
    DRIVE_ROOT,
    MIME_DOC,
    MIME_FOLDER,
    MIME_SHEET,
    _parse_range,
)
from functional_tests.test_config import TERAPIA_INITIAL_CONTENT


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def drive():
    d = FakeDrive(DRIVE_ROOT)
    snap = d.snapshot()
    yield d
    d.restore(snap)


@pytest.fixture
def client(drive):
    return FakeMCPClient(drive)


# ─── FakeDrive: loading from disk ─────────────────────────────────────────────

class TestFakeDriveLoading:
    def test_root_item_exists(self, drive):
        assert "root" in drive.items

    def test_top_level_folders_loaded(self, drive):
        names = {v["name"] for v in drive.items.values()}
        for expected in (
            "Ideas Sueltas", "Terapia-Autoanálisis", "Cuentos", "PrototipoJuego",
            "Trabajo", "Finanzas", "Viajes", "Libros", "Recetas",
            "Fotografías", "Proyectos Personales", "Salud", "Música", "Cursos",
        ):
            assert expected in names

    def test_consciencia_subfolder_loaded(self, drive):
        item = drive.find_by_name("consciencia")
        assert item is not None
        assert item["mimeType"] == MIME_FOLDER

    def test_consciencia_parent_is_ideas_sueltas(self, drive):
        consciencia = drive.find_by_name("consciencia")
        ideas_sueltas = drive.find_by_name("Ideas Sueltas")
        assert consciencia["parent_id"] == ideas_sueltas["id"]

    def test_terapia_doc_loaded(self, drive):
        item = drive.find_by_name("Terapia")
        assert item is not None
        assert item["mimeType"] == MIME_DOC

    def test_terapia_content_matches_file(self, drive):
        item = drive.find_by_name("Terapia")
        assert drive.contents[item["id"]] == TERAPIA_INITIAL_CONTENT

    def test_spreadsheet_loaded(self, drive):
        item = drive.find_by_name("Prototipo Juego")
        assert item is not None
        assert item["mimeType"] == MIME_SHEET

    def test_spreadsheet_tab_loaded(self, drive):
        item = drive.find_by_name("Prototipo Juego")
        rows = drive.sheets[item["id"]]["Ideas suelta 1"]
        assert rows == [["idea anterior 1"], ["idea anterior 2"]]

    def test_gitkeep_files_ignored(self, drive):
        names = {v["name"] for v in drive.items.values()}
        assert ".gitkeep" not in names

    def test_dotfiles_ignored(self, drive):
        for name in drive.items:
            assert not name.startswith(".")


# ─── FakeDrive: snapshot / restore ────────────────────────────────────────────

class TestSnapshotRestore:
    def test_restore_reverts_new_doc(self, drive):
        snap = drive.snapshot()
        drive.create_doc("temp_doc", "root", "hello")
        assert drive.find_by_name("temp_doc") is not None
        drive.restore(snap)
        assert drive.find_by_name("temp_doc") is None

    def test_restore_reverts_content_change(self, drive):
        snap = drive.snapshot()
        terapia = drive.find_by_name("Terapia")
        drive.write_doc(terapia["id"], "replaced content")
        drive.restore(snap)
        terapia = drive.find_by_name("Terapia")
        assert drive.contents[terapia["id"]] == TERAPIA_INITIAL_CONTENT

    def test_restore_reverts_filesystem(self, drive):
        snap = drive.snapshot()
        terapia_path = drive._path_map[drive.find_by_name("Terapia")["id"]]
        drive.write_doc(drive.find_by_name("Terapia")["id"], "changed")
        drive.restore(snap)
        assert terapia_path.read_text(encoding="utf-8") == TERAPIA_INITIAL_CONTENT

    def test_restore_reverts_new_folder(self, drive):
        snap = drive.snapshot()
        ideas = drive.find_by_name("Ideas Sueltas")
        drive.create_folder("new_topic", ideas["id"])
        drive.restore(snap)
        assert drive.find_by_name("new_topic") is None


# ─── FakeDrive: mutations ─────────────────────────────────────────────────────

class TestMutations:
    def test_create_doc_in_memory(self, drive):
        item = drive.create_doc("my note", "root", "content here")
        assert drive.find_by_name("my note") is not None
        assert drive.contents[item["id"]] == "content here"

    def test_create_doc_on_disk(self, drive, tmp_path):
        # Use a real temp dir as root to avoid polluting fixture drive
        d = FakeDrive(tmp_path)
        item = d.create_doc("test", "root", "hello disk")
        assert (tmp_path / "test.txt").read_text(encoding="utf-8") == "hello disk"

    def test_create_folder_in_memory(self, drive):
        item = drive.create_folder("new_folder", "root")
        assert drive.find_by_name("new_folder") is not None
        assert item["mimeType"] == MIME_FOLDER

    def test_create_folder_on_disk(self, drive):
        drive.create_folder("new_folder", "root")
        assert (drive.root / "new_folder").is_dir()

    def test_write_doc_updates_content(self, drive):
        terapia = drive.find_by_name("Terapia")
        drive.write_doc(terapia["id"], "new content")
        assert drive.contents[terapia["id"]] == "new content"

    def test_write_doc_updates_file(self, drive):
        terapia = drive.find_by_name("Terapia")
        drive.write_doc(terapia["id"], "written to disk")
        path = drive._path_map[terapia["id"]]
        assert path.read_text(encoding="utf-8") == "written to disk"

    def test_write_sheet_tab(self, drive):
        sheet = drive.find_by_name("Prototipo Juego")
        drive.write_sheet_tab(sheet["id"], "Ideas suelta 1", [["a"], ["b"], ["c"]])
        assert drive.sheets[sheet["id"]]["Ideas suelta 1"] == [["a"], ["b"], ["c"]]

    def test_write_sheet_tab_on_disk(self, drive):
        sheet = drive.find_by_name("Prototipo Juego")
        drive.write_sheet_tab(sheet["id"], "Ideas suelta 1", [["x", "y"]])
        sheet_path = drive._path_map[sheet["id"]]
        csv_path = sheet_path / "Ideas suelta 1.csv"
        content = csv_path.read_text(encoding="utf-8")
        assert "x" in content and "y" in content


# ─── Range parser ─────────────────────────────────────────────────────────────

class TestParseRange:
    def test_simple_cell(self):
        sn, sr, sc, er, ec = _parse_range("A1")
        assert sn is None and sr == 0 and sc == 0 and er is None and ec is None

    def test_range(self):
        sn, sr, sc, er, ec = _parse_range("B2:C5")
        assert sc == 1 and sr == 1 and ec == 2 and er == 4

    def test_sheet_prefix(self):
        sn, sr, sc, er, ec = _parse_range("Sheet1!A1:Z10")
        assert sn == "Sheet1" and sr == 0 and sc == 0 and er == 9


# ─── FakeMCPClient: tool listing ──────────────────────────────────────────────

class TestFakeMCPClientTools:
    async def test_list_tools_returns_tools(self, client):
        tools = await client.list_tools()
        assert len(tools) > 0

    async def test_list_tools_have_name_and_schema(self, client):
        tools = await client.list_tools()
        for t in tools:
            assert "name" in t
            assert "inputSchema" in t

    async def test_expected_tools_present(self, client):
        tools = await client.list_tools()
        names = {t["name"] for t in tools}
        for expected in (
            "search_drive_files", "list_drive_items", "create_drive_file",
            "create_drive_folder", "update_drive_file",
            "get_doc_content", "modify_doc_text", "find_and_replace_doc",
            "read_sheet_values", "modify_sheet_values",
        ):
            assert expected in names, f"Tool '{expected}' missing from FakeMCPClient"


# ─── FakeMCPClient: tool dispatch ─────────────────────────────────────────────

class TestFakeMCPClientDispatch:
    async def _call(self, client, tool, **kwargs):
        return await client.call_tool(token="t", tool_name=tool, arguments=kwargs)

    async def test_search_drive_files_finds_terapia(self, client, drive):
        result = await self._call(client, "search_drive_files", query="name contains 'Terapia'")
        assert "Terapia" in result

    async def test_search_drive_files_no_match(self, client, drive):
        result = await self._call(client, "search_drive_files", query="name contains 'nonexistent_xyz'")
        assert "No files found" in result

    async def test_list_drive_items_root(self, client, drive):
        result = await self._call(client, "list_drive_items", folder_id="root")
        assert "Ideas Sueltas" in result or "PrototipoJuego" in result

    async def test_get_doc_content(self, client, drive):
        terapia = drive.find_by_name("Terapia")
        result = await self._call(client, "get_doc_content", document_id=terapia["id"])
        assert "infancia" in result

    async def test_create_drive_file_creates_doc(self, client, drive):
        ideas = drive.find_by_name("Ideas Sueltas")
        result = await self._call(
            client, "create_drive_file",
            file_name="Nueva nota", folder_id=ideas["id"],
            mime_type="application/vnd.google-apps.document",
            content="idea aquí",
        )
        assert "Nueva nota" in result
        item = drive.find_by_name("Nueva nota")
        assert item is not None
        assert drive.contents[item["id"]] == "idea aquí"

    async def test_create_drive_folder(self, client, drive):
        ideas = drive.find_by_name("Ideas Sueltas")
        result = await self._call(
            client, "create_drive_folder",
            folder_name="nuevo tema", parent_folder_id=ideas["id"],
        )
        assert "nuevo tema" in result
        assert drive.find_by_name("nuevo tema") is not None

    async def test_update_drive_file_trashed(self, client, drive):
        terapia = drive.find_by_name("Terapia")
        result = await self._call(client, "update_drive_file", file_id=terapia["id"], trashed=True)
        assert "Updated" in result
        assert terapia.get("trashed") is True

    async def test_modify_doc_text_appends(self, client, drive):
        terapia = drive.find_by_name("Terapia")
        original = drive.contents[terapia["id"]]
        result = await self._call(
            client, "modify_doc_text",
            document_id=terapia["id"],
            start_index=len(original),
            text="\nappended text",
        )
        assert "Inserted" in result
        assert "appended text" in drive.contents[terapia["id"]]
        assert original in drive.contents[terapia["id"]]

    async def test_modify_doc_text_range_replace(self, client, drive):
        terapia = drive.find_by_name("Terapia")
        content = drive.contents[terapia["id"]]
        result = await self._call(
            client, "modify_doc_text",
            document_id=terapia["id"],
            start_index=1,
            end_index=len(content) - 1,
            text="hello",
        )
        assert "Inserted" in result
        new_content = drive.contents[terapia["id"]]
        # First char preserved + "hello" + last char preserved
        assert new_content == content[0] + "hello" + content[-1]

    async def test_find_and_replace_doc(self, client, drive):
        terapia = drive.find_by_name("Terapia")
        result = await self._call(
            client, "find_and_replace_doc",
            document_id=terapia["id"],
            find_text="infancia",
            replace_text="niñez",
        )
        assert "1" in result
        assert "niñez" in drive.contents[terapia["id"]]
        assert "infancia" not in drive.contents[terapia["id"]]

    async def test_read_sheet_values(self, client, drive):
        sheet = drive.find_by_name("Prototipo Juego")
        result = await self._call(
            client, "read_sheet_values",
            spreadsheet_id=sheet["id"], range_name="A1:A10",
        )
        assert "idea anterior 1" in result
        assert "idea anterior 2" in result

    async def test_modify_sheet_values_appends_row(self, client, drive):
        sheet = drive.find_by_name("Prototipo Juego")
        result = await self._call(
            client, "modify_sheet_values",
            spreadsheet_id=sheet["id"],
            range_name="A3",
            values=[["nueva idea"]],
        )
        assert "updated" in result.lower()
        rows = drive.sheets[sheet["id"]]["Ideas suelta 1"]
        assert rows[2] == ["nueva idea"]

    async def test_search_drive_files_rejects_invalid_field_name(self, client, drive):
        # mime_type (underscore) is invalid — Drive uses mimeType (camelCase)
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia' and mime_type = 'application/vnd.google-apps.document'",
        )
        assert "[Tool error]" in result
        assert "mime_type" in result

    async def test_search_drive_files_accepts_valid_mimeType(self, client, drive):
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia' and mimeType = 'application/vnd.google-apps.document'",
        )
        assert "[Tool error]" not in result
        assert "Terapia" in result

    async def test_unknown_tool_returns_message(self, client, drive):
        result = await self._call(client, "nonexistent_tool")
        assert "not implemented" in result.lower()

    async def test_call_log_records_calls(self, client, drive):
        terapia = drive.find_by_name("Terapia")
        await self._call(client, "get_doc_content", document_id=terapia["id"])
        calls = drive.get_calls("get_doc_content")
        assert len(calls) == 1
        assert calls[0]["args"]["document_id"] == terapia["id"]


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
        assert "[Tool error]" in result
        assert "mime_type" in result

    async def test_query_unknown_field_rejected(self, client, drive):
        # Real failure: Google API 400 "Invalid Value" for any unrecognised field name.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia' and type = 'document'",
        )
        assert "[Tool error]" in result

    async def test_query_valid_fields_accepted(self, client, drive):
        # Sanity check: a well-formed query must not be rejected.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia'",
        )
        assert "[Tool error]" not in result
        assert "Terapia" in result

    # ── search_drive_files: file_type validation ──────────────────────────────

    async def test_search_invalid_file_type_rejected(self, client, drive):
        # Real failure: server's resolve_file_type_mime() raises ValueError before
        # the Google API is called.
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia'",
            file_type="texto",  # not a valid friendly name or MIME type
        )
        assert "[Tool error]" in result
        assert "texto" in result

    async def test_search_valid_file_type_accepted(self, client, drive):
        result = await self._call(
            client, "search_drive_files",
            query="name contains 'Terapia'",
            file_type="document",
        )
        assert "[Tool error]" not in result

    # ── list_drive_items: file_type validation ────────────────────────────────

    async def test_list_invalid_file_type_rejected(self, client, drive):
        # Real failure: same resolve_file_type_mime() ValueError path.
        result = await self._call(
            client, "list_drive_items",
            folder_id="root",
            file_type="archivo",  # Spanish, not a valid value
        )
        assert "[Tool error]" in result
        assert "archivo" in result

    async def test_list_valid_file_type_accepted(self, client, drive):
        result = await self._call(
            client, "list_drive_items",
            folder_id="root",
            file_type="folder",
        )
        assert "[Tool error]" not in result

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
        assert "[Tool error]" in result
        assert "documento" in result

    async def test_create_valid_mime_type_accepted(self, client, drive):
        result = await self._call(
            client, "create_drive_file",
            file_name="test doc",
            folder_id="root",
            mime_type="application/vnd.google-apps.document",
        )
        assert "[Tool error]" not in result
