"""
In-memory fake Google Drive + MCP client stub.

The template is loaded once from functional_tests/drive/ on first
FakeDrive() instantiation. Subsequent instances deep-copy the template —
each test gets a fresh, isolated drive state with no shared mutable data,
so tests are safe to run in parallel.

Directory layout under functional_tests/drive/ (read-only, loaded once):
  - Regular directories  → Google Drive folders  (MIME_FOLDER)
  - *.txt files          → Google Docs            (MIME_DOC)
  - *.sheet directories  → Google Sheets          (MIME_SHEET)
      children *.csv     → individual tabs
  - .gitkeep / dotfiles  → ignored
"""
import copy
import csv as _csv
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

MIME_FOLDER = "application/vnd.google-apps.folder"
MIME_DOC    = "application/vnd.google-apps.document"
MIME_SHEET  = "application/vnd.google-apps.spreadsheet"

FILE_TYPE_MIME: dict[str, str] = {
    "folder":        MIME_FOLDER,
    "document":      MIME_DOC,
    "doc":           MIME_DOC,
    "spreadsheet":   MIME_SHEET,
    "sheet":         MIME_SHEET,
    "presentation":  "application/vnd.google-apps.presentation",
    "slides":        "application/vnd.google-apps.presentation",
    "form":          "application/vnd.google-apps.form",
    "drawing":       "application/vnd.google-apps.drawing",
    "pdf":           "application/pdf",
    "shortcut":      "application/vnd.google-apps.shortcut",
    "script":        "application/vnd.google-apps.script",
    "site":          "application/vnd.google-apps.site",
    "jam":           "application/vnd.google-apps.jam",
    "jamboard":      "application/vnd.google-apps.jam",
}

_MIME_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")


def _resolve_file_type(file_type: str) -> tuple[str | None, str | None]:
    """
    Mirrors the server's resolve_file_type_mime().
    Returns (mime, None) on success, (None, error_message) on failure.
    """
    normalized = file_type.strip()
    if "/" in normalized:
        if not _MIME_TYPE_PATTERN.fullmatch(normalized):
            return None, (
                f"[Tool error] Invalid MIME type '{file_type}'. "
                f"Expected format like 'application/pdf'."
            )
        return normalized, None
    lower = normalized.lower()
    if lower not in FILE_TYPE_MIME:
        valid = ", ".join(sorted(FILE_TYPE_MIME.keys()))
        return None, (
            f"[Tool error] Unknown file_type '{file_type}'. "
            f"Use a MIME type (e.g. 'application/pdf') or one of: {valid}"
        )
    return FILE_TYPE_MIME[lower], None

DRIVE_ROOT = Path(__file__).parent / "drive"


# ─── CSV helper (template load only) ──────────────────────────────────────────

def _read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in _csv.reader(f) if row]  # skip blank lines


# ─── FakeDrive ────────────────────────────────────────────────────────────────

class FakeDrive:
    """
    In-memory fake Drive. Template is loaded once from disk; every instance
    gets a deep-copied, isolated snapshot of that template — no disk writes,
    no shared state, safe to run in parallel.
    """

    _TEMPLATE: dict | None = None

    def __init__(self) -> None:
        if FakeDrive._TEMPLATE is None:
            FakeDrive._build_template()
        tmpl = FakeDrive._TEMPLATE
        # id → {id, name, mimeType, parent_id}
        self.items: dict[str, dict] = copy.deepcopy(tmpl["items"])
        # doc id → text content
        self.contents: dict[str, str] = copy.deepcopy(tmpl["contents"])
        # sheet id → {tab_name → [[row...]]}
        self.sheets: dict[str, dict[str, list]] = copy.deepcopy(tmpl["sheets"])
        self._counter: int = tmpl["counter"]
        # doc id → [{"title", "tab_id"}] — docs marked as tab-based (newer Google Docs)
        self.doc_tabs: dict[str, list[dict]] = {}
        self._call_log: list[dict] = []
        self._step: int = 0

    # ── template loader ──────────────────────────────────────────────────────

    @classmethod
    def _build_template(cls) -> None:
        """Walk DRIVE_ROOT once and cache the resulting in-memory state."""
        items: dict[str, dict] = {
            "root": {"id": "root", "name": "root", "mimeType": MIME_FOLDER, "parent_id": None},
        }
        contents: dict[str, str] = {}
        sheets: dict[str, dict[str, list]] = {}
        counter = [1000]

        def next_id(name: str) -> str:
            counter[0] += 1
            slug = re.sub(r"[^A-Za-z0-9]", "", name)[:12]
            suffix = f"_{slug}" if slug else ""
            return f"item_{counter[0]}{suffix}"

        def walk(directory: Path, parent_id: str) -> None:
            for child in sorted(directory.iterdir()):
                if child.name.startswith("."):
                    continue  # ignore .gitkeep and other dotfiles
                if child.is_dir():
                    if child.name.endswith(".sheet"):
                        load_sheet(child, parent_id)
                    else:
                        load_folder(child, parent_id)
                elif child.suffix == ".txt":
                    load_doc(child, parent_id)

        def load_folder(path: Path, parent_id: str) -> None:
            item_id = next_id(path.name)
            items[item_id] = {
                "id": item_id, "name": path.name,
                "mimeType": MIME_FOLDER, "parent_id": parent_id,
            }
            walk(path, item_id)

        def load_doc(path: Path, parent_id: str) -> None:
            item_id = next_id(path.stem)
            items[item_id] = {
                "id": item_id, "name": path.stem,
                "mimeType": MIME_DOC, "parent_id": parent_id,
            }
            contents[item_id] = path.read_text(encoding="utf-8")

        def load_sheet(path: Path, parent_id: str) -> None:
            display_name = path.name[:-6]  # strip ".sheet"
            item_id = next_id(display_name)
            items[item_id] = {
                "id": item_id, "name": display_name,
                "mimeType": MIME_SHEET, "parent_id": parent_id,
            }
            sheets[item_id] = {
                csv.stem: _read_csv(csv)
                for csv in sorted(path.glob("*.csv"))
            }

        walk(DRIVE_ROOT, "root")
        cls._TEMPLATE = {
            "items": items,
            "contents": contents,
            "sheets": sheets,
            "counter": counter[0],
        }

    # ── id allocator ─────────────────────────────────────────────────────────

    def _next_id(self, name: str = "") -> str:
        self._counter += 1
        slug = re.sub(r"[^A-Za-z0-9]", "", name)[:12]
        suffix = f"_{slug}" if slug else ""
        return f"item_{self._counter}{suffix}"

    # ── call log ─────────────────────────────────────────────────────────────

    def log_call(self, tool: str, args: dict) -> None:
        self._step += 1
        self._call_log.append({"step": self._step, "tool": tool, "args": args})

    def get_calls(self, tool: str | None = None) -> list[dict]:
        return [c for c in self._call_log if not tool or c["tool"] == tool]

    def clear_calls(self) -> None:
        self._call_log.clear()
        self._step = 0

    # ── mutation helpers (in-memory only) ────────────────────────────────────

    def create_folder(self, name: str, parent_id: str) -> dict:
        item_id = self._next_id(name)
        item = {"id": item_id, "name": name, "mimeType": MIME_FOLDER, "parent_id": parent_id}
        self.items[item_id] = item
        return item

    def create_doc(self, name: str, parent_id: str, content: str = "") -> dict:
        item_id = self._next_id(name)
        item = {"id": item_id, "name": name, "mimeType": MIME_DOC, "parent_id": parent_id}
        self.items[item_id] = item
        self.contents[item_id] = content
        return item

    def create_sheet(self, name: str, parent_id: str) -> dict:
        item_id = self._next_id(name)
        item = {"id": item_id, "name": name, "mimeType": MIME_SHEET, "parent_id": parent_id}
        self.items[item_id] = item
        self.sheets[item_id] = {}
        return item

    def write_doc(self, item_id: str, content: str) -> None:
        self.contents[item_id] = content

    def make_tabbed(self, item_id: str, tab_title: str | None = None) -> dict:
        """Mark a doc as tab-based; its content stays in self.contents."""
        tab = {"title": tab_title or self.items[item_id]["name"], "tab_id": "t.0"}
        self.doc_tabs[item_id] = [tab]
        return tab

    def write_sheet_tab(self, item_id: str, tab_name: str, rows: list) -> None:
        self.sheets.setdefault(item_id, {})[tab_name] = rows

    # ── query helpers ─────────────────────────────────────────────────────────

    def find_by_name(self, name: str, mime: str | None = None) -> dict | None:
        for item in self.items.values():
            if item["name"] == name and (mime is None or item["mimeType"] == mime):
                return item
        return None


# ─── output formatter ─────────────────────────────────────────────────────────

def _fmt(item: dict) -> str:
    return f'- Name: "{item["name"]}" (ID: {item["id"]}, Type: {item["mimeType"]})'


# ─── range parser ─────────────────────────────────────────────────────────────

def _col_to_idx(col: str) -> int:
    r = 0
    for c in col.upper():
        r = r * 26 + (ord(c) - ord("A") + 1)
    return r - 1


def _a1_range_error(range_str: str) -> str | None:
    """
    Mirror the real Sheets API grammar: a tab name containing spaces or other
    non-alphanumeric characters MUST be wrapped in single quotes, otherwise the
    API rejects the range with 400 'Unable to parse range'.
    """
    if "!" not in range_str:
        return None
    sheet, _ = range_str.split("!", 1)
    if sheet.startswith("'") and sheet.endswith("'") and len(sheet) >= 2:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_]+", sheet):
        return f"[Tool error] API error: Unable to parse range: {range_str}"
    return None


def _parse_range(range_str: str) -> tuple[str | None, int, int, int | None, int | None]:
    """'Sheet1!A2:B5' -> (sheet_name, start_row, start_col, end_row, end_col) 0-based."""
    sheet_name = None
    if "!" in range_str:
        sheet_name, range_str = range_str.split("!", 1)
        # Strip A1 single-quoting ('Ideas suelta 1' → Ideas suelta 1, '' → ')
        if sheet_name.startswith("'") and sheet_name.endswith("'") and len(sheet_name) >= 2:
            sheet_name = sheet_name[1:-1].replace("''", "'")
    m = re.match(r"([A-Za-z]+)(\d+)(?::([A-Za-z]+)(\d+))?", range_str)
    if not m:
        return sheet_name, 0, 0, None, None
    sc = _col_to_idx(m.group(1))
    sr = int(m.group(2)) - 1
    ec = _col_to_idx(m.group(3)) if m.group(3) else None
    er = int(m.group(4)) - 1    if m.group(4) else None
    return sheet_name, sr, sc, er, ec


# ─── FakeMCPClient ────────────────────────────────────────────────────────────

_SERVER_ROOT = Path(__file__).resolve().parent.parent / "servers" / "google_workspace_mcp"
_SERVER_MODULES = ("gdrive.drive_tools", "gdocs.docs_tools", "gsheets.sheets_tools")


async def _load_schemas() -> dict[str, Any]:
    """
    Import the google_workspace_mcp tool modules and extract JSON schemas
    directly from the FastMCP server's tool registry.
    Always up-to-date; no static file to maintain.
    """
    import importlib
    import os
    import sys

    sys.path.insert(0, str(_SERVER_ROOT))
    os.environ.setdefault("GOOGLE_CLIENT_ID", "x")
    os.environ.setdefault("GOOGLE_CLIENT_SECRET", "x")

    from core.server import server  # type: ignore[import]  # resolved via runtime sys.path

    for mod in _SERVER_MODULES:
        importlib.import_module(mod)

    tools = await server.list_tools()
    schemas = {}
    for t in tools:
        params = t.parameters
        # In production (OAuth 2.1 mode) service_decorator.py strips user_google_email from
        # tool schemas before the LLM sees them — the decorator injects it from the auth token.
        # Mirror that here so the LLM behaves the same way in tests as in production.
        if isinstance(params, dict) and "user_google_email" in params.get("properties", {}):
            import copy
            params = copy.deepcopy(params)
            params["properties"].pop("user_google_email", None)
            if "required" in params:
                params["required"] = [r for r in params["required"] if r != "user_google_email"]
        schemas[t.name] = {"description": t.description, "parameters": params}
    return schemas


class FakeMCPClient:
    """
    Drop-in stub for MCPClient.
    Tool schemas loaded live from the google_workspace_mcp server modules.
    """

    _SCHEMAS: dict[str, Any] = {}

    def __init__(self, drive: FakeDrive) -> None:
        self._drive = drive

    @asynccontextmanager
    async def session(self, *, token: str | None = None):  # noqa: ARG002
        yield self

    async def list_tools(self, *, token: str | None = None) -> list[dict]:  # noqa: ARG002
        if not FakeMCPClient._SCHEMAS:
            FakeMCPClient._SCHEMAS = await _load_schemas()
        return [{"name": n, "description": s["description"], "inputSchema": s["parameters"]} for n, s in self._SCHEMAS.items()]

    async def call_tool(self, *, token: str, tool_name: str, arguments: dict) -> str:  # noqa: ARG002
        self._drive.log_call(tool_name, arguments)
        step = self._drive._step
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            result = f"[Fake server] Tool '{tool_name}' is not implemented in the fake MCP."
        else:
            result = await handler(**arguments)
        print(f"  [{step}] → {tool_name}  {arguments}", flush=True)
        print(f"      ↳ {result}", flush=True)
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Drive tools
    # ═══════════════════════════════════════════════════════════════════════════

    # Valid field names in the Drive query language
    _VALID_QUERY_FIELDS = {
        "name", "mimeType", "fullText", "trashed", "starred",
        "parents", "owners", "modifiedTime", "createdTime", "sharedWithMe",
    }

    async def _tool_search_drive_files(
        self,
        query: str,
        page_size: int = 10,
        file_type: str | None = None,
        **_,
    ) -> str:
        # Reject queries using invalid field names (e.g. mime_type instead of mimeType)
        found_fields = re.findall(r'\b([a-zA-Z_]+)\b(?=\s*(?:contains|=|!=|<|>))', query)
        invalid = [f for f in found_fields if f not in self._VALID_QUERY_FIELDS]
        if invalid:
            return (
                f"[Tool error] Invalid Drive query syntax: unknown field(s) {invalid}. "
                f"Drive query language uses camelCase fields (e.g. 'mimeType', not 'mime_type'). "
                f"Query was: {query!r}"
            )
        d = self._drive
        mime_filter = None
        if file_type:
            mime_filter, err = _resolve_file_type(file_type)
            if err:
                return err
        name_re   = re.search(r"name contains '([^']+)'",    query, re.IGNORECASE)
        parent_re = re.search(r"'([^']+)'\s+in\s+parents",   query, re.IGNORECASE)
        full_re   = re.search(r"fullText contains '([^']+)'", query, re.IGNORECASE)
        results = []
        for item in d.items.values():
            if item["id"] == "root":
                continue
            if mime_filter and item["mimeType"] != mime_filter:
                continue
            if name_re and name_re.group(1).lower() not in item["name"].lower():
                continue
            if parent_re and item.get("parent_id") != parent_re.group(1):
                continue
            if full_re:
                text = d.contents.get(item["id"], "")
                if full_re.group(1).lower() not in text.lower():
                    continue
            results.append(item)
        results = results[:page_size]
        if not results:
            return f"No files found for '{query}'."
        return "\n".join([f"Found {len(results)} files matching '{query}':"] + [_fmt(i) for i in results])

    async def _tool_list_drive_items(
        self,
        folder_id: str = "root",
        page_size: int = 100,
        file_type: str | None = None,
        **_,
    ) -> str:
        d = self._drive
        mime_filter = None
        if file_type:
            mime_filter, err = _resolve_file_type(file_type)
            if err:
                return err
        results = [
            item for item in d.items.values()
            if item.get("parent_id") == folder_id
            and item["id"] != "root"
            and (mime_filter is None or item["mimeType"] == mime_filter)
        ][:page_size]
        if not results:
            return f"No items found in folder '{folder_id}'."
        return "\n".join([f"Found {len(results)} items in folder '{folder_id}':"] + [_fmt(i) for i in results])

    async def _tool_get_drive_file_content(self, file_id: str, **_) -> str:
        item = self._drive.items.get(file_id)
        if not item:
            return f"[Tool error] File '{file_id}' not found."
        content = self._drive.contents.get(file_id, "")
        return f'File: "{item["name"]}" (ID: {file_id})\n{content}'

    async def _tool_create_drive_file(
        self,
        file_name: str,
        content: str | None = None,
        folder_id: str = "root",
        mime_type: str = "text/plain",
        **_,
    ) -> str:
        _, err = _resolve_file_type(mime_type)
        if err:
            return err
        if mime_type == MIME_SHEET:
            item = self._drive.create_sheet(file_name, folder_id)
        else:
            item = self._drive.create_doc(file_name, folder_id, content or "")
        return (
            f"Created file '{file_name}' (ID: {item['id']}) in folder '{folder_id}'. "
            f"Link: https://docs.example/d/{item['id']}/edit"
        )

    async def _tool_create_drive_folder(
        self,
        folder_name: str,
        parent_folder_id: str = "root",
        **_,
    ) -> str:
        item = self._drive.create_folder(folder_name, parent_folder_id)
        return f"Created folder '{folder_name}' (ID: {item['id']}) in folder '{parent_folder_id}'."

    async def _tool_update_drive_file(
        self,
        file_id: str,
        trashed: bool | None = None,
        name: str | None = None,
        **_,
    ) -> str:
        item = self._drive.items.get(file_id)
        if not item:
            return f"[Tool error] File '{file_id}' not found."
        if trashed is True:
            item["trashed"] = True
        if name:
            item["name"] = name
        return f"Updated file '{item['name']}' (ID: {file_id})."

    async def _tool_copy_drive_file(
        self,
        file_id: str,
        new_name: str | None = None,
        parent_folder_id: str = "root",
        **_,
    ) -> str:
        item = self._drive.items.get(file_id)
        if not item:
            return f"[Tool error] File '{file_id}' not found."
        copy_name = new_name or f"{item['name']} (copy)"
        new_item = self._drive.create_doc(
            copy_name, parent_folder_id, self._drive.contents.get(file_id, "")
        )
        return f"Copied '{item['name']}' to '{copy_name}' (ID: {new_item['id']})."

    # ═══════════════════════════════════════════════════════════════════════════
    # Drive tools
    # ═══════════════════════════════════════════════════════════════════════════

    async def _tool_search_docs(self, query: str, page_size: int = 10, **_) -> str:
        results = [
            item for item in self._drive.items.values()
            if item["mimeType"] == MIME_DOC
            and (
                query.lower() in item["name"].lower()
                or query.lower() in self._drive.contents.get(item["id"], "").lower()
            )
        ][:page_size]
        if not results:
            return f"No docs found for '{query}'."
        return "\n".join([f"Found {len(results)} docs:"] + [_fmt(i) for i in results])

    async def _tool_start_google_auth(self, service_name: str, **_) -> str:
        return f"Authentication for '{service_name}' is not required in the fake server."

    async def _tool_get_drive_file_download_url(self, file_id: str, **_) -> str:
        item = self._drive.items.get(file_id)
        if not item:
            return f"[Tool error] File '{file_id}' not found."
        return f"Download URL for '{item['name']}': https://drive.example/download/{file_id}"

    async def _tool_import_to_google_doc(
        self,
        file_name: str,
        content: str | None = None,
        folder_id: str = "root",
        **_,
    ) -> str:
        item = self._drive.create_doc(file_name, folder_id, content or "")
        return (
            f"Imported '{file_name}' as Google Doc (ID: {item['id']}). "
            f"Link: https://docs.example/d/{item['id']}/edit"
        )

    async def _tool_get_drive_file_permissions(self, file_id: str, **_) -> str:
        item = self._drive.items.get(file_id)
        if not item:
            return f"[Tool error] File '{file_id}' not found."
        return f"Permissions for '{item['name']}' (ID: {file_id}): owner only (fake server)."

    async def _tool_check_drive_file_public_access(self, file_name: str, **_) -> str:
        item = self._drive.find_by_name(file_name)
        if not item:
            return f"File '{file_name}' not found."
        return f"'{file_name}' is not publicly shared (fake server)."

    async def _tool_get_drive_shareable_link(self, file_id: str, **_) -> str:
        item = self._drive.items.get(file_id)
        if not item:
            return f"[Tool error] File '{file_id}' not found."
        return f"Shareable link for '{item['name']}': https://drive.example/share/{file_id}"

    async def _tool_share_drive_file(self, file_id: str, **_) -> str:
        item = self._drive.items.get(file_id)
        if not item:
            return f"[Tool error] File '{file_id}' not found."
        return f"Shared '{item['name']}' (fake server — no actual sharing performed)."

    async def _tool_batch_share_drive_file(self, file_id: str, **_) -> str:
        return await self._tool_share_drive_file(file_id)

    async def _tool_update_drive_permission(self, file_id: str, permission_id: str, **_) -> str:
        return f"Updated permission '{permission_id}' on file '{file_id}' (fake server)."

    async def _tool_remove_drive_permission(self, file_id: str, permission_id: str, **_) -> str:
        return f"Removed permission '{permission_id}' from file '{file_id}' (fake server)."

    async def _tool_transfer_drive_ownership(self, file_id: str, new_owner_email: str, **_) -> str:
        return f"Transferred ownership of '{file_id}' to '{new_owner_email}' (fake server)."

    async def _tool_set_drive_file_permissions(self, file_id: str, **_) -> str:
        return f"Updated file-level permissions for '{file_id}' (fake server)."

    # ═══════════════════════════════════════════════════════════════════════════
    # Docs tools
    # ═══════════════════════════════════════════════════════════════════════════

    async def _tool_get_doc_content(self, document_id: str, **_) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        content = self._drive.contents.get(document_id, "")
        return f'Document: "{item["name"]}" (ID: {document_id})\n\n{content}'

    async def _tool_list_docs_in_folder(
        self,
        folder_id: str = "root",
        page_size: int = 100,
        **_,
    ) -> str:
        results = [
            item for item in self._drive.items.values()
            if item.get("parent_id") == folder_id and item["mimeType"] == MIME_DOC
        ][:page_size]
        if not results:
            return f"No documents found in folder '{folder_id}'."
        return "\n".join([f"Found {len(results)} docs in '{folder_id}':"] + [_fmt(i) for i in results])

    async def _tool_create_doc(self, title: str, content: str = "", folder_id: str = "root", **_) -> str:
        item = self._drive.create_doc(title, folder_id, content)
        return (
            f"Created Google Doc '{title}' (ID: {item['id']}) in folder '{folder_id}'. "
            f"Link: https://docs.example/d/{item['id']}/edit"
        )

    async def _tool_inspect_doc_structure(self, document_id: str, tab_id: str | None = None, **_) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        content = self._drive.contents.get(document_id, "")
        tabs = self._drive.doc_tabs.get(document_id)
        if tabs and tab_id is None:
            # Mirror the real server: documents.get(includeTabsContent=True) leaves
            # the top-level body empty for tab-based docs, so a no-tab_id inspection
            # reports total_length=0 and only lists the available tabs. The agent's
            # TabAwareInspectRefiner is expected to re-call with the first tab_id.
            result = {
                "total_elements": 0,
                "tables": 0,
                "paragraphs": 0,
                "total_length": 0,
                "tabs": tabs,
            }
        elif tabs:
            if not any(t["tab_id"] == tab_id for t in tabs):
                return f"Error: Tab {tab_id} not found in document."
            result = {
                "total_length": max(1, len(content)),
                "tables": 0,
                "statistics": {"elements": content.count("\n") + 1},
                "inspected_tab_id": tab_id,
            }
        else:
            result = {
                "total_length": max(1, len(content)),
                "tables": 0,
                "statistics": {"elements": content.count("\n") + 1},
            }
        return (
            f"Document structure analysis for {document_id}:\n\n"
            f"{json.dumps(result, indent=2)}\n\n"
            f"Link: https://docs.example/d/{document_id}/edit"
        )

    async def _tool_modify_doc_text(
        self,
        document_id: str,
        start_index: int,
        end_index: int | None = None,
        text: str | None = None,
        tab_id: str | None = None,
        **_,
    ) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        if text is None:
            return f"Applied formatting to document {document_id}."
        if tab_id is not None:
            tabs = self._drive.doc_tabs.get(document_id)
            if tabs and not any(t["tab_id"] == tab_id for t in tabs):
                return f"[Tool error] Tab {tab_id} not found in document."
            # Mirror the real server: its 0→1 index auto-correction only fires when
            # tab_id is None; with a tab_id the raw 0 reaches the Docs API → 400.
            if start_index == 0:
                return (
                    "[Tool error] API error in modify_doc_text: <HttpError 400 ... returned "
                    '"Invalid requests[0].insertText: The insertion index must be inside the '
                    "bounds of an existing paragraph. You can still create new paragraphs by "
                    'inserting newlines.">'
                )
        content = self._drive.contents.get(document_id, "")
        if end_index is not None and end_index > start_index:
            new_content = content[:start_index] + text + content[end_index:]
        else:
            # Mirror the real Docs API: an insertion index must be strictly below
            # the segment end (= inspect's total_length). Inserting AT the end
            # index is rejected; the last valid index sits before the final newline.
            segment_end = max(1, len(content))
            if start_index >= segment_end:
                return (
                    f"[Tool error] API error in modify_doc_text: <HttpError 400 ... returned "
                    f'"Invalid requests[0].insertText: Index {start_index} must be less than '
                    f'the end index of the referenced segment, {segment_end}.">'
                )
            new_content = content[:start_index] + text + content[start_index:]
        self._drive.write_doc(document_id, new_content)
        return (
            f"Inserted text at index {start_index} in document {document_id}. "
            f"Text length: {len(text)} characters. "
            f"Link: https://docs.example/d/{document_id}/edit"
        )

    async def _tool_find_and_replace_doc(
        self,
        document_id: str,
        find_text: str,
        replace_text: str,
        match_case: bool = False,
        **_,
    ) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        content = self._drive.contents.get(document_id, "")
        flags = 0 if match_case else re.IGNORECASE
        count = len(re.findall(re.escape(find_text), content, flags=flags))
        new_content = re.sub(re.escape(find_text), replace_text, content, flags=flags)
        self._drive.write_doc(document_id, new_content)
        return f"Replaced {count} occurrence(s) of '{find_text}' in document {document_id}."

    async def _tool_insert_doc_elements(
        self,
        document_id: str,
        element_type: str,
        index: int,
        rows: int | None = None,
        columns: int | None = None,
        list_type: str | None = None,
        text: str | None = None,
        **_,
    ) -> str:
        """Inserts structural elements like tables, lists, or page breaks into a Google Doc."""
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        content = self._drive.contents.get(document_id, "")

        if element_type == "table":
            if not rows or not columns:
                return "Error: 'rows' and 'columns' parameters are required for table insertion."
            placeholder = "\n" + "\n".join(
                "\t".join([""] * columns) for _ in range(rows)
            ) + "\n"
            insert_at = min(index, len(content))
            new_content = content[:insert_at] + placeholder + content[insert_at:]
            description = f"table ({rows}x{columns})"
        elif element_type == "list":
            if not list_type:
                return "Error: 'list_type' parameter is required for list insertion ('UNORDERED' or 'ORDERED')."
            item_text = text or "List item"
            insert_at = min(index, len(content))
            new_content = content[:insert_at] + f"\n• {item_text}\n" + content[insert_at:]
            description = f"{list_type.lower()} list"
        elif element_type == "page_break":
            insert_at = min(index, len(content))
            new_content = content[:insert_at] + "\n---page-break---\n" + content[insert_at:]
            description = "page break"
        elif element_type == "paragraph" or text is not None:
            # LLMs sometimes call this with element_type="paragraph" to insert text
            insert_at = len(content) if index >= len(content) else index
            new_content = content[:insert_at] + (text or "") + content[insert_at:]
            description = "paragraph"
        else:
            return f"Error: Unsupported element type '{element_type}'. Supported: 'table', 'list', 'page_break'."

        self._drive.write_doc(document_id, new_content)
        link = f"https://docs.example/d/{document_id}/edit"
        return f"Inserted {description} at index {index} in document {document_id}. Link: {link}"

    async def _tool_insert_doc_image(self, document_id: str, image_source: str, index: int, **_) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        return f"Inserted image from '{image_source}' at index {index} in document {document_id} (fake server)."

    async def _tool_update_doc_headers_footers(
        self, document_id: str, section_type: str, **_
    ) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        return f"Updated {section_type} in document {document_id} (fake server)."

    async def _tool_batch_update_doc(self, document_id: str, operations: list, **_) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        content = self._drive.contents.get(document_id, "")
        for op in operations:
            insert = op.get("insertText")
            if insert:
                idx = insert.get("location", {}).get("index", len(content))
                t   = insert.get("text", "")
                content = content + t if idx >= len(content) else content[:idx] + t + content[idx:]
        self._drive.write_doc(document_id, content)
        return f"Applied {len(operations)} batch operation(s) to document {document_id}."

    async def _tool_create_table_with_data(
        self, document_id: str, table_data: list, index: int, **_
    ) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        rows = len(table_data)
        cols = max((len(r) for r in table_data), default=0)
        content = self._drive.contents.get(document_id, "")
        placeholder = "\n" + "\n".join("\t".join(row) for row in table_data) + "\n"
        insert_at = min(index, len(content))
        self._drive.write_doc(document_id, content[:insert_at] + placeholder + content[insert_at:])
        link = f"https://docs.example/d/{document_id}/edit"
        return f"Created table ({rows}x{cols}) at index {index} in document {document_id}. Link: {link}"

    async def _tool_debug_table_structure(self, document_id: str, table_index: int = 0, **_) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        return json.dumps({"error": "No tables in fake document", "table_index": table_index})

    async def _tool_export_doc_to_pdf(self, document_id: str, **_) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        return f"Exported '{item['name']}' to PDF (fake server — no file created)."

    async def _tool_update_paragraph_style(
        self, document_id: str, start_index: int, end_index: int, **_
    ) -> str:
        item = self._drive.items.get(document_id)
        if not item:
            return f"[Tool error] Document '{document_id}' not found."
        return f"Applied paragraph style to range {start_index}:{end_index} in document {document_id}."

    async def _tool_get_doc_as_markdown(self, document_id: str, **_) -> str:
        return await self._tool_get_doc_content(document_id)

    async def _tool_read_document_comments(self, document_id: str, **_) -> str:
        return f"No comments found in document '{document_id}' (fake server)."

    async def _tool_create_document_comment(self, document_id: str, comment_content: str, **_) -> str:
        return f"Created comment on document '{document_id}': {comment_content!r} (fake server)."

    async def _tool_reply_to_document_comment(self, document_id: str, comment_id: str, **_) -> str:
        return f"Replied to comment '{comment_id}' in document '{document_id}' (fake server)."

    async def _tool_resolve_document_comment(self, document_id: str, comment_id: str, **_) -> str:
        return f"Resolved comment '{comment_id}' in document '{document_id}' (fake server)."

    # ═══════════════════════════════════════════════════════════════════════════
    # Sheets tools
    # ═══════════════════════════════════════════════════════════════════════════

    async def _tool_list_spreadsheets(self, max_results: int = 25, **_) -> str:
        sheets = [
            i for i in self._drive.items.values() if i["mimeType"] == MIME_SHEET
        ][:max_results]
        if not sheets:
            return "No spreadsheets found."
        return "\n".join(
            [f"Successfully listed {len(sheets)} spreadsheets:"]
            + [f'- "{s["name"]}" (ID: {s["id"]})' for s in sheets]
        )

    async def _tool_get_spreadsheet_info(self, spreadsheet_id: str, **_) -> str:
        item = self._drive.items.get(spreadsheet_id)
        if not item:
            return f"[Tool error] Spreadsheet '{spreadsheet_id}' not found."
        sheet_names = list(self._drive.sheets.get(spreadsheet_id, {}).keys())
        return f'Spreadsheet: "{item["name"]}" (ID: {spreadsheet_id})\nSheets: {sheet_names}'

    def _resolve_sheet(
        self, spreadsheet_id: str, sheet_name: str | None
    ) -> tuple[dict | None, str, list]:
        item = self._drive.items.get(spreadsheet_id)
        if not item:
            return None, "", []
        sheet_data = self._drive.sheets.setdefault(spreadsheet_id, {})
        if not sheet_name:
            sheet_name = next(iter(sheet_data), "Sheet1")
        rows = sheet_data.setdefault(sheet_name, [])
        return item, sheet_name, rows

    async def _tool_read_sheet_values(
        self,
        spreadsheet_id: str,
        range_name: str = "A1:Z1000",
        **_,
    ) -> str:
        err = _a1_range_error(range_name)
        if err:
            return err
        sn, sr, _, er, _ = _parse_range(range_name)
        item, _, rows = self._resolve_sheet(spreadsheet_id, sn)
        if item is None:
            return f"[Tool error] Spreadsheet '{spreadsheet_id}' not found."
        r_end   = (er + 1) if er is not None else len(rows)
        visible = rows[sr:r_end]
        if not visible:
            return f"No data found in range '{range_name}' in spreadsheet {spreadsheet_id}."
        formatted = "\n".join(f"Row {sr + i + 1:2d}: {row}" for i, row in enumerate(visible))
        return (
            f"Successfully read {len(visible)} rows from range '{range_name}' "
            f"in spreadsheet {spreadsheet_id}:\n{formatted}"
        )

    async def _tool_modify_sheet_values(
        self,
        spreadsheet_id: str,
        range_name: str,
        values: Any = None,
        clear_values: bool = False,
        **_,
    ) -> str:
        err = _a1_range_error(range_name)
        if err:
            return err
        sn, sr, sc, _, _ = _parse_range(range_name)
        item, sheet_name, rows = self._resolve_sheet(spreadsheet_id, sn)
        if item is None:
            return f"[Tool error] Spreadsheet '{spreadsheet_id}' not found."
        if clear_values:
            self._drive.write_sheet_tab(spreadsheet_id, sheet_name, [])
            return f"Cleared range '{range_name}' in spreadsheet {spreadsheet_id}."
        if isinstance(values, str):
            values = json.loads(values)
        if not values:
            return "[Tool error] No values provided."
        for ri, row in enumerate(values):
            abs_row = sr + ri
            while len(rows) <= abs_row:
                rows.append([])
            cur = rows[abs_row]
            for ci, val in enumerate(row):
                abs_col = sc + ci
                while len(cur) <= abs_col:
                    cur.append("")
                cur[abs_col] = val
        self._drive.write_sheet_tab(spreadsheet_id, sheet_name, rows)
        total_cells = sum(len(r) for r in values)
        return (
            f"Successfully updated range '{range_name}' in spreadsheet {spreadsheet_id}. "
            f"Updated: {total_cells} cells, {len(values)} rows."
        )

    async def _tool_format_sheet_range(self, spreadsheet_id: str, range_name: str, **_) -> str:
        item = self._drive.items.get(spreadsheet_id)
        if not item:
            return f"[Tool error] Spreadsheet '{spreadsheet_id}' not found."
        return f"Applied formatting to range '{range_name}' in spreadsheet {spreadsheet_id} (fake server)."

    async def _tool_add_conditional_formatting(self, spreadsheet_id: str, **_) -> str:
        return f"Added conditional formatting to spreadsheet '{spreadsheet_id}' (fake server)."

    async def _tool_update_conditional_formatting(self, spreadsheet_id: str, **_) -> str:
        return f"Updated conditional formatting in spreadsheet '{spreadsheet_id}' (fake server)."

    async def _tool_delete_conditional_formatting(self, spreadsheet_id: str, **_) -> str:
        return f"Deleted conditional formatting in spreadsheet '{spreadsheet_id}' (fake server)."

    async def _tool_create_spreadsheet(
        self,
        title: str,
        sheet_names: list | None = None,
        **_,
    ) -> str:
        """Creates a new Google Spreadsheet."""
        item = self._drive.create_sheet(title, "root")
        if sheet_names:
            for name in sheet_names:
                self._drive.sheets.setdefault(item["id"], {})[name] = []
        else:
            self._drive.sheets.setdefault(item["id"], {})["Sheet1"] = []
        spreadsheet_url = f"https://sheets.example/d/{item['id']}/edit"
        return (
            f"Successfully created spreadsheet '{title}' for user. "
            f"ID: {item['id']} | URL: {spreadsheet_url} | Locale: en_US"
        )

    async def _tool_create_sheet(self, spreadsheet_id: str, sheet_name: str, **_) -> str:
        """Creates a new sheet within an existing spreadsheet."""
        item = self._drive.items.get(spreadsheet_id)
        if not item:
            return f"[Tool error] Spreadsheet '{spreadsheet_id}' not found."
        self._drive.sheets.setdefault(spreadsheet_id, {})[sheet_name] = []
        return f"Successfully created sheet '{sheet_name}' in spreadsheet '{item['name']}' (ID: {spreadsheet_id})."

    async def _tool_read_spreadsheet_comments(self, spreadsheet_id: str, **_) -> str:
        return f"No comments found in spreadsheet '{spreadsheet_id}' (fake server)."

    async def _tool_create_spreadsheet_comment(self, spreadsheet_id: str, **_) -> str:
        return f"Created comment on spreadsheet '{spreadsheet_id}' (fake server)."

    async def _tool_reply_to_spreadsheet_comment(self, spreadsheet_id: str, comment_id: str, **_) -> str:
        return f"Replied to comment '{comment_id}' in spreadsheet '{spreadsheet_id}' (fake server)."

    async def _tool_resolve_spreadsheet_comment(self, spreadsheet_id: str, comment_id: str, **_) -> str:
        return f"Resolved comment '{comment_id}' in spreadsheet '{spreadsheet_id}' (fake server)."
