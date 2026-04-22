import json
import pytest
from pathlib import Path

from app.server_registry import ServerRegistry


@pytest.fixture
def write_servers(tmp_path: Path):
    """Helper to write a servers.json and return its path as str."""
    def _write(data: dict) -> str:
        p = tmp_path / "servers.json"
        p.write_text(json.dumps(data))
        return str(p)
    return _write


class TestLoadFromFile:
    """ServerRegistry.load_from_file: parses and validates servers.json."""

    def test_loads_valid_single_server(self, write_servers):
        # Minimal valid entry (id + url) round-trips through the registry.
        path = write_servers({"servers": [{"id": "srv", "url": "http://srv/mcp/"}]})
        registry = ServerRegistry.load_from_file(path)
        server = registry.all()[0]
        assert server.id == "srv", f"Expected id='srv', got {server.id!r}"
        assert server.url == "http://srv/mcp/", f"Expected url unchanged, got {server.url!r}"

    def test_loads_allowed_tools(self, write_servers):
        # allowed_tools is kept verbatim for later filtering during discovery.
        path = write_servers({"servers": [
            {"id": "srv", "url": "http://srv/mcp/", "allowed_tools": ["search", "create"]}
        ]})
        registry = ServerRegistry.load_from_file(path)
        tools = registry.all()[0].allowed_tools
        assert tools == ["search", "create"], (
            f"Expected allowed_tools=['search','create'], got {tools!r}"
        )

    def test_no_allowed_tools_is_none(self, write_servers):
        # Missing optional key → None, not [] (distinguishes "no filter" from "empty filter").
        path = write_servers({"servers": [{"id": "srv", "url": "http://srv/mcp/"}]})
        registry = ServerRegistry.load_from_file(path)
        tools = registry.all()[0].allowed_tools
        assert tools is None, (
            f"Missing 'allowed_tools' must parse as None, got {tools!r}"
        )

    def test_empty_servers_list_raises(self, write_servers):
        # An empty list is almost always a config error — fail fast.
        path = write_servers({"servers": []})
        with pytest.raises(ValueError, match="non-empty"):
            ServerRegistry.load_from_file(path)

    def test_missing_id_raises(self, write_servers):
        path = write_servers({"servers": [{"url": "http://srv/mcp/"}]})
        with pytest.raises(ValueError, match="'id' and 'url'"):
            ServerRegistry.load_from_file(path)

    def test_missing_url_raises(self, write_servers):
        path = write_servers({"servers": [{"id": "srv"}]})
        with pytest.raises(ValueError, match="'id' and 'url'"):
            ServerRegistry.load_from_file(path)

    def test_duplicate_server_id_raises(self, write_servers):
        # Server IDs are used as dict keys elsewhere — duplicates must be rejected.
        path = write_servers({"servers": [
            {"id": "srv", "url": "http://srv1/mcp/"},
            {"id": "srv", "url": "http://srv2/mcp/"},
        ]})
        with pytest.raises(ValueError, match="Duplicate"):
            ServerRegistry.load_from_file(path)

    def test_invalid_allowed_tools_type_raises(self, write_servers):
        # allowed_tools must be a list; a string is a common mistake — reject.
        path = write_servers({"servers": [
            {"id": "srv", "url": "http://srv/mcp/", "allowed_tools": "not-a-list"}
        ]})
        with pytest.raises(ValueError, match="invalid allowed_tools"):
            ServerRegistry.load_from_file(path)

    def test_url_overridden_by_env_var(self, write_servers, monkeypatch):
        # When the entry declares url_env_var_name and the env var is set, the env value wins.
        monkeypatch.setenv("SERVER_URL_X", "http://override/mcp/")
        path = write_servers({"servers": [{"id": "srvx", "url": "http://original/mcp/", "url_env_var_name": "SERVER_URL_X"}]})
        registry = ServerRegistry.load_from_file(path)
        url = registry.all()[0].url
        assert url == "http://override/mcp/", (
            f"Expected env-var override to replace the file URL, got {url!r}"
        )


class TestAll:
    """ServerRegistry.all: returns all loaded server configs."""

    @pytest.fixture
    def registry(self, write_servers):
        path = write_servers({"servers": [
            {"id": "primary", "url": "http://primary/mcp/"},
            {"id": "secondary", "url": "http://secondary/mcp/"},
        ]})
        return ServerRegistry.load_from_file(path)

    def test_all_returns_all_servers(self, registry):
        ids = {s.id for s in registry.all()}
        assert ids == {"primary", "secondary"}, (
            f"Expected ids {{'primary','secondary'}}, got {ids!r}"
        )

    def test_all_returns_correct_urls(self, registry):
        urls = {s.url for s in registry.all()}
        assert "http://primary/mcp/" in urls, f"'primary' URL missing from all(): {urls!r}"
        assert "http://secondary/mcp/" in urls, f"'secondary' URL missing from all(): {urls!r}"

    def test_all_returns_list(self, registry):
        result = registry.all()
        assert isinstance(result, list), (
            f"Expected all() to return a list, got {type(result).__name__}: {result!r}"
        )
