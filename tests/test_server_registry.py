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
    def test_loads_valid_single_server(self, write_servers):
        path = write_servers({"servers": [{"id": "srv", "url": "http://srv/mcp/"}]})
        registry = ServerRegistry.load_from_file(path)
        server = registry.all()[0]
        assert server.id == "srv"
        assert server.url == "http://srv/mcp/"

    def test_loads_allowed_tools(self, write_servers):
        path = write_servers({"servers": [
            {"id": "srv", "url": "http://srv/mcp/", "allowed_tools": ["search", "create"]}
        ]})
        registry = ServerRegistry.load_from_file(path)
        assert registry.all()[0].allowed_tools == ["search", "create"]

    def test_no_allowed_tools_is_none(self, write_servers):
        path = write_servers({"servers": [{"id": "srv", "url": "http://srv/mcp/"}]})
        registry = ServerRegistry.load_from_file(path)
        assert registry.all()[0].allowed_tools is None

    def test_empty_servers_list_raises(self, write_servers):
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
        path = write_servers({"servers": [
            {"id": "srv", "url": "http://srv1/mcp/"},
            {"id": "srv", "url": "http://srv2/mcp/"},
        ]})
        with pytest.raises(ValueError, match="Duplicate"):
            ServerRegistry.load_from_file(path)

    def test_invalid_allowed_tools_type_raises(self, write_servers):
        path = write_servers({"servers": [
            {"id": "srv", "url": "http://srv/mcp/", "allowed_tools": "not-a-list"}
        ]})
        with pytest.raises(ValueError, match="invalid allowed_tools"):
            ServerRegistry.load_from_file(path)

    def test_url_overridden_by_env_var(self, write_servers, monkeypatch):
        monkeypatch.setenv("SRV_URL", "http://override/mcp/")
        path = write_servers({"servers": [{"id": "srv", "url": "http://original/mcp/"}]})
        registry = ServerRegistry.load_from_file(path)
        assert registry.all()[0].url == "http://override/mcp/"


class TestAll:
    @pytest.fixture
    def registry(self, write_servers):
        path = write_servers({"servers": [
            {"id": "primary", "url": "http://primary/mcp/"},
            {"id": "secondary", "url": "http://secondary/mcp/"},
        ]})
        return ServerRegistry.load_from_file(path)

    def test_all_returns_all_servers(self, registry):
        ids = {s.id for s in registry.all()}
        assert ids == {"primary", "secondary"}

    def test_all_returns_correct_urls(self, registry):
        urls = {s.url for s in registry.all()}
        assert "http://primary/mcp/" in urls
        assert "http://secondary/mcp/" in urls

    def test_all_returns_list(self, registry):
        assert isinstance(registry.all(), list)
