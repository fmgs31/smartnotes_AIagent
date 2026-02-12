
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    id: str
    url: str


class ServerRegistry:
    def __init__(self, servers: dict[str, ServerConfig], default_server_id: str):
        self._servers = servers
        self._default = default_server_id

    @classmethod
    def load_from_file(cls, path: str) -> "ServerRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        servers_list = data.get("servers", [])
        if not servers_list:
            raise ValueError("servers.json must contain a non-empty 'servers' list")

        servers: dict[str, ServerConfig] = {}
        for s in servers_list:
            sid = s.get("id")
            url = s.get("url")
            if not sid or not url:
                raise ValueError("Each server must have 'id' and 'url'")
            if sid in servers:
                raise ValueError(f"Duplicate server id: {sid}")
            servers[sid] = ServerConfig(id=sid, url=url)

        default_id = data.get("default_server_id") or servers_list[0]["id"]
        if default_id not in servers:
            raise ValueError(f"default_server_id '{default_id}' not found in servers")

        return cls(servers=servers, default_server_id=default_id)

    def get(self, server_id: str | None) -> ServerConfig:
        sid = server_id or self._default
        if sid not in self._servers:
            raise KeyError(f"Unknown server_id: {sid}")
        return self._servers[sid]
