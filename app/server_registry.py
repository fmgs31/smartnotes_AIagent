from dataclasses import dataclass
from app.utils import load_json


@dataclass(frozen=True)
class ServerConfig:
    id: str
    url: str
    allowed_tools: list[str] | None = None


class ServerRegistry:
    def __init__(self, servers: dict[str, ServerConfig], default_server_id: str):
        self._servers = servers
        self._default = default_server_id

    @classmethod
    def load_from_file(cls, path: str) -> "ServerRegistry":
        data = load_json(path)

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

            allowed_tools = s.get("allowed_tools")
            if allowed_tools is not None:
                if not isinstance(allowed_tools, list) or not all(isinstance(t, str) and t for t in allowed_tools):
                    raise ValueError(f"Server '{sid}' has invalid allowed_tools; expected non-empty strings")

            servers[sid] = ServerConfig(id=sid, url=url, allowed_tools=allowed_tools)

        default_id = data.get("default_server_id") or servers_list[0]["id"]
        if default_id not in servers:
            raise ValueError(f"default_server_id '{default_id}' not found in servers")

        return cls(servers=servers, default_server_id=default_id)

    def get(self, server_id: str | None) -> ServerConfig:
        sid = server_id or self._default
        if sid not in self._servers:
            raise KeyError(f"Unknown server_id: {sid}")
        return self._servers[sid]

    def all(self) -> list[ServerConfig]:
        return list(self._servers.values())
