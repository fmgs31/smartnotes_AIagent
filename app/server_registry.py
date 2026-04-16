import os
from dataclasses import dataclass

from app.utils import load_json


@dataclass(frozen=True)
class ServerConfig:
    id: str
    url: str
    allowed_tools: list[str] | None = None
    blocked_tools: tuple[str, ...] = ()


class ServerRegistry:
    def __init__(self, servers: dict[str, ServerConfig]):
        self._servers = servers

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

            blocked_tools = s.get("blocked_tools", [])
            if not isinstance(blocked_tools, list) or not all(isinstance(t, str) and t for t in blocked_tools):
                raise ValueError(f"Server '{sid}' has invalid blocked_tools; expected non-empty strings")

            # Allow the URL to be overridden at runtime via an env var.
            url_env_var_name = s.get("url_env_var_name")
            if url_env_var_name:
                url = os.environ.get(url_env_var_name) or url

            servers[sid] = ServerConfig(
                id=sid,
                url=url,
                allowed_tools=allowed_tools,
                blocked_tools=tuple(blocked_tools),
            )

        return cls(servers=servers)

    def all(self) -> list[ServerConfig]:
        return list(self._servers.values())
