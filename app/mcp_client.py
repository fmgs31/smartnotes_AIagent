import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


class MCPClient:
    def __init__(self, server_url: str, timeout_s: float, max_retries: int):
        # Normalize trailing slash to avoid redirect hops like /mcp/ -> /mcp.
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client: Client | None = None
        self._session_token: str | None = None

    def _retry_decorator(self):
        # attempts = 1 + max_retries
        # initial/max chosen to cover Cloud Run cold-start windows (typically 5-15s).
        return retry(
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=1.5, max=10.0),
        )

    def _build_transport(self, token: str | None):
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return StreamableHttpTransport(url=self.server_url, headers=headers)

    async def _open_session(self, token: str | None) -> None:
        if self._client is not None:
            if token != self._session_token:
                raise ValueError("MCPClient session is already bound to a different token")
            return

        @_retry_decorator_wrapper(self._retry_decorator())
        async def _do_open():
            transport = self._build_transport(token)
            client = Client(transport=transport)
            try:
                await client.__aenter__()
            except Exception:
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass
                raise
            return client

        client = await _do_open()
        self._client = client
        self._session_token = token

    async def _close_session(self) -> None:
        if self._client is None:
            return

        client = self._client
        self._client = None
        self._session_token = None
        await client.__aexit__(None, None, None)

    @asynccontextmanager
    async def session(self, *, token: str | None = None) -> AsyncIterator["MCPClient"]:
        await self._open_session(token)
        try:
            yield self
        finally:
            await self._close_session()

    def _active_client(self, token: str | None) -> Client | None:
        if self._client is None:
            return None
        if token != self._session_token:
            raise ValueError("MCPClient session token mismatch")
        return self._client

    async def list_tools(self, *, token: str | None = None):
        active_client = self._active_client(token)

        if active_client is not None:
            @_retry_decorator_wrapper(self._retry_decorator())
            async def _do_list_connected():
                return await asyncio.wait_for(active_client.list_tools(), timeout=self.timeout_s)

            return await _do_list_connected()

        @_retry_decorator_wrapper(self._retry_decorator())
        async def _do_list():
            transport = self._build_transport(token)
            async with Client(transport=transport) as client:
                return await asyncio.wait_for(client.list_tools(), timeout=self.timeout_s)

        return await _do_list()

    async def call_tool(self, *, token: str, tool_name: str, arguments: dict):
        active_client = self._active_client(token)

        if active_client is not None:
            @_retry_decorator_wrapper(self._retry_decorator())
            async def _do_call_connected():
                return await asyncio.wait_for(
                    active_client.call_tool(name=tool_name, arguments=arguments),
                    timeout=self.timeout_s,
                )

            return await _do_call_connected()

        @_retry_decorator_wrapper(self._retry_decorator())
        async def _do_call():
            transport = self._build_transport(token)
            async with Client(transport=transport) as client:
                return await asyncio.wait_for(
                    client.call_tool(name=tool_name, arguments=arguments),
                    timeout=self.timeout_s,
                )

        return await _do_call()


def _retry_decorator_wrapper(deco):
    # tiny helper so we can build the retry decorator dynamically
    def wrap(fn):
        return deco(fn)

    return wrap
