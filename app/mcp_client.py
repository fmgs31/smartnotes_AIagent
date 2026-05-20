import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from app.config import settings


class MCPClient:
    def __init__(self, server_url: str):
        # Normalize trailing slash to avoid redirect hops like /mcp/ -> /mcp.
        self.server_url = server_url.rstrip("/")
        self._client: Client | None = None
        self._session_token: str | None = None
        self._exit_stack: AsyncExitStack | None = None

    def _retry_decorator(self):
        # Retry config reflects Cloud Run platform characteristics (cold-start windows),
        # not anything specific to individual servers — so it lives in global settings.
        return retry(
            stop=stop_after_attempt(1 + settings.mcp_max_retries),
            wait=wait_exponential_jitter(
                initial=settings.mcp_retry_initial_wait_s,
                max=settings.mcp_retry_max_wait_s,
            ),
            reraise=True,
        )

    def _build_transport(self, token: str | None):
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return StreamableHttpTransport(url=self.server_url, headers=headers)

    async def _open_session(self, token: str | None) -> None:
        if self._client is not None:
            if token != self._session_token:
                raise ValueError("MCPClient session is already bound to a different token")
            return

        @self._retry_decorator()
        async def _do_open():
            stack = AsyncExitStack()
            client = await stack.enter_async_context(
                Client(transport=self._build_transport(token))
            )
            return stack, client

        self._exit_stack, self._client = await _do_open()
        self._session_token = token

    async def _close_session(self) -> None:
        if self._exit_stack is None:
            return

        # Null out before awaiting so a concurrent _close_session call sees None
        # and returns immediately, preventing a double-close.
        stack, self._exit_stack = self._exit_stack, None
        self._client = None
        self._session_token = None
        await stack.aclose()

    @asynccontextmanager
    async def session(self, *, token: str | None = None) -> AsyncIterator["MCPClient"]:
        await self._open_session(token)
        try:
            yield self
        finally:
            await self._close_session()

    def _require_client(self, token: str | None) -> Client:
        """Return the active client or raise if no session is open."""
        if self._client is None:
            raise RuntimeError("No MCP session is open; call session() first")
        if token != self._session_token:
            raise ValueError("MCPClient session token mismatch")
        return self._client

    async def list_tools(self, *, token: str | None = None):
        client = self._require_client(token)

        @self._retry_decorator() 
        async def _do_list():
            return await asyncio.wait_for(client.list_tools(), timeout=settings.mcp_timeout_s)

        return await _do_list()

    async def call_tool(self, *, token: str, tool_name: str, arguments: dict):
        client = self._require_client(token)

        @self._retry_decorator() # won't retry on tool error because those are wrapped as string in the server
        async def _do_call():
            return await asyncio.wait_for(
                client.call_tool(name=tool_name, arguments=arguments),
                timeout=settings.mcp_timeout_s,
            )

        return await _do_call()
