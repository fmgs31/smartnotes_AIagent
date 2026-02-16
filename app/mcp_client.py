import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


class MCPClient:
    def __init__(self, server_url: str, timeout_s: float, max_retries: int):
        # Normalize trailing slash to avoid redirect hops like /mcp/ -> /mcp.
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _retry_decorator(self):
        # attempts = 1 + max_retries
        return retry(
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=0.3, max=3.0),
        )

    def _build_transport(self, token: str | None):
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return StreamableHttpTransport(url=self.server_url, headers=headers)

    async def list_tools(self, *, token: str | None = None):
        @_retry_decorator_wrapper(self._retry_decorator())
        async def _do_list():
            transport = self._build_transport(token)
            async with Client(transport=transport) as client:
                return await asyncio.wait_for(client.list_tools(), timeout=self.timeout_s)

        return await _do_list()

    async def call_tool(self, *, token: str, tool_name: str, arguments: dict):
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
