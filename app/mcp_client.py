import asyncio
import re
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator
from urllib.parse import urlsplit
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token as google_id_token

from app.config import settings

_HTTP_4XX = re.compile(r"HttpError (4\d\d)")
_RATE_LIMIT = re.compile(r"ratelimitexceeded", re.IGNORECASE)


def _is_permanent(exc: BaseException) -> bool:
    """True for a 4xx from the upstream Google API: the request itself was rejected,
    so the same arguments get the same answer. Rate limits are the exception, as
    they clear after a wait: 429, and the 403 rateLimitExceeded /
    userRateLimitExceeded that Drive uses instead of 429."""
    text = str(exc)
    match = _HTTP_4XX.search(text)
    if match is None or match.group(1) == "429":
        return False
    return not _RATE_LIMIT.search(text)


def _retry_transient(retry_state) -> bool:
    """tenacity `retry` predicate. A plain callable over RetryCallState rather than
    retry_if_exception(...), which only landed in tenacity 8.2 while requirements.txt
    pins tenacity>=8.0.0. The parameter must be named `retry_state`: tenacity 8.0/8.1
    pass it as a keyword, later versions positionally."""
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None and outcome.failed else None
    return exc is not None and not _is_permanent(exc)


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
        # The predicate keeps that backoff for transient failures only: a 4xx from the
        # upstream Google API can never succeed on a second attempt with the same
        # arguments, and retrying it cost ~19s per Docs append.
        return retry(
            stop=stop_after_attempt(1 + settings.mcp_max_retries),
            wait=wait_exponential_jitter(
                initial=settings.mcp_retry_initial_wait_s,
                max=settings.mcp_retry_max_wait_s,
            ),
            retry=_retry_transient,
            reraise=True,
        )

    def _fetch_id_token(self) -> str:
        """ID token proving the gateway's service-account identity to Cloud Run IAM.
        Audience is the service origin: Cloud Run matches it against the service URL."""
        parts = urlsplit(self.server_url)
        audience = f"{parts.scheme}://{parts.netloc}"
        return google_id_token.fetch_id_token(GoogleAuthRequest(), audience)

    def _build_transport(self, token: str | None, id_token: str | None = None):
        # Cloud Run reads X-Serverless-Authorization for IAM and forwards Authorization
        # untouched, so the user's Google token still reaches the MCP's OAuth 2.1 layer.
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if id_token:
            headers["X-Serverless-Authorization"] = f"Bearer {id_token}"
        return StreamableHttpTransport(url=self.server_url, headers=headers or None)

    async def _open_session(self, token: str | None) -> None:
        if self._client is not None:
            if token != self._session_token:
                raise ValueError("MCPClient session is already bound to a different token")
            return

        @self._retry_decorator()
        async def _do_open():
            # fetch_id_token is blocking (metadata-server HTTP call), hence the thread.
            id_token = (
                await asyncio.to_thread(self._fetch_id_token) if settings.mcp_iam_auth else None
            )
            stack = AsyncExitStack()
            client = await stack.enter_async_context(
                Client(transport=self._build_transport(token, id_token))
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

        # google_workspace_mcp raises on API errors rather than returning them as a
        # string result, so a Docs 400 arrives here as a ToolError carrying the
        # stringified HttpError — which _retry_transient reads to stop the backoff.
        # If the server ever carries a structured error code, match on that instead.
        @self._retry_decorator()
        async def _do_call():
            return await asyncio.wait_for(
                client.call_tool(name=tool_name, arguments=arguments),
                timeout=settings.mcp_timeout_s,
            )

        return await _do_call()
