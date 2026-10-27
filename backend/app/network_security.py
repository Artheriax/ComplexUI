"""Host and browser-origin validation for the local ComplexUI service."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "DEFAULT_CORS_ORIGINS",
    "ENV_ALLOWED_HOSTS",
    "ENV_CORS_ORIGINS",
    "MAX_REQUEST_BYTES",
    "RequestSizeLimitMiddleware",
    "configured_allowed_hosts",
    "configured_cors_origins",
    "origin_is_allowed",
]

ENV_ALLOWED_HOSTS = "COMPLEXUI_ALLOWED_HOSTS"
ENV_CORS_ORIGINS = "COMPLEXUI_CORS_ORIGINS"
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]", "testserver")
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
MAX_REQUEST_BYTES = 16 * 1024 * 1024


def configured_allowed_hosts() -> list[str]:
    """Return the configured Host/Origin allowlist.

    The default is loopback plus the in-process test client. Set
    ``COMPLEXUI_ALLOWED_HOSTS`` to a comma-separated list of additional hosts
    (``*`` explicitly disables the restriction) when using a custom hostname or
    a LAN address.
    """
    raw = os.environ.get(ENV_ALLOWED_HOSTS)
    if raw is None:
        return list(DEFAULT_ALLOWED_HOSTS)
    hosts = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return hosts or list(DEFAULT_ALLOWED_HOSTS)


def configured_cors_origins() -> list[str]:
    """Return browser origins allowed to call the REST API.

    Only the Vite dev server origins are allowed by default. The launcher sets
    ``COMPLEXUI_CORS_ORIGINS`` to match ``--frontend-port``; set it explicitly
    when running Vite behind a custom hostname. An empty value disables CORS
    (useful for same-origin production deployments).
    """
    raw = os.environ.get(ENV_CORS_ORIGINS)
    if raw is None:
        return list(DEFAULT_CORS_ORIGINS)
    origins: list[str] = []
    for item in raw.split(","):
        value = item.strip().rstrip("/").lower()
        if not value:
            continue
        try:
            parsed = urlsplit(value)
        except ValueError:
            continue
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            continue
        if parsed.path or parsed.query or parsed.fragment:
            continue
        origins.append(value)
    return origins


def _normalize_host(host: str) -> str:
    value = host.strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def _host_matches(host: str, pattern: str) -> bool:
    candidate = _normalize_host(host)
    normalized_pattern = pattern.strip().lower()
    if normalized_pattern == "*":
        return True
    if normalized_pattern.startswith("*."):
        return candidate.endswith(normalized_pattern[1:])
    return candidate == _normalize_host(normalized_pattern)


def origin_is_allowed(
    origin: str | None, allowed_hosts: Sequence[str]
) -> bool:
    """Validate a browser Origin header against the Host allowlist.

    Missing Origin is accepted for non-browser clients such as tests and CLI
    tools. Browsers always send Origin for WebSocket handshakes, so a malicious
    page cannot omit it to bypass this check.
    """
    if origin is None:
        return True
    if not origin or origin == "null":
        return False
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return False
    return any(_host_matches(parsed.hostname, pattern) for pattern in allowed_hosts)


class RequestSizeLimitMiddleware:
    """Reject oversized HTTP requests.

    Requests with ``Content-Length`` are rejected before body streaming.
    Chunked requests without that header are buffered only when their route
    uses the normal JSON limit; upload routes with a larger override keep
    streaming so their per-file caps can apply without buffering the payload.
    ``overrides`` maps path prefixes (for example ``"/api/uploads"``) to a
    larger or smaller limit. The longest matching prefix wins.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = MAX_REQUEST_BYTES,
        overrides: Mapping[str, int] | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.overrides = tuple(
            sorted(
                (
                    (prefix.rstrip("/") or "/", limit)
                    for prefix, limit in (overrides or {}).items()
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )

    def _limit_for(self, path: str) -> int:
        for prefix, limit in self.overrides:
            if path == prefix or path.startswith(f"{prefix}/"):
                return limit
        return self.max_bytes

    @staticmethod
    async def _buffer_body(
        receive: Receive, limit: int
    ) -> tuple[list[Message], bool]:
        messages: list[Message] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                return messages, True
            total += len(message.get("body", b""))
            if total > limit:
                return messages, False
            if not message.get("more_body", False):
                return messages, True

    @staticmethod
    def _replay_body(messages: list[Message], receive: Receive) -> Receive:
        async def replay() -> Message:
            if messages:
                return messages.pop(0)
            return await receive()

        return replay

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit_for(scope["path"])
        raw_length = Headers(scope=scope).get("content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                await PlainTextResponse(
                    "Invalid Content-Length", status_code=400
                )(scope, receive, send)
                return
            if content_length > limit:
                await PlainTextResponse(
                    "Request body too large", status_code=413
                )(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        if limit > self.max_bytes:
            await self.app(scope, receive, send)
            return

        messages, accepted = await self._buffer_body(receive, limit)
        if not accepted:
            await PlainTextResponse(
                "Request body too large", status_code=413
            )(scope, receive, send)
            return
        await self.app(scope, self._replay_body(messages, receive), send)
