import time
from collections import deque, defaultdict
from typing import Callable, Deque, Dict, Tuple, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

# Simple in-memory rate limiter (token bucket / fixed window hybrid)
# Keyed by user (JWT sub) if available, else client IP
# Defaults: 10 requests per 60s

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, requests: int = 10, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.requests = max(1, int(requests))
        self.window = max(1, int(window_seconds))
        self.buckets: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        key = await self._make_key(request)
        now = time.time()
        dq = self.buckets[key]
        # purge old
        cutoff = now - self.window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self.requests:
            from fastapi import status
            return Response(
                content="Rate limit exceeded. Try again later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="text/plain",
            )
        dq.append(now)
        return await call_next(request)

    async def _make_key(self, request: Request) -> str:
        sub: Optional[str] = None
        try:
            from auth.jwt_handler import verify_token  # local import to avoid circular deps
            auth = request.headers.get("authorization") or request.headers.get("Authorization")
            if auth and auth.lower().startswith("bearer "):
                token = auth.split(" ", 1)[1]
                payload = verify_token(token)
                # Use username (sub) if available
                sub = str(payload.get("sub"))
        except Exception:
            sub = None
        if sub:
            return f"user:{sub}"
        # Fallback to client IP
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.time()
        method = request.method
        path = request.url.path
        client = request.client.host if request.client else "-"
        user = "-"
        try:
            from auth.jwt_handler import verify_token
            auth = request.headers.get("authorization") or request.headers.get("Authorization")
            if auth and auth.lower().startswith("bearer "):
                token = auth.split(" ", 1)[1]
                payload = verify_token(token)
                user = str(payload.get("sub") or "-")
        except Exception:
            pass
        try:
            response = await call_next(request)
            status_code = response.status_code
            elapsed_ms = int((time.time() - start) * 1000)
            print(f"[HTTP] {method} {path} {status_code} user={user} ip={client} {elapsed_ms}ms")
            return response
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            print(f"[HTTP] {method} {path} 500 user={user} ip={client} {elapsed_ms}ms error={e}")
            raise

