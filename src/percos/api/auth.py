"""Bearer token authentication middleware (§11 Security).

When `secret_key` is set to something other than the default, all requests
(except /health) must include `Authorization: Bearer <secret_key>`.
"""

from __future__ import annotations

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from percos.config import get_settings

# Paths that skip auth
PUBLIC_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc"}

DEFAULT_KEY = "change-me-in-production"


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Simple bearer-token auth middleware.

    Auth is enforced only when secret_key != default placeholder.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()

        # Skip auth if key is the default placeholder (dev mode)
        if settings.secret_key == DEFAULT_KEY:
            return await call_next(request)

        # Skip for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")

        token = auth_header[7:]  # strip "Bearer "
        if token != settings.secret_key:
            raise HTTPException(status_code=403, detail="Invalid bearer token")

        return await call_next(request)
