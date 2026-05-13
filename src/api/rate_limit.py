"""
Rate-limiting setup using slowapi.

Per-user limit: read from the ApiKey.rate_limit_rpm field when an API key
is presented; fall back to 60 rpm for JWT-authenticated requests.

Usage
-----
In main.py:
    from .rate_limit import limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

On an endpoint:
    @router.post("/score")
    @limiter.limit(_per_user_limit)
    async def score(..., request: Request, ...):
        ...
"""

from slowapi import Limiter
from slowapi.util import get_remote_address


def _get_key(request) -> str:  # type: ignore[override]
    """Return a rate-limit key: X-Api-Key prefix if present, else remote IP."""
    api_key = request.headers.get("X-Api-Key")
    if api_key:
        return f"apikey:{api_key[:8]}"
    return get_remote_address(request)


limiter = Limiter(key_func=_get_key, default_limits=["60/minute"])
