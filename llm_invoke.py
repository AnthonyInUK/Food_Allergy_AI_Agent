"""
Thin retry wrapper for LangChain Runnable.invoke / tool.invoke.

Goals (minimal surface):
- Retry transient provider errors (429 / 5xx / timeouts / connection resets).
- Do not retry auth/config errors (401 / 403).
- Exponential backoff + jitter between attempts.
"""

from __future__ import annotations

import os
import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw.strip()))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw.strip()))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _http_status(exc: BaseException) -> Optional[int]:
    s = getattr(exc, "status_code", None)
    if isinstance(s, int):
        return s
    resp = getattr(exc, "response", None)
    s2 = getattr(resp, "status_code", None) if resp is not None else None
    if isinstance(s2, int):
        return s2
    return None


def _is_auth_error(exc: BaseException) -> bool:
    status = _http_status(exc)
    if status in (401, 403):
        return True
    low = str(exc).lower()
    return "invalid api key" in low or "incorrect api key" in low or "unauthorized" in low


def _is_retryable_error(exc: BaseException) -> bool:
    if _is_auth_error(exc):
        return False
    status = _http_status(exc)
    if status is not None:
        if status in (401, 403):
            return False
        if status in (408, 409, 425, 429, 500, 502, 503, 504):
            return True
    name = type(exc).__name__.lower()
    low = str(exc).lower()
    if "ratelimit" in name or "rate limit" in low or "429" in low:
        return True
    if "timeout" in name or "timed out" in low or "readtimeout" in low:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError)):
        return True
    if "connection" in low and ("reset" in low or "refused" in low or "aborted" in low):
        return True
    if "temporarily unavailable" in low or "overloaded" in low or "503" in low or "502" in low:
        return True
    return False


def invoke_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: Optional[int] = None,
    initial_interval: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    max_interval: Optional[float] = None,
) -> T:
    """
    Run fn() with retries. Disabled when LLM_INVOKE_RETRY=0/false.

    Env (optional):
    - LLM_INVOKE_RETRY: default true
    - LLM_INVOKE_MAX_ATTEMPTS: default 3 (first try + 2 retries)
    - LLM_INVOKE_INITIAL_INTERVAL / LLM_INVOKE_BACKOFF_FACTOR / LLM_INVOKE_MAX_INTERVAL
    """
    if not _env_bool("LLM_INVOKE_RETRY", True):
        return fn()

    attempts = max_attempts if max_attempts is not None else _env_int("LLM_INVOKE_MAX_ATTEMPTS", 3)
    t0 = initial_interval if initial_interval is not None else _env_float("LLM_INVOKE_INITIAL_INTERVAL", 0.45)
    bf = backoff_factor if backoff_factor is not None else _env_float("LLM_INVOKE_BACKOFF_FACTOR", 2.0)
    cap = max_interval if max_interval is not None else _env_float("LLM_INVOKE_MAX_INTERVAL", 8.0)

    last: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return fn()
        except BaseException as e:
            last = e
            if attempt >= attempts - 1 or not _is_retryable_error(e):
                raise
            sleep_s = min(cap, t0 * (bf**attempt))
            sleep_s += random.uniform(0, min(0.35, sleep_s * 0.25))
            time.sleep(sleep_s)
    assert last is not None
    raise last
