"""
Password gate for the Alpha Lab dashboard.

When ``ALPHA_LAB_PASSWORD`` is set the dashboard is read-only: everything is
browsable, but running a backtest requires a session token granted by typing
the password.  Tokens live in memory only, so restarting the service logs
everyone out.
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading

_PASSWORD = os.environ.get("ALPHA_LAB_PASSWORD", "")
LOCKED = bool(_PASSWORD)

_SESSIONS: set = set()
_SESSION_LOCK = threading.Lock()


def grant_session() -> str:
    token = secrets.token_urlsafe(24)
    with _SESSION_LOCK:
        _SESSIONS.add(token)
    return token


def is_unlocked(token) -> bool:
    if not LOCKED:
        return True
    with _SESSION_LOCK:
        return isinstance(token, str) and token in _SESSIONS


def password_matches(candidate) -> bool:
    return bool(candidate) and hmac.compare_digest(str(candidate), _PASSWORD)
