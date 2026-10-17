"""
Process-wide configuration: filesystem locations and environment settings.

Historically each module derived its paths from ``__file__``.  Now that the
code lives inside a package that no longer coincides with the project root,
locations are resolved here instead — from ``RELEVANCE_DATA_DIR`` when set,
otherwise relative to the working directory (which is what the systemd units
and all documented CLI invocations use).
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    env = os.environ.get("RELEVANCE_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


DATA_ROOT = _root()

LAB_DIR = DATA_ROOT / "lab"
RUNS_DIR = LAB_DIR / "runs"
CACHE_DIR = LAB_DIR / "cache"
SIGNALS_DIR = DATA_ROOT / "signals"
RESULTS_DIR = DATA_ROOT / "results"

# Legacy location kept readable so previously cached prices are still found.
LEGACY_CACHE_DIR = DATA_ROOT / "tests" / "cache"
CACHE_DIRS = [CACHE_DIR, LEGACY_CACHE_DIR]


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
