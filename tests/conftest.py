"""
Shared fixtures and the snapshot helper used by the characterisation tests.

These tests pin the *current* numerical behaviour of the library so that the
restructuring work can be verified to be behaviour-preserving.  Nothing here
touches the network: all prices are generated from a seeded RNG.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


# ---------------------------------------------------------------------------
# Deterministic synthetic market data
# ---------------------------------------------------------------------------

# Kept deliberately small: the relevance model is O(n_train x n_test) pure Python,
# so the panel size dominates total suite runtime.
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def make_prices(n_days: int = 1200, tickers=None, seed: int = 12345) -> pd.DataFrame:
    """A reproducible geometric random walk panel of adjusted closes."""
    tickers = list(tickers or TICKERS)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", periods=n_days)

    # A common market factor plus idiosyncratic noise, so cross-sectional
    # features (beta, relative strength) have something to bite on.
    market = rng.normal(0.0003, 0.010, size=n_days)
    betas = np.linspace(0.6, 1.4, len(tickers))
    idio = rng.normal(0.0, 0.012, size=(n_days, len(tickers)))

    rets = market[:, None] * betas[None, :] + idio
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    return list(TICKERS)


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture(scope="session")
def attributes_outcomes():
    """A small fixed attributes/outcomes/circumstances set for the core model."""
    rng = np.random.default_rng(7)
    n, k = 60, 3
    X = rng.normal(size=(n, k))
    beta = np.array([0.8, -0.4, 0.25])
    y = X @ beta + rng.normal(scale=0.5, size=n)
    X_t = rng.normal(size=(5, k))

    cols = [f"attr_{i}" for i in range(k)]
    return (
        pd.DataFrame(X, columns=cols),
        pd.DataFrame(X_t, columns=cols),
        pd.DataFrame({"outcome": y}),
    )


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

def _normalise(obj):
    """Convert numpy/pandas objects into plain JSON-friendly Python."""
    if isinstance(obj, pd.DataFrame):
        return {
            "__frame__": True,
            "columns": [str(c) for c in obj.columns],
            "index": [str(i) for i in obj.index],
            "values": _normalise(obj.to_numpy().tolist()),
        }
    if isinstance(obj, pd.Series):
        return {
            "__series__": True,
            "index": [str(i) for i in obj.index],
            "values": _normalise(obj.to_numpy().tolist()),
        }
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _normalise(obj.tolist())
    if isinstance(obj, np.generic):
        return _normalise(obj.item())
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float):
        # Guard against -0.0 / 0.0 churn and platform ULP noise.
        return round(obj, 10) + 0.0
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _compare(expected, actual, path, tol, failures):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                failures.append(f"{path}.{key}: unexpected new key")
            elif key not in actual:
                failures.append(f"{path}.{key}: key disappeared")
            else:
                _compare(expected[key], actual[key], f"{path}.{key}", tol, failures)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            failures.append(f"{path}: length {len(expected)} -> {len(actual)}")
            return
        for i, (e, a) in enumerate(zip(expected, actual)):
            _compare(e, a, f"{path}[{i}]", tol, failures)
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        e, a = float(expected), float(actual)
        if math.isnan(e) and math.isnan(a):
            return
        if math.isnan(e) != math.isnan(a):
            failures.append(f"{path}: {e!r} -> {a!r} (NaN mismatch)")
            return
        if not math.isclose(e, a, rel_tol=tol, abs_tol=tol):
            failures.append(f"{path}: {e!r} -> {a!r}")
        return
    if expected != actual:
        failures.append(f"{path}: {expected!r} -> {actual!r}")


def assert_snapshot(name: str, payload, tol: float = 1e-8):
    """
    Compare ``payload`` against the stored snapshot.

    If the snapshot does not exist it is written and the test is skipped, so
    the first run bootstraps the baseline.  Snapshots are committed.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    actual = _normalise(payload)

    if not path.exists():
        path.write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")
        pytest.skip(f"baseline snapshot written: {path.name}")

    expected = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    _compare(expected, actual, name, tol, failures)
    if failures:
        shown = "\n  ".join(failures[:25])
        more = "" if len(failures) <= 25 else f"\n  ... and {len(failures) - 25} more"
        pytest.fail(f"snapshot '{name}' changed:\n  {shown}{more}")


@pytest.fixture
def snapshot():
    return assert_snapshot
