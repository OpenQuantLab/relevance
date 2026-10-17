"""Characterisation tests for the performance metrics in alpha_signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alpha_signal as alpha


@pytest.fixture(scope="module")
def results():
    """A fixed set of walk-forward predictions, independent of any data feed."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2015-01-30", periods=48, freq="BME")
    rows = []
    for date in dates:
        for i in range(12):
            pred = rng.normal()
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{i:02d}",
                    "predicted": pred,
                    "actual": 0.3 * pred + rng.normal(scale=0.05),
                }
            )
    return pd.DataFrame(rows)


def test_annualized_sharpe_snapshot(snapshot):
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0.01, 0.04, size=120))
    snapshot(
        "metrics_sharpe",
        {
            "monthly": alpha.annualized_sharpe(series, 12),
            "weekly": alpha.annualized_sharpe(series, 52),
            "quarterly": alpha.annualized_sharpe(series, 4),
            "too_short": alpha.annualized_sharpe(pd.Series([0.01]), 12),
            "zero_vol": alpha.annualized_sharpe(pd.Series([0.01] * 10), 12),
        },
    )


def test_max_drawdown_snapshot(snapshot):
    cum = pd.Series([0.0, 0.10, 0.25, 0.05, -0.10, 0.02, 0.30])
    snapshot("metrics_max_drawdown", {"max_drawdown": alpha.max_drawdown(cum)})


def test_turnover_snapshot(results, snapshot):
    snapshot(
        "metrics_turnover",
        {
            "q5": alpha.compute_turnover(results, n_bins=5),
            "q3": alpha.compute_turnover(results, n_bins=3),
        },
    )


def test_summarize_performance_snapshot(results, snapshot):
    summary, metrics, ls = alpha.summarize_performance(results)
    snapshot(
        "metrics_summary",
        {
            "summary": summary,
            "metrics_head": metrics.head(10),
            "ls_head": ls.head(10),
        },
    )
