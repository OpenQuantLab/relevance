"""Characterisation tests for the Study 2 walk-forward pipeline and analytics."""

from __future__ import annotations

import pandas as pd
import pytest

import paper_replication as repl


@pytest.fixture(scope="module")
def panel(prices):
    return repl.build_backtest_panel(prices, forward_period=21)


@pytest.fixture(scope="module")
def results(panel):
    # Last 24 months only; the walk-forward is quadratic in training size.
    keep = sorted(panel["date"].unique())[-24:]
    reduced = panel[panel["date"].isin(keep)].reset_index(drop=True)
    return repl.run_walk_forward(
        reduced, thresh=0.75, most=True, min_training_months=8, progress=False
    )


def test_constants_snapshot(snapshot):
    snapshot(
        "repl_constants",
        {
            "DEFAULT_STOCKS": list(repl.DEFAULT_STOCKS),
            "FUNDAMENTAL_ATTRS": list(repl.FUNDAMENTAL_ATTRS),
            "PRICE_ATTRS": list(repl.PRICE_ATTRS),
            "BACKTEST_FEATURE_COLS": list(repl.BACKTEST_FEATURE_COLS),
        },
    )


def test_build_backtest_panel_snapshot(panel, snapshot):
    snapshot(
        "repl_panel",
        {
            "columns": list(panel.columns),
            "n_rows": len(panel),
            "n_dates": int(panel["date"].nunique()),
            "first_date": str(pd.Timestamp(panel["date"].min()).date()),
            "last_date": str(pd.Timestamp(panel["date"].max()).date()),
            "describe": panel.drop(columns=["date", "ticker"]).describe(),
        },
    )


def test_latest_period_features_snapshot(prices, snapshot):
    latest = repl.build_latest_period_features(prices)
    snapshot("repl_latest_features", latest)


def test_walk_forward_snapshot(results, snapshot):
    snapshot(
        "repl_walk_forward",
        {
            "n_predictions": len(results),
            "columns": list(results.columns),
            "head": results.head(10),
            "tail": results.tail(10),
        },
    )


def test_analytics_snapshot(results, snapshot):
    snapshot(
        "repl_analytics",
        {
            "period_metrics_head": repl.compute_period_metrics(results).head(12),
            "cumulative_ls_head": repl.compute_cumulative_ls(results).head(12),
            "quintile_analysis": repl.compute_quintile_analysis(
                results, pred_col="predicted", actual_col="actual"
            ),
        },
    )


def test_predict_live_signal_snapshot(panel, prices, snapshot):
    latest = repl.build_latest_period_features(prices)
    signal = repl.predict_live_signal(panel, latest, thresh=0.75, most=True)
    snapshot("repl_live_signal", signal)
