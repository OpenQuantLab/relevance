"""Characterisation tests for the Alpha Lab panel/backtest/evaluate pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

import alpha_lab as lab


def _spec(tickers, **overrides) -> lab.SignalSpec:
    base = dict(
        name="charac",
        universe="custom",
        tickers=list(tickers),
        from_date="2008-01-01",
        features=["return_21d", "return_63d", "return_252d", "vol_63d"],
        method="relevance",
        horizon=21,
        rebalance="M",
        thresh=0.75,
        most=True,
        min_training_periods=12,
        train_window=36,
    )
    base.update(overrides)
    return lab.SignalSpec(**base)


@pytest.fixture(scope="module")
def make_spec(tickers):
    def _factory(**overrides):
        return _spec(tickers, **overrides)

    return _factory


@pytest.fixture(scope="module")
def panel(prices, make_spec):
    return lab.build_panel(prices, make_spec())


@pytest.fixture(scope="module")
def small_panel(panel):
    """Last 24 rebalance dates only — the relevance model is O(n_train x n_test)."""
    keep = sorted(panel["date"].unique())[-24:]
    return panel[panel["date"].isin(keep)].reset_index(drop=True)


def test_feature_library_inventory(snapshot):
    """The registry's ids, groups and sign priors must survive refactoring."""
    snapshot(
        "lab_feature_registry",
        {
            fid: {"label": f.label, "group": f.group, "sign": f.sign}
            for fid, f in sorted(lab.FEATURES.items())
        },
    )


def test_universes_snapshot(snapshot):
    snapshot("lab_universes", {k: list(v) for k, v in sorted(lab.UNIVERSES.items())})


def test_spec_fingerprint_is_stable(make_spec):
    assert make_spec().fingerprint() == make_spec().fingerprint()
    assert make_spec().fingerprint() != make_spec(horizon=63).fingerprint()


def test_build_panel_snapshot(panel, snapshot):
    snapshot(
        "lab_panel",
        {
            "columns": list(panel.columns),
            "n_rows": len(panel),
            "n_dates": int(panel["date"].nunique()),
            "first_date": str(pd.Timestamp(panel["date"].min()).date()),
            "last_date": str(pd.Timestamp(panel["date"].max()).date()),
            "describe": panel.drop(columns=["date", "ticker"]).describe(),
            "head": panel.head(8),
        },
    )


def test_relevance_backtest_snapshot(small_panel, make_spec, snapshot):
    spec = make_spec(min_training_periods=8, train_window=12)
    results = lab.run_backtest(small_panel, spec)
    summary = lab.evaluate(results, spec)
    snapshot(
        "lab_backtest_relevance",
        {
            "n_predictions": len(results),
            "head": results.head(10),
            "tail": results.tail(10),
            "summary": summary["summary"],
        },
    )


def test_linear_backtest_snapshot(panel, make_spec, snapshot):
    spec = make_spec(method="linear")
    results = lab.run_backtest(panel, spec)
    summary = lab.evaluate(results, spec)
    snapshot(
        "lab_backtest_linear",
        {
            "n_predictions": len(results),
            "head": results.head(10),
            "summary": summary["summary"],
        },
    )


def test_evaluate_components_snapshot(panel, make_spec, snapshot):
    spec = make_spec(method="linear")
    results = lab.run_backtest(panel, spec)
    snapshot(
        "lab_evaluate_components",
        {
            "ic_series_head": lab.ic_series(results).head(12),
            "ls_series_head": lab.ls_series(results, spec.n_bins).head(6),
            "bin_profile": lab.bin_profile(results, spec.n_bins),
            "turnover": lab.turnover(results, spec.n_bins),
            "yearly_stability": lab.yearly_stability(results, spec.n_bins),
        },
    )


def test_permutation_test_snapshot(panel, make_spec, snapshot):
    spec = make_spec(method="linear")
    results = lab.run_backtest(panel, spec)
    perm = lab.permutation_test(results, n_perm=25, n_bins=spec.n_bins, seed=0)
    perm.pop("null_ic_dist")
    perm.pop("null_sharpe_dist")
    snapshot("lab_permutation", perm)


def test_spec_roundtrip(make_spec):
    spec = make_spec()
    assert lab.SignalSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_spec_rejects_unknown_feature(make_spec):
    with pytest.raises(ValueError, match="Unknown features"):
        make_spec(features=["not_a_real_feature"])
