"""
Performance analytics for walk-forward prediction results.

Every function here takes a results frame with at least the columns
``date``, ``ticker``, ``predicted`` and ``actual``.

This module is the single home for metrics that were previously duplicated
across the backtest engine, the CLI harness and the study replication.  Note
that two genuinely different turnover definitions existed; both are kept and
named explicitly rather than silently reconciled, because they produce
different numbers and both appear in published output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Return statistics
# -----------------------------------------------------------------------

def annualized_sharpe(returns, periods_per_year: int = 12):
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def max_drawdown(cum_returns):
    """Max drawdown from a cumulative return series (0-based, not 1-based)."""
    wealth = 1 + cum_returns
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return dd.min()


# -----------------------------------------------------------------------
# Cross-sectional ranking helpers
# -----------------------------------------------------------------------

def _ranked(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.dropna(subset=["predicted", "actual"]).copy()
    valid["rank"] = valid.groupby("date")["predicted"].rank(pct=True)
    return valid


def ls_series(results: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Equal-weight long top-bin / short bottom-bin return series."""
    valid = _ranked(results)
    hi = valid[valid["rank"] > 1 - 1 / n_bins].groupby("date")["actual"].mean()
    lo = valid[valid["rank"] <= 1 / n_bins].groupby("date")["actual"].mean()
    out = pd.DataFrame({"long": hi, "short": lo}).dropna()
    out["long_short"] = out["long"] - out["short"]
    out["benchmark"] = valid.groupby("date")["actual"].mean()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    for col in ("long_short", "long", "short", "benchmark"):
        out[f"cum_{col}"] = (1 + out[col]).cumprod() - 1
    return out


def bin_profile(results: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    valid = _ranked(results)
    valid["bin"] = np.clip((valid["rank"] * n_bins).astype(int), 0, n_bins - 1)
    prof = (valid.groupby("bin")["actual"]
            .agg(["mean", "std", "count"]).reset_index())
    prof["label"] = [f"Q{i + 1}" for i in prof["bin"]]
    return prof


def ic_series(results: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation of predictions and outcomes, per date."""
    valid = results.dropna(subset=["predicted", "actual"])
    ranks = pd.DataFrame({
        "date": valid["date"].to_numpy(),
        "p": valid.groupby("date")["predicted"].rank().to_numpy(),
        "a": valid.groupby("date")["actual"].rank().to_numpy(),
    })
    ranks["pa"] = ranks["p"] * ranks["a"]
    agg = ranks.groupby("date").agg(
        mp=("p", "mean"), ma=("a", "mean"), mpa=("pa", "mean"),
        sp=("p", "std"), sa=("a", "std"), n=("p", "size"))
    agg = agg[agg["n"] > 2]
    cov = (agg["mpa"] - agg["mp"] * agg["ma"]) * agg["n"] / (agg["n"] - 1)
    ic = cov / (agg["sp"] * agg["sa"]).replace(0, np.nan)
    ic.index = pd.to_datetime(ic.index)
    return ic.sort_index()


def yearly_stability(results: pd.DataFrame, n_bins: int = 5,
                     ppy: int = 12) -> pd.DataFrame:
    ls = ls_series(results, n_bins)
    df = pd.DataFrame({"long_short": ls["long_short"],
                       "ic": ic_series(results)}).dropna()
    if df.empty:
        return pd.DataFrame()
    return df.groupby(df.index.year).agg(
        periods=("long_short", "size"),
        mean_ic=("ic", "mean"),
        ls_total=("long_short", lambda s: (1 + s).prod() - 1),
        ls_sharpe=("long_short", lambda s: annualized_sharpe(s, ppy)),
    ).reset_index(names="year")


# -----------------------------------------------------------------------
# Turnover
#
# Two definitions are in use and they are NOT interchangeable: for equally
# sized portfolios the double-sided figure is twice the single-sided one.
# -----------------------------------------------------------------------

def turnover(results: pd.DataFrame, n_bins: int = 5) -> float:
    """Single-sided turnover of the long book, using percentile ranks."""
    valid = _ranked(results)
    prev: set = set()
    moves = []
    for _, group in valid.groupby("date"):
        longs = set(group[group["rank"] > 1 - 1 / n_bins]["ticker"])
        if prev and longs:
            kept = len(prev & longs)
            moves.append(1 - kept / max(len(prev), len(longs)))
        prev = longs or prev
    return float(np.mean(moves)) if moves else np.nan


def compute_turnover(backtest_results: pd.DataFrame, n_bins: int = 5) -> float:
    """Double-sided turnover of the long book, using ``pd.qcut`` bins."""
    prev_long = set()
    turnovers = []
    for _date, group in backtest_results.groupby("date"):
        valid = group.dropna(subset=["predicted", "actual"]).copy()
        nb = min(n_bins, len(valid))
        if nb < 2:
            continue
        valid["q"] = pd.qcut(
            valid["predicted"], nb, labels=False, duplicates="drop"
        )
        long_tickers = set(valid[valid["q"] == valid["q"].max()]["ticker"])
        if prev_long:
            union = prev_long | long_tickers
            changed = len(union - (prev_long & long_tickers))
            turnovers.append(changed / max(len(prev_long), len(long_tickers)))
        prev_long = long_tickers
    return np.mean(turnovers) if turnovers else np.nan


# -----------------------------------------------------------------------
# Quantile analytics (qcut based)
# -----------------------------------------------------------------------

def compute_quintile_analysis(results, pred_col="predicted_return",
                              actual_col="actual_return"):
    """Average actual return per predicted-return quintile."""
    valid = results.dropna(subset=[pred_col, actual_col]).copy()
    if len(valid) < 5:
        return None

    n_bins = min(5, len(valid))
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    valid["quintile"] = pd.qcut(
        valid[pred_col], n_bins, labels=labels, duplicates="drop"
    )
    return (
        valid.groupby("quintile", observed=True)[actual_col]
        .agg(["mean", "count", "std"])
    )


def compute_period_metrics(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """Per-date IC, hit rate and quintile spread."""
    metrics = []
    for date, group in backtest_results.groupby("date"):
        valid = group.dropna(subset=["predicted", "actual"])
        if len(valid) < 5:
            continue

        ic = valid["predicted"].corr(valid["actual"], method="spearman")
        median_pred = valid["predicted"].median()
        median_actual = valid["actual"].median()
        hit_rate = (
            (valid["predicted"] > median_pred)
            == (valid["actual"] > median_actual)
        ).mean()

        valid = valid.copy()
        n_bins = min(5, len(valid))
        valid["q"] = pd.qcut(
            valid["predicted"], n_bins, labels=False, duplicates="drop"
        )
        q_hi = valid[valid["q"] == valid["q"].max()]["actual"].mean()
        q_lo = valid[valid["q"] == valid["q"].min()]["actual"].mean()

        metrics.append({
            "date": date,
            "ic": ic,
            "hit_rate": hit_rate,
            "quintile_spread": q_hi - q_lo,
            "q_high": q_hi,
            "q_low": q_lo,
            "n": len(valid),
        })

    return pd.DataFrame(
        metrics,
        columns=["date", "ic", "hit_rate", "quintile_spread",
                 "q_high", "q_low", "n"],
    )


def compute_cumulative_ls(backtest_results: pd.DataFrame,
                          n_bins: int = 5) -> pd.DataFrame:
    """Cumulative long-Q5 / short-Q1 returns, using ``pd.qcut`` bins."""
    ls_returns = []
    for date, group in backtest_results.groupby("date"):
        valid = group.dropna(subset=["predicted", "actual"]).copy()
        nb = min(n_bins, len(valid))
        if nb < 2:
            continue
        valid["q"] = pd.qcut(
            valid["predicted"], nb, labels=False, duplicates="drop"
        )
        q_hi = valid[valid["q"] == valid["q"].max()]["actual"].mean()
        q_lo = valid[valid["q"] == valid["q"].min()]["actual"].mean()
        ls_returns.append({"date": date, "long_short": q_hi - q_lo,
                           "long": q_hi, "short": q_lo})

    df = pd.DataFrame(ls_returns).set_index("date").sort_index()
    df["cumulative_ls"] = (1 + df["long_short"]).cumprod() - 1
    df["cumulative_long"] = (1 + df["long"]).cumprod() - 1
    df["cumulative_short"] = (1 + df["short"]).cumprod() - 1
    return df


# -----------------------------------------------------------------------
# Significance
# -----------------------------------------------------------------------

def permutation_test(results: pd.DataFrame, n_perm: int = 200,
                     n_bins: int = 5, ppy: int = 12, seed: int = 0) -> dict:
    """Empirical p-values under the null that predictions carry no signal."""
    rng = np.random.default_rng(seed)
    valid = results.dropna(subset=["predicted", "actual"])
    obs_ls = ls_series(valid, n_bins)["long_short"]
    obs = dict(ic=float(ic_series(valid).mean()),
               sharpe=float(annualized_sharpe(obs_ls, ppy)))

    groups = [g[["predicted", "actual"]].to_numpy()
              for _, g in valid.groupby("date")]
    null_ic, null_sharpe = [], []
    for _ in range(n_perm):
        ics, his, los = [], [], []
        for g in groups:
            if len(g) < n_bins:
                continue
            pred = rng.permutation(g[:, 0])
            act = g[:, 1]
            ics.append(pd.Series(pred).corr(pd.Series(act), method="spearman"))
            r = pd.Series(pred).rank(pct=True).to_numpy()
            his.append(act[r > 1 - 1 / n_bins].mean())
            los.append(act[r <= 1 / n_bins].mean())
        if not ics:
            continue
        null_ic.append(np.nanmean(ics))
        null_sharpe.append(annualized_sharpe(
            pd.Series(np.array(his) - np.array(los)), ppy))

    null_ic = np.array(null_ic, dtype=float)
    null_sharpe = np.array(null_sharpe, dtype=float)
    return dict(
        n_perm=int(len(null_ic)),
        obs_ic=obs["ic"],
        obs_sharpe=obs["sharpe"],
        p_ic=float(np.mean(null_ic >= obs["ic"])) if len(null_ic) else np.nan,
        p_sharpe=(float(np.mean(null_sharpe >= obs["sharpe"]))
                  if len(null_sharpe) else np.nan),
        null_sharpe_mean=(float(np.nanmean(null_sharpe))
                          if len(null_sharpe) else np.nan),
        null_sharpe_p95=(float(np.nanpercentile(null_sharpe, 95))
                         if len(null_sharpe) else np.nan),
        null_ic_dist=null_ic.tolist(),
        null_sharpe_dist=null_sharpe.tolist(),
    )
