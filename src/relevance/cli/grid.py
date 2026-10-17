"""
Relevance Alpha Signal – US Market Test Suite
=============================================

Runs a battery of alpha tests on the relevance signal over US equity
universes:

  1. Baseline walk-forward backtest per universe
  2. Parameter sweep (relevance threshold, most/least relevant, horizon)
  3. Statistical significance: IC t-stat / IR, permutation test on the
     long-short Sharpe (null = shuffled predictions within each date)
  4. Quintile monotonicity and decile spread
  5. Sub-period (per-year) stability
  6. Benchmark comparison vs equal-weight universe

Prices are cached to tests/cache/*.parquet so re-runs are fast.

Usage
-----
    relevance-grid                 # full suite
    relevance-grid --quick         # mega-cap only, small grid
    relevance-grid --universe sp100
"""

import argparse
import time

import numpy as np
import pandas as pd

from relevance import config
from relevance.backtest import harness as alpha
from relevance.data import cache, universes
from relevance.studies import paper as repl

CACHE_DIR = config.CACHE_DIR
OUT_DIR = config.RESULTS_DIR


# -----------------------------------------------------------------------
# Universes
# -----------------------------------------------------------------------

MEGA30 = universes.MEGA30
SP100 = universes.SP100
UNIVERSES = universes.UNIVERSES


# -----------------------------------------------------------------------
# Cached price loading
# -----------------------------------------------------------------------

def load_prices(tickers, from_date, to_date, exchange="US", refresh=False):
    return cache.load_prices(tickers, from_date, to_date, exchange,
                             refresh=refresh, verbose=True)


# -----------------------------------------------------------------------
# Extra analytics
# -----------------------------------------------------------------------

def ic_stats(metrics):
    """Mean IC, standard error, t-stat and annualised information ratio."""
    ic = metrics["ic"].dropna()
    n = len(ic)
    if n < 3 or ic.std(ddof=1) == 0:
        return dict(mean_ic=np.nan, ic_std=np.nan, ic_t=np.nan, ic_ir=np.nan,
                    n_periods=n)
    mean, sd = ic.mean(), ic.std(ddof=1)
    return dict(
        mean_ic=mean,
        ic_std=sd,
        ic_t=mean / (sd / np.sqrt(n)),
        ic_ir=mean / sd * np.sqrt(12),
        n_periods=n,
    )


def bin_profile(results, n_bins=5):
    """Mean actual return per predicted bin, pooled, plus monotonicity."""
    valid = results.dropna(subset=["predicted", "actual"]).copy()
    # Rank within each date so bins are cross-sectional, not pooled-in-time.
    valid["rank"] = valid.groupby("date")["predicted"].rank(pct=True)
    valid["bin"] = np.clip((valid["rank"] * n_bins).astype(int), 0, n_bins - 1)
    prof = valid.groupby("bin")["actual"].agg(["mean", "std", "count"])
    # Spearman of bin index vs mean return: +1 = perfectly monotonic.
    mono = prof["mean"].reset_index().corr(method="spearman").iloc[0, 1]
    spread = prof["mean"].iloc[-1] - prof["mean"].iloc[0]
    return prof, mono, spread


def ls_series(results, n_bins=5):
    """Cross-sectional long-short return series using percentile ranks."""
    valid = results.dropna(subset=["predicted", "actual"]).copy()
    valid["rank"] = valid.groupby("date")["predicted"].rank(pct=True)
    hi = valid[valid["rank"] > 1 - 1 / n_bins].groupby("date")["actual"].mean()
    lo = valid[valid["rank"] <= 1 / n_bins].groupby("date")["actual"].mean()
    out = pd.DataFrame({"long": hi, "short": lo})
    out["long_short"] = out["long"] - out["short"]
    return out.dropna()


def permutation_test(results, n_perm=500, n_bins=5, seed=0):
    """
    Null: predictions carry no cross-sectional information.
    Shuffle predicted values within each date, recompute mean IC and
    long-short Sharpe, and report empirical p-values.
    """
    rng = np.random.default_rng(seed)
    valid = results.dropna(subset=["predicted", "actual"]).copy()

    obs_ls = ls_series(valid, n_bins)["long_short"]
    obs_sharpe = alpha.annualized_sharpe(obs_ls)
    obs_ic = (
        valid.groupby("date")
        .apply(lambda g: g["predicted"].corr(g["actual"], method="spearman"),
               include_groups=False)
        .mean()
    )

    groups = [g for _, g in valid.groupby("date")]
    null_sharpe, null_ic = [], []
    for _ in range(n_perm):
        ics, his, los = [], [], []
        for g in groups:
            pred = rng.permutation(g["predicted"].to_numpy())
            act = g["actual"].to_numpy()
            if len(act) < n_bins:
                continue
            ics.append(pd.Series(pred).corr(pd.Series(act), method="spearman"))
            r = pd.Series(pred).rank(pct=True).to_numpy()
            his.append(act[r > 1 - 1 / n_bins].mean())
            los.append(act[r <= 1 / n_bins].mean())
        ls = pd.Series(np.array(his) - np.array(los))
        null_sharpe.append(alpha.annualized_sharpe(ls))
        null_ic.append(np.nanmean(ics))

    null_sharpe = np.array(null_sharpe, dtype=float)
    null_ic = np.array(null_ic, dtype=float)
    return dict(
        obs_ic=obs_ic,
        obs_sharpe=obs_sharpe,
        p_ic=float(np.mean(null_ic >= obs_ic)),
        p_sharpe=float(np.mean(null_sharpe >= obs_sharpe)),
        null_sharpe_mean=float(np.nanmean(null_sharpe)),
        null_sharpe_p95=float(np.nanpercentile(null_sharpe, 95)),
    )


def yearly_stability(results, n_bins=5):
    ls = ls_series(results, n_bins)
    ls.index = pd.to_datetime(ls.index)
    ic = (
        results.dropna(subset=["predicted", "actual"])
        .groupby("date")
        .apply(lambda g: g["predicted"].corr(g["actual"], method="spearman"),
               include_groups=False)
    )
    ic.index = pd.to_datetime(ic.index)
    df = pd.DataFrame({"long_short": ls["long_short"], "ic": ic}).dropna()
    by_year = df.groupby(df.index.year).agg(
        months=("long_short", "size"),
        mean_ic=("ic", "mean"),
        ls_total=("long_short", lambda s: (1 + s).prod() - 1),
        ls_sharpe=("long_short", alpha.annualized_sharpe),
    )
    return by_year


def benchmark_return(panel, dates):
    """Equal-weight universe forward return, matched to backtest dates."""
    bm = (
        panel[panel["date"].isin(dates)]
        .groupby("date")["forward_return"].mean()
    )
    return bm


# -----------------------------------------------------------------------
# Single configuration run
# -----------------------------------------------------------------------

def run_config(panel, label, thresh, most, min_training_months=12,
               n_bins=5, n_perm=0, progress=True):
    t0 = time.time()
    results = repl.run_walk_forward(
        panel,
        feature_cols=repl.BACKTEST_FEATURE_COLS,
        thresh=thresh,
        most=most,
        min_training_months=min_training_months,
        progress=progress,
    )
    metrics = repl.compute_period_metrics(results)
    ls = ls_series(results, n_bins)
    prof, mono, spread = bin_profile(results, n_bins)

    row = dict(config=label, thresh=thresh, most=most)
    row.update(ic_stats(metrics))
    row["hit_rate"] = metrics["hit_rate"].mean()
    row["ic_pos_rate"] = (metrics["ic"] > 0).mean()
    row["ls_mean_m"] = ls["long_short"].mean()
    row["ls_sharpe"] = alpha.annualized_sharpe(ls["long_short"])
    row["ls_total"] = (1 + ls["long_short"]).prod() - 1
    row["ls_maxdd"] = alpha.max_drawdown((1 + ls["long_short"]).cumprod() - 1)
    row["bin_spread"] = spread
    row["monotonicity"] = mono
    row["turnover"] = alpha.compute_turnover(results, n_bins)
    row["secs"] = time.time() - t0

    if n_perm:
        row.update({f"perm_{k}": v for k, v in
                    permutation_test(results, n_perm=n_perm,
                                     n_bins=n_bins).items()})
    return row, results, metrics, ls, prof


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="US alpha test suite")
    ap.add_argument("--universe", nargs="+", default=["mega30", "sp100"],
                    choices=list(UNIVERSES))
    ap.add_argument("--from-date", default="2010-01-01")
    ap.add_argument("--to-date", default=None)
    ap.add_argument("--horizons", nargs="+", type=int, default=[21, 63])
    ap.add_argument("--threshs", nargs="+", type=float,
                    default=[0.50, 0.75, 0.90])
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.universe = ["mega30"]
        args.horizons = [21]
        args.threshs = [0.75]
        args.n_perm = 100

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    all_rows = []
    best = {}

    for uni in args.universe:
        tickers = UNIVERSES[uni]
        print(f"\n{'=' * 78}\nUNIVERSE: {uni}  ({len(tickers)} tickers)"
              f"\n{'=' * 78}")
        prices = load_prices(tickers, args.from_date, args.to_date,
                             refresh=args.refresh)

        for h in args.horizons:
            print(f"\n-- building panel, horizon = {h}d --")
            panel = repl.build_backtest_panel(prices, forward_period=h)
            print(f"   panel: {len(panel)} obs, "
                  f"{panel['date'].nunique()} months, "
                  f"{panel['ticker'].nunique()} tickers")

            for thresh in args.threshs:
                for most in (True, False):
                    label = f"{uni}|h{h}|t{thresh:.2f}|{'most' if most else 'least'}"
                    print(f"   running {label} …", flush=True)
                    try:
                        row, results, metrics, ls, prof = run_config(
                            panel, label, thresh, most, n_perm=0
                        )
                    except Exception as exc:
                        print(f"   {label} FAILED: {exc}")
                        continue
                    row["universe"] = uni
                    row["horizon"] = h
                    all_rows.append(row)
                    print(f"   -> IC={row['mean_ic']:+.4f} "
                          f"(t={row['ic_t']:+.2f}) "
                          f"Sharpe={row['ls_sharpe']:+.2f} "
                          f"[{row['secs']:.0f}s]", flush=True)

                    key = (uni, h)
                    if key not in best or (
                        row["ls_sharpe"] > best[key][0]["ls_sharpe"]
                    ):
                        best[key] = (row, results, metrics, ls, prof, panel)

    if not all_rows:
        print("No configurations completed.")
        return

    grid = pd.DataFrame(all_rows)
    cols = ["config", "n_periods", "mean_ic", "ic_t", "ic_ir", "ic_pos_rate",
            "hit_rate", "ls_mean_m", "ls_sharpe", "ls_total", "ls_maxdd",
            "bin_spread", "monotonicity", "turnover"]
    print(f"\n\n{'=' * 78}\nPARAMETER GRID SUMMARY\n{'=' * 78}")
    print(grid[cols].to_string(index=False,
                               float_format=lambda v: f"{v:,.4f}"))
    grid.to_csv(OUT_DIR / "grid_summary.csv", index=False)

    # ---- deep dive on the best config per universe/horizon ----
    for (_uni, _h), (row, results, _metrics, ls, prof, panel) in best.items():
        print(f"\n\n{'=' * 78}\nDEEP DIVE – best config: {row['config']}"
              f"\n{'=' * 78}")

        print("\n--- Pooled quintile profile (cross-sectional ranks) ---")
        print(prof.to_string(float_format=lambda v: f"{v:,.4f}"))
        print(f"Monotonicity (Spearman bin vs mean): {row['monotonicity']:+.3f}")

        print("\n--- Benchmark comparison ---")
        bm = benchmark_return(panel, ls.index)
        bm = bm.reindex(ls.index).dropna()
        comp = pd.DataFrame({
            "long_short": ls["long_short"],
            "long_only": ls["long"],
            "equal_weight_bm": bm,
        }).dropna()
        comp["long_vs_bm"] = comp["long_only"] - comp["equal_weight_bm"]
        stats = pd.DataFrame({
            "mean_m": comp.mean(),
            "sharpe": comp.apply(alpha.annualized_sharpe),
            "total": (1 + comp).prod() - 1,
            "maxdd": comp.apply(lambda s: alpha.max_drawdown(
                (1 + s).cumprod() - 1)),
        })
        print(stats.to_string(float_format=lambda v: f"{v:,.4f}"))

        print(f"\n--- Permutation test ({args.n_perm} shuffles) ---")
        if args.n_perm:
            pt = permutation_test(results, n_perm=args.n_perm)
            print(f"  observed mean IC        {pt['obs_ic']:+.4f}  "
                  f"p = {pt['p_ic']:.3f}")
            print(f"  observed L/S Sharpe     {pt['obs_sharpe']:+.3f}  "
                  f"p = {pt['p_sharpe']:.3f}")
            print(f"  null Sharpe mean / p95  {pt['null_sharpe_mean']:+.3f}"
                  f" / {pt['null_sharpe_p95']:+.3f}")

        print("\n--- Yearly stability ---")
        print(yearly_stability(results).to_string(
            float_format=lambda v: f"{v:,.4f}"))

        tag = row["config"].replace("|", "_").replace(".", "")
        results.to_csv(OUT_DIR / f"preds_{tag}.csv", index=False)
        ls.to_csv(OUT_DIR / f"ls_{tag}.csv")

    print(f"\n\nArtifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
