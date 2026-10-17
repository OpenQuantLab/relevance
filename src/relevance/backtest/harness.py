"""
Relevance Alpha Signal – Backtest Harness
==========================================

Generates cross-sectional alpha signals from the relevance-based prediction
model and evaluates them with standard quant metrics:
  - Information Coefficient (IC) time series
  - Quintile spread returns
  - Cumulative long-short equity curve
  - Sharpe ratio, max drawdown, hit rate, turnover

Usage
-----
    python alpha_signal.py                          # defaults
    python alpha_signal.py --tickers AAPL MSFT ...  # custom universe
    python alpha_signal.py --from-date 2015-01-01 --thresh 0.70
    python alpha_signal.py --output results.csv     # save signal + returns
"""

import argparse

import numpy as np

from relevance.backtest import metrics as _metrics
from relevance.data import eodhd as eodhd_data
from relevance.studies import paper as repl

# -----------------------------------------------------------------------
# Performance metrics (re-exported from relevance.backtest.metrics)
# -----------------------------------------------------------------------

annualized_sharpe = _metrics.annualized_sharpe
max_drawdown = _metrics.max_drawdown
compute_turnover = _metrics.compute_turnover


def summarize_performance(backtest_results):
    """Compute a full summary dict from walk-forward backtest results."""
    metrics = repl.compute_period_metrics(backtest_results)
    ls = repl.compute_cumulative_ls(backtest_results)

    summary = {}

    # IC stats
    summary["mean_ic"] = metrics["ic"].mean()
    summary["median_ic"] = metrics["ic"].median()
    summary["ic_hit_rate"] = (metrics["ic"] > 0).mean()
    summary["mean_hit_rate"] = metrics["hit_rate"].mean()

    # Long-short returns
    if not ls.empty:
        summary["total_ls_return"] = ls["cumulative_ls"].iloc[-1]
        summary["annualized_ls_sharpe"] = annualized_sharpe(ls["long_short"])
        summary["max_drawdown"] = max_drawdown(ls["cumulative_ls"])
        summary["mean_monthly_ls"] = ls["long_short"].mean()
        summary["total_long_return"] = ls["cumulative_long"].iloc[-1]
        summary["total_short_return"] = ls["cumulative_short"].iloc[-1]
    else:
        summary["total_ls_return"] = np.nan
        summary["annualized_ls_sharpe"] = np.nan
        summary["max_drawdown"] = np.nan

    # Quintile analysis (pooled)
    qa = repl.compute_quintile_analysis(
        backtest_results, pred_col="predicted", actual_col="actual"
    )
    if qa is not None:
        summary["quintile_spread"] = qa["mean"].iloc[-1] - qa["mean"].iloc[0]

    # Turnover
    summary["avg_turnover"] = compute_turnover(backtest_results)

    # Coverage
    summary["n_periods"] = len(metrics)
    summary["avg_stocks_per_period"] = metrics["n"].mean()

    return summary, metrics, ls


def print_report(summary, metrics, ls, qa=None):
    """Pretty-print the backtest report."""
    print("\n" + "=" * 60)
    print("  RELEVANCE ALPHA SIGNAL – BACKTEST REPORT")
    print("=" * 60)

    print(f"\n{'Periods':30s} {summary['n_periods']:>10.0f}")
    print(f"{'Avg stocks / period':30s} {summary['avg_stocks_per_period']:>10.1f}")

    print("\n--- Information Coefficient ---")
    print(f"{'Mean IC':30s} {summary['mean_ic']:>10.4f}")
    print(f"{'Median IC':30s} {summary['median_ic']:>10.4f}")
    print(f"{'IC > 0 (hit rate)':30s} {summary['ic_hit_rate']:>10.1%}")
    print(f"{'Directional hit rate':30s} {summary['mean_hit_rate']:>10.1%}")

    print("\n--- Long-Short Portfolio (Q5 - Q1) ---")
    print(f"{'Total L/S return':30s} {summary.get('total_ls_return', np.nan):>10.2%}")
    print(f"{'Mean monthly L/S':30s} {summary.get('mean_monthly_ls', np.nan):>10.4f}")
    print(f"{'Annualized Sharpe':30s} {summary.get('annualized_ls_sharpe', np.nan):>10.2f}")
    print(f"{'Max drawdown':30s} {summary.get('max_drawdown', np.nan):>10.2%}")
    print(f"{'Total long return':30s} {summary.get('total_long_return', np.nan):>10.2%}")
    print(f"{'Total short return':30s} {summary.get('total_short_return', np.nan):>10.2%}")

    print("\n--- Turnover ---")
    print(f"{'Avg monthly turnover':30s} {summary.get('avg_turnover', np.nan):>10.1%}")

    if qa is not None:
        print("\n--- Quintile Analysis (pooled) ---")
        print(qa.to_string())

    print("\n" + "=" * 60)


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def run_alpha_backtest(
    tickers=None,
    exchange="US",
    api_token=None,
    from_date="2015-01-01",
    to_date=None,
    thresh=0.75,
    most=True,
    forward_period=21,
    min_training_months=12,
    feature_cols=None,
    output_path=None,
    plot=False,
):
    """
    End-to-end alpha signal backtest.

    1. Fetch daily prices from EODHD
    2. Build monthly panel with momentum / volatility features
    3. Walk-forward predict next-month cross-sectional returns
    4. Evaluate signal quality

    Returns
    -------
    backtest_results : DataFrame  (date, ticker, predicted, actual, fit)
    summary          : dict       performance metrics
    """
    if tickers is None:
        tickers = repl.DEFAULT_STOCKS

    if feature_cols is None:
        feature_cols = repl.BACKTEST_FEATURE_COLS

    # --- Step 1: fetch prices ---
    print(f"Fetching prices for {len(tickers)} tickers …")
    prices = eodhd_data.fetch_multiple(
        tickers, exchange=exchange, api_token=api_token,
        from_date=from_date, to_date=to_date,
    )
    print(f"  Got {len(prices)} trading days, {len(prices.columns)} tickers")

    # --- Step 2: build panel ---
    print("Building monthly feature panel …")
    panel = repl.build_backtest_panel(prices, forward_period=forward_period)
    print(f"  Panel: {len(panel)} obs, "
          f"{panel['date'].nunique()} months, "
          f"{panel['ticker'].nunique()} tickers")

    # --- Step 3: walk-forward backtest ---
    print("Running walk-forward backtest …")
    backtest_results = repl.run_walk_forward(
        panel,
        feature_cols=feature_cols,
        thresh=thresh,
        most=most,
        min_training_months=min_training_months,
    )
    print(f"  {len(backtest_results)} predictions across "
          f"{backtest_results['date'].nunique()} months")

    # --- Step 4: evaluate ---
    summary, metrics, ls = summarize_performance(backtest_results)

    qa = repl.compute_quintile_analysis(
        backtest_results, pred_col="predicted", actual_col="actual"
    )
    print_report(summary, metrics, ls, qa)

    # --- Optional outputs ---
    if output_path:
        backtest_results.to_csv(output_path, index=False)
        metrics.to_csv(output_path.replace(".csv", "_metrics.csv"), index=False)
        ls.to_csv(output_path.replace(".csv", "_ls.csv"))
        print(f"\nResults saved to {output_path}")

    if plot:
        _plot_results(metrics, ls, qa)

    return backtest_results, summary


def _plot_results(metrics, ls, qa):
    """Matplotlib plots for quick visual inspection."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for plots: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # IC time series
    ax = axes[0, 0]
    ax.bar(range(len(metrics)), metrics["ic"], color="steelblue", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(metrics["ic"].mean(), color="red", linestyle="--",
               label=f'Mean IC = {metrics["ic"].mean():.3f}')
    ax.set_title("Information Coefficient by Period")
    ax.set_ylabel("Spearman IC")
    ax.legend()

    # Cumulative L/S
    ax = axes[0, 1]
    if not ls.empty:
        ax.plot(ls.index, ls["cumulative_ls"], label="Long-Short", linewidth=2)
        ax.plot(ls.index, ls["cumulative_long"], label="Long", alpha=0.6)
        ax.plot(ls.index, ls["cumulative_short"], label="Short", alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    # Quintile bar chart
    ax = axes[1, 0]
    if qa is not None:
        ax.bar(qa.index.astype(str), qa["mean"], color="steelblue")
        ax.set_title("Mean Actual Return by Predicted Quintile")
        ax.set_ylabel("Mean Return")
        ax.set_xlabel("Quintile (Q1=lowest predicted)")

    # Hit rate time series
    ax = axes[1, 1]
    ax.plot(range(len(metrics)), metrics["hit_rate"], color="green", alpha=0.7)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.5)
    ax.axhline(metrics["hit_rate"].mean(), color="red", linestyle="--",
               label=f'Mean = {metrics["hit_rate"].mean():.1%}')
    ax.set_title("Directional Hit Rate by Period")
    ax.set_ylabel("Hit Rate")
    ax.legend()

    plt.tight_layout()
    plt.savefig("alpha_backtest.png", dpi=150)
    print("Plot saved to alpha_backtest.png")
    plt.show()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Relevance Alpha Signal – Walk-Forward Backtest"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Ticker symbols (default: 30-stock universe)",
    )
    parser.add_argument("--exchange", default="US")
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--from-date", default="2015-01-01")
    parser.add_argument("--to-date", default=None)
    parser.add_argument(
        "--thresh", type=float, default=0.75,
        help="Relevance threshold (percentile, 0-1)",
    )
    parser.add_argument(
        "--forward-period", type=int, default=21,
        help="Forward return horizon in trading days",
    )
    parser.add_argument(
        "--min-training", type=int, default=12,
        help="Minimum training months before first prediction",
    )
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument(
        "--plot", action="store_true", help="Show matplotlib plots"
    )
    args = parser.parse_args()

    run_alpha_backtest(
        tickers=args.tickers,
        exchange=args.exchange,
        api_token=args.api_token,
        from_date=args.from_date,
        to_date=args.to_date,
        thresh=args.thresh,
        forward_period=args.forward_period,
        min_training_months=args.min_training,
        output_path=args.output,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
