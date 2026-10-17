"""
Signal History – Live Alpha Signal Generation & Persistence
=============================================================
Generates the *current* (outcome-unknown) cross-sectional alpha signal from
the relevance-based prediction model and appends it to a persisted time
series on disk, so a real signal history accumulates run over run.

Two signal families, matching the two studies in paper_replication.py:
  price_v1        – Study 2 (momentum/volatility) forward-return prediction
  fundamental_v1  – Study 1 (fundamentals + price) cross-sectional score

Usage
-----
    python signal_history.py --model price
    python signal_history.py --model fundamental
    python signal_history.py --model both --tickers AAPL MSFT ...
"""

import argparse
import os

import pandas as pd

from relevance import config
from relevance.data import eodhd as eodhd_data
from relevance.studies import paper as repl

SIGNAL_DIR = str(config.SIGNALS_DIR)
PRICE_SIGNAL_PATH = os.path.join(SIGNAL_DIR, "price_signal_history.parquet")
FUNDAMENTAL_SIGNAL_PATH = os.path.join(
    SIGNAL_DIR, "fundamental_signal_history.parquet"
)


def _append_and_save(new_rows, path, key_cols):
    """Append new_rows to the parquet history at path, keeping the latest
    generated_at value for any duplicate key_cols."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = (
        combined.sort_values("generated_at")
        .drop_duplicates(subset=key_cols, keep="last")
        .sort_values(key_cols[0])
        .reset_index(drop=True)
    )
    combined.to_parquet(path, index=False)
    return combined


def generate_price_signal(
    tickers=None, exchange="US", api_token=None, from_date="2015-01-01",
    to_date=None, thresh=0.75, most=True, forward_period=21,
    feature_cols=None, store_path=PRICE_SIGNAL_PATH,
):
    """Generate today's live price-based alpha signal and persist it."""
    if tickers is None:
        tickers = repl.DEFAULT_STOCKS
    if feature_cols is None:
        feature_cols = repl.BACKTEST_FEATURE_COLS

    prices = eodhd_data.fetch_multiple(
        tickers, exchange=exchange, api_token=api_token,
        from_date=from_date, to_date=to_date,
    )
    panel = repl.build_backtest_panel(prices, forward_period=forward_period)
    latest = repl.build_latest_period_features(prices, feature_cols=feature_cols)

    signal = repl.predict_live_signal(
        panel, latest, feature_cols=feature_cols, thresh=thresh, most=most,
    )
    signal["model"] = "price_v1"
    signal["generated_at"] = pd.Timestamp.now()

    return _append_and_save(signal, store_path, key_cols=["date", "ticker", "model"])


def generate_fundamental_signal(
    tickers=None, exchange="US", api_token=None, from_date=None,
    thresh=0.75, most=True, store_path=FUNDAMENTAL_SIGNAL_PATH,
):
    """Generate today's live fundamental+price cross-sectional alpha signal
    and persist it. Uses the current cross-section as both train and test
    set (full-sample relevance), so signal quality depends on universe size."""
    if tickers is None:
        tickers = repl.DEFAULT_STOCKS

    attributes_df, outcomes_df, _ = repl.build_cross_sectional_dataset(
        tickers, exchange=exchange, api_token=api_token, from_date=from_date,
    )
    pred, _ = repl.run_full_prediction(
        attributes_df, outcomes_df, thresh=thresh, most=most
    )

    signal = pd.DataFrame({
        "date": pd.Timestamp.now().normalize(),
        "ticker": pred.index,
        "signal": pred["yhat"].values,
        "fit": pred["fit"].values,
    })
    signal["model"] = "fundamental_v1"
    signal["generated_at"] = pd.Timestamp.now()

    return _append_and_save(signal, store_path, key_cols=["date", "ticker", "model"])


def load_signal_history(store_path):
    """Load a persisted signal time series from disk (empty frame if none yet)."""
    if not os.path.exists(store_path):
        return pd.DataFrame(
            columns=["date", "ticker", "signal", "fit", "model", "generated_at"]
        )
    return pd.read_parquet(store_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and persist the live relevance alpha signal."
    )
    parser.add_argument(
        "--model", choices=["price", "fundamental", "both"], default="price"
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--exchange", default="US")
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--from-date", default="2015-01-01")
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--thresh", type=float, default=0.75)
    args = parser.parse_args()

    if args.model in ("price", "both"):
        history = generate_price_signal(
            tickers=args.tickers, exchange=args.exchange,
            api_token=args.api_token, from_date=args.from_date,
            to_date=args.to_date, thresh=args.thresh,
        )
        latest_date = history["date"].max()
        latest = history[history["date"] == latest_date].sort_values(
            "signal", ascending=False
        )
        print(f"Price signal updated. {len(history)} rows total, latest date: {latest_date}")
        print(latest.to_string(index=False))

    if args.model in ("fundamental", "both"):
        history = generate_fundamental_signal(
            tickers=args.tickers, exchange=args.exchange,
            api_token=args.api_token, from_date=args.from_date,
            thresh=args.thresh,
        )
        latest_date = history["date"].max()
        latest = history[history["date"] == latest_date].sort_values(
            "signal", ascending=False
        )
        print(f"\nFundamental signal updated. {len(history)} rows total, "
              f"latest date: {latest_date}")
        print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
