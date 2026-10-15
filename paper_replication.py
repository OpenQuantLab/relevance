"""
Paper Replication: Relevance-Based Prediction
==============================================

Replicates the empirical methodology from:
"Relevance-Based Prediction: A Transparent and Adaptive Alternative
 to Machine Learning"
by Megan Czasonis, Mark Kritzman, and David Turkington
(Journal of Financial Data Science, 2023)

Two studies implemented:
  Study 1 – Cross-Sectional: Uses current fundamental + price features
            for a universe of stocks. Leave-one-out cross-validation
            demonstrates the model's ability to identify return-predictive
            patterns across stocks.

  Study 2 – Walk-Forward Backtest: Uses only price-derived features
            (momentum and volatility at multiple horizons) in a monthly
            panel. Walk-forward out-of-sample prediction produces
            quintile portfolios, IC time series, and cumulative
            long-short returns.
"""

import sys
import time

import numpy as np
import pandas as pd
import predictionrevisited as pr
import eodhd_data


# -----------------------------------------------------------------------
# Default universes
# -----------------------------------------------------------------------

DEFAULT_STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "JNJ", "V", "UNH", "PG", "HD", "BAC", "XOM",
    "PFE", "CSCO", "KO", "PEP", "MRK", "WMT", "CRM",
    "NFLX", "AMD", "QCOM", "COST", "LLY", "AVGO", "ORCL", "CVX",
]

FUNDAMENTAL_ATTRS = [
    "log_market_cap", "earnings_yield", "pb_ratio", "ev_ebitda",
    "dividend_yield", "roe", "profit_margin", "beta",
]

PRICE_ATTRS = [
    "momentum_12m", "momentum_6m", "momentum_1m", "volatility_3m",
]

BACKTEST_FEATURE_COLS = [
    "return_21d", "return_63d", "return_126d", "return_252d",
    "vol_63d", "vol_252d",
]


# -----------------------------------------------------------------------
# Study 1 – Cross-Sectional (Fundamental + Price)
# -----------------------------------------------------------------------

def build_cross_sectional_dataset(symbols, exchange="US", api_token=None,
                                  from_date=None):
    """
    Build a cross-sectional dataset from EODHD fundamentals and prices.

    Each stock becomes one observation.
      Attributes: fundamental ratios + price momentum/volatility
      Outcome: trailing 3-month return (known)

    Returns
    -------
    attributes_df : DataFrame  (N x K)
    outcomes_df   : DataFrame  (N x 1)
    stock_info    : dict
    """
    fundamentals = eodhd_data.fetch_fundamentals_batch(
        symbols, exchange, api_token
    )
    prices = eodhd_data.fetch_multiple(
        symbols, exchange, api_token, from_date=from_date
    )
    daily_returns = prices.pct_change()

    records = {}
    for sym in symbols:
        if sym not in fundamentals or sym not in prices.columns:
            continue

        fund = fundamentals[sym]
        px = prices[sym].dropna()
        ret = daily_returns[sym].dropna()

        if len(px) < 63:
            continue

        mc = fund.get("market_cap")
        pe = fund.get("pe_ratio")

        records[sym] = {
            # Fundamental attributes
            "log_market_cap": (
                np.log(mc) if mc and mc > 0 else np.nan
            ),
            "earnings_yield": (
                1.0 / pe if pe and pe > 0 else np.nan
            ),
            "pb_ratio": fund.get("pb_ratio"),
            "ev_ebitda": fund.get("ev_ebitda"),
            "dividend_yield": fund.get("dividend_yield"),
            "roe": fund.get("roe"),
            "profit_margin": fund.get("profit_margin"),
            "beta": fund.get("beta"),
            # Price-based attributes
            "momentum_12m": (
                px.pct_change(min(252, len(px) - 1)).iloc[-1]
                if len(px) > 252 else np.nan
            ),
            "momentum_6m": (
                px.pct_change(min(126, len(px) - 1)).iloc[-1]
                if len(px) > 126 else np.nan
            ),
            "momentum_1m": px.pct_change(min(21, len(px) - 1)).iloc[-1],
            "volatility_3m": (
                ret.iloc[-63:].std() * np.sqrt(252)
                if len(ret) >= 63 else np.nan
            ),
            # Outcome
            "trailing_3m_return": px.pct_change(min(63, len(px) - 1)).iloc[-1],
        }

    df = pd.DataFrame(records).T

    all_attrs = FUNDAMENTAL_ATTRS + PRICE_ATTRS
    available = [
        c for c in all_attrs
        if c in df.columns and df[c].notna().sum() > len(df) * 0.3
    ]

    df = df.dropna(subset=available + ["trailing_3m_return"])

    if len(df) < 5:
        raise ValueError(
            f"Only {len(df)} stocks have complete data. Need at least 5."
        )

    attributes_df = df[available].astype(float)
    outcomes_df = df[["trailing_3m_return"]].astype(float)

    stock_info = {
        "symbols": list(df.index),
        "n_stocks": len(df),
        "n_attributes": len(available),
        "attribute_names": available,
    }
    return attributes_df, outcomes_df, stock_info


def run_loo_prediction(attributes_df, outcomes_df,
                       thresh=0.75, most=True):
    """
    Leave-one-out cross-sectional prediction.

    For each stock *i*, predict its return using all other stocks
    as observations.  This replicates the paper's cross-sectional study.

    Returns
    -------
    DataFrame with columns: predicted_return, actual_return, fit,
                            prediction_error
    """
    N = len(attributes_df)
    predicted = np.full(N, np.nan)
    fit_vals = np.full(N, np.nan)

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False

        try:
            pred, _ = pr.predict_df(
                attributes_df.iloc[mask],
                attributes_df.iloc[[i]],
                outcomes_df.iloc[mask],
                thresh=thresh, most=most, pct_thresh=True,
                pairwise_fits=False,
            )
            predicted[i] = pred["yhat"].iloc[0]
            fit_vals[i] = pred["fit"].iloc[0]
        except Exception:
            pass

    results = pd.DataFrame({
        "symbol": list(attributes_df.index),
        "predicted_return": predicted,
        "actual_return": outcomes_df.iloc[:, 0].values,
        "fit": fit_vals,
    }).set_index("symbol")

    results["prediction_error"] = (
        results["predicted_return"] - results["actual_return"]
    )
    return results


def run_full_prediction(attributes_df, outcomes_df,
                        thresh=0.75, most=True):
    """
    Full-sample prediction: predict every stock using *all* stocks
    (including itself) as observations.  Returns the relevance matrix
    alongside predictions for heatmap visualisation.
    """
    pred, obs_rel = pr.predict_df(
        attributes_df, attributes_df, outcomes_df,
        thresh=thresh, most=most, pct_thresh=True,
        pairwise_fits=True,
    )

    pred["symbol"] = list(attributes_df.index)
    pred["actual_return"] = outcomes_df.iloc[:, 0].values
    pred["prediction_error"] = pred["yhat"] - pred["actual_return"]
    pred = pred.set_index("symbol")

    return pred, obs_rel


# -----------------------------------------------------------------------
# Study 2 – Walk-Forward Backtest (Price-Based Panel)
# -----------------------------------------------------------------------

def build_backtest_panel(prices, forward_period=21):
    """
    Build a monthly panel from daily prices.

    At each month-end compute for every ticker:
      Features  – trailing returns (21d, 63d, 126d, 252d)
                  and realised volatility (63d, 252d)
      Outcome   – forward return over *forward_period* trading days

    Returns
    -------
    DataFrame  with columns:
        date, ticker, return_21d … vol_252d, forward_return
    """
    daily_ret = prices.pct_change()

    # Last actual trading day of each month (avoids KeyError from calendar
    # month-end dates that may not exist in the trading calendar).
    month_labels = prices.index.to_period("M")
    month_end_idx = prices.groupby(month_labels).apply(
        lambda g: g.index[-1]
    ).values

    records = []
    for date in month_end_idx:
        future_dates = prices.index[prices.index > date]
        if len(future_dates) < forward_period:
            continue
        forward_date = future_dates[forward_period - 1]

        for ticker in prices.columns:
            px = prices[ticker].loc[:date].dropna()
            ret = daily_ret[ticker].loc[:date].dropna()

            if len(px) < 252:
                continue

            row = {
                "date": date,
                "ticker": ticker,
                "return_21d": px.iloc[-1] / px.iloc[-21] - 1,
                "return_63d": px.iloc[-1] / px.iloc[-63] - 1,
                "return_126d": px.iloc[-1] / px.iloc[-126] - 1,
                "return_252d": px.iloc[-1] / px.iloc[-252] - 1,
                "vol_63d": ret.iloc[-63:].std() * np.sqrt(252),
                "vol_252d": ret.iloc[-252:].std() * np.sqrt(252),
                "forward_return": (
                    prices[ticker].loc[forward_date]
                    / prices[ticker].loc[date] - 1
                ),
            }
            records.append(row)

    panel = pd.DataFrame(records)
    if panel.empty:
        raise ValueError(
            "Could not build panel. Need >1 year of price history."
        )
    return panel


def build_latest_period_features(prices, feature_cols=None):
    """
    Compute Study 2 features for the most recent trading day only, with no
    forward outcome (it hasn't happened yet). Used to generate a live signal.
    """
    if feature_cols is None:
        feature_cols = BACKTEST_FEATURE_COLS

    daily_ret = prices.pct_change()
    date = prices.index[-1]

    records = []
    for ticker in prices.columns:
        px = prices[ticker].loc[:date].dropna()
        ret = daily_ret[ticker].loc[:date].dropna()
        if len(px) < 252:
            continue
        records.append({
            "date": date,
            "ticker": ticker,
            "return_21d": px.iloc[-1] / px.iloc[-21] - 1,
            "return_63d": px.iloc[-1] / px.iloc[-63] - 1,
            "return_126d": px.iloc[-1] / px.iloc[-126] - 1,
            "return_252d": px.iloc[-1] / px.iloc[-252] - 1,
            "vol_63d": ret.iloc[-63:].std() * np.sqrt(252),
            "vol_252d": ret.iloc[-252:].std() * np.sqrt(252),
        })
    return pd.DataFrame(records, columns=["date", "ticker"] + feature_cols)


def predict_live_signal(panel, latest_features, feature_cols=None,
                        thresh=0.75, most=True, max_training_obs=1500):
    """
    Predict the alpha signal for the current (outcome-unknown) cross-section,
    training on the full realised history in *panel*.

    Returns
    -------
    DataFrame with columns: date, ticker, signal, fit
    """
    if feature_cols is None:
        feature_cols = BACKTEST_FEATURE_COLS

    train = panel.dropna(subset=feature_cols + ["forward_return"])
    test = latest_features.dropna(subset=feature_cols)

    if len(train) < 20 or len(test) < 3:
        raise ValueError("Not enough data to generate a live signal.")

    if len(train) > max_training_obs:
        train = train.tail(max_training_obs)

    pred_result, _ = pr.predict_df(
        train[feature_cols],
        test[feature_cols],
        train[["forward_return"]],
        thresh=thresh, most=most, pct_thresh=True,
        pairwise_fits=False,
    )
    return pd.DataFrame({
        "date": test["date"].values,
        "ticker": test["ticker"].values,
        "signal": pred_result["yhat"].values,
        "fit": pred_result["fit"].values,
    })


def run_walk_forward(panel, feature_cols=None, thresh=0.75, most=True,
                     min_training_months=12, max_training_obs=1500,
                     progress=False):
    """
    Walk-forward backtest.

    At each month-end, train on all prior observations and predict
    the current cross-section of stocks.

    Returns
    -------
    DataFrame  with columns: date, ticker, predicted, actual, fit
    """
    if feature_cols is None:
        feature_cols = BACKTEST_FEATURE_COLS

    dates = sorted(panel["date"].unique())

    if len(dates) < min_training_months + 1:
        raise ValueError(
            f"Need ≥ {min_training_months + 1} months; "
            f"have {len(dates)}."
        )

    n_steps = len(dates) - min_training_months
    t_start = time.time()

    all_preds = []
    for t_idx in range(min_training_months, len(dates)):
        current_date = dates[t_idx]

        if progress:
            done = t_idx - min_training_months
            elapsed = time.time() - t_start
            eta = (elapsed / done * (n_steps - done)) if done else float("nan")
            print(f"\r    month {done + 1}/{n_steps} "
                  f"({pd.Timestamp(current_date):%Y-%m}) "
                  f"elapsed {elapsed:6.0f}s eta {eta:6.0f}s",
                  end="", file=sys.stderr, flush=True)
        train_dates = dates[max(0, t_idx - 36):t_idx]

        train = (
            panel[panel["date"].isin(train_dates)]
            .dropna(subset=feature_cols + ["forward_return"])
        )
        test = (
            panel[panel["date"] == current_date]
            .dropna(subset=feature_cols + ["forward_return"])
        )

        if len(train) < 20 or len(test) < 3:
            continue

        if len(train) > max_training_obs:
            train = train.tail(max_training_obs)

        try:
            pred_result, _ = pr.predict_df(
                train[feature_cols],
                test[feature_cols],
                train[["forward_return"]],
                thresh=thresh, most=most, pct_thresh=True,
                pairwise_fits=False,
            )
            preds = pd.DataFrame({
                "date": current_date,
                "ticker": test["ticker"].values,
                "predicted": pred_result["yhat"].values,
                "actual": test["forward_return"].values,
                "fit": pred_result["fit"].values,
            })
            all_preds.append(preds)
        except Exception:
            continue

    if progress:
        print(file=sys.stderr, flush=True)

    if not all_preds:
        raise ValueError("No successful predictions during backtest.")

    return pd.concat(all_preds, ignore_index=True)


# -----------------------------------------------------------------------
# Performance analytics
# -----------------------------------------------------------------------

def compute_quintile_analysis(results, pred_col="predicted_return",
                              actual_col="actual_return"):
    """
    Sort stocks by predicted return into quintiles and compute
    average actual return per quintile.  Works for both LOO results
    and individual backtest periods.
    """
    valid = results.dropna(subset=[pred_col, actual_col]).copy()
    if len(valid) < 5:
        return None

    n_bins = min(5, len(valid))
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    valid["quintile"] = pd.qcut(
        valid[pred_col], n_bins, labels=labels, duplicates="drop"
    )

    summary = (
        valid.groupby("quintile", observed=True)[actual_col]
        .agg(["mean", "count", "std"])
    )
    return summary


def compute_period_metrics(backtest_results):
    """
    For each backtest date compute:
      IC            – Spearman rank correlation (predicted vs actual)
      hit_rate      – fraction with correct directional call
      quintile_spread – average return Q5 minus Q1
    """
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


def compute_cumulative_ls(backtest_results, n_bins=5):
    """
    Compute cumulative long-short returns from the backtest.

    At each date, go long Q5 (highest predicted) and short Q1 (lowest).
    """
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
