"""
EODHD Data Fetcher
==================
Fetches end-of-day historical market data from the EODHD API and transforms it
into the attributes/outcomes format required by predictionrevisited.
"""

import os
import numpy as np
import pandas as pd
import requests


EODHD_BASE_URL = "https://eodhd.com/api/eod"
DEFAULT_API_TOKEN = "69bb99ebc59573.31118566"


def fetch_eod(symbol, exchange="US", api_token=None,
              from_date=None, to_date=None, period="d"):
    """
    Fetch end-of-day historical data for a single ticker from EODHD.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    exchange : str
        Exchange code (e.g. "US", "LSE"). Default "US".
    api_token : str or None
        EODHD API key. Falls back to EODHD_API_TOKEN env var.
    from_date : str or None
        Start date in YYYY-MM-DD format.
    to_date : str or None
        End date in YYYY-MM-DD format.
    period : str
        "d" for daily, "w" for weekly, "m" for monthly.

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, adjusted_close, volume.
    """
    token = api_token or os.environ.get("EODHD_API_TOKEN", "") or DEFAULT_API_TOKEN

    ticker = f"{symbol}.{exchange}"
    params = {"api_token": token, "fmt": "json", "period": period}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    resp = requests.get(f"{EODHD_BASE_URL}/{ticker}", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_multiple(symbols, exchange="US", api_token=None,
                   from_date=None, to_date=None, period="d"):
    """
    Fetch EOD data for multiple tickers and return aligned adjusted closes.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    exchange : str
        Exchange code.
    api_token : str or None
        EODHD API key.
    from_date, to_date : str or None
        Date range in YYYY-MM-DD.
    period : str
        "d", "w", or "m".

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, one column per ticker with adjusted close prices.
    """
    frames = {}
    for sym in symbols:
        try:
            df = fetch_eod(sym, exchange, api_token, from_date, to_date, period)
            frames[sym] = df.set_index("date")["adjusted_close"]
        except Exception:
            continue
    if not frames:
        raise ValueError("No data retrieved for any of the requested tickers.")
    prices = pd.DataFrame(frames).dropna()
    return prices


def compute_features(prices, windows=None):
    """
    Compute feature attributes from a price DataFrame for use with the
    prediction model.

    Features per ticker per date:
    - return_{w}d : rolling return over w days
    - vol_{w}d    : rolling volatility (std of daily returns) over w days

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are tickers, index is datetime, values are adjusted close.
    windows : list of int or None
        Lookback windows in trading days. Default [5, 21, 63].

    Returns
    -------
    pd.DataFrame
        Multi-level columns: (ticker, feature). Rows are dates.
    """
    if windows is None:
        windows = [5, 21, 63]

    daily_returns = prices.pct_change()
    feature_frames = {}

    for sym in prices.columns:
        for w in windows:
            feature_frames[(sym, f"return_{w}d")] = (
                prices[sym].pct_change(w)
            )
            feature_frames[(sym, f"vol_{w}d")] = (
                daily_returns[sym].rolling(w).std()
            )

    features = pd.DataFrame(feature_frames).dropna()
    features.columns = pd.MultiIndex.from_tuples(
        list(features.columns), names=["ticker", "feature"]
    )
    return features


def build_prediction_dataset(prices, target_ticker, target_horizon=21,
                             feature_tickers=None, windows=None):
    """
    Build attributes, outcomes, and circumstances DataFrames suitable for
    predictionrevisited.predict_df().

    Uses a rolling window approach: each past date is an "observation" whose
    attributes are the feature values at that date, and whose outcome is the
    forward return of the target ticker.

    The most recent date becomes the "prediction circumstance" (no outcome yet).

    Parameters
    ----------
    prices : pd.DataFrame
        Aligned adjusted-close prices (from fetch_multiple).
    target_ticker : str
        The ticker whose forward return we want to predict.
    target_horizon : int
        Forward return horizon in trading days (default 21 ~ 1 month).
    feature_tickers : list of str or None
        Tickers to use as feature sources. Default: all columns in prices.
    windows : list of int or None
        Feature lookback windows. Default [5, 21, 63].

    Returns
    -------
    attributes_df : pd.DataFrame
        Feature attributes for past observations.
    outcomes_df : pd.DataFrame
        Forward returns for past observations.
    circumstances_df : pd.DataFrame
        Feature attributes for the most recent date (prediction target).
    info : dict
        Contains 'dates', 'target_ticker', 'target_horizon'.
    """
    if feature_tickers is None:
        feature_tickers = list(prices.columns)

    features = compute_features(prices, windows)

    # Forward return of the target ticker
    fwd_return = prices[target_ticker].pct_change(target_horizon).shift(-target_horizon)
    fwd_return.name = "forward_return"

    # Flatten multi-level columns to simple feature names
    flat_features = features.copy()
    flat_features.columns = [
        f"{ticker}_{feat}" for ticker, feat in features.columns
    ]

    # Align features and forward returns
    combined = flat_features.join(fwd_return, how="inner").dropna(subset=flat_features.columns)

    # Observations: rows where we have the forward return (not NaN)
    obs_mask = combined["forward_return"].notna()
    obs_data = combined[obs_mask]

    # Circumstances: rows where forward return is NaN (most recent dates)
    circ_data = combined[~obs_mask]

    feature_cols = list(flat_features.columns)

    if obs_data.empty:
        raise ValueError("Not enough data to build observations. Try a longer date range.")
    if circ_data.empty:
        raise ValueError("No prediction circumstances available. Data may be stale.")

    attributes_df = obs_data[feature_cols].copy()
    outcomes_df = obs_data[["forward_return"]].copy()
    circumstances_df = circ_data[feature_cols].tail(1).copy()

    info = {
        "dates": {
            "obs_start": str(attributes_df.index[0].date()),
            "obs_end": str(attributes_df.index[-1].date()),
            "prediction_date": str(circumstances_df.index[0].date()),
        },
        "target_ticker": target_ticker,
        "target_horizon": target_horizon,
        "n_observations": len(attributes_df),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
    }

    return attributes_df, outcomes_df, circumstances_df, info


# ---------------------------------------------------------------------------
# Fundamentals API
# ---------------------------------------------------------------------------

EODHD_FUNDAMENTALS_URL = "https://eodhd.com/api/v1.1/fundamentals"


def _resolve_token(api_token):
    return api_token or os.environ.get("EODHD_API_TOKEN", "") or DEFAULT_API_TOKEN


def fetch_fundamentals(symbol, exchange="US", api_token=None, filters=None):
    """
    Fetch fundamental data for a single ticker.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    exchange : str
        Exchange code.
    api_token : str or None
        EODHD API key.
    filters : str or None
        Comma-separated filter sections (e.g. "Highlights,Valuation,Technicals").

    Returns
    -------
    dict
        JSON response from EODHD fundamentals API.
    """
    token = _resolve_token(api_token)
    ticker = f"{symbol}.{exchange}"
    params = {"api_token": token, "fmt": "json"}
    if filters:
        params["filter"] = filters

    resp = requests.get(
        f"{EODHD_FUNDAMENTALS_URL}/{ticker}", params=params, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def fetch_fundamentals_batch(symbols, exchange="US", api_token=None):
    """
    Fetch key fundamental metrics for multiple tickers.

    Extracts Highlights, Valuation, and Technicals sections.

    Returns
    -------
    dict
        {symbol: {metric_name: value, ...}, ...}
    """
    filters = "Highlights,Valuation,Technicals"
    results = {}

    for sym in symbols:
        try:
            data = fetch_fundamentals(sym, exchange, api_token, filters)
            h = data.get("Highlights", {})
            v = data.get("Valuation", {})
            t = data.get("Technicals", {})

            results[sym] = {
                "market_cap": h.get("MarketCapitalization"),
                "pe_ratio": h.get("PERatio"),
                "peg_ratio": h.get("PEGRatio"),
                "eps": h.get("EarningsShare"),
                "dividend_yield": h.get("DividendYield"),
                "book_value": h.get("BookValue"),
                "profit_margin": h.get("ProfitMargin"),
                "roe": h.get("ReturnOnEquityTTM"),
                "revenue_ttm": h.get("RevenueTTM"),
                "pb_ratio": v.get("PriceBookMRQ"),
                "ps_ratio": v.get("PriceSalesTTM"),
                "ev_ebitda": v.get("EnterpriseValueEbitda"),
                "ev_revenue": v.get("EnterpriseValueRevenue"),
                "forward_pe": v.get("ForwardPE"),
                "beta": t.get("Beta"),
                "week_52_high": t.get("52WeekHigh"),
                "week_52_low": t.get("52WeekLow"),
                "ma_50": t.get("50DayMA"),
                "ma_200": t.get("200DayMA"),
            }
        except Exception:
            continue

    return results


def fetch_index_constituents(index_code="GSPC", api_token=None):
    """
    Fetch constituents of a major index (e.g. GSPC for S&P 500).

    Returns
    -------
    list of dict
        Each dict has keys: symbol, name, exchange, sector, industry, weight.
    """
    token = _resolve_token(api_token)
    resp = requests.get(
        f"{EODHD_FUNDAMENTALS_URL}/{index_code}.INDX",
        params={"api_token": token, "fmt": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    components = data.get("Components", data)
    tickers = []
    for _key, comp in components.items():
        if not isinstance(comp, dict):
            continue
        tickers.append({
            "symbol": comp.get("Code"),
            "name": comp.get("Name"),
            "exchange": comp.get("Exchange", "US"),
            "sector": comp.get("Sector"),
            "industry": comp.get("Industry"),
            "weight": comp.get("Weight"),
        })

    return tickers
