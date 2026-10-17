"""
Ticker universes and attribute sets.

Single source of truth: these lists were previously duplicated across the
backtest engine, the grid runner and the study replication module.
"""

from __future__ import annotations

# The 30 mega-cap names used by the published cross-sectional study.
MEGA30 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "JNJ", "V", "UNH", "PG", "HD", "BAC", "XOM",
    "PFE", "CSCO", "KO", "PEP", "MRK", "WMT", "CRM",
    "NFLX", "AMD", "QCOM", "COST", "LLY", "AVGO", "ORCL", "CVX",
]

# S&P 100 constituents (dot-class tickers such as BRK.B omitted).
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

DOW30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
    "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT", "DOW", "INTC",
]

SECTOR_ETFS = [
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE",
    "XLC",
]

UNIVERSES: dict[str, list[str]] = {
    "mega30": MEGA30,
    "dow30": DOW30,
    "sp100": SP100,
    "sector_etfs": SECTOR_ETFS,
}


# -----------------------------------------------------------------------
# Attribute sets
# -----------------------------------------------------------------------

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
