"""
On-disk price cache.

The backtest engine and the grid runner previously carried separate copies of
this logic pointing at different directories.  They used the same cache key,
so a single implementation serves both; all known cache directories are
searched on read and the primary one is written to.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from relevance import config
from relevance.data import eodhd


def price_cache_name(tickers, from_date, to_date, exchange) -> str:
    key = hashlib.md5(
        json.dumps([sorted(tickers), from_date, to_date, exchange]).encode()
    ).hexdigest()[:12]
    return f"prices_{key}.parquet"


def load_prices(tickers, from_date, to_date=None, exchange="US",
                refresh: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Adjusted closes for ``tickers``, cached on disk as parquet."""
    name = price_cache_name(tickers, from_date, to_date, exchange)

    if not refresh:
        for directory in config.CACHE_DIRS:
            path = directory / name
            if path.exists():
                prices = pd.read_parquet(path)
                if verbose:
                    print(f"  [cache] {path.name}: {prices.shape[0]} days x "
                          f"{prices.shape[1]} tickers")
                return prices

    if verbose:
        print(f"  [fetch] {len(tickers)} tickers from EODHD …")

    prices = eodhd.fetch_multiple(
        tickers, exchange=exchange, from_date=from_date, to_date=to_date
    )
    config.ensure(config.CACHE_DIRS[0])
    prices.to_parquet(config.CACHE_DIRS[0] / name)

    if verbose:
        print(f"  [fetch] got {prices.shape[0]} days x "
              f"{prices.shape[1]} tickers")
    return prices
