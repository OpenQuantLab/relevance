"""
Alpha Lab – signal design, backtest and evaluation engine
=========================================================

Backend for the interactive Dash app (``alpha_lab_app.py``).  Lets you define
a cross-sectional alpha signal from a library of price-based features, run a
walk-forward backtest with either the relevance model or a linear composite,
and evaluate it with standard quant diagnostics.

A signal is fully described by a :class:`SignalSpec`, which is JSON
serialisable, so every run can be saved, reloaded and compared.

    spec = SignalSpec(name="momentum", tickers=UNIVERSES["mega30"],
                      features=["return_21d", "return_252d", "vol_63d"])
    prices = load_prices(spec)
    panel = build_panel(prices, spec)
    results = run_backtest(panel, spec)
    report = evaluate(results, spec)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import alpha_signal as alpha
import eodhd_data
import paper_replication as repl
import predictionrevisited as pr

ROOT = Path(__file__).resolve().parent
LAB_DIR = ROOT / "lab"
RUNS_DIR = LAB_DIR / "runs"
CACHE_DIRS = [LAB_DIR / "cache", ROOT / "tests" / "cache"]

PERIODS_PER_YEAR = {"W": 52, "M": 12, "Q": 4}


# -----------------------------------------------------------------------
# Universes
# -----------------------------------------------------------------------

MEGA30 = list(repl.DEFAULT_STOCKS)

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

UNIVERSES: Dict[str, List[str]] = {
    "mega30": MEGA30,
    "dow30": DOW30,
    "sp100": SP100,
    "sector_etfs": SECTOR_ETFS,
}


# -----------------------------------------------------------------------
# Feature library
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureDef:
    id: str
    label: str
    group: str
    fn: Callable[[pd.DataFrame, pd.DataFrame, "_Ctx"], pd.DataFrame]
    sign: int = 1          # prior on the sign of the cross-sectional payoff
    desc: str = ""


class _Ctx:
    """Lazily computed quantities shared by several features."""

    def __init__(self, prices: pd.DataFrame, rets: pd.DataFrame):
        self.prices = prices
        self.rets = rets
        self._mkt: Optional[pd.Series] = None
        self._beta: Optional[pd.DataFrame] = None

    @property
    def mkt(self) -> pd.Series:
        if self._mkt is None:
            self._mkt = self.rets.mean(axis=1)
        return self._mkt

    @property
    def beta(self) -> pd.DataFrame:
        if self._beta is None:
            cov = self.rets.rolling(252, min_periods=120).cov(self.mkt)
            var = self.mkt.rolling(252, min_periods=120).var()
            self._beta = cov.div(var, axis=0)
        return self._beta


FEATURES: Dict[str, FeatureDef] = {}


def _register(fid, label, group, fn, sign=1, desc=""):
    FEATURES[fid] = FeatureDef(fid, label, group, fn, sign, desc)


def _ann_vol(rets: pd.DataFrame, w: int) -> pd.DataFrame:
    return rets.rolling(w).std() * np.sqrt(252)


for _w in (5, 21, 63, 126, 252):
    _register(
        f"return_{_w}d", f"Return {_w}d", "Momentum",
        (lambda w: lambda px, rt, c: px.pct_change(w))(_w),
        sign=1, desc=f"Trailing {_w}-day total return",
    )

_register(
    "mom_12_1", "Momentum 12-1m", "Momentum",
    lambda px, rt, c: px.shift(21) / px.shift(252) - 1,
    sign=1, desc="12-month return skipping the most recent month",
)
_register(
    "mom_6_1", "Momentum 6-1m", "Momentum",
    lambda px, rt, c: px.shift(21) / px.shift(126) - 1,
    sign=1, desc="6-month return skipping the most recent month",
)
_register(
    "rev_21d", "Short-term reversal 21d", "Momentum",
    lambda px, rt, c: -(px.pct_change(21)),
    sign=1, desc="Negated 1-month return",
)

for _w in (21, 63, 126, 252):
    _register(
        f"vol_{_w}d", f"Volatility {_w}d", "Risk",
        (lambda w: lambda px, rt, c: _ann_vol(rt, w))(_w),
        sign=-1, desc=f"Annualised realised volatility over {_w} days",
    )

_register(
    "downside_vol_63d", "Downside vol 63d", "Risk",
    lambda px, rt, c: rt.clip(upper=0).rolling(63).std() * np.sqrt(252),
    sign=-1, desc="Annualised std of negative daily returns",
)
_register(
    "beta_252d", "Beta 252d", "Risk",
    lambda px, rt, c: c.beta,
    sign=-1, desc="Rolling beta vs the equal-weight universe",
)
_register(
    "idio_vol_63d", "Idiosyncratic vol 63d", "Risk",
    lambda px, rt, c: (
        rt.sub(c.beta.mul(c.mkt, axis=0)).rolling(63).std() * np.sqrt(252)
    ),
    sign=-1, desc="Residual volatility after removing market beta",
)
_register(
    "skew_63d", "Skew 63d", "Risk",
    lambda px, rt, c: rt.rolling(63).skew(),
    sign=-1, desc="Skewness of daily returns",
)
_register(
    "kurt_63d", "Kurtosis 63d", "Risk",
    lambda px, rt, c: rt.rolling(63).kurt(),
    sign=-1, desc="Excess kurtosis of daily returns",
)

_register(
    "px_ma_50", "Price / MA50", "Trend",
    lambda px, rt, c: px / px.rolling(50).mean() - 1,
    sign=1, desc="Distance of price above its 50-day moving average",
)
_register(
    "px_ma_200", "Price / MA200", "Trend",
    lambda px, rt, c: px / px.rolling(200).mean() - 1,
    sign=1, desc="Distance of price above its 200-day moving average",
)
_register(
    "ma_50_200", "MA50 / MA200", "Trend",
    lambda px, rt, c: px.rolling(50).mean() / px.rolling(200).mean() - 1,
    sign=1, desc="Golden-cross style trend strength",
)
_register(
    "dist_52w_high", "Distance to 52w high", "Trend",
    lambda px, rt, c: px / px.rolling(252).max() - 1,
    sign=1, desc="0 when at the 52-week high, negative below it",
)
_register(
    "dist_52w_low", "Distance to 52w low", "Trend",
    lambda px, rt, c: px / px.rolling(252).min() - 1,
    sign=-1, desc="0 when at the 52-week low, positive above it",
)


def _rsi(px: pd.DataFrame, w: int = 14) -> pd.DataFrame:
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / w, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / w, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


_register(
    "rsi_14", "RSI 14d", "Mean reversion",
    lambda px, rt, c: _rsi(px, 14),
    sign=-1, desc="Relative strength index",
)
_register(
    "zscore_21d", "Price z-score 21d", "Mean reversion",
    lambda px, rt, c: (
        (px - px.rolling(21).mean()) / px.rolling(21).std()
    ),
    sign=-1, desc="Price deviation from its 21-day mean in std units",
)
_register(
    "rel_strength_63d", "Relative strength 63d", "Mean reversion",
    lambda px, rt, c: px.pct_change(63).sub(px.pct_change(63).mean(axis=1),
                                            axis=0),
    sign=1, desc="63-day return minus the universe average",
)

FEATURE_GROUPS = ["Momentum", "Risk", "Trend", "Mean reversion"]

DEFAULT_FEATURES = [
    "return_21d", "return_63d", "return_126d", "return_252d",
    "vol_63d", "vol_252d",
]


def feature_options(group: str) -> List[dict]:
    return [
        {"label": f.label, "value": f.id, "title": f.desc}
        for f in FEATURES.values() if f.group == group
    ]


# -----------------------------------------------------------------------
# Signal specification
# -----------------------------------------------------------------------

@dataclass
class SignalSpec:
    """Everything needed to reproduce a backtest."""

    name: str = "untitled"
    universe: str = "mega30"
    tickers: List[str] = field(default_factory=lambda: list(MEGA30))
    exchange: str = "US"
    from_date: str = "2010-01-01"
    to_date: Optional[str] = None

    features: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    weights: Dict[str, float] = field(default_factory=dict)
    standardize: str = "zscore"      # none | zscore | rank
    target: str = "excess"           # raw | excess | rank

    method: str = "relevance"        # relevance | linear
    horizon: int = 21
    rebalance: str = "M"             # W | M | Q
    thresh: float = 0.75
    most: bool = True
    pct_thresh: bool = True
    train_window: int = 36           # rebalance periods of training history
    min_training_periods: int = 12
    max_training_obs: int = 1500
    n_bins: int = 5
    min_history: int = 252

    def __post_init__(self):
        if not self.features:
            raise ValueError("Select at least one feature.")
        unknown = [f for f in self.features if f not in FEATURES]
        if unknown:
            raise ValueError(f"Unknown features: {', '.join(unknown)}")
        if not self.tickers:
            raise ValueError("Select at least one ticker.")
        self.tickers = sorted({t.strip().upper() for t in self.tickers if t.strip()})

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR.get(self.rebalance, 12)

    @property
    def label(self) -> str:
        mode = "most" if self.most else "least"
        core = (f"{self.universe}|h{self.horizon}|{self.rebalance}"
                f"|{self.method}")
        if self.method == "relevance":
            core += f"|t{self.thresh:.2f}|{mode}"
        return core

    def effective_weights(self) -> Dict[str, float]:
        """Linear weights, defaulting to the prior sign of each feature."""
        return {
            f: float(self.weights.get(f, FEATURES[f].sign))
            for f in self.features
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalSpec":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(payload.encode()).hexdigest()[:10]


# -----------------------------------------------------------------------
# Price loading (cached)
# -----------------------------------------------------------------------

def _price_cache_name(tickers, from_date, to_date, exchange) -> str:
    key = hashlib.md5(
        json.dumps([sorted(tickers), from_date, to_date, exchange]).encode()
    ).hexdigest()[:12]
    return f"prices_{key}.parquet"


def load_prices(spec: SignalSpec, refresh: bool = False) -> pd.DataFrame:
    """Adjusted closes for the spec universe, cached on disk as parquet."""
    name = _price_cache_name(spec.tickers, spec.from_date, spec.to_date,
                             spec.exchange)
    if not refresh:
        for d in CACHE_DIRS:
            path = d / name
            if path.exists():
                return pd.read_parquet(path)

    prices = eodhd_data.fetch_multiple(
        spec.tickers, exchange=spec.exchange,
        from_date=spec.from_date, to_date=spec.to_date,
    )
    CACHE_DIRS[0].mkdir(parents=True, exist_ok=True)
    prices.to_parquet(CACHE_DIRS[0] / name)
    return prices


# -----------------------------------------------------------------------
# Panel construction
# -----------------------------------------------------------------------

def _to_long(frame: pd.DataFrame, name: str, dates: pd.DatetimeIndex
             ) -> pd.Series:
    sub = frame.reindex(dates).copy()
    sub.index.name = "date"
    sub.columns.name = "ticker"
    long = sub.reset_index().melt(id_vars="date", var_name="ticker",
                                  value_name=name)
    return long.set_index(["date", "ticker"])[name]


def rebalance_dates(prices: pd.DataFrame, rebalance: str) -> pd.DatetimeIndex:
    """Last actual trading day of each rebalance period."""
    periods = prices.index.to_period(rebalance)
    last = pd.Series(prices.index, index=periods).groupby(level=0).last()
    return pd.DatetimeIndex(sorted(last.values))


def build_panel(prices: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """
    Long-format panel sampled at each rebalance date.

    Columns: date, ticker, <features…>, forward_return, target
    """
    prices = prices.sort_index().astype(float)
    prices.index = pd.to_datetime(prices.index)
    rets = prices.pct_change()
    ctx = _Ctx(prices, rets)

    dates = rebalance_dates(prices, spec.rebalance)
    forward = prices.shift(-spec.horizon) / prices - 1
    history = prices.notna().cumsum()

    cols = {f: _to_long(FEATURES[f].fn(prices, rets, ctx), f, dates)
            for f in spec.features}
    cols["forward_return"] = _to_long(forward, "forward_return", dates)
    cols["_history"] = _to_long(history, "_history", dates)

    panel = pd.concat(cols, axis=1)
    panel = panel[panel["_history"] >= spec.min_history].drop(columns="_history")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=list(spec.features) + ["forward_return"])
    panel = panel.reset_index().sort_values(["date", "ticker"])

    if panel.empty:
        raise ValueError(
            "Empty panel. Widen the date range or pick shorter-window features."
        )

    panel = _standardize(panel, spec)
    panel["target"] = _make_target(panel, spec)
    return panel.reset_index(drop=True)


def _standardize(panel: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    if spec.standardize == "none":
        return panel
    grouped = panel.groupby("date")
    for f in spec.features:
        if spec.standardize == "rank":
            panel[f] = grouped[f].rank(pct=True) - 0.5
        else:
            mean = grouped[f].transform("mean")
            std = grouped[f].transform("std")
            panel[f] = ((panel[f] - mean) / std.replace(0, np.nan)).clip(-3, 3)
    return panel.dropna(subset=list(spec.features))


def _make_target(panel: pd.DataFrame, spec: SignalSpec) -> pd.Series:
    fwd = panel["forward_return"]
    if spec.target == "excess":
        return fwd - panel.groupby("date")["forward_return"].transform("mean")
    if spec.target == "rank":
        return panel.groupby("date")["forward_return"].rank(pct=True) - 0.5
    return fwd


# -----------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------

def run_backtest(panel: pd.DataFrame, spec: SignalSpec,
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None
                 ) -> pd.DataFrame:
    """
    Walk-forward backtest.

    Returns a DataFrame with columns: date, ticker, predicted, actual, fit.
    """
    if spec.method == "linear":
        return _run_linear(panel, spec)
    return _run_relevance(panel, spec, progress, should_cancel)


def _run_linear(panel: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    if spec.standardize == "none":
        raise ValueError(
            "The linear composite needs standardised features "
            "(choose z-score or rank)."
        )
    weights = spec.effective_weights()
    score = sum(panel[f] * w for f, w in weights.items())
    dates = sorted(panel["date"].unique())
    warmup = set(dates[: spec.min_training_periods])
    out = pd.DataFrame({
        "date": panel["date"],
        "ticker": panel["ticker"],
        "predicted": score,
        "actual": panel["forward_return"],
        "fit": np.nan,
    })
    return (out[~out["date"].isin(warmup)]
            .dropna(subset=["predicted", "actual"])
            .reset_index(drop=True))


def _run_relevance(panel: pd.DataFrame, spec: SignalSpec,
                   progress=None, should_cancel=None) -> pd.DataFrame:
    features = list(spec.features)
    dates = sorted(panel["date"].unique())

    if len(dates) < spec.min_training_periods + 1:
        raise ValueError(
            f"Need at least {spec.min_training_periods + 1} rebalance dates; "
            f"the panel has {len(dates)}."
        )

    by_date = {d: g for d, g in panel.groupby("date")}
    total = len(dates) - spec.min_training_periods
    chunks: List[pd.DataFrame] = []

    for step, idx in enumerate(range(spec.min_training_periods, len(dates)), 1):
        if should_cancel is not None and should_cancel():
            break

        current = dates[idx]
        if progress is not None:
            progress(step, total, pd.Timestamp(current).strftime("%Y-%m-%d"))

        window = dates[max(0, idx - spec.train_window):idx]
        train = pd.concat([by_date[d] for d in window]) if window else None
        test = by_date[current]

        if train is None or len(train) < 20 or len(test) < 3:
            continue
        if len(train) > spec.max_training_obs:
            train = train.tail(spec.max_training_obs)

        try:
            preds, _ = pr.predict_df(
                train[features],
                test[features],
                train[["target"]],
                thresh=spec.thresh, most=spec.most,
                pct_thresh=spec.pct_thresh, pairwise_fits=False,
            )
        except Exception:
            continue

        chunks.append(pd.DataFrame({
            "date": current,
            "ticker": test["ticker"].to_numpy(),
            "predicted": np.asarray(preds["yhat"], dtype=float),
            "actual": test["forward_return"].to_numpy(),
            "fit": pd.to_numeric(preds["fit"], errors="coerce").to_numpy(),
        }))

    if not chunks:
        raise ValueError("No successful predictions – try a wider date range.")

    return pd.concat(chunks, ignore_index=True)


# -----------------------------------------------------------------------
# Evaluation
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
    by_year = df.groupby(df.index.year).agg(
        periods=("long_short", "size"),
        mean_ic=("ic", "mean"),
        ls_total=("long_short", lambda s: (1 + s).prod() - 1),
        ls_sharpe=("long_short", lambda s: alpha.annualized_sharpe(s, ppy)),
    ).reset_index(names="year")
    return by_year


def turnover(results: pd.DataFrame, n_bins: int = 5) -> float:
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


def permutation_test(results: pd.DataFrame, n_perm: int = 200,
                     n_bins: int = 5, ppy: int = 12, seed: int = 0) -> dict:
    """Empirical p-values under the null that predictions carry no signal."""
    rng = np.random.default_rng(seed)
    valid = results.dropna(subset=["predicted", "actual"])
    obs_ls = ls_series(valid, n_bins)["long_short"]
    obs = dict(ic=float(ic_series(valid).mean()),
               sharpe=float(alpha.annualized_sharpe(obs_ls, ppy)))

    groups = [g[["predicted", "actual"]].to_numpy() for _, g in valid.groupby("date")]
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
        null_sharpe.append(alpha.annualized_sharpe(
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
        null_sharpe_mean=float(np.nanmean(null_sharpe)) if len(null_sharpe) else np.nan,
        null_sharpe_p95=(float(np.nanpercentile(null_sharpe, 95))
                         if len(null_sharpe) else np.nan),
        null_ic_dist=null_ic.tolist(),
        null_sharpe_dist=null_sharpe.tolist(),
    )


def evaluate(results: pd.DataFrame, spec: SignalSpec,
             n_perm: int = 0) -> dict:
    """Full diagnostic bundle for a set of walk-forward predictions."""
    ppy = spec.periods_per_year
    ic = ic_series(results)
    ls = ls_series(results, spec.n_bins)
    prof = bin_profile(results, spec.n_bins)
    metrics = repl.compute_period_metrics(results)

    n = len(ic.dropna())
    ic_sd = float(ic.std(ddof=1)) if n > 2 else np.nan
    has_sd = bool(n > 2 and np.isfinite(ic_sd) and ic_sd > 0)
    summary = {
        "n_periods": int(n),
        "n_names": int(results["ticker"].nunique()),
        "avg_names": float(results.groupby("date")["ticker"].size().mean()),
        "start": str(pd.Timestamp(results["date"].min()).date()),
        "end": str(pd.Timestamp(results["date"].max()).date()),
        "mean_ic": float(ic.mean()),
        "ic_std": ic_sd,
        "ic_t": float(ic.mean() / (ic_sd / np.sqrt(n))) if has_sd else np.nan,
        "ic_ir": float(ic.mean() / ic_sd * np.sqrt(ppy)) if has_sd else np.nan,
        "ic_pos_rate": float((ic > 0).mean()),
        "hit_rate": float(metrics["hit_rate"].mean()) if not metrics.empty else np.nan,
    }

    if not ls.empty:
        lsr = ls["long_short"]
        summary.update({
            "ls_mean": float(lsr.mean()),
            "ls_vol": float(lsr.std(ddof=1) * np.sqrt(ppy)),
            "ls_total": float(ls["cum_long_short"].iloc[-1]),
            "ls_cagr": _cagr(ls["cum_long_short"].iloc[-1], len(lsr), ppy),
            "ls_sharpe": float(alpha.annualized_sharpe(lsr, ppy)),
            "ls_maxdd": float(alpha.max_drawdown(ls["cum_long_short"])),
            "ls_win_rate": float((lsr > 0).mean()),
            "long_total": float(ls["cum_long"].iloc[-1]),
            "short_total": float(ls["cum_short"].iloc[-1]),
            "bench_total": float(ls["cum_benchmark"].iloc[-1]),
            "bench_sharpe": float(alpha.annualized_sharpe(ls["benchmark"], ppy)),
        })

    prof_means = prof["mean"].to_numpy()
    summary["bin_spread"] = (float(prof_means[-1] - prof_means[0])
                             if len(prof_means) > 1 else np.nan)
    summary["monotonicity"] = (float(
        pd.Series(prof_means).corr(pd.Series(range(len(prof_means))),
                                   method="spearman")
    ) if len(prof_means) > 2 else np.nan)
    summary["turnover"] = turnover(results, spec.n_bins)

    perm = (permutation_test(results, n_perm, spec.n_bins, ppy)
            if n_perm else None)

    return {
        "summary": summary,
        "ic": ic,
        "ls": ls,
        "bins": prof,
        "metrics": metrics,
        "yearly": yearly_stability(results, spec.n_bins, ppy),
        "perm": perm,
    }


def _cagr(total_return: float, n_periods: int, ppy: int) -> float:
    years = n_periods / ppy
    if years <= 0:
        return np.nan
    return float((1 + total_return) ** (1 / years) - 1)


# -----------------------------------------------------------------------
# Run store
# -----------------------------------------------------------------------

def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def save_run(spec: SignalSpec, results: pd.DataFrame, summary: dict,
             elapsed: float = 0.0, perm: Optional[dict] = None) -> str:
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{spec.fingerprint()}"
    path = _run_dir(run_id)
    path.mkdir(parents=True, exist_ok=True)

    results.to_parquet(path / "predictions.parquet", index=False)
    (path / "spec.json").write_text(
        json.dumps(spec.to_dict(), indent=2, default=str), encoding="utf-8")
    meta = {
        "run_id": run_id,
        "name": spec.name,
        "label": spec.label,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 1),
        "summary": _jsonable(summary),
        "perm": _jsonable(perm) if perm else None,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_id


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        out[k] = v
    return out


def load_run(run_id: str):
    """Returns (spec, results, meta)."""
    path = _run_dir(run_id)
    spec = SignalSpec.from_dict(json.loads((path / "spec.json").read_text()))
    results = pd.read_parquet(path / "predictions.parquet")
    meta = json.loads((path / "meta.json").read_text())
    return spec, results, meta


def list_runs() -> pd.DataFrame:
    if not RUNS_DIR.exists():
        return pd.DataFrame()
    rows = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        row = {k: meta.get(k) for k in
               ("run_id", "name", "label", "created", "elapsed_sec")}
        row.update(meta.get("summary", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def load_report(run_id: str) -> dict:
    """Reload a saved run and recompute its diagnostics."""
    spec, results, meta = load_run(run_id)
    report = evaluate(results, spec)
    report["perm"] = meta.get("perm")
    report["spec"] = spec
    report["results"] = results
    report["meta"] = meta
    return report


def delete_run(run_id: str) -> None:
    path = _run_dir(run_id)
    if not path.exists():
        return
    if path.resolve().parent != RUNS_DIR.resolve():
        raise ValueError("Refusing to delete outside the runs directory.")
    for f in path.iterdir():
        f.unlink()
    path.rmdir()
