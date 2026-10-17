"""
Alpha Lab – signal design, backtest and evaluation engine
=========================================================

Backend for the interactive Dash app (``relevance.webapp.alpha_lab.app``).
Lets you define
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
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from relevance import config
from relevance.backtest import harness as alpha
from relevance.backtest import metrics
from relevance.core import prediction as pr
from relevance.data import cache, universes
from relevance.studies import paper as repl

LAB_DIR = config.LAB_DIR
RUNS_DIR = config.RUNS_DIR
CACHE_DIRS = config.CACHE_DIRS

PERIODS_PER_YEAR = {"W": 52, "M": 12, "Q": 4}


# -----------------------------------------------------------------------
# Universes (re-exported for backwards compatibility)
# -----------------------------------------------------------------------

MEGA30 = universes.MEGA30
SP100 = universes.SP100
DOW30 = universes.DOW30
SECTOR_ETFS = universes.SECTOR_ETFS
UNIVERSES = universes.UNIVERSES


# -----------------------------------------------------------------------
# Feature library
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureDef:
    id: str
    label: str
    group: str
    fn: Callable[[pd.DataFrame, pd.DataFrame, _Ctx], pd.DataFrame]
    sign: int = 1          # prior on the sign of the cross-sectional payoff
    desc: str = ""


class _Ctx:
    """Lazily computed quantities shared by several features."""

    def __init__(self, prices: pd.DataFrame, rets: pd.DataFrame):
        self.prices = prices
        self.rets = rets
        self._mkt: pd.Series | None = None
        self._beta: pd.DataFrame | None = None

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


FEATURES: dict[str, FeatureDef] = {}


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


def feature_options(group: str) -> list[dict]:
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
    tickers: list[str] = field(default_factory=lambda: list(MEGA30))
    exchange: str = "US"
    from_date: str = "2010-01-01"
    to_date: str | None = None

    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    weights: dict[str, float] = field(default_factory=dict)
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

    def effective_weights(self) -> dict[str, float]:
        """Linear weights, defaulting to the prior sign of each feature."""
        return {
            f: float(self.weights.get(f, FEATURES[f].sign))
            for f in self.features
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SignalSpec:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(payload.encode()).hexdigest()[:10]


# -----------------------------------------------------------------------
# Price loading (cached)
# -----------------------------------------------------------------------

def _price_cache_name(tickers, from_date, to_date, exchange) -> str:
    return cache.price_cache_name(tickers, from_date, to_date, exchange)


def load_prices(spec: SignalSpec, refresh: bool = False) -> pd.DataFrame:
    """Adjusted closes for the spec universe, cached on disk as parquet."""
    return cache.load_prices(
        spec.tickers, spec.from_date, spec.to_date, spec.exchange,
        refresh=refresh,
    )


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
                 progress: Callable[[int, int, str], None] | None = None,
                 should_cancel: Callable[[], bool] | None = None
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
    chunks: list[pd.DataFrame] = []

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
    return metrics._ranked(results)


ls_series = metrics.ls_series
bin_profile = metrics.bin_profile
ic_series = metrics.ic_series
turnover = metrics.turnover
permutation_test = metrics.permutation_test


def yearly_stability(results: pd.DataFrame, n_bins: int = 5,
                     ppy: int = 12) -> pd.DataFrame:
    return metrics.yearly_stability(results, n_bins, ppy)


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
             elapsed: float = 0.0, perm: dict | None = None) -> str:
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
