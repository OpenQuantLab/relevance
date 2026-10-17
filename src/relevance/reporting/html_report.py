"""
Relevance Alpha Signal - Standalone HTML Report
================================================

Renders a walk-forward alpha test as a single self-contained HTML file.
Charts are inline SVG, so the report needs no internet access, no CDN and
no JavaScript.

Usage
-----
    relevance-report                        # replay cached results
    relevance-report --live                 # run a fresh backtest
    relevance-report --live --tickers AAPL MSFT NVDA --from-date 2016-01-01
    relevance-report --preds results/preds_mega30_h21_t075_most.csv
    relevance-report --output docs/alpha_demo.html --open
"""

import argparse
import datetime as dt
import html
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from relevance import config
from relevance.backtest import harness as alpha
from relevance.studies import paper as repl

DEFAULT_PREDS = config.RESULTS_DIR / "preds_mega30_h21_t075_most.csv"
DEFAULT_OUTPUT = config.DATA_ROOT / "alpha_report.html"

INK = "#16213e"
POS = "#2a9d8f"
NEG = "#e07a5f"
GRID = "#e3e6ea"
MUTED = "#6b7280"


# -----------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------

def _pct(x, dp=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{dp}%}"


def _num(x, dp=3):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{dp}f}"


def _esc(x):
    return html.escape(str(x))


# -----------------------------------------------------------------------
# Inline SVG charts
# -----------------------------------------------------------------------

def _nice_bounds(values, pad_frac=0.08):
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return -1.0, 1.0
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _axes(w, h, pad, lo, hi, y_fmt, x_labels, zero_line=True):
    """Return (svg_parts, y_of, x_of) for a chart frame."""
    left, right, top, bottom = pad
    plot_w = w - left - right
    plot_h = h - top - bottom

    def y_of(v):
        return top + plot_h * (hi - v) / (hi - lo)

    def x_of(i, n):
        if n <= 1:
            return left + plot_w / 2
        return left + plot_w * i / (n - 1)

    parts = []
    for frac in np.linspace(0, 1, 5):
        v = hi - (hi - lo) * frac
        y = y_of(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="ax">{_esc(y_fmt(v))}</text>'
        )
    if zero_line and lo < 0 < hi:
        y0 = y_of(0)
        parts.append(
            f'<line x1="{left}" y1="{y0:.1f}" x2="{w - right}" y2="{y0:.1f}" '
            f'stroke="#9aa3af" stroke-width="1"/>'
        )

    n = len(x_labels)
    if n:
        step = max(1, n // 6)
        for i in range(0, n, step):
            x = x_of(i, n)
            parts.append(
                f'<text x="{x:.1f}" y="{h - bottom + 18}" text-anchor="middle" '
                f'class="ax">{_esc(x_labels[i])}</text>'
            )
    return parts, y_of, x_of


def svg_line(x_labels, series, width=860, height=300, y_fmt=_pct):
    """series: list of (name, values, colour)."""
    pad = (66, 18, 14, 34)
    all_vals = [v for _, vals, _ in series for v in vals]
    lo, hi = _nice_bounds(all_vals)
    parts, y_of, x_of = _axes(width, height, pad, lo, hi, y_fmt, x_labels)

    n = len(x_labels)
    legend = []
    for k, (name, vals, colour) in enumerate(series):
        pts = " ".join(
            f"{x_of(i, n):.1f},{y_of(v):.1f}"
            for i, v in enumerate(vals)
            if not pd.isna(v)
        )
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )
        lx = pad[0] + 10 + k * 150
        legend.append(
            f'<rect x="{lx}" y="{pad[2] + 2}" width="18" height="3" fill="{colour}"/>'
            f'<text x="{lx + 24}" y="{pad[2] + 8}" class="ax">{_esc(name)}</text>'
        )
    return _wrap_svg(width, height, parts + legend)


def svg_bars(x_labels, values, width=860, height=260, y_fmt=_num,
             mean_line=True):
    pad = (66, 18, 14, 34)
    lo, hi = _nice_bounds(list(values) + [0.0])
    parts, y_of, x_of = _axes(width, height, pad, lo, hi, y_fmt, x_labels)

    n = len(values)
    plot_w = width - pad[0] - pad[1]
    bar_w = max(1.5, plot_w / max(n, 1) * 0.72)
    y0 = y_of(0)
    for i, v in enumerate(values):
        if pd.isna(v):
            continue
        y = y_of(v)
        top, h = (y, y0 - y) if v >= 0 else (y0, y - y0)
        x = x_of(i, n) - bar_w / 2
        colour = POS if v >= 0 else NEG
        parts.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{max(h, 0.5):.1f}" fill="{colour}" opacity="0.85"/>'
        )

    if mean_line and n:
        m = float(np.nanmean(values))
        ym = y_of(m)
        parts.append(
            f'<line x1="{pad[0]}" y1="{ym:.1f}" x2="{width - pad[1]}" '
            f'y2="{ym:.1f}" stroke="{INK}" stroke-width="1.5" '
            f'stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text x="{width - pad[1]}" y="{ym - 6:.1f}" text-anchor="end" '
            f'class="ax">mean {_esc(y_fmt(m))}</text>'
        )
    return _wrap_svg(width, height, parts)


def _wrap_svg(width, height, parts):
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg>"
    )


# -----------------------------------------------------------------------
# Report assembly
# -----------------------------------------------------------------------

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f8f9fa; color: #1a1a2e;
       font: 15px/1.55 'Inter', 'Segoe UI', system-ui, sans-serif; }
header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
         color: #fff; padding: 26px 40px; }
header h1 { margin: 0; font-size: 26px; font-weight: 600; }
header p { margin: 6px 0 0; opacity: .75; font-size: 14px; }
main { max-width: 1040px; margin: 0 auto; padding: 28px 20px 60px; }
section { background: #fff; border: 1px solid #e3e6ea; border-radius: 10px;
          padding: 20px 24px; margin-bottom: 22px; }
h2 { font-size: 17px; margin: 0 0 14px; }
p.note { color: #6b7280; font-size: 13.5px; margin: 10px 0 0; }
.cards { display: grid; gap: 12px;
         grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
.card { background: #f8f9fa; border: 1px solid #e3e6ea; border-radius: 8px;
        padding: 12px 14px; }
.card .label { font-size: 11.5px; letter-spacing: .04em; text-transform: uppercase;
               color: #6b7280; }
.card .value { font-size: 21px; font-weight: 600; margin-top: 3px; }
.pos { color: #2a9d8f; } .neg { color: #e07a5f; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 7px 10px; border-bottom: 1px solid #eceff2; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f3f5f7; font-size: 12px; text-transform: uppercase;
           letter-spacing: .03em; color: #4b5563; }
text.ax { font: 11px 'Inter', 'Segoe UI', sans-serif; fill: #6b7280; }
ol, ul { margin: 8px 0 0 18px; padding: 0; }
li { margin-bottom: 4px; }
footer { color: #6b7280; font-size: 12.5px; text-align: center; padding-bottom: 30px; }
"""


def _card(label, value, tone=None):
    cls = f' class="value {tone}"' if tone else ' class="value"'
    return (
        f'<div class="card"><div class="label">{_esc(label)}</div>'
        f"<div{cls}>{_esc(value)}</div></div>"
    )


def _tone(x, invert=False):
    if x is None or pd.isna(x):
        return None
    good = x < 0 if invert else x > 0
    return "pos" if good else "neg"


def _table(df, formatters):
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = "".join(
            f"<td>{_esc(formatters.get(c, str)(r[c]))}</td>" for c in df.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_html(results, summary, metrics, ls, qa, meta):
    dates = [d.strftime("%Y-%m") for d in pd.to_datetime(ls.index)]
    ic_dates = [d.strftime("%Y-%m") for d in pd.to_datetime(metrics["date"])]

    cards = "".join([
        _card("Periods", f"{summary['n_periods']:.0f}"),
        _card("Stocks / period", f"{summary['avg_stocks_per_period']:.0f}"),
        _card("Mean IC", _num(summary["mean_ic"]), _tone(summary["mean_ic"])),
        _card("IC > 0", _pct(summary["ic_hit_rate"], 1)),
        _card("Hit rate", _pct(summary["mean_hit_rate"], 1)),
        _card("L/S total return", _pct(summary.get("total_ls_return")),
              _tone(summary.get("total_ls_return"))),
        _card("L/S Sharpe", _num(summary.get("annualized_ls_sharpe"), 2),
              _tone(summary.get("annualized_ls_sharpe"))),
        _card("Max drawdown", _pct(summary.get("max_drawdown"))),
        _card("Quintile spread", _pct(summary.get("quintile_spread")),
              _tone(summary.get("quintile_spread"))),
        _card("Avg turnover", _pct(summary.get("avg_turnover"), 1)),
    ])

    equity = svg_line(
        dates,
        [
            ("Long-short (Q5-Q1)", ls["cumulative_ls"].tolist(), INK),
            ("Long Q5", ls["cumulative_long"].tolist(), POS),
            ("Short Q1", ls["cumulative_short"].tolist(), NEG),
        ],
    )
    ic_chart = svg_bars(ic_dates, metrics["ic"].tolist(), y_fmt=lambda v: f"{v:.2f}")

    qa_tbl = ""
    if qa is not None:
        q = qa.reset_index()
        q.columns = ["Quintile", "Mean actual return", "Observations", "Std dev"]
        qa_tbl = svg_bars(
            q["Quintile"].tolist(), q["Mean actual return"].tolist(),
            height=220, y_fmt=_pct, mean_line=False,
        ) + _table(q, {
            "Mean actual return": lambda v: _pct(v),
            "Std dev": lambda v: _pct(v),
            "Observations": lambda v: f"{v:.0f}",
        })

    last_date = results["date"].max()
    latest = (
        results[results["date"] == last_date]
        .sort_values("predicted", ascending=False)
        .head(10)
        .rename(columns={
            "ticker": "Ticker", "predicted": "Predicted return",
            "actual": "Actual return", "fit": "Fit",
        })[["Ticker", "Predicted return", "Actual return", "Fit"]]
    )
    latest_tbl = _table(latest, {
        "Predicted return": lambda v: _pct(v),
        "Actual return": lambda v: _pct(v),
        "Fit": lambda v: _num(v, 4),
    })

    meta_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in meta.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relevance Alpha Signal - Demo Report</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Relevance Alpha Signal</h1>
  <p>Walk-forward out-of-sample test of relevance-based prediction &middot;
     generated {dt.datetime.now():%Y-%m-%d %H:%M}</p>
</header>
<main>

<section>
  <h2>Headline metrics</h2>
  <div class="cards">{cards}</div>
  <p class="note">Each month the model predicts the next-period cross-sectional
  return for every stock, using only data available at that time. Q5 holds the
  highest predicted returns, Q1 the lowest.</p>
</section>

<section>
  <h2>Cumulative returns</h2>
  {equity}
</section>

<section>
  <h2>Information coefficient by period</h2>
  {ic_chart}
  <p class="note">Spearman rank correlation between predicted and realised
  returns within each month. A positive mean indicates a usable ranking signal.</p>
</section>

<section>
  <h2>Quintile analysis (pooled)</h2>
  {qa_tbl}
  <p class="note">Monotonic growth from Q1 to Q5 indicates the prediction
  ranks stocks in the correct order.</p>
</section>

<section>
  <h2>Last prediction date: {_esc(pd.to_datetime(last_date).date())}</h2>
  {latest_tbl}
  <p class="note">Fit measures how well the relevant observations explain this
  particular prediction. It is available before the outcome is known.</p>
</section>

<section>
  <h2>Method</h2>
  <ol>
    <li>Fetch daily prices and build a monthly panel of momentum and
        volatility features.</li>
    <li>For each month, compute the relevance of every past observation to the
        current circumstances (informativeness plus similarity, in Mahalanobis
        space).</li>
    <li>Predict the forward return as the relevance-weighted average of past
        outcomes, keeping only the most relevant subset.</li>
    <li>Rank the predictions, form quintile portfolios, and measure IC,
        spread, Sharpe ratio, drawdown and turnover.</li>
  </ol>
  <table><tbody>{meta_rows}</tbody></table>
</section>

<footer>No training, no fitted parameters - every prediction is a weighted
average of observed outcomes.</footer>
</main>
</body>
</html>
"""


# -----------------------------------------------------------------------
# Data sources
# -----------------------------------------------------------------------

def load_cached(path):
    results = pd.read_csv(path, parse_dates=["date"])
    return results


def run_live(args):
    results, _ = alpha.run_alpha_backtest(
        tickers=args.tickers,
        exchange=args.exchange,
        api_token=args.api_token,
        from_date=args.from_date,
        to_date=args.to_date,
        thresh=args.thresh,
        forward_period=args.forward_period,
        min_training_months=args.min_training,
    )
    results["date"] = pd.to_datetime(results["date"])
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--live", action="store_true",
                   help="Run a fresh backtest instead of replaying a CSV")
    p.add_argument("--preds", default=str(DEFAULT_PREDS),
                   help="Prediction CSV (date,ticker,predicted,actual,fit)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--open", action="store_true", dest="open_browser",
                   help="Open the report in the default browser")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--exchange", default="US")
    p.add_argument("--api-token", default=None)
    p.add_argument("--from-date", default="2015-01-01")
    p.add_argument("--to-date", default=None)
    p.add_argument("--thresh", type=float, default=0.75)
    p.add_argument("--forward-period", type=int, default=21)
    p.add_argument("--min-training", type=int, default=12)
    args = p.parse_args()

    if args.live:
        results = run_live(args)
        source = "live backtest"
    else:
        results = load_cached(args.preds)
        source = Path(args.preds).name

    summary, metrics, ls = alpha.summarize_performance(results)
    qa = repl.compute_quintile_analysis(
        results, pred_col="predicted", actual_col="actual"
    )

    meta = {
        "Source": source,
        "Universe": f"{results['ticker'].nunique()} tickers",
        "Sample": f"{results['date'].min():%Y-%m} to {results['date'].max():%Y-%m}",
        "Predictions": f"{len(results):,}",
        "Relevance threshold": f"top {(1 - args.thresh):.0%} most relevant",
        "Forward horizon": f"{args.forward_period} trading days",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(results, summary, metrics, ls, qa, meta),
                   encoding="utf-8")
    print(f"Report written to {out.resolve()}")
    if args.open_browser:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
