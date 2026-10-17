"""
Alpha Lab – interactive Dash workbench
======================================

Design a cross-sectional alpha signal, run a walk-forward test on it, and
read the results in a dashboard.

    python alpha_lab_app.py        # http://127.0.0.1:8051

Backtests run in a worker thread inside this process, so serve the app with a
single worker (``gunicorn -w 1 --threads 8 alpha_lab_app:server``).

Environment
-----------
ALPHA_LAB_PASSWORD    when set, visitors get a read-only dashboard and only
                      someone with the password can start a backtest
ALPHA_LAB_URL_PREFIX  path the app is served under, e.g. /projects/alpha-lab/
ALPHA_LAB_HOST        dev-server bind address (default 127.0.0.1)
ALPHA_LAB_PORT        dev-server port (default 8051)
ALPHA_LAB_DEBUG       "0" to turn off the dev-server debug tools
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (ALL, Dash, Input, Output, State, callback, dash_table, dcc,
                  html, no_update)

import alpha_lab as lab

# ---------------------------------------------------------------------------
# Deployment configuration
# ---------------------------------------------------------------------------

URL_PREFIX = os.environ.get("ALPHA_LAB_URL_PREFIX", "/").strip() or "/"
if not URL_PREFIX.startswith("/"):
    URL_PREFIX = "/" + URL_PREFIX
if not URL_PREFIX.endswith("/"):
    URL_PREFIX += "/"

_PASSWORD = os.environ.get("ALPHA_LAB_PASSWORD", "")
LOCKED = bool(_PASSWORD)

# Upper bounds applied to every submitted spec, unlocked or not.
MAX_TICKERS = 150
MAX_SWEEP = 8
MAX_PERM = 2000

_SESSIONS: set = set()
_SESSION_LOCK = threading.Lock()


def grant_session() -> str:
    token = secrets.token_urlsafe(24)
    with _SESSION_LOCK:
        _SESSIONS.add(token)
    return token


def is_unlocked(token) -> bool:
    if not LOCKED:
        return True
    with _SESSION_LOCK:
        return isinstance(token, str) and token in _SESSIONS


def password_matches(candidate) -> bool:
    return bool(candidate) and hmac.compare_digest(str(candidate), _PASSWORD)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

INK = "#16213e"
ACCENT = "#2a9d8f"
NEG = "#e07a5f"
BLUE = "#4361ee"
MUTED = "#6b7280"
BG = "#f4f5f7"
LINE = "#e5e7eb"
FONT = "'Inter', 'Segoe UI', sans-serif"

CARD = {
    "backgroundColor": "white",
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
}
LABEL = {"fontWeight": "600", "fontSize": "12px", "color": INK,
         "display": "block", "marginBottom": "4px", "marginTop": "14px"}
SECTION = {"fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.08em",
           "textTransform": "uppercase", "color": MUTED,
           "margin": "22px 0 2px 0", "borderTop": f"1px solid {LINE}",
           "paddingTop": "14px"}
INPUT = {"width": "100%", "padding": "6px 8px", "fontSize": "13px",
         "border": f"1px solid {LINE}", "borderRadius": "6px",
         "boxSizing": "border-box"}

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=FONT, size=12, color=INK),
    margin=dict(l=55, r=20, t=78, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    title=dict(x=0, xanchor="left", y=0.97, yanchor="top",
               font=dict(size=15)),
)


def _pct(x, digits=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _num(x, digits=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _positive(x) -> Optional[bool]:
    """True/False for colouring a metric, None when the value is missing."""
    if x is None or not np.isfinite(x):
        return None
    return bool(x > 0)


# ---------------------------------------------------------------------------
# Background job manager
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, specs: List[lab.SignalSpec], n_perm: int):
        self.id = uuid.uuid4().hex[:12]
        self.specs = specs
        self.n_perm = n_perm
        self.status = "running"          # running | done | error | cancelled
        self.stage = "Loading prices…"
        self.step = 0
        self.total = 0
        self.spec_index = 0
        self.run_ids: List[str] = []
        self.error: Optional[str] = None
        self.cancelled = False

    @property
    def fraction(self) -> float:
        per_spec = 1 / len(self.specs)
        inner = (self.step / self.total) if self.total else 0.0
        return min(1.0, self.spec_index * per_spec + inner * per_spec)


JOBS: Dict[str, Job] = {}
_JOB_LOCK = threading.Lock()


def _run_job(job: Job) -> None:
    try:
        for i, spec in enumerate(job.specs):
            if job.cancelled:
                break
            job.spec_index = i
            job.step, job.total = 0, 0
            job.stage = f"[{i + 1}/{len(job.specs)}] loading prices…"
            prices = lab.load_prices(spec)

            job.stage = f"[{i + 1}/{len(job.specs)}] building panel…"
            panel = lab.build_panel(prices, spec)

            def progress(step, total, when, _i=i):
                job.step, job.total = step, total
                job.stage = (f"[{_i + 1}/{len(job.specs)}] "
                             f"backtesting {when} ({step}/{total})")

            t0 = time.time()
            results = lab.run_backtest(
                panel, spec, progress=progress,
                should_cancel=lambda: job.cancelled,
            )
            if job.cancelled:
                break

            job.stage = f"[{i + 1}/{len(job.specs)}] evaluating…"
            report = lab.evaluate(results, spec, n_perm=job.n_perm)
            summary = dict(report["summary"])
            if report["perm"]:
                summary["p_ic"] = report["perm"]["p_ic"]
                summary["p_sharpe"] = report["perm"]["p_sharpe"]
            job.run_ids.append(
                lab.save_run(spec, results, summary,
                             elapsed=time.time() - t0, perm=report["perm"])
            )
        job.status = "cancelled" if job.cancelled else "done"
        job.stage = ("Cancelled." if job.cancelled
                     else f"Finished {len(job.run_ids)} run(s).")
    except Exception as exc:                      # surfaced in the UI
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.stage = "Failed."
        traceback.print_exc()


def start_job(specs: List[lab.SignalSpec], n_perm: int) -> Job:
    job = Job(specs, n_perm)
    with _JOB_LOCK:
        JOBS[job.id] = job
        for old in [k for k, v in JOBS.items()
                    if v.status != "running" and k != job.id][:-10]:
            JOBS.pop(old, None)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def job_in_flight() -> bool:
    with _JOB_LOCK:
        return any(j.status == "running" for j in JOBS.values())


# ---------------------------------------------------------------------------
# Designer sidebar
# ---------------------------------------------------------------------------

def _field(label, control, hint=None):
    kids = [html.Label(label, style=LABEL), control]
    if hint:
        kids.append(html.Div(hint, style={"fontSize": "11px", "color": MUTED,
                                          "marginTop": "3px"}))
    return html.Div(kids)


def _run_btn_style(enabled: bool) -> dict:
    return {"flex": "2", "padding": "10px", "border": "none",
            "borderRadius": "8px",
            "backgroundColor": INK if enabled else LINE,
            "color": "white" if enabled else MUTED,
            "fontWeight": "600", "fontSize": "13px",
            "cursor": "pointer" if enabled else "not-allowed"}


def lock_panel(unlocked: bool):
    """Password box shown when ALPHA_LAB_PASSWORD is configured."""
    if not LOCKED:
        return None
    if unlocked:
        return html.Div("Unlocked – you can start backtests.",
                        style={"fontSize": "12px", "color": ACCENT,
                               "marginTop": "12px", "fontWeight": "600"})
    return html.Div(style={"marginTop": "12px"}, children=[
        html.Label("Password", style=LABEL),
        dcc.Input(id="password", type="password", style=INPUT,
                  placeholder="to enable running", debounce=True),
        html.Button("Unlock", id="unlock-btn", n_clicks=0, style={
            "width": "100%", "marginTop": "8px", "padding": "8px",
            "border": f"1px solid {LINE}", "borderRadius": "8px",
            "backgroundColor": "white", "color": INK, "fontSize": "13px",
            "cursor": "pointer"}),
        html.Div(id="unlock-msg", style={"fontSize": "11px", "color": MUTED,
                                         "marginTop": "6px"}),
    ])


def designer():
    return html.Div(
        style={**CARD, "width": "350px", "flexShrink": "0",
               "maxHeight": "calc(100vh - 140px)", "overflowY": "auto",
               "position": "sticky", "top": "20px"},
        children=[
            html.H3("Signal designer", style={"margin": "0", "fontSize": "17px",
                                              "color": INK}),

            html.Div("Universe", style=SECTION),
            _field("Preset", dcc.Dropdown(
                id="universe", clearable=False, style={"fontSize": "13px"},
                options=[{"label": k, "value": k} for k in lab.UNIVERSES]
                        + [{"label": "custom", "value": "custom"}],
                value="mega30")),
            _field("Tickers", dcc.Textarea(
                id="tickers", style={**INPUT, "height": "70px",
                                     "fontFamily": "monospace",
                                     "fontSize": "11px"},
                value=", ".join(lab.MEGA30)),
                "Comma or space separated."),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Div(_field("From", dcc.Input(
                    id="from-date", type="text", value="2010-01-01",
                    style=INPUT)), style={"flex": "1"}),
                html.Div(_field("To", dcc.Input(
                    id="to-date", type="text", value="", placeholder="today",
                    style=INPUT)), style={"flex": "1"}),
            ]),

            html.Div("Features", style=SECTION),
            html.Div([
                html.Div([
                    html.Div(group, style={"fontSize": "11px",
                                           "fontWeight": "700",
                                           "color": INK, "marginTop": "10px"}),
                    dcc.Checklist(
                        id={"type": "feature-group", "group": group},
                        options=lab.feature_options(group),
                        value=[f for f in lab.DEFAULT_FEATURES
                               if lab.FEATURES[f].group == group],
                        style={"fontSize": "12px"},
                        labelStyle={"display": "block", "marginBottom": "1px"},
                    ),
                ]) for group in lab.FEATURE_GROUPS
            ]),
            _field("Cross-sectional scaling", dcc.Dropdown(
                id="standardize", clearable=False, style={"fontSize": "13px"},
                options=[{"label": "Z-score (clipped ±3)", "value": "zscore"},
                         {"label": "Percentile rank", "value": "rank"},
                         {"label": "Raw values", "value": "none"}],
                value="zscore")),

            html.Div("Model", style=SECTION),
            _field("Prediction method", dcc.RadioItems(
                id="method",
                options=[{"label": " Relevance model", "value": "relevance"},
                         {"label": " Linear composite", "value": "linear"}],
                value="relevance", style={"fontSize": "13px"},
                labelStyle={"display": "block"})),
            html.Div(id="weights-panel"),
            html.Div(id="relevance-panel", children=[
                _field("Relevance threshold", dcc.Slider(
                    id="thresh", min=0.05, max=0.95, step=0.05, value=0.75,
                    marks={i / 10: f"{i / 10:.1f}" for i in range(1, 10, 2)},
                    tooltip={"placement": "bottom", "always_visible": True})),
                _field("Predict from", dcc.RadioItems(
                    id="most",
                    options=[{"label": " Most relevant", "value": "most"},
                             {"label": " Least relevant", "value": "least"}],
                    value="most", style={"fontSize": "13px"},
                    labelStyle={"display": "inline-block",
                                "marginRight": "12px"})),
                _field("Sweep thresholds", dcc.Input(
                    id="sweep", type="text", value="", style=INPUT,
                    placeholder="e.g. 0.5, 0.75, 0.9"),
                    "Optional – runs one backtest per value."),
            ]),

            html.Div("Target & schedule", style=SECTION),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Div(_field("Horizon (days)", dcc.Input(
                    id="horizon", type="number", value=21, min=1, step=1,
                    style=INPUT)), style={"flex": "1"}),
                html.Div(_field("Rebalance", dcc.Dropdown(
                    id="rebalance", clearable=False,
                    style={"fontSize": "13px"},
                    options=[{"label": "Weekly", "value": "W"},
                             {"label": "Monthly", "value": "M"},
                             {"label": "Quarterly", "value": "Q"}],
                    value="M")), style={"flex": "1"}),
            ]),
            _field("Target variable", dcc.Dropdown(
                id="target", clearable=False, style={"fontSize": "13px"},
                options=[{"label": "Forward return", "value": "raw"},
                         {"label": "Excess vs universe", "value": "excess"},
                         {"label": "Cross-sectional rank", "value": "rank"}],
                value="excess")),

            html.Div("Backtest", style=SECTION),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Div(_field("Train window", dcc.Input(
                    id="train-window", type="number", value=36, min=2,
                    style=INPUT)), style={"flex": "1"}),
                html.Div(_field("Warm-up", dcc.Input(
                    id="min-train", type="number", value=12, min=1,
                    style=INPUT)), style={"flex": "1"}),
            ]),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                html.Div(_field("Max train obs", dcc.Input(
                    id="max-obs", type="number", value=1500, min=50,
                    step=50, style=INPUT)), style={"flex": "1"}),
                html.Div(_field("Bins", dcc.Input(
                    id="n-bins", type="number", value=5, min=2, max=10,
                    style=INPUT)), style={"flex": "1"}),
            ]),
            _field("Permutation tests", dcc.Input(
                id="n-perm", type="number", value=0, min=0, max=2000, step=50,
                style=INPUT), "0 = skip. 200 is a good trade-off."),

            html.Div("Run", style=SECTION),
            _field("Name", dcc.Input(id="run-name", type="text",
                                     value="my-alpha", style=INPUT)),
            html.Div(id="lock-panel"),
            html.Div(style={"display": "flex", "gap": "8px",
                            "marginTop": "16px"}, children=[
                html.Button("Run test", id="run-btn", n_clicks=0,
                            disabled=LOCKED, style=_run_btn_style(not LOCKED)),
                html.Button("Cancel", id="cancel-btn", n_clicks=0, style={
                    "flex": "1", "padding": "10px",
                    "border": f"1px solid {LINE}", "borderRadius": "8px",
                    "backgroundColor": "white", "color": MUTED,
                    "fontSize": "13px", "cursor": "pointer"}),
            ]),
            html.Div(id="progress-wrap", style={"marginTop": "14px"}, children=[
                html.Div(style={"height": "6px", "backgroundColor": LINE,
                                "borderRadius": "3px", "overflow": "hidden"},
                         children=html.Div(id="progress-bar", style={
                             "height": "100%", "width": "0%",
                             "backgroundColor": ACCENT,
                             "transition": "width .3s"})),
                html.Div(id="progress-text",
                         style={"fontSize": "11px", "color": MUTED,
                                "marginTop": "6px", "minHeight": "16px"}),
            ]),
        ],
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_equity(ls: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    series = [("cum_long_short", "Long / short", INK, 2.5),
              ("cum_long", "Long bin", ACCENT, 1.5),
              ("cum_short", "Short bin", NEG, 1.5),
              ("cum_benchmark", "Equal-weight universe", MUTED, 1.5)]
    for col, name, color, width in series:
        fig.add_trace(go.Scatter(
            x=ls.index, y=ls[col], name=name, mode="lines",
            line=dict(color=color, width=width,
                      dash="dot" if name.endswith("universe") else "solid")))
    fig.update_layout(**PLOT_LAYOUT, title_text="Cumulative return",
                      yaxis_tickformat=".0%", height=400)
    return fig


def fig_drawdown(ls: pd.DataFrame) -> go.Figure:
    wealth = 1 + ls["cum_long_short"]
    dd = wealth / wealth.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", mode="lines",
                               line=dict(color=NEG, width=1),
                               name="Drawdown"))
    fig.update_layout(**PLOT_LAYOUT, title_text="Long/short drawdown",
                      yaxis_tickformat=".0%", height=280)
    return fig


def fig_ic(ic: pd.Series, window: int = 12) -> go.Figure:
    colors = [ACCENT if v >= 0 else NEG for v in ic.fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ic.index, y=ic, marker_color=colors,
                         name="Period IC"))
    fig.add_trace(go.Scatter(x=ic.index, y=ic.rolling(window).mean(),
                             mode="lines", name=f"{window}-period mean",
                             line=dict(color=INK, width=2)))
    fig.add_hline(y=float(ic.mean()), line_dash="dash", line_color=BLUE,
                  annotation_text=f"mean {ic.mean():.3f}")
    fig.update_layout(**PLOT_LAYOUT, title_text="Information coefficient",
                      height=340)
    return fig


def fig_bins(prof: pd.DataFrame) -> go.Figure:
    colors = [ACCENT if v >= 0 else NEG for v in prof["mean"]]
    fig = go.Figure(go.Bar(x=prof["label"], y=prof["mean"],
                           marker_color=colors,
                           text=[f"{v * 100:.2f}%" for v in prof["mean"]],
                           textposition="outside",
                           customdata=prof["count"],
                           hovertemplate="%{x}: %{y:.2%}<br>n=%{customdata}"))
    layout = {**PLOT_LAYOUT, "hovermode": "closest"}
    fig.update_layout(**layout,
                      title_text="Mean forward return by predicted bin",
                      yaxis_tickformat=".1%", height=340)
    return fig


def fig_rolling_sharpe(ls: pd.DataFrame, ppy: int, window: int = 24
                       ) -> go.Figure:
    r = ls["long_short"]
    roll = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(ppy)
    fig = go.Figure(go.Scatter(x=roll.index, y=roll, mode="lines",
                               line=dict(color=BLUE, width=2),
                               name="Rolling Sharpe"))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.update_layout(**PLOT_LAYOUT,
                      title_text=f"Rolling {window}-period Sharpe", height=300)
    return fig


def fig_yearly(yearly: pd.DataFrame) -> go.Figure:
    colors = [ACCENT if v >= 0 else NEG for v in yearly["ls_total"]]
    fig = go.Figure(go.Bar(x=yearly["year"], y=yearly["ls_total"],
                           marker_color=colors,
                           text=[f"{v * 100:.0f}%" for v in yearly["ls_total"]],
                           textposition="outside"))
    layout = {**PLOT_LAYOUT, "hovermode": "closest"}
    fig.update_layout(**layout,
                      title_text="Long/short return by calendar year",
                      yaxis_tickformat=".0%", height=320)
    return fig


def fig_scatter(results: pd.DataFrame) -> go.Figure:
    valid = results.dropna(subset=["predicted", "actual"]).copy()
    valid["pred_rank"] = valid.groupby("date")["predicted"].rank(pct=True)
    sample = valid.sample(min(4000, len(valid)), random_state=0)
    fig = go.Figure(go.Scattergl(
        x=sample["pred_rank"], y=sample["actual"], mode="markers",
        marker=dict(size=4, color=BLUE, opacity=0.35),
        name="Observations",
        hovertemplate="rank %{x:.2f} → %{y:.2%}<extra></extra>"))
    binned = valid.groupby(pd.cut(valid["pred_rank"], 20),
                           observed=False)["actual"].mean()
    centers = [iv.mid for iv in binned.index]
    fig.add_trace(go.Scatter(x=centers, y=binned.to_numpy(), mode="lines+markers",
                             line=dict(color=INK, width=2.5),
                             name="Binned mean"))
    layout = {**PLOT_LAYOUT, "hovermode": "closest"}
    fig.update_layout(**layout, title_text="Predicted rank vs realised return",
                      xaxis_title="Cross-sectional rank of prediction",
                      yaxis_tickformat=".0%", height=380)
    return fig


def fig_perm(perm: dict) -> go.Figure:
    fig = go.Figure(go.Histogram(x=perm.get("null_sharpe_dist", []),
                                 marker_color=LINE, name="Null"))
    fig.add_vline(x=perm["obs_sharpe"], line_color=NEG, line_width=2,
                  annotation_text=f"observed {perm['obs_sharpe']:.2f}")
    layout = {**PLOT_LAYOUT, "hovermode": "closest"}
    fig.update_layout(**layout, height=300,
                      title_text=(f"Permutation null for L/S Sharpe "
                                  f"(p = {perm['p_sharpe']:.3f})"))
    return fig


def fig_heatmap(ls: pd.DataFrame) -> go.Figure:
    df = pd.DataFrame({"r": ls["long_short"]})
    df["year"] = df.index.year
    pivot = df.pivot_table(index="year", columns=df.index.month,
                           values="r", aggfunc="mean")
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b")
                     for m in pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=list(pivot.columns), y=list(pivot.index),
        colorscale=[[0, NEG], [0.5, "#ffffff"], [1, ACCENT]], zmid=0,
        hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        colorbar=dict(tickformat=".0%")))
    layout = {**PLOT_LAYOUT, "hovermode": "closest"}
    fig.update_layout(**layout, title_text="Long/short return by period",
                      height=max(260, 28 * len(pivot) + 120))
    return fig


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def metric_card(label, value, hint=None, good=None):
    color = INK if good is None else (ACCENT if good else NEG)
    return html.Div(style={**CARD, "padding": "14px 16px", "flex": "1",
                           "minWidth": "140px"}, children=[
        html.Div(label, style={"fontSize": "11px", "color": MUTED,
                               "textTransform": "uppercase",
                               "letterSpacing": "0.05em"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "700",
                               "color": color, "marginTop": "4px"}),
        html.Div(hint or "", style={"fontSize": "11px", "color": MUTED}),
    ])


def _table(df: pd.DataFrame, fmt: Optional[dict] = None, height="320px"):
    fmt = fmt or {}
    show = df.copy()
    for col, fn in fmt.items():
        if col in show.columns:
            show[col] = show[col].map(fn)
    return dash_table.DataTable(
        data=show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in show.columns],
        page_size=25, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto", "maxHeight": height,
                     "overflowY": "auto"},
        style_cell={"fontFamily": FONT, "fontSize": "12px",
                    "padding": "6px 10px", "textAlign": "right"},
        style_header={"backgroundColor": BG, "fontWeight": "700",
                      "color": INK, "border": "none"},
        style_data={"borderBottom": f"1px solid {LINE}", "border": "none"},
    )


def overview_tab(report: dict) -> html.Div:
    s, spec = report["summary"], report["spec"]
    perm = report.get("perm")
    ic_t = s.get("ic_t")
    ic_significant = (None if ic_t is None or not np.isfinite(ic_t)
                      else abs(ic_t) > 2)
    cards = [
        metric_card("L/S total", _pct(s.get("ls_total")),
                    f"{s['start']} → {s['end']}",
                    good=_positive(s.get("ls_total"))),
        metric_card("Sharpe", _num(s.get("ls_sharpe")),
                    f"vs {_num(s.get('bench_sharpe'))} benchmark",
                    good=_positive(s.get("ls_sharpe"))),
        metric_card("CAGR", _pct(s.get("ls_cagr")),
                    f"vol {_pct(s.get('ls_vol'), 0)}"),
        metric_card("Max drawdown", _pct(s.get("ls_maxdd")), good=False),
        metric_card("Mean IC", _num(s.get("mean_ic"), 3),
                    f"t = {_num(ic_t)}", good=ic_significant),
        metric_card("Bin spread", _pct(s.get("bin_spread"), 2),
                    f"monotonicity {_num(s.get('monotonicity'))}"),
        metric_card("Turnover", _pct(s.get("turnover"), 0),
                    "per rebalance"),
        metric_card("Periods", f"{s.get('n_periods', 0)}",
                    f"{_num(s.get('avg_names'), 0)} names each"),
    ]
    if perm:
        p = perm.get("p_sharpe")
        cards.append(metric_card(
            "p (Sharpe)", _num(p, 3),
            f"{perm.get('n_perm')} permutations",
            good=_positive(0.05 - p) if p is not None else None))

    return html.Div([
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
                 children=cards),
        html.Div(style={**CARD, "marginTop": "16px"},
                 children=dcc.Graph(figure=fig_equity(report["ls"]),
                                    config={"displaylogo": False})),
        html.Div(style={"display": "flex", "gap": "16px",
                        "marginTop": "16px"}, children=[
            html.Div(style={**CARD, "flex": "1"},
                     children=dcc.Graph(figure=fig_drawdown(report["ls"]),
                                        config={"displaylogo": False})),
            html.Div(style={**CARD, "flex": "1"},
                     children=dcc.Graph(
                         figure=fig_rolling_sharpe(report["ls"],
                                                   spec.periods_per_year),
                         config={"displaylogo": False})),
        ]),
        html.Div(style={**CARD, "marginTop": "16px"}, children=[
            html.H4("Specification", style={"marginTop": "0", "fontSize": "14px"}),
            html.Pre(_spec_text(spec), style={
                "fontSize": "12px", "fontFamily": "monospace",
                "whiteSpace": "pre-wrap", "color": MUTED, "margin": "0"}),
        ]),
    ])


def _spec_text(spec: lab.SignalSpec) -> str:
    lines = [
        f"universe      {spec.universe} ({len(spec.tickers)} names)",
        f"window        {spec.from_date} → {spec.to_date or 'today'}",
        f"features      {', '.join(spec.features)}",
        f"scaling       {spec.standardize}    target: {spec.target}",
        f"method        {spec.method}",
    ]
    if spec.method == "relevance":
        lines.append(f"relevance     thresh={spec.thresh:.2f} "
                     f"{'most' if spec.most else 'least'} relevant, "
                     f"pct_thresh={spec.pct_thresh}")
    else:
        lines.append("weights       " + ", ".join(
            f"{k}={v:+.2f}" for k, v in spec.effective_weights().items()))
    lines.append(f"schedule      {spec.rebalance} rebalance, "
                 f"{spec.horizon}d horizon, train {spec.train_window} periods")
    return "\n".join(lines)


def signal_tab(report: dict) -> html.Div:
    return html.Div([
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**CARD, "flex": "1"},
                     children=dcc.Graph(figure=fig_bins(report["bins"]),
                                        config={"displaylogo": False})),
            html.Div(style={**CARD, "flex": "1"},
                     children=dcc.Graph(figure=fig_scatter(report["results"]),
                                        config={"displaylogo": False})),
        ]),
        html.Div(style={**CARD, "marginTop": "16px"},
                 children=dcc.Graph(figure=fig_ic(report["ic"]),
                                    config={"displaylogo": False})),
        html.Div(style={**CARD, "marginTop": "16px"}, children=[
            html.H4("Bin profile", style={"marginTop": "0", "fontSize": "14px"}),
            _table(report["bins"][["label", "mean", "std", "count"]]
                   .rename(columns={"label": "bin", "mean": "mean return",
                                    "std": "std", "count": "obs"}),
                   fmt={"mean return": lambda v: _pct(v, 2),
                        "std": lambda v: _pct(v, 2)}, height="260px"),
        ]),
    ])


def stability_tab(report: dict) -> html.Div:
    s = report["summary"]
    perm = report.get("perm")
    blocks = [
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
                 children=[
            metric_card("IC t-stat", _num(s.get("ic_t")),
                        f"IR {_num(s.get('ic_ir'))}"),
            metric_card("IC > 0", _pct(s.get("ic_pos_rate"), 0),
                        "of periods"),
            metric_card("L/S win rate", _pct(s.get("ls_win_rate"), 0)),
            metric_card("Directional hit", _pct(s.get("hit_rate"), 1)),
            metric_card("Long leg", _pct(s.get("long_total"))),
            metric_card("Short leg", _pct(s.get("short_total"))),
            metric_card("Benchmark", _pct(s.get("bench_total")),
                        "equal weight"),
        ]),
        html.Div(style={**CARD, "marginTop": "16px"},
                 children=dcc.Graph(figure=fig_heatmap(report["ls"]),
                                    config={"displaylogo": False})),
    ]
    yearly = report["yearly"]
    if not yearly.empty:
        blocks.append(html.Div(style={**CARD, "marginTop": "16px"},
                               children=dcc.Graph(figure=fig_yearly(yearly),
                                                  config={"displaylogo": False})))
        blocks.append(html.Div(style={**CARD, "marginTop": "16px"}, children=[
            html.H4("Year by year", style={"marginTop": "0",
                                           "fontSize": "14px"}),
            _table(yearly, fmt={"mean_ic": lambda v: _num(v, 3),
                                "ls_total": _pct,
                                "ls_sharpe": lambda v: _num(v, 2)},
                   height="300px"),
        ]))
    if perm:
        blocks.append(html.Div(style={**CARD, "marginTop": "16px"}, children=[
            dcc.Graph(figure=fig_perm(perm), config={"displaylogo": False}),
            html.Div(f"Observed mean IC {perm['obs_ic']:.4f} "
                     f"(p = {perm['p_ic']:.3f}). Null Sharpe mean "
                     f"{perm['null_sharpe_mean']:.2f}, 95th pct "
                     f"{perm['null_sharpe_p95']:.2f}.",
                     style={"fontSize": "12px", "color": MUTED}),
        ]))
    return html.Div(blocks)


def predictions_tab(report: dict) -> html.Div:
    results = report["results"].copy()
    results["date"] = pd.to_datetime(results["date"]).dt.strftime("%Y-%m-%d")
    latest_date = results["date"].max()
    latest = (results[results["date"] == latest_date]
              .sort_values("predicted", ascending=False))
    return html.Div([
        html.Div(style={**CARD}, children=[
            html.H4(f"Cross-section on {latest_date}",
                    style={"marginTop": "0", "fontSize": "14px"}),
            _table(latest[["ticker", "predicted", "actual", "fit"]],
                   fmt={"predicted": lambda v: _num(v, 4),
                        "actual": lambda v: _pct(v, 2),
                        "fit": lambda v: _num(v, 3)}, height="400px"),
        ]),
        html.Div(style={**CARD, "marginTop": "16px"}, children=[
            html.H4("All predictions", style={"marginTop": "0",
                                              "fontSize": "14px"}),
            html.Div("Sortable and filterable. Use the filter row to drill "
                     "into a ticker or date.",
                     style={"fontSize": "12px", "color": MUTED,
                            "marginBottom": "8px"}),
            _table(results[["date", "ticker", "predicted", "actual", "fit"]],
                   fmt={"predicted": lambda v: _num(v, 4),
                        "actual": lambda v: _pct(v, 2),
                        "fit": lambda v: _num(v, 3)}, height="520px"),
        ]),
    ])


COMPARE_COLS = ["name", "label", "created", "n_periods", "mean_ic", "ic_t",
                "ls_total", "ls_sharpe", "ls_maxdd", "bin_spread", "turnover",
                "p_sharpe", "elapsed_sec"]


def _compare_figure(selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for rid in (selected or [])[:8]:
        try:
            spec, results, meta = lab.load_run(rid)
        except (FileNotFoundError, OSError):
            continue
        ls = lab.ls_series(results, spec.n_bins)
        fig.add_trace(go.Scatter(x=ls.index, y=ls["cum_long_short"],
                                 mode="lines", name=meta.get("name", rid)))
    fig.update_layout(**PLOT_LAYOUT, title_text="Cumulative long/short by run",
                      yaxis_tickformat=".0%", height=420)
    return fig


def compare_tab(selected: List[str]) -> html.Div:
    runs = lab.list_runs()
    if runs.empty:
        return html.Div("No saved runs yet.", style={**CARD, "color": MUTED})

    cols = [c for c in COMPARE_COLS if c in runs.columns]
    table = runs[["run_id"] + cols]

    return html.Div([
        html.Div(style={**CARD}, children=[
            html.H4("Saved runs", style={"marginTop": "0", "fontSize": "14px"}),
            dcc.Dropdown(id="compare-select", multi=True, value=selected or [],
                         options=[{"label": f"{r['name']} · {r['label']} · "
                                            f"{r['created']}",
                                   "value": r["run_id"]}
                                  for _, r in runs.iterrows()],
                         placeholder="Pick runs to overlay",
                         style={"fontSize": "13px"}),
            html.Div(style={"height": "12px"}),
            _table(table, fmt={
                "mean_ic": lambda v: _num(v, 3),
                "ic_t": lambda v: _num(v, 2),
                "ls_total": _pct, "ls_maxdd": _pct,
                "bin_spread": lambda v: _pct(v, 2),
                "turnover": lambda v: _pct(v, 0),
                "ls_sharpe": lambda v: _num(v, 2),
                "p_sharpe": lambda v: _num(v, 3),
            }, height="340px"),
        ]),
        html.Div(style={**CARD, "marginTop": "16px"},
                 children=dcc.Graph(id="compare-graph",
                                    figure=_compare_figure(selected or []),
                                    config={"displaylogo": False})),
    ])


def sweep_table(run_ids: List[str]) -> Optional[html.Div]:
    if len(run_ids) < 2:
        return None
    rows = []
    for rid in run_ids:
        try:
            _, _, meta = lab.load_run(rid)
        except (FileNotFoundError, OSError):
            continue
        row = {"run": meta.get("label", rid)}
        row.update({k: meta["summary"].get(k) for k in
                    ("mean_ic", "ic_t", "ls_total", "ls_sharpe", "ls_maxdd",
                     "bin_spread", "turnover")})
        rows.append(row)
    return html.Div(style={**CARD, "marginBottom": "16px"}, children=[
        html.H4("Sweep results", style={"marginTop": "0", "fontSize": "14px"}),
        _table(pd.DataFrame(rows), fmt={
            "mean_ic": lambda v: _num(v, 3), "ic_t": lambda v: _num(v, 2),
            "ls_total": _pct, "ls_maxdd": _pct,
            "bin_spread": lambda v: _pct(v, 2),
            "turnover": lambda v: _pct(v, 0),
            "ls_sharpe": lambda v: _num(v, 2)}, height="260px"),
    ])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, title="Alpha Lab",
           requests_pathname_prefix=URL_PREFIX,
           routes_pathname_prefix=URL_PREFIX)
app.config.suppress_callback_exceptions = True
server = app.server


def _saved_run_options():
    runs = lab.list_runs()
    if runs.empty:
        return []
    return [{"label": f"{r['name']} · {r['label']} · {r['created']}",
             "value": r["run_id"]} for _, r in runs.iterrows()]


_INITIAL_RUNS = _saved_run_options()

app.layout = html.Div(
    style={"fontFamily": FONT, "backgroundColor": BG, "minHeight": "100vh"},
    children=[
        dcc.Store(id="job-id"),
        dcc.Store(id="run-ids", data=[]),
        dcc.Store(id="compare-store", data=[]),
        dcc.Store(id="auth-token", storage_type="session"),
        dcc.Interval(id="poll", interval=900, disabled=True),

        html.Div(style={"background":
                        "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                        "padding": "20px 32px", "color": "white"}, children=[
            html.H1("Alpha Lab", style={"margin": "0", "fontSize": "26px",
                                        "fontWeight": "600"}),
            html.P("Design a cross-sectional signal, walk it forward, "
                   "and read the diagnostics",
                   style={"margin": "4px 0 0 0", "opacity": "0.75",
                          "fontSize": "13px"}),
            html.Div(id="mode-banner", style={"marginTop": "10px"}),
        ]),

        html.Div(style={"display": "flex", "gap": "20px",
                        "padding": "20px 32px", "alignItems": "flex-start"},
                 children=[
            designer(),
            html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                html.Div(style={"display": "flex", "gap": "12px",
                                "alignItems": "center",
                                "marginBottom": "12px"}, children=[
                    html.Div("Run", style={"fontSize": "12px",
                                           "fontWeight": "600",
                                           "color": MUTED}),
                    html.Div(dcc.Dropdown(
                        id="run-select", clearable=False,
                        style={"fontSize": "13px"},
                        options=_INITIAL_RUNS,
                        value=_INITIAL_RUNS[0]["value"] if _INITIAL_RUNS else None,
                        placeholder="No runs yet"),
                        style={"flex": "1"}),
                ]),
                dcc.Tabs(id="tabs", value="overview", children=[
                    dcc.Tab(label="Overview", value="overview"),
                    dcc.Tab(label="Signal quality", value="signal"),
                    dcc.Tab(label="Stability", value="stability"),
                    dcc.Tab(label="Predictions", value="predictions"),
                    dcc.Tab(label="Compare runs", value="compare"),
                ]),
                html.Div(id="tab-content", style={"marginTop": "16px"}),
            ]),
        ]),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(Output("tickers", "value"), Input("universe", "value"),
          prevent_initial_call=True)
def fill_universe(universe):
    if universe == "custom":
        return no_update
    return ", ".join(lab.UNIVERSES[universe])


@callback(Output("mode-banner", "children"),
          Output("lock-panel", "children"),
          Output("run-btn", "disabled"),
          Output("run-btn", "style"),
          Input("auth-token", "data"))
def render_mode(token):
    unlocked = is_unlocked(token)
    if not LOCKED:
        banner = None
    elif unlocked:
        banner = html.Div("Editing enabled", style={
            "display": "inline-block", "fontSize": "12px",
            "backgroundColor": "rgba(42,157,143,0.25)", "color": "#8ee4d8",
            "padding": "4px 10px", "borderRadius": "999px"})
    else:
        banner = html.Div(
            "Read-only view – browse the saved backtests below. "
            "Running new tests requires the password.",
            style={"display": "inline-block", "fontSize": "12px",
                   "backgroundColor": "rgba(255,255,255,0.12)",
                   "padding": "4px 10px", "borderRadius": "999px"})
    return (banner, lock_panel(unlocked), not unlocked,
            _run_btn_style(unlocked))


@callback(Output("auth-token", "data"), Output("unlock-msg", "children"),
          Input("unlock-btn", "n_clicks"), Input("password", "n_submit"),
          State("password", "value"), prevent_initial_call=True)
def unlock(clicks, submits, password):
    if not (clicks or submits):       # fires once when the panel is rendered
        return no_update, ""
    if password_matches(password):
        return grant_session(), ""
    time.sleep(0.5)                      # slow down brute-force attempts
    return no_update, "Incorrect password."


@callback(Output("relevance-panel", "style"), Input("method", "value"))
def toggle_relevance(method):
    return {} if method == "relevance" else {"display": "none"}


@callback(Output("weights-panel", "children"),
          Input("method", "value"),
          Input({"type": "feature-group", "group": ALL}, "value"))
def render_weights(method, group_values):
    if method != "linear":
        return None
    features = [f for group in group_values or [] for f in group]
    if not features:
        return html.Div("Select features to set weights.",
                        style={"fontSize": "12px", "color": MUTED})
    rows = [
        html.Div(style={"display": "flex", "alignItems": "center",
                        "gap": "8px", "marginTop": "6px"}, children=[
            html.Div(lab.FEATURES[f].label,
                     style={"flex": "1", "fontSize": "12px"}),
            dcc.Input(id={"type": "weight", "feature": f}, type="number",
                      value=lab.FEATURES[f].sign, step=0.25,
                      style={**INPUT, "width": "80px"}),
        ]) for f in features
    ]
    return html.Div([html.Label("Feature weights", style=LABEL)] + rows + [
        html.Div("Signal = Σ weight × standardised feature.",
                 style={"fontSize": "11px", "color": MUTED,
                        "marginTop": "6px"})])


@callback(
    Output("job-id", "data"),
    Output("poll", "disabled"),
    Output("progress-text", "children", allow_duplicate=True),
    Input("run-btn", "n_clicks"),
    State("run-name", "value"), State("universe", "value"),
    State("tickers", "value"), State("from-date", "value"),
    State("to-date", "value"),
    State({"type": "feature-group", "group": ALL}, "value"),
    State("standardize", "value"), State("target", "value"),
    State("method", "value"),
    State({"type": "weight", "feature": ALL}, "value"),
    State({"type": "weight", "feature": ALL}, "id"),
    State("thresh", "value"), State("most", "value"), State("sweep", "value"),
    State("horizon", "value"), State("rebalance", "value"),
    State("train-window", "value"), State("min-train", "value"),
    State("max-obs", "value"), State("n-bins", "value"),
    State("n-perm", "value"), State("auth-token", "data"),
    prevent_initial_call=True,
)
def launch(n_clicks, name, universe, tickers_text, from_date, to_date,
           group_values, standardize, target, method, weight_values,
           weight_ids, thresh, most, sweep, horizon, rebalance, train_window,
           min_train, max_obs, n_bins, n_perm, token):
    if not n_clicks:
        return no_update, True, ""
    if not is_unlocked(token):
        return no_update, True, "Enter the password to run a backtest."
    if job_in_flight():
        return no_update, True, "Another backtest is already running."

    tickers = [t for t in
               str(tickers_text or "").replace("\n", ",").replace(" ", ",")
               .split(",") if t.strip()]
    if len(tickers) > MAX_TICKERS:
        return no_update, True, f"At most {MAX_TICKERS} tickers."

    features = [f for group in group_values or [] for f in group]
    weights = {wid["feature"]: float(v)
               for wid, v in zip(weight_ids or [], weight_values or [])
               if v is not None}

    base = dict(
        name=(name or "untitled").strip(), universe=universe,
        tickers=tickers, from_date=(from_date or "2010-01-01").strip(),
        to_date=(to_date or "").strip() or None,
        features=features, weights=weights, standardize=standardize,
        target=target, method=method, horizon=int(horizon or 21),
        rebalance=rebalance, most=(most == "most"),
        train_window=int(train_window or 36),
        min_training_periods=int(min_train or 12),
        max_training_obs=int(max_obs or 1500), n_bins=int(n_bins or 5),
    )

    threshes = [float(thresh)]
    if method == "relevance" and (sweep or "").strip():
        try:
            threshes = [float(v) for v in sweep.replace(" ", "").split(",") if v]
        except ValueError:
            return no_update, True, "Could not parse the sweep thresholds."
        if len(threshes) > MAX_SWEEP:
            return no_update, True, f"At most {MAX_SWEEP} sweep values."

    try:
        specs = [lab.SignalSpec(**{**base, "thresh": t}) for t in threshes]
    except ValueError as exc:
        return no_update, True, str(exc)

    job = start_job(specs, min(int(n_perm or 0), MAX_PERM))
    return job.id, False, "Starting…"


@callback(Output("job-id", "data", allow_duplicate=True),
          Input("cancel-btn", "n_clicks"), State("job-id", "data"),
          State("auth-token", "data"), prevent_initial_call=True)
def cancel(n_clicks, job_id, token):
    if not is_unlocked(token):
        return no_update
    job = JOBS.get(job_id or "")
    if job and job.status == "running":
        job.cancelled = True
    return no_update


@callback(
    Output("progress-bar", "style"),
    Output("progress-text", "children"),
    Output("poll", "disabled", allow_duplicate=True),
    Output("run-ids", "data"),
    Output("run-select", "options"),
    Output("run-select", "value"),
    Input("poll", "n_intervals"),
    State("job-id", "data"), State("run-select", "value"),
    prevent_initial_call=True,
)
def poll(_n, job_id, current_run):
    job = JOBS.get(job_id or "")
    if job is None:
        return no_update, "", True, no_update, no_update, no_update

    bar = {"height": "100%", "width": f"{job.fraction * 100:.0f}%",
           "backgroundColor": NEG if job.status == "error" else ACCENT,
           "transition": "width .3s"}

    if job.status == "running":
        return bar, job.stage, False, no_update, no_update, no_update

    bar["width"] = "100%"
    if job.status == "error":
        return bar, job.error, True, no_update, no_update, no_update

    options = []
    for rid in job.run_ids:
        try:
            _, _, meta = lab.load_run(rid)
            options.append({"label": f"{meta['name']} · {meta['label']}",
                            "value": rid})
        except (FileNotFoundError, OSError):
            continue
    value = (current_run if current_run in job.run_ids
             else (job.run_ids[0] if job.run_ids else None))
    return bar, job.stage, True, job.run_ids, options, value


@callback(Output("tab-content", "children"),
          Input("tabs", "value"), Input("run-select", "value"),
          State("run-ids", "data"), State("compare-store", "data"))
def render_tab(tab, run_id, run_ids, compare_selected):
    if tab == "compare":
        return compare_tab(compare_selected or [])

    if not run_id:
        runs = lab.list_runs()
        hint = ("Pick a saved run above or design a new one on the left."
                if not runs.empty else
                "Design a signal on the left and press “Run test”.")
        return html.Div(style={**CARD, "color": MUTED}, children=[
            html.H4("No run selected", style={"marginTop": "0", "color": INK}),
            html.Div(hint, style={"fontSize": "13px"}),
        ])

    try:
        report = lab.load_report(run_id)
    except Exception as exc:
        return html.Div(f"Could not load this run: {exc}",
                        style={**CARD, "color": NEG})

    body = {"overview": overview_tab, "signal": signal_tab,
            "stability": stability_tab,
            "predictions": predictions_tab}[tab](report)

    sweep = sweep_table(run_ids or []) if tab == "overview" else None
    return html.Div([sweep, body]) if sweep else body


@callback(Output("compare-store", "data"),
          Output("compare-graph", "figure"),
          Input("compare-select", "value"), prevent_initial_call=True)
def store_compare(selected):
    return selected or [], _compare_figure(selected or [])


@callback(Output("run-select", "options", allow_duplicate=True),
          Output("run-select", "value", allow_duplicate=True),
          Input("tabs", "value"), State("run-select", "options"),
          prevent_initial_call=True)
def seed_runs(_tab, options):
    if options:
        return no_update, no_update
    opts = _saved_run_options()
    if not opts:
        return no_update, no_update
    return opts, opts[0]["value"]


if __name__ == "__main__":
    app.run(debug=os.environ.get("ALPHA_LAB_DEBUG", "1") == "1",
            host=os.environ.get("ALPHA_LAB_HOST", "127.0.0.1"),
            port=int(os.environ.get("ALPHA_LAB_PORT", "8051")))
