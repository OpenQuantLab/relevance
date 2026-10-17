"""Plotly figure builders for the Alpha Lab dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from relevance.webapp.alpha_lab.theme import (
    ACCENT,
    BLUE,
    INK,
    LINE,
    MUTED,
    NEG,
    PLOT_LAYOUT,
)


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
