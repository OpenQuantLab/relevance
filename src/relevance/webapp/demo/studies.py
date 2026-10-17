"""
Runners for the two published studies, plus the Dash components that present
their results.

Kept out of ``app.py`` so the dashboard module is layout and callback wiring
rather than analysis code.  ``card`` and ``ok_style`` are passed in so the
styling stays owned by the app.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dash_table, dcc, html

from relevance.data import eodhd as eodhd_data
from relevance.studies import paper as repl

# ---- Cross-sectional study ----

def _run_cross_sectional_study(symbols, exchange, token, from_date,
                               thresh, most, card, ok_style):
    attrs_df, outcomes_df, info = repl.build_cross_sectional_dataset(
        symbols, exchange, token, from_date,
    )

    loo = repl.run_loo_prediction(attrs_df, outcomes_df, thresh, most)
    full, obs_rel = repl.run_full_prediction(attrs_df, outcomes_df, thresh, most)
    quintiles = repl.compute_quintile_analysis(loo)

    status = html.Div(
        style=ok_style,
        children=[
            html.Strong("Cross-sectional study complete. "),
            f"{info['n_stocks']} stocks, {info['n_attributes']} attributes. ",
            f"Features: {', '.join(info['attribute_names'])}.",
        ],
    )

    results_div = _build_cross_sectional_results(loo, full, obs_rel, quintiles, card)

    return loo.reset_index().to_dict("records"), status, results_div


def _build_cross_sectional_results(loo, full, obs_rel, quintiles, card):
    children = []

    # --- LOO Scatter ---
    valid_loo = loo.dropna(subset=["predicted_return", "actual_return"])
    ic = valid_loo["predicted_return"].corr(valid_loo["actual_return"], method="spearman")

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=valid_loo["actual_return"] * 100,
        y=valid_loo["predicted_return"] * 100,
        mode="markers+text",
        text=valid_loo.index,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=10, color="#1a1a2e"),
    ))
    rng = max(
        abs(valid_loo["actual_return"].max()),
        abs(valid_loo["predicted_return"].max()),
    ) * 100 + 2
    fig_scatter.add_trace(go.Scatter(
        x=[-rng, rng], y=[-rng, rng],
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ))
    fig_scatter.update_layout(
        title=f"Leave-One-Out: Predicted vs Actual Returns  (IC = {ic:.3f})",
        xaxis_title="Actual Return (%)",
        yaxis_title="Predicted Return (%)",
        template="plotly_white", height=400, margin=dict(t=50, b=40),
    )
    children.append(html.Div(style=card, children=[dcc.Graph(figure=fig_scatter)]))

    # --- Quintile chart ---
    if quintiles is not None and not quintiles.empty:
        fig_q = go.Figure(go.Bar(
            x=quintiles.index.tolist(),
            y=(quintiles["mean"] * 100).tolist(),
            marker_color=["#e94560" if i < len(quintiles) // 2
                          else "#0f3460" for i in range(len(quintiles))],
            text=[f"{v:.2f}%" for v in quintiles["mean"] * 100],
            textposition="outside",
        ))
        fig_q.update_layout(
            title="Quintile Analysis: Avg Actual Return by Predicted-Return Quintile",
            yaxis_title="Avg Actual Return (%)",
            template="plotly_white", height=350, margin=dict(t=50, b=40),
        )
        children.append(html.Div(style=card, children=[dcc.Graph(figure=fig_q)]))

    # --- Relevance heatmap ---
    rel_matrix = obs_rel.get("relevance")
    if rel_matrix is not None:
        syms = list(full.index)
        fig_hm = go.Figure(data=go.Heatmap(
            z=rel_matrix, x=syms, y=syms,
            colorscale="RdBu", zmid=0,
            hovertemplate="Obs: %{y}<br>Pred: %{x}<br>Rel: %{z:.4f}<extra></extra>",
        ))
        fig_hm.update_layout(
            title="Relevance Matrix (Observation → Prediction)",
            template="plotly_white", height=500,
            margin=dict(t=50, b=40, l=100),
        )
        children.append(html.Div(style=card, children=[dcc.Graph(figure=fig_hm)]))

    # --- Detailed table ---
    display_loo = loo.reset_index().round(4)
    children.append(html.Div(
        style=card,
        children=[
            html.H4("Stock Rankings by Predicted Return", style={"marginTop": "0"}),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in display_loo.columns],
                data=display_loo.sort_values("predicted_return", ascending=False).to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "13px", "padding": "8px"},
                style_header={"fontWeight": "600", "backgroundColor": "#f1f3f5"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{predicted_return} > 0", "column_id": "predicted_return"},
                        "color": "#155724",
                    },
                    {
                        "if": {"filter_query": "{predicted_return} < 0", "column_id": "predicted_return"},
                        "color": "#721c24",
                    },
                ],
            ),
        ],
    ))

    return html.Div(children)


# ---- Backtest study ----

def _run_backtest_study(symbols, exchange, token, from_date, to_date,
                        fwd, thresh, most, card, ok_style):
    prices = eodhd_data.fetch_multiple(
        symbols, exchange=exchange, api_token=token,
        from_date=from_date or None, to_date=to_date or None,
    )

    panel = repl.build_backtest_panel(prices, forward_period=fwd)
    bt = repl.run_walk_forward(panel, thresh=thresh, most=most)
    metrics = repl.compute_period_metrics(bt)
    cum = repl.compute_cumulative_ls(bt)

    n_periods = len(metrics)
    avg_ic = metrics["ic"].mean() if n_periods else 0
    avg_hr = metrics["hit_rate"].mean() if n_periods else 0
    avg_spread = metrics["quintile_spread"].mean() if n_periods else 0
    n_tickers = bt["ticker"].nunique()

    status = html.Div(
        style=ok_style,
        children=[
            html.Strong("Walk-forward backtest complete. "),
            f"{n_tickers} stocks, {n_periods} monthly periods. "
            f"Avg IC: {avg_ic:.3f}, Avg Hit Rate: {avg_hr:.1%}, "
            f"Avg Quintile Spread: {avg_spread:.2%}.",
        ],
    )

    results_div = _build_backtest_results(bt, metrics, cum, card)

    return bt.to_dict("records"), status, results_div


def _build_backtest_results(bt, metrics, cum, card):
    children = []

    # --- Summary metrics ---
    avg_ic = metrics["ic"].mean()
    avg_hr = metrics["hit_rate"].mean()
    avg_spread = metrics["quintile_spread"].mean()
    total_ls = cum["cumulative_ls"].iloc[-1] if len(cum) > 0 else 0

    metric_style = {"textAlign": "center", "padding": "16px"}
    num_style = {"fontSize": "28px", "fontWeight": "700"}
    lbl_style = {"fontSize": "12px", "color": "#666"}

    children.append(html.Div(
        style={**card, "display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "16px"},
        children=[
            html.Div(style=metric_style, children=[
                html.Div(f"{avg_ic:.3f}", style=num_style),
                html.Div("Avg IC (Spearman)", style=lbl_style),
            ]),
            html.Div(style=metric_style, children=[
                html.Div(f"{avg_hr:.1%}", style=num_style),
                html.Div("Avg Hit Rate", style=lbl_style),
            ]),
            html.Div(style=metric_style, children=[
                html.Div(f"{avg_spread:.2%}", style=num_style),
                html.Div("Avg Quintile Spread", style=lbl_style),
            ]),
            html.Div(style=metric_style, children=[
                html.Div(
                    f"{total_ls:+.1%}",
                    style={**num_style, "color": "#155724" if total_ls > 0 else "#721c24"},
                ),
                html.Div("Cumulative Long-Short", style=lbl_style),
            ]),
        ],
    ))

    # --- Quintile chart (pooled across all periods) ---
    overall_q = repl.compute_quintile_analysis(bt, pred_col="predicted", actual_col="actual")
    if overall_q is not None and not overall_q.empty:
        fig_q = go.Figure(go.Bar(
            x=overall_q.index.tolist(),
            y=(overall_q["mean"] * 100).tolist(),
            marker_color=[
                "#e94560" if i < len(overall_q) // 2
                else "#0f3460" for i in range(len(overall_q))
            ],
            text=[f"{v:.2f}%" for v in overall_q["mean"] * 100],
            textposition="outside",
        ))
        fig_q.update_layout(
            title="Quintile Analysis: Avg Forward Return by Predicted-Return Quintile (All Periods)",
            yaxis_title="Avg Forward Return (%)",
            template="plotly_white", height=350, margin=dict(t=50, b=40),
        )
        children.append(html.Div(style=card, children=[dcc.Graph(figure=fig_q)]))

    # --- IC time series ---
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(
        x=metrics["date"], y=metrics["ic"],
        marker_color=[
            "#0f3460" if v > 0 else "#e94560" for v in metrics["ic"]
        ],
        name="IC",
    ))
    fig_ic.add_hline(y=avg_ic, line_dash="dash", line_color="gray",
                     annotation_text=f"Avg = {avg_ic:.3f}")
    fig_ic.update_layout(
        title="Information Coefficient (Spearman) Over Time",
        yaxis_title="IC", template="plotly_white",
        height=320, margin=dict(t=50, b=40),
    )
    children.append(html.Div(style=card, children=[dcc.Graph(figure=fig_ic)]))

    # --- Cumulative returns ---
    if len(cum) > 1:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum.index, y=cum["cumulative_long"] * 100,
            mode="lines", name="Long (Q5)",
            line=dict(color="#0f3460", width=2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum.index, y=cum["cumulative_short"] * 100,
            mode="lines", name="Short (Q1)",
            line=dict(color="#e94560", width=2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum.index, y=cum["cumulative_ls"] * 100,
            mode="lines", name="Long-Short",
            line=dict(color="#2ecc71", width=2.5),
        ))
        fig_cum.update_layout(
            title="Cumulative Portfolio Returns",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white", height=380,
            margin=dict(t=50, b=40),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        children.append(html.Div(style=card, children=[dcc.Graph(figure=fig_cum)]))

    # --- Hit rate time series ---
    fig_hr = go.Figure()
    fig_hr.add_trace(go.Scatter(
        x=metrics["date"], y=metrics["hit_rate"] * 100,
        mode="lines+markers", name="Hit Rate",
        line=dict(color="#1a1a2e", width=2),
        marker=dict(size=5),
    ))
    fig_hr.add_hline(y=50, line_dash="dash", line_color="gray",
                     annotation_text="50%")
    fig_hr.update_layout(
        title="Directional Hit Rate Over Time",
        yaxis_title="Hit Rate (%)", template="plotly_white",
        height=300, margin=dict(t=50, b=40),
    )
    children.append(html.Div(style=card, children=[dcc.Graph(figure=fig_hr)]))

    return html.Div(children)
