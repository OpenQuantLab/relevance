"""
Prediction Revisited – Interactive Dashboard
=============================================
Plotly Dash application for exploring the relevance-based prediction model.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html

from relevance.core import prediction as pr
from relevance.data import eodhd as eodhd_data
from relevance.studies import paper as repl
from relevance.webapp.demo.data import DEFAULT_OBSERVATIONS, DEFAULT_PREDICTIONS
from relevance.webapp.demo.studies import (
    _run_backtest_study,
    _run_cross_sectional_study,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    requests_pathname_prefix="/projects/relevance/",
    routes_pathname_prefix="/projects/relevance/",
)
app.title = "Prediction Revisited"
app.config.suppress_callback_exceptions = True

# WSGI entry point for gunicorn: gunicorn app:server
server = app.server

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"fontFamily": "'Inter', sans-serif", "backgroundColor": "#f8f9fa"},
    children=[
        # Header
        html.Div(
            style={
                "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                "padding": "24px 40px",
                "color": "white",
            },
            children=[
                html.H1(
                    "Prediction Revisited",
                    style={"margin": "0", "fontSize": "28px", "fontWeight": "600"},
                ),
                html.P(
                    "Interactive exploration of relevance-based statistical prediction",
                    style={"margin": "4px 0 0 0", "opacity": "0.8", "fontSize": "14px"},
                ),
            ],
        ),
        # Main content
        html.Div(
            style={"display": "flex", "gap": "24px", "padding": "24px 40px"},
            children=[
                # Sidebar – controls
                html.Div(
                    style={
                        "width": "300px",
                        "flexShrink": "0",
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "padding": "24px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.H3("Parameters", style={"marginTop": "0"}),
                        html.Label("Relevance Threshold", style={"fontWeight": "500", "fontSize": "13px"}),
                        dcc.Slider(
                            id="thresh-slider",
                            min=0.1,
                            max=0.95,
                            step=0.05,
                            value=0.75,
                            marks={i / 10: f"{i / 10:.1f}" for i in range(1, 10)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(style={"height": "16px"}),
                        html.Label("Predict from", style={"fontWeight": "500", "fontSize": "13px"}),
                        dcc.RadioItems(
                            id="most-radio",
                            options=[
                                {"label": " Most relevant", "value": "most"},
                                {"label": " Least relevant", "value": "least"},
                            ],
                            value="most",
                            style={"fontSize": "13px"},
                        ),
                        html.Div(style={"height": "16px"}),
                        html.Label("Threshold type", style={"fontWeight": "500", "fontSize": "13px"}),
                        dcc.RadioItems(
                            id="pct-thresh-radio",
                            options=[
                                {"label": " Percentile", "value": "pct"},
                                {"label": " Raw value", "value": "raw"},
                            ],
                            value="pct",
                            style={"fontSize": "13px"},
                        ),
                        html.Div(style={"height": "16px"}),
                        dcc.Checklist(
                            id="pairwise-check",
                            options=[{"label": " Compute pairwise fits", "value": "pairwise"}],
                            value=["pairwise"],
                            style={"fontSize": "13px"},
                        ),
                        html.Div(style={"height": "16px"}),
                        dcc.Checklist(
                            id="binary-check",
                            options=[{"label": " Binary outcome mode", "value": "binary"}],
                            value=[],
                            style={"fontSize": "13px"},
                        ),
                        html.Div(style={"height": "24px"}),
                        html.Button(
                            "Run Prediction",
                            id="run-btn",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "backgroundColor": "#1a1a2e",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "8px",
                                "fontSize": "14px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                ),
                # Main panel – tabs
                html.Div(
                    style={"flex": "1", "minWidth": "0"},
                    children=[
                        dcc.Tabs(
                            id="main-tabs",
                            value="tab-data",
                            children=[
                                dcc.Tab(label="Data", value="tab-data"),
                                dcc.Tab(label="Market Data", value="tab-market"),
                                dcc.Tab(label="Paper Replication", value="tab-paper"),
                                dcc.Tab(label="Predictions", value="tab-predictions"),
                                dcc.Tab(label="Relevance", value="tab-relevance"),
                                dcc.Tab(label="Diagnostics", value="tab-diagnostics"),
                            ],
                            style={"marginBottom": "16px"},
                        ),
                        html.Div(id="tab-content"),
                    ],
                ),
            ],
        ),
        # Hidden stores
        dcc.Store(id="predictions-store"),
        dcc.Store(id="relevance-store"),
        dcc.Store(id="obs-data-store", data=DEFAULT_OBSERVATIONS.to_dict("records")),
        dcc.Store(id="pred-data-store", data=DEFAULT_PREDICTIONS.to_dict("records")),
        dcc.Store(id="market-predictions-store"),
        dcc.Store(id="market-relevance-store"),
        dcc.Store(id="market-info-store"),
        dcc.Store(id="paper-results-store"),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("predictions-store", "data"),
    Output("relevance-store", "data"),
    Input("run-btn", "n_clicks"),
    State("obs-data-store", "data"),
    State("pred-data-store", "data"),
    State("thresh-slider", "value"),
    State("most-radio", "value"),
    State("pct-thresh-radio", "value"),
    State("pairwise-check", "value"),
    State("binary-check", "value"),
    prevent_initial_call=False,
)
def run_prediction(n_clicks, obs_records, pred_records, thresh, most_val, pct_val, pairwise_val, binary_val):
    obs_df = pd.DataFrame(obs_records)
    pred_df = pd.DataFrame(pred_records)

    attributes_df = obs_df[["latitude", "january_or_july"]]
    outcomes_df = obs_df[["avg_temp_celsius"]]
    circumstances_df = pred_df[["latitude", "january_or_july"]]

    most = most_val == "most"
    pct_thresh = pct_val == "pct"
    pairwise_fits = "pairwise" in (pairwise_val or [])
    predict_binary = "binary" in (binary_val or [])

    predictions_result, obs_relevance = pr.predict_df(
        attributes_df,
        circumstances_df,
        outcomes_df,
        thresh=thresh,
        most=most,
        pct_thresh=pct_thresh,
        pairwise_fits=pairwise_fits,
        predict_binary_outcome=predict_binary,
    )

    predictions_result["city"] = pred_df["city"].values
    predictions_result["actual"] = pred_df["avg_temp_celsius"].values
    predictions_result["error"] = predictions_result["yhat"] - predictions_result["actual"]

    relevance_matrix = obs_relevance["relevance"]
    obs_labels = [
        f"{row['city']}, {'Jan' if row['january_or_july'] == 0 else 'Jul'}"
        for _, row in obs_df.iterrows()
    ]
    pred_labels = [
        f"{row['city']}, {'Jan' if row['january_or_july'] == 0 else 'Jul'}"
        for _, row in pred_df.iterrows()
    ]

    rel_data = {
        "matrix": relevance_matrix.tolist(),
        "obs_labels": obs_labels,
        "pred_labels": pred_labels,
    }

    return predictions_result.to_dict("records"), rel_data


@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("predictions-store", "data"),
    Input("relevance-store", "data"),
    State("obs-data-store", "data"),
    State("pred-data-store", "data"),
)
def render_tab(tab, predictions_data, relevance_data, obs_records, pred_records):
    if tab == "tab-data":
        return render_data_tab(obs_records, pred_records)
    elif tab == "tab-market":
        return render_market_tab()
    elif tab == "tab-paper":
        return render_paper_tab()
    elif tab == "tab-predictions":
        return render_predictions_tab(predictions_data)
    elif tab == "tab-relevance":
        return render_relevance_tab(relevance_data)
    elif tab == "tab-diagnostics":
        return render_diagnostics_tab(predictions_data)
    return html.Div()


def render_data_tab(obs_records, pred_records):
    obs_df = pd.DataFrame(obs_records)
    pred_df = pd.DataFrame(pred_records)

    card_style = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }

    return html.Div(
        [
            html.Div(
                style=card_style,
                children=[
                    html.H4("Prior Observations (Training Data)", style={"marginTop": "0"}),
                    dash_table.DataTable(
                        id="obs-table",
                        columns=[{"name": c, "id": c, "editable": True} for c in obs_df.columns],
                        data=obs_records,
                        editable=True,
                        row_deletable=True,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "13px", "padding": "8px"},
                        style_header={"fontWeight": "600", "backgroundColor": "#f1f3f5"},
                    ),
                ],
            ),
            html.Div(
                style=card_style,
                children=[
                    html.H4("Prediction Circumstances (with Actuals)", style={"marginTop": "0"}),
                    dash_table.DataTable(
                        id="pred-table",
                        columns=[{"name": c, "id": c, "editable": True} for c in pred_df.columns],
                        data=pred_records,
                        editable=True,
                        row_deletable=True,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "13px", "padding": "8px"},
                        style_header={"fontWeight": "600", "backgroundColor": "#f1f3f5"},
                    ),
                ],
            ),
        ]
    )


def render_predictions_tab(predictions_data):
    card_style = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }

    if not predictions_data:
        return html.Div(style=card_style, children=[html.P("Click 'Run Prediction' to see results.")])

    df = pd.DataFrame(predictions_data)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Predicted",
            x=df["city"],
            y=df["yhat"],
            marker_color="#1a1a2e",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Actual",
            x=df["city"],
            y=df["actual"],
            marker_color="#e94560",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Predicted vs Actual",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        margin=dict(t=50, b=40),
        height=350,
    )

    display_cols = ["city", "yhat", "actual", "error", "fit", "agreement", "outlier_influence", "precision"]
    display_df = df[[c for c in display_cols if c in df.columns]].round(4)

    return html.Div(
        [
            html.Div(style=card_style, children=[dcc.Graph(figure=fig)]),
            html.Div(
                style=card_style,
                children=[
                    html.H4("Prediction Details", style={"marginTop": "0"}),
                    dash_table.DataTable(
                        columns=[{"name": c, "id": c} for c in display_df.columns],
                        data=display_df.to_dict("records"),
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "13px", "padding": "8px"},
                        style_header={"fontWeight": "600", "backgroundColor": "#f1f3f5"},
                    ),
                ],
            ),
        ]
    )


def render_relevance_tab(relevance_data):
    card_style = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }

    if not relevance_data:
        return html.Div(style=card_style, children=[html.P("Click 'Run Prediction' to see results.")])

    matrix = np.array(relevance_data["matrix"])
    obs_labels = relevance_data["obs_labels"]
    pred_labels = relevance_data["pred_labels"]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=pred_labels,
            y=obs_labels,
            colorscale="RdBu",
            zmid=0,
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            hovertemplate="Observation: %{y}<br>Prediction: %{x}<br>Relevance: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Relevance of Each Observation to Each Prediction Task",
        xaxis_title="Prediction Circumstance",
        yaxis_title="Prior Observation",
        template="plotly_white",
        height=420,
        margin=dict(t=50, b=40, l=160),
    )

    return html.Div(
        [
            html.Div(style=card_style, children=[dcc.Graph(figure=fig)]),
        ]
    )


def render_diagnostics_tab(predictions_data):
    card_style = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }

    if not predictions_data:
        return html.Div(style=card_style, children=[html.P("Click 'Run Prediction' to see results.")])

    df = pd.DataFrame(predictions_data)

    # Fit breakdown chart
    fig_fit = go.Figure()
    if "agreement" in df.columns and "outlier_influence" in df.columns:
        fig_fit.add_trace(
            go.Bar(name="Agreement", x=df["city"], y=df["agreement"], marker_color="#0f3460")
        )
        fig_fit.add_trace(
            go.Bar(name="Outlier Influence", x=df["city"], y=df["outlier_influence"], marker_color="#e94560")
        )
        fig_fit.update_layout(
            barmode="stack",
            title="Fit Decomposition (Agreement + Outlier Influence)",
            yaxis_title="Contribution to Fit",
            template="plotly_white",
            height=320,
            margin=dict(t=50, b=40),
        )

    # Error scatter
    fig_error = go.Figure()
    fig_error.add_trace(
        go.Scatter(
            x=df["actual"],
            y=df["yhat"],
            mode="markers+text",
            text=df["city"],
            textposition="top center",
            marker=dict(size=12, color="#1a1a2e"),
        )
    )
    min_val = min(df["actual"].min(), df["yhat"].min()) - 2
    max_val = max(df["actual"].max(), df["yhat"].max()) + 2
    fig_error.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        )
    )
    fig_error.update_layout(
        title="Predicted vs Actual (Scatter)",
        xaxis_title="Actual Temperature (°C)",
        yaxis_title="Predicted Temperature (°C)",
        template="plotly_white",
        height=380,
        margin=dict(t=50, b=40),
    )

    return html.Div(
        [
            html.Div(style=card_style, children=[dcc.Graph(figure=fig_error)]),
            html.Div(style=card_style, children=[dcc.Graph(figure=fig_fit)]),
        ]
    )


# Sync editable tables back to stores
@callback(
    Output("obs-data-store", "data"),
    Input("obs-table", "data"),
    prevent_initial_call=True,
)
def update_obs_store(rows):
    return rows


@callback(
    Output("pred-data-store", "data"),
    Input("pred-table", "data"),
    prevent_initial_call=True,
)
def update_pred_store(rows):
    return rows


# ---------------------------------------------------------------------------
# Market Data tab
# ---------------------------------------------------------------------------

def render_market_tab():
    card_style = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }
    input_style = {
        "width": "100%",
        "padding": "8px 12px",
        "border": "1px solid #dee2e6",
        "borderRadius": "6px",
        "fontSize": "13px",
    }

    return html.Div(
        [
            html.Div(
                style=card_style,
                children=[
                    html.H4("Fetch Market Data from EODHD", style={"marginTop": "0"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                        children=[
                            html.Div([
                                html.Label("EODHD API Token", style={"fontWeight": "500", "fontSize": "13px"}),
                                dcc.Input(
                                    id="eodhd-api-token",
                                    type="password",
                                    value=eodhd_data.DEFAULT_API_TOKEN,
                                    placeholder="Your API token",
                                    style=input_style,
                                ),
                            ]),
                            html.Div([
                                html.Label("Exchange", style={"fontWeight": "500", "fontSize": "13px"}),
                                dcc.Input(
                                    id="eodhd-exchange",
                                    type="text",
                                    value="US",
                                    placeholder="US, LSE, TSE, ...",
                                    style=input_style,
                                ),
                            ]),
                        ],
                    ),
                    html.Div(style={"height": "12px"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px"},
                        children=[
                            html.Div([
                                html.Label(
                                    "Feature Tickers (comma-separated)",
                                    style={"fontWeight": "500", "fontSize": "13px"},
                                ),
                                dcc.Input(
                                    id="eodhd-feature-tickers",
                                    type="text",
                                    value="AAPL, MSFT, AMZN, GOOGL, SPY",
                                    placeholder="AAPL, MSFT, SPY",
                                    style=input_style,
                                ),
                            ]),
                            html.Div([
                                html.Label(
                                    "Target Ticker",
                                    style={"fontWeight": "500", "fontSize": "13px"},
                                ),
                                dcc.Input(
                                    id="eodhd-target-ticker",
                                    type="text",
                                    value="AAPL",
                                    placeholder="AAPL",
                                    style=input_style,
                                ),
                            ]),
                        ],
                    ),
                    html.Div(style={"height": "12px"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"},
                        children=[
                            html.Div([
                                html.Label("From Date", style={"fontWeight": "500", "fontSize": "13px"}),
                                dcc.Input(
                                    id="eodhd-from-date",
                                    type="text",
                                    value="2023-01-01",
                                    placeholder="YYYY-MM-DD",
                                    style=input_style,
                                ),
                            ]),
                            html.Div([
                                html.Label("To Date", style={"fontWeight": "500", "fontSize": "13px"}),
                                dcc.Input(
                                    id="eodhd-to-date",
                                    type="text",
                                    value="",
                                    placeholder="YYYY-MM-DD (blank = today)",
                                    style=input_style,
                                ),
                            ]),
                            html.Div([
                                html.Label(
                                    "Prediction Horizon (days)",
                                    style={"fontWeight": "500", "fontSize": "13px"},
                                ),
                                dcc.Input(
                                    id="eodhd-horizon",
                                    type="number",
                                    value=21,
                                    min=1,
                                    max=252,
                                    style=input_style,
                                ),
                            ]),
                        ],
                    ),
                    html.Div(style={"height": "12px"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                        children=[
                            html.Div([
                                html.Label("Feature Types", style={"fontWeight": "500", "fontSize": "13px"}),
                                dcc.Checklist(
                                    id="eodhd-feature-types",
                                    options=[
                                        {"label": f" {v}", "value": k}
                                        for k, v in eodhd_data.AVAILABLE_FEATURE_TYPES.items()
                                    ],
                                    value=["return", "vol"],
                                    style={"fontSize": "13px"},
                                ),
                            ]),
                            html.Div([
                                html.Label("Lookback Windows (days)", style={"fontWeight": "500", "fontSize": "13px"}),
                                dcc.Checklist(
                                    id="eodhd-windows",
                                    options=[
                                        {"label": f" {w}d", "value": w}
                                        for w in eodhd_data.AVAILABLE_WINDOWS
                                    ],
                                    value=[21],
                                    style={"fontSize": "13px"},
                                ),
                            ]),
                        ],
                    ),
                    html.Div(style={"height": "16px"}),
                    html.Button(
                        "Fetch & Predict",
                        id="eodhd-fetch-btn",
                        n_clicks=0,
                        style={
                            "padding": "10px 24px",
                            "backgroundColor": "#0f3460",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),
            # Status / error messages
            html.Div(id="eodhd-status"),
            # Results area
            html.Div(id="eodhd-results"),
        ]
    )


@callback(
    Output("market-predictions-store", "data"),
    Output("market-relevance-store", "data"),
    Output("market-info-store", "data"),
    Output("eodhd-status", "children"),
    Output("eodhd-results", "children"),
    Input("eodhd-fetch-btn", "n_clicks"),
    State("eodhd-api-token", "value"),
    State("eodhd-exchange", "value"),
    State("eodhd-feature-tickers", "value"),
    State("eodhd-target-ticker", "value"),
    State("eodhd-from-date", "value"),
    State("eodhd-to-date", "value"),
    State("eodhd-horizon", "value"),
    State("eodhd-feature-types", "value"),
    State("eodhd-windows", "value"),
    State("thresh-slider", "value"),
    State("most-radio", "value"),
    State("pct-thresh-radio", "value"),
    State("pairwise-check", "value"),
    prevent_initial_call=True,
)
def fetch_and_predict_market(
    n_clicks, api_token, exchange, feature_tickers_str, target_ticker,
    from_date, to_date, horizon, feature_types, windows,
    thresh, most_val, pct_val, pairwise_val,
):
    card_style = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }
    error_style = {
        "backgroundColor": "#fff3cd",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "marginBottom": "16px",
        "color": "#856404",
        "fontSize": "13px",
    }

    if not n_clicks:
        return None, None, None, html.Div(), html.Div()

    # Parse tickers
    feature_tickers = [t.strip().upper() for t in feature_tickers_str.split(",") if t.strip()]
    target_ticker = target_ticker.strip().upper()
    if target_ticker not in feature_tickers:
        feature_tickers.append(target_ticker)

    # Resolve token
    token = api_token or None
    exchange = (exchange or "US").strip().upper()
    horizon = int(horizon or 21)

    try:
        prices = eodhd_data.fetch_multiple(
            feature_tickers, exchange=exchange, api_token=token,
            from_date=from_date or None, to_date=to_date or None,
        )

        attributes_df, outcomes_df, circumstances_df, info = eodhd_data.build_prediction_dataset(
            prices, target_ticker, target_horizon=horizon,
            feature_tickers=feature_tickers,
            windows=windows or [21],
            feature_types=feature_types or ["return", "vol"],
        )

        most = most_val == "most"
        pct_thresh = pct_val == "pct"
        pairwise_fits = "pairwise" in (pairwise_val or [])

        predictions_result, obs_relevance = pr.predict_df(
            attributes_df,
            circumstances_df,
            outcomes_df,
            thresh=thresh,
            most=most,
            pct_thresh=pct_thresh,
            pairwise_fits=pairwise_fits,
        )

        yhat = float(predictions_result["yhat"].iloc[0])
        fit_val = float(predictions_result["fit"].iloc[0])

        status = html.Div(
            style={
                "backgroundColor": "#d4edda",
                "borderRadius": "8px",
                "padding": "12px 16px",
                "marginBottom": "16px",
                "color": "#155724",
                "fontSize": "13px",
            },
            children=[
                html.Strong("Success. "),
                f"Loaded {info['n_observations']} observations with "
                f"{info['n_features']} features "
                f"({', '.join(info['feature_names'])}). "
                f"Date range: {info['dates']['obs_start']} to {info['dates']['obs_end']}.",
            ],
        )

        # Build results display
        pred_pct = yhat * 100
        direction = "UP" if yhat > 0 else "DOWN"
        direction_color = "#155724" if yhat > 0 else "#721c24"

        # Price history chart
        price_fig = go.Figure()
        price_fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices[target_ticker],
                mode="lines", name=target_ticker,
                line=dict(color="#1a1a2e", width=2),
            )
        )
        price_fig.update_layout(
            title=f"{target_ticker} Price History",
            yaxis_title="Adjusted Close",
            template="plotly_white",
            height=320, margin=dict(t=50, b=40),
        )

        # Feature importance: show top relevance contributors
        results_div = html.Div(
            [
                html.Div(
                    style=card_style,
                    children=[
                        html.H4("Prediction Result", style={"marginTop": "0"}),
                        html.Div(
                            style={"display": "flex", "gap": "40px", "alignItems": "center"},
                            children=[
                                html.Div([
                                    html.Div("Predicted Forward Return", style={"fontSize": "12px", "color": "#666"}),
                                    html.Div(
                                        f"{pred_pct:+.2f}%",
                                        style={"fontSize": "32px", "fontWeight": "700", "color": direction_color},
                                    ),
                                ]),
                                html.Div([
                                    html.Div("Direction", style={"fontSize": "12px", "color": "#666"}),
                                    html.Div(
                                        direction,
                                        style={"fontSize": "24px", "fontWeight": "600", "color": direction_color},
                                    ),
                                ]),
                                html.Div([
                                    html.Div("Prediction Fit", style={"fontSize": "12px", "color": "#666"}),
                                    html.Div(
                                        f"{fit_val:.4f}",
                                        style={"fontSize": "24px", "fontWeight": "600"},
                                    ),
                                ]),
                                html.Div([
                                    html.Div("Horizon", style={"fontSize": "12px", "color": "#666"}),
                                    html.Div(
                                        f"{horizon} days",
                                        style={"fontSize": "24px", "fontWeight": "600"},
                                    ),
                                ]),
                                html.Div([
                                    html.Div("Prediction Date", style={"fontSize": "12px", "color": "#666"}),
                                    html.Div(
                                        info["dates"]["prediction_date"],
                                        style={"fontSize": "18px", "fontWeight": "600"},
                                    ),
                                ]),
                            ],
                        ),
                    ],
                ),
                html.Div(style=card_style, children=[dcc.Graph(figure=price_fig)]),
                html.Div(
                    style=card_style,
                    children=[
                        html.H4("Prediction Details", style={"marginTop": "0"}),
                        dash_table.DataTable(
                            columns=[{"name": c, "id": c} for c in predictions_result.columns],
                            data=predictions_result.round(6).to_dict("records"),
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": "13px", "padding": "8px"},
                            style_header={"fontWeight": "600", "backgroundColor": "#f1f3f5"},
                        ),
                    ],
                ),
            ]
        )

        return (
            predictions_result.to_dict("records"),
            {"relevance": obs_relevance["relevance"].tolist()} if obs_relevance else None,
            info,
            status,
            results_div,
        )

    except Exception as e:
        return (
            None, None, None,
            html.Div(style=error_style, children=[html.Strong("Error: "), str(e)]),
            html.Div(),
        )


# ---------------------------------------------------------------------------
# Paper Replication tab
# ---------------------------------------------------------------------------

def render_paper_tab():
    card_style = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }
    input_style = {
        "width": "100%",
        "padding": "8px 12px",
        "border": "1px solid #dee2e6",
        "borderRadius": "6px",
        "fontSize": "13px",
    }
    label_style = {"fontWeight": "500", "fontSize": "13px"}

    return html.Div(
        [
            # --- Configuration card ---
            html.Div(
                style=card_style,
                children=[
                    html.H4(
                        "Relevance-Based Prediction – Paper Replication",
                        style={"marginTop": "0"},
                    ),
                    html.P(
                        "Czasonis, Kritzman & Turkington (JFDS 2023). "
                        "Cross-sectional stock return prediction using "
                        "relevance-weighted partial-sample regression.",
                        style={"fontSize": "13px", "color": "#666", "marginBottom": "16px"},
                    ),
                    # Study type
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                        children=[
                            html.Div([
                                html.Label("Study Type", style=label_style),
                                dcc.RadioItems(
                                    id="paper-study-type",
                                    options=[
                                        {"label": " Cross-Sectional (Fundamentals + Price)",
                                         "value": "cross_sectional"},
                                        {"label": " Walk-Forward Backtest (Price Only)",
                                         "value": "backtest"},
                                    ],
                                    value="backtest",
                                    style={"fontSize": "13px"},
                                ),
                            ]),
                            html.Div([
                                html.Label("EODHD API Token", style=label_style),
                                dcc.Input(
                                    id="paper-api-token",
                                    type="password",
                                    value=eodhd_data.DEFAULT_API_TOKEN,
                                    placeholder="Your API token",
                                    style=input_style,
                                ),
                            ]),
                        ],
                    ),
                    html.Div(style={"height": "12px"}),
                    # Tickers
                    html.Div([
                        html.Label("Stock Universe (comma-separated)", style=label_style),
                        dcc.Input(
                            id="paper-tickers",
                            type="text",
                            value=", ".join(repl.DEFAULT_STOCKS),
                            style={**input_style, "width": "100%"},
                        ),
                    ]),
                    html.Div(style={"height": "12px"}),
                    # Date range & parameters
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "16px"},
                        children=[
                            html.Div([
                                html.Label("From Date", style=label_style),
                                dcc.Input(
                                    id="paper-from-date",
                                    type="text",
                                    value="2022-01-01",
                                    placeholder="YYYY-MM-DD",
                                    style=input_style,
                                ),
                            ]),
                            html.Div([
                                html.Label("To Date", style=label_style),
                                dcc.Input(
                                    id="paper-to-date",
                                    type="text",
                                    value="",
                                    placeholder="YYYY-MM-DD (blank = today)",
                                    style=input_style,
                                ),
                            ]),
                            html.Div([
                                html.Label("Exchange", style=label_style),
                                dcc.Input(
                                    id="paper-exchange",
                                    type="text",
                                    value="US",
                                    style=input_style,
                                ),
                            ]),
                            html.Div([
                                html.Label("Forward Period (days)", style=label_style),
                                dcc.Input(
                                    id="paper-forward-period",
                                    type="number",
                                    value=21,
                                    min=5, max=126,
                                    style=input_style,
                                ),
                            ]),
                        ],
                    ),
                    html.Div(style={"height": "16px"}),
                    html.Button(
                        "Run Study",
                        id="paper-run-btn",
                        n_clicks=0,
                        style={
                            "padding": "10px 24px",
                            "backgroundColor": "#0f3460",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "cursor": "pointer",
                        },
                    ),
                    html.Span(
                        "  Backtest may take 30–60 s depending on universe size.",
                        style={"fontSize": "12px", "color": "#888", "marginLeft": "12px"},
                    ),
                ],
            ),
            # Status & results (filled by callback)
            html.Div(id="paper-status"),
            html.Div(id="paper-results"),
        ]
    )


@callback(
    Output("paper-results-store", "data"),
    Output("paper-status", "children"),
    Output("paper-results", "children"),
    Input("paper-run-btn", "n_clicks"),
    State("paper-study-type", "value"),
    State("paper-api-token", "value"),
    State("paper-tickers", "value"),
    State("paper-from-date", "value"),
    State("paper-to-date", "value"),
    State("paper-exchange", "value"),
    State("paper-forward-period", "value"),
    State("thresh-slider", "value"),
    State("most-radio", "value"),
    prevent_initial_call=True,
)
def run_paper_study(
    n_clicks, study_type, api_token, tickers_str,
    from_date, to_date, exchange, forward_period,
    thresh, most_val,
):
    card = {
        "backgroundColor": "white",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    }
    err_style = {
        "backgroundColor": "#fff3cd",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "marginBottom": "16px",
        "color": "#856404",
        "fontSize": "13px",
    }
    ok_style = {
        "backgroundColor": "#d4edda",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "marginBottom": "16px",
        "color": "#155724",
        "fontSize": "13px",
    }

    if not n_clicks:
        return None, html.Div(), html.Div()

    symbols = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    token = api_token or None
    exchange = (exchange or "US").strip().upper()
    most = most_val == "most"
    fwd = int(forward_period or 21)

    try:
        if study_type == "cross_sectional":
            return _run_cross_sectional_study(
                symbols, exchange, token, from_date,
                thresh, most, card, ok_style,
            )
        else:
            return _run_backtest_study(
                symbols, exchange, token, from_date, to_date,
                fwd, thresh, most, card, ok_style,
            )
    except Exception as e:
        return (
            None,
            html.Div(style=err_style, children=[html.Strong("Error: "), str(e)]),
            html.Div(),
        )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
