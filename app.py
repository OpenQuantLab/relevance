"""
Prediction Revisited – Interactive Dashboard
=============================================
Plotly Dash application for exploring the relevance-based prediction model.
"""

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State
import plotly.graph_objects as go
import plotly.express as px

import predictionrevisited as pr

# ---------------------------------------------------------------------------
# Default dataset (from the demo)
# ---------------------------------------------------------------------------

DEFAULT_OBSERVATIONS = pd.DataFrame(
    [
        ["London", 51.25, 0, 1.1],
        ["Tokyo", 35.75, 0, 6.3],
        ["Rio de Janeiro", -22.75, 0, 27.9],
        ["Buenos Aires", -34.75, 0, 24.8],
        ["London", 51.25, 1, 18.2],
        ["Tokyo", 35.75, 1, 28.0],
        ["Rio de Janeiro", -22.75, 1, 21.1],
        ["Buenos Aires", -34.75, 1, 10.0],
    ],
    columns=["city", "latitude", "january_or_july", "avg_temp_celsius"],
)

DEFAULT_PREDICTIONS = pd.DataFrame(
    [
        ["New York", 40.75, 0, 0.1],
        ["Sydney", -33.75, 0, 23.4],
        ["New York", 40.75, 1, 25.9],
        ["Sydney", -33.75, 1, 12.5],
    ],
    columns=["city", "latitude", "january_or_july", "avg_temp_celsius"],
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Prediction Revisited"

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
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8000)
