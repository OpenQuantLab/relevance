"""Shared visual language for the Alpha Lab dashboard."""

from __future__ import annotations

import numpy as np

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


def _positive(x) -> bool | None:
    """True/False for colouring a metric, None when the value is missing."""
    if x is None or not np.isfinite(x):
        return None
    return bool(x > 0)
