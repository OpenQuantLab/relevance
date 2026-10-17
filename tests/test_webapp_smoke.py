"""
Structural guards for the Dash dashboards.

The numerical snapshots do not cover the web layer, so these tests pin the
things that a refactor can silently break: the set of registered callbacks
and the component ids present in each layout.  No server is started and no
network call is made.
"""

from __future__ import annotations

import pytest


def _component_ids(node, found=None):
    """Collect every ``id`` in a Dash component tree."""
    if found is None:
        found = set()

    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    elif isinstance(cid, dict):  # pattern-matching id
        found.add("{" + ",".join(f"{k}={v}" for k, v in sorted(cid.items())) + "}")

    children = getattr(node, "children", None)
    if children is None:
        return found
    if isinstance(children, (list, tuple)):
        for child in children:
            _component_ids(child, found)
    else:
        _component_ids(children, found)
    return found


def _callback_signature(app):
    """Stable description of every registered callback."""
    out = []
    for key, spec in app.callback_map.items():
        inputs = sorted(
            f"{i['id']}.{i['property']}"
            for i in spec.get("inputs", [])
            if isinstance(i.get("id"), str)
        )
        states = sorted(
            f"{s['id']}.{s['property']}"
            for s in spec.get("state", [])
            if isinstance(s.get("id"), str)
        )
        out.append({"output": key, "inputs": inputs, "state": states})
    return sorted(out, key=lambda d: d["output"])


@pytest.fixture(scope="module")
def alpha_lab_app():
    from relevance.webapp.alpha_lab import app as module

    return module


@pytest.fixture(scope="module")
def demo_app():
    from relevance.webapp.demo import app as module

    return module


def test_alpha_lab_exposes_wsgi_server(alpha_lab_app):
    assert alpha_lab_app.server is alpha_lab_app.app.server


def test_demo_exposes_wsgi_server(demo_app):
    assert demo_app.server is demo_app.app.server


def test_alpha_lab_layout_ids(alpha_lab_app, snapshot):
    snapshot("webapp_alpha_lab_layout_ids",
             sorted(_component_ids(alpha_lab_app.app.layout)))


def test_demo_layout_ids(demo_app, snapshot):
    snapshot("webapp_demo_layout_ids", sorted(_component_ids(demo_app.app.layout)))


def test_alpha_lab_callbacks(alpha_lab_app, snapshot):
    snapshot("webapp_alpha_lab_callbacks", _callback_signature(alpha_lab_app.app))


def test_demo_callbacks(demo_app, snapshot):
    snapshot("webapp_demo_callbacks", _callback_signature(demo_app.app))


def test_alpha_lab_figure_builders(alpha_lab_app, prices, tickers):
    """The figure helpers must keep working once moved into their own module."""
    import relevance.backtest.lab as lab

    spec = lab.SignalSpec(
        name="smoke", universe="custom", tickers=list(tickers),
        from_date="2008-01-01",
        features=["return_21d", "return_63d"],
        method="linear", min_training_periods=8,
    )
    panel = lab.build_panel(prices, spec)
    results = lab.run_backtest(panel, spec)
    report = lab.evaluate(results, spec)

    figures = {
        "fig_equity": (report["ls"],),
        "fig_drawdown": (report["ls"],),
        "fig_ic": (report["ic"],),
        "fig_bins": (report["bins"],),
        "fig_rolling_sharpe": (report["ls"], spec.periods_per_year),
        "fig_yearly": (report["yearly"],),
        "fig_scatter": (results,),
        "fig_heatmap": (report["ls"],),
    }
    for name, args in figures.items():
        builder = getattr(alpha_lab_app, name)
        fig = builder(*args)
        assert fig is not None, f"{name} returned nothing"
