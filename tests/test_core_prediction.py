"""Characterisation tests for the core relevance/prediction maths."""

from __future__ import annotations

import numpy as np
import pytest

import predictionrevisited as pr


def test_predict_df_snapshot(attributes_outcomes, snapshot):
    X, X_t, y = attributes_outcomes
    preds, obs_rel = pr.predict_df(
        X, X_t, y, thresh=0.75, most=True, pct_thresh=True, pairwise_fits=True
    )
    snapshot(
        "core_predict_df",
        {
            "columns": list(preds.columns),
            "predictions": preds,
            "relevance_shape": list(np.shape(obs_rel["relevance"])),
            "info_t": obs_rel["info_t"],
        },
    )


def test_predict_df_least_relevant_snapshot(attributes_outcomes, snapshot):
    X, X_t, y = attributes_outcomes
    preds, _ = pr.predict_df(
        X, X_t, y, thresh=0.75, most=False, pct_thresh=True, pairwise_fits=False
    )
    snapshot("core_predict_df_least", preds)


def test_linear_algebra_primitives_snapshot(attributes_outcomes, snapshot):
    X, _, _ = attributes_outcomes
    Xm = X.to_numpy()
    x_bar = pr.average(Xm)
    cov, cov_inv = pr.covariance(Xm)

    snapshot(
        "core_primitives",
        {
            "average": x_bar,
            "spread": pr.spread(Xm),
            "standard_deviation": pr.standard_deviation(Xm),
            "correlation": pr.correlation(Xm),
            "covariance": cov,
            "covariance_inverse": cov_inv,
            "mahalanobis_0_1": pr.mahalanobis(Xm[0], Xm[1], cov_inv, Xm),
            "similarity_0_1": pr.similarity(Xm[0], Xm[1], cov_inv, Xm),
            "informativeness_0": pr.informativeness(Xm[0], x_bar, cov_inv, Xm),
            "relevance_0_1": pr.relevance(Xm[0], Xm[1], x_bar, cov_inv, Xm),
        },
    )


@pytest.mark.parametrize("shortcut", [False, True])
def test_spread_shortcut_agrees(attributes_outcomes, shortcut):
    """The pairwise and shortcut formulations must agree — a paper invariant."""
    X, _, _ = attributes_outcomes
    Xm = X.to_numpy()
    assert np.allclose(pr.spread(Xm, shortcut=shortcut), pr.spread(Xm, shortcut=False))
