"""
Prediction diagnostics: fit, precision and reliability.

These describe how well a prediction is supported by the observations that
informed it.  They depend only on the primitives in
:mod:`relevance.core.linalg`, so :mod:`relevance.core.prediction` can import
them without creating a cycle.
"""

import numpy as np

from relevance.core.linalg import (
    average,
    informativeness,
    spread,
    standard_deviation,
)


def fit(pred_details, shortcut=False, yhat=False):
    """
    FIT:
    Computes the fit of a single prediction.

    Fit may be computed as pairwise alignment, or more efficiently as a ratio
    of informativeness. However, the latter does not provide agreement or
    outlier influence.

    Parameters
    ----------
    pred_details : dict
        Calculation is based on the detailed output from the predict function.
    shortcut : bool, optional(default=False)
        Compute fit pairwise if True, use ratio of informativeness if False.
    yhat : ndarray [1-by-1], optional(default=False)
        Prediction is a required input for the 'shortcut' calculation.

    Returns
    -------
    f : ndarray [1-by-1]
        The fit of this individual prediction.
    fit_details: dict
        Includes agreement and outlier_influence subcomponents of fit.
    """
    # unpack prediction details that are needed
    y = pred_details.get('y')
    rel = pred_details.get('relevance')
    include = pred_details.get('include')
    lambda_sq = pred_details.get('lambda_sq')
    info_t = pred_details.get('info_t')
    K = pred_details.get('K')

    if shortcut is True:
        if yhat is False:
            raise ValueError('yhat is a required input when shortcut is used.')
        y_bar = average(y)
        spread_y = spread(y)
        if np.any(spread_y == 0):
            f = np.array([[np.nan]])
            fit_details = {'agreement': np.nan, 'outlier_influence': np.nan,
                           'fit_scaled': np.nan, 'agreement_scaled': np.nan,
                           'outlier_influence_scaled': np.nan}
            return f, fit_details
        info_y = informativeness(yhat, y_bar, 1/spread_y)
        info_x_t = info_t[0]
        f = info_y / info_x_t if info_x_t != 0 else np.array([[np.nan]])
        agreement = np.nan
        outlier_influence = np.nan

    else:

        y_keep = np.transpose(np.atleast_2d(np.extract(include, y)))
        rel_keep = np.transpose(np.atleast_2d(np.extract(include, rel)))
        n = np.sum(include)

        # standardize y and relevance to z-scores
        stdev_y = standard_deviation(y)
        stdev_rel = standard_deviation(rel)
        if np.any(stdev_y == 0) or np.any(stdev_rel == 0):
            f = np.array([[np.nan]])
            fit_details = {'agreement': np.nan, 'outlier_influence': np.nan,
                           'fit_scaled': np.nan, 'agreement_scaled': np.nan,
                           'outlier_influence_scaled': np.nan}
            return f, fit_details
        z_y = (y_keep - average(y)) / stdev_y
        z_r = (rel_keep - average(rel)) / stdev_rel

        # initialize agreement and outlier_influence, to sum incrementally
        agreement = 0
        outlier_influence = 0

        # loop through all pairs
        for i in range(n):
            for j in range(n):
                to_add = (lambda_sq / (n-1))**2 * z_r[i] * z_r[j] * z_y[i] * z_y[j]
                if i == j:
                    outlier_influence = outlier_influence + to_add
                else:
                    agreement = agreement + to_add

        f = agreement + outlier_influence

    # compute a scaled version of fit which is in units similar to R-squared
    # (because it equates to R-squared when summed with weights that sum to 1)
    fit_scaled = f * K
    if bool(np.isnan(agreement)) is False:
        agreement_scaled = agreement * K
    else:
        agreement_scaled = np.nan
    if bool(np.isnan(outlier_influence)) is False:
        outlier_influence_scaled = outlier_influence * K
    else:
        outlier_influence_scaled = np.nan

    # store and pass detailed outputs
    fit_details = {
        'agreement': agreement,
        'outlier_influence': outlier_influence,
        'fit_scaled': fit_scaled,
        'agreement_scaled': agreement_scaled,
        'outlier_influence_scaled': outlier_influence_scaled
        }

    return f, fit_details



def precision(pred_details, fit_details=None):
    """
    PRECISION:
    Computes the precision for a single prediction.

    Parameters
    ----------
    pred_details : dict
        Calculation is based on the detailed output from the prediction routine.
    fit_details : dict, optional(default=None)
        Precision may be computed faster from the same components as fit.

    Returns
    -------
    precision : ndarray [1-by-1]
        The precision of this individual prediction.
    prec_details: dict
        Includes the subcomponents of precision: n_minus_1, info_t,
        outlier_minus_agree.
    """
    if fit_details is None: # compute here only if needed
        _, fit_details = fit(pred_details)
    elif bool(np.isnan(fit_details.get('agreement'))) is True or \
         bool(np.isnan(fit_details.get('outlier_influence'))) is True:
        _, fit_details = fit(pred_details)

    # unpack prediction and fit details that are needed
    info_t = pred_details.get('info_t')[0]
    n = np.sum(pred_details.get('include'))
    agreement = fit_details.get('agreement')
    outlier_influence = fit_details.get('outlier_influence')

    # compute precision from its components
    n_minus_1 = n - 1
    outlier_minus_agree = n * outlier_influence - agreement
    denom = info_t * outlier_minus_agree
    if denom == 0:
        precision = np.array([[np.nan]])
    else:
        precision = n_minus_1 / denom

    # store and pass detailed outputs
    prec_details = {
        'n_minus_1': n_minus_1,
        'info_t': info_t,
        'outlier_minus_agree': outlier_minus_agree
        }

    return precision, prec_details



def reliability(fits, info_ts, agreements=None, outlier_influences=None):
    """
    RELIABILITY:
    Computes the reliability across multiple prediction tasks.

    Reliability is the information-weighted average fit.

    Parameters
    ----------
    fits : ndarray [P-by-1]
        Fits from a series of predictions.
    info_ts : ndarray [P-by-1]
        Informativeness of each prediction task's circustances, as a column.
    agreements: ndarray [P-by-1], optional(default=None)
        The weighted-average agreement may be computed, if available.
    outlier_influences: ndarray [P-by-1], optional(default=None)
        The weighted-average outlier influence may be computed, if available.

    Returns
    -------
    rely : ndarray [1-by-1]
        The precision of this individual prediction.
    rely_details : dict
        Includes reliability_agreement, reliability_outier_influence.
    """
    T = fits.shape[0]

    if T < 2:
        rely = np.array([[np.nan]])
        rely_details = {
            'reliability_agreement': None,
            'reliability_outlier_influence': None,
        }
        return rely, rely_details

    rely = (1/(T-1)) * np.matmul(np.transpose(fits), info_ts)

    # reliability based on agreement, if desired
    if agreements is None:
        rely_agreement = None
    else:
        rely_agreement = (1/(T-1)) * np.matmul(
            np.transpose(agreements), info_ts)

    # reliability based on outlier influence, if desired
    if agreements is None:
        rely_outlier_influence = None
    else:
        rely_outlier_influence = (1/(T-1)) * np.matmul(
            np.transpose(outlier_influences), info_ts)

    # store and pass detailed outputs
    rely_details = {
        'reliability_agreement': rely_agreement,
        'reliability_outlier_influence': rely_outlier_influence
        }

    return rely, rely_details
