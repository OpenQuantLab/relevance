"""
Observation-centric linear algebra primitives.

These are the building blocks of the relevance approach described in
"Prediction Revisited: The Importance of Observation": averages, spread,
co-occurrence, correlation, covariance, Mahalanobis distance, similarity,
informativeness and relevance itself.

Nothing here depends on the prediction or diagnostic layers, which keeps the
import graph acyclic: ``linalg`` <- ``diagnostics`` <- ``prediction``.
"""

import numpy as np


def average(X):
    """
    AVERAGE:
    Computes the average for one or many attributes (columns).

    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows.

    Returns
    -------
    avg : ndarray [1-by-K]
        Row vector containing the equally weighted average by column.
    """
    N, K = X.shape

    avg = (1/N) * np.atleast_2d(np.sum(X, axis=0)) # keep as a 2d row vector

    return avg



def spread(X, shortcut=False):
    """
    SPREAD:
    Computes the spread (variance) for one or many attributes (columns).

    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows.
    shortcut : bool, optional(default=False)
        Pairwise calculation if False, versus average if True.

    Returns
    -------
    spread : ndarray [1-by-K]
        Row vector containing the spread (or variance) by column.
    """
    N, K = X.shape

    # spread may be computed as a pairwise sum, or as standard variance
    if shortcut is True:
        spr = np.var(X, axis=0, ddof=1) # ddof applies N-1 as divisor
        spr = np.atleast_2d(spr) # keep as a 2d row vector
    else:
        spr = np.empty([1,K]) # initialize
        for k in range(K):
            spr_k = 0
            for i in range(N):
                for j in range(N):
                    spr_k = spr_k + 0.5 * (X[i,k] - X[j,k])**2
            spr[0,k] = (1/(N*(N-1))) * spr_k

    return spr



def standard_deviation(X, shortcut=False):
    """
    STANDARD_DEVIATION:
    Computes the standard deviation for attributes (columns).

    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows.
    shortcut : bool, optional(default=False)
        Pairwise calculation if False, versus average if True.

    Returns
    -------
    stdev : ndarray [1-by-K]
        Row vector containing the standard deviation by column.
    """
    stdev = np.power(spread(X, shortcut), 0.5) # square root of spread (variance)

    return stdev



def co_occurrence(x_i, x_bar, stdev, scalar_output_for_pair=False):
    """
    CO_OCCURRENCE:
    Computes the co-occurences for an observation's (row's) attributes.

    Returns a scalar if x_i contains two attributes, otherwise a matrix.

    Parameters
    ----------
    x_i : ndarray [1-by-K]
        Row vector of attributes for a given observation.
    x_bar : ndarray [1-by-K]
        Row vector of average attribute values.
    stdev : ndarray [1-by-K]
        Row vector of standard deviations of attributes.
    scalar_output_for_pair : bool(default=False)
        Return scalar correlation for 2-by-2 case if True, 2-by-2 matrix if False.

    Returns
    -------
    co_occur : ndarray [K-by-K] or scalar value
        The co-occurrences for each pair of attributes.
    avg_sq_z : ndarray [K-by-K] or scalar value
        The average squared z-scores for each pair of attributes.
    """
    K = x_i.shape[0]

    # standardize x_i and relevance to z-scores
    safe_stdev = np.where(stdev == 0, 1.0, stdev)
    z_i = np.atleast_2d(np.divide(x_i - x_bar, safe_stdev))

    # compute either a scalar or a matrix
    if K == 1:
        co_occur = 1 # trivial case by definition
    elif K == 2 and scalar_output_for_pair is True:
        avg_sq_z = 0.5 * (z_i[0,0]**2 + z_i[0,1]**2)
        co_occur = z_i[0,0] * z_i[0,1] / avg_sq_z
    else:
        # start with an identity matrix (and fill in the off-diagonals)
        avg_sq_z = np.full([K, K], np.nan)
        co_occur = np.full([K, K], np.nan)
        for k in range(K):
            for l in range(k, K):
                avg_sq_z[k,l] = 0.5 * (z_i[0,k]**2 + z_i[0,l]**2)
                avg_sq_z[l,k] = avg_sq_z[k,l] # fill in symmetrically
                co_occur[k,l] = z_i[0,k] * z_i[0,l] / avg_sq_z[k,l]
                co_occur[l,k] = co_occur[k,l] # fill in symmetrically

    return co_occur, avg_sq_z



def correlation(X, scalar_output_for_pair=False):
    """
    CORRELATION:
    Computes the correlation for attributes (columns).

    Also returns detail on the co-occurrences and informativeness (avg sq z-score)

    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows.
    scalar_output_for_pair : bool(default=False)
        Return scalar correlation for 2-by-2 case if True, 2-by-2 matrix if False.

    Returns
    -------
    corr : ndarray [K-by-K] or scalar value
        The correlation matrix (or scalar) for the attributes.
    corr_details : dict
        Contains 'co_occurrence' and 'average_sq_zscore'
    """
    N, K = X.shape

    x_bar = average(X)
    stdev = standard_deviation(X)

    # preallocate
    if scalar_output_for_pair is True:
        co_occur = np.full([1, 1, N], np.nan)
        avg_sq_z = np.full([1, 1, N], np.nan)
    else:
        co_occur = np.full([K, K, N], np.nan)
        avg_sq_z = np.full([K, K, N], np.nan)

    for i in range(N):
        # compute co-occurrence for each observation, a matrix for each
        co_occur[:,:,i], avg_sq_z[:,:,i] = co_occurrence(X[i,:], x_bar, stdev,
                                                         scalar_output_for_pair)

    # sum across all the co-occurrence matrices
    corr = (1/(N-1)) * np.sum(np.multiply(co_occur, avg_sq_z), axis=2)

    # squeeze into column vectors for a single pair, if desired
    if K == 2 and scalar_output_for_pair is True:
        co_occur = np.squeeze(co_occur)
        avg_sq_z = np.squeeze(avg_sq_z)
        corr = np.squeeze(corr)

    # define a dictionary to store and pass detailed outputs
    corr_details = {
        'co_occurrence': co_occur,
        'average_sq_zscore': avg_sq_z
        }

    return corr, corr_details



def covariance(X, shortcut=False):
    """
    COVARIANCE:
    Computes the covariance matrix and its inverse.

    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows.
    shortcut : bool, optional(default=False)
        Pairwise calculation if False, versus average if True.

    Returns
    -------
    cov : ndarray
        Covariance matrix [K-by-K].
    cov_inv : dict
        Inverse covariance matrix [K-by-K].
    """
    if shortcut is True:
        # compute using traditional method
        cov = np.cov(X, rowvar=False)
    else:
        # compute from observation-centric components in this module
        corr = correlation(X)[0]
        stdev = standard_deviation(X)
        K = corr.shape[0]

        stdev_diag = np.zeros((K,K))
        np.fill_diagonal(stdev_diag, stdev)
        cov = np.matmul(np.matmul(stdev_diag, corr), stdev_diag)

    cov_inv = np.linalg.inv(np.atleast_2d(cov)) # atleast_2d ensures it works for scalar

    return cov, cov_inv



def mahalanobis(x_i, x_j, cov_inv=None, X=None):
    """
    MAHALANOBIS:
    Computes the Mahalanobis distance (in 'squared' form).

    Use the inverse covariance matrix if it is provided, otherwise compute the
    inverse covariance matrix based on the attributes in X.

    Parameters
    ----------
    x_i : ndarray [1-by-K]
        Row vector of attributes for one observation.
    x_j : ndarray [1-by-K]
        Row vector of attributes for a second observation.
    cov_inv : ndarray [K-by-K], optional(default=None)
        Code executes faster when covariance inverse is already computed.
    X : ndarray [N-by-K], optional(default=None)
        Matrix of attributes in columns, for observations in rows. Required if
        cov_inv is not provided.

    Returns
    -------
    mahal : ndarray [1-by-1]
        Mahalanobis distance as a single number, formatted in a 2-dim array.
    """
    # allow to calculate using X as input instead of covar_inv
    if cov_inv is None:
        if X is None:
            raise ValueError("Either cov_inv or X is required as an input.")
        else:
            cov, cov_inv = covariance(X, shortcut=True)

    vec_diff = np.atleast_2d(x_i - x_j)
    mahal = np.matmul(vec_diff, np.matmul(cov_inv, np.transpose(vec_diff)))
    return mahal



def similarity(x_i, x_j, cov_inv=None, X=None):
    """
    SIMILARITY:
    Computes the similarity between two observations.

    Use the inverse covariance matrix if provided, otherwise compute from X.

    Parameters
    ----------
    x_i : ndarray [1-by-K]
        Row vector of attributes for one observation.
    x_j : ndarray [1-by-K]
        Row vector of attributes for a second observation.
    cov_inv : ndarray [K-by-K], optional(default=None)
        Code executes faster when covariance inverse is already computed.
    X : ndarray [N-by-K], optional(default=None)
        Matrix of attributes in columns, for observations in rows. Required if
        cov_inv is not provided.

    Returns
    -------
    sim : ndarray [1-by-1]
        Similarity as a single number, formatted in a 2-dim array.
    """
    # allow to calculate using X as input instead of cov_inv
    if cov_inv is None:
        if X is None:
            raise ValueError("Either cov_inv or X is required as an input.")
        else:
            _, cov_inv = covariance(X, shortcut=True)

    sim = -0.5 * mahalanobis(x_i, x_j, cov_inv)

    return sim



def informativeness(x_i, x_bar=None, cov_inv=None, X=None):
    """
    INFORMATIVENESS:
    Computes the informativeness of an observations.

    Use the inverse covariance matrix and x_bar if provided, otherwise compute
    from X.

    Parameters
    ----------
    x_i : ndarray [1-by-K]
        Row vector of attributes for one observation.
    x_bar : ndarray [1-by-K]
        Row vector of average attribute values.
    cov_inv : ndarray [K-by-K], optional(default=None)
        Code executes faster when covariance inverse is already computed.
    X : ndarray [N-by-K], optional(default=None)
        Matrix of attributes in columns, for observations in rows. Required if
        cov_inv is not provided.

    Returns
    -------
    info : ndarray [1-by-1]
        Informativeness as a single number, formatted in a 2-dim array.
    """
    # allow to calculate using X as input instead of cov_inv and x_bar
    if x_bar is None:
        if X is None:
            raise ValueError("Either x_bar or X is required as an input.")
        else:
            x_bar = average(X)

    if cov_inv is None:
        if X is None:
            raise ValueError("Either cov_inv or X is required as an input.")
        else:
            _, cov_inv = covariance(X, shortcut=True)

    info = mahalanobis(x_i, x_bar, cov_inv)

    return info



def relevance(x_i, x_j, x_bar=None, cov_inv=None, X=None):
    """
    RELEVANCE:
    Computes the relevance of one observation to another.

    Use the inverse covariance matrix and x_bar if provided, otherwise compute
    from X.

    Parameters
    ----------
    x_i : ndarray [1-by-K]
        Row vector of attributes for one observation.
    x_j : ndarray [1-by-K]
        Row vector of attributes for a second observation.
    x_bar : ndarray [1-by-K]
        Row vector of average attribute values.
    cov_inv : ndarray [K-by-K], optional(default=None)
        Code executes faster when covariance inverse is already computed.
    X : ndarray [N-by-K], optional(default=None)
        Matrix of attributes in columns, for observations in rows. Required if
        cov_inv is not provided.

    Returns
    -------
    rel : ndarray [1-by-1]
        Relevance as a single number, formatted in a 2-dim array.
    rel_details: dict
        Contains sim_ij, info_i, info_j.
    """
    # similarity and informativeness will handle whichever inputs are given
    sim_ij = similarity(x_i, x_j, cov_inv, X)
    info_i = informativeness(x_i, x_bar, cov_inv, X)
    info_j = informativeness(x_j, x_bar, cov_inv, X)

    rel = sim_ij + 0.5 * (info_i + info_j)

    # define a dictionary to store and pass detailed outputs
    rel_details = {
        'sim_ij': sim_ij,
        'info_i': info_i,
        'info_j': info_j
        }

    return rel, rel_details
