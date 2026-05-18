"""
Prediction Revisited
====================

This module implements statistical prediction from an observation-centric
perspective, as outlined in the book "Prediction Revisited: The Importance of
Observation" (see also: predictionrevisited.com). 

The key to this approach is computing relevance based on observed attributes 
(X) and current circumstacnes (x_t) to predict a future outcome (y_hat) as a 
relevance-weighted average of previously observed outcomes (y). This approach 
also allows us to compute the fit of each individual prediction task. 

"""


# imports
import numpy as np     # for matrix manipulations and math
import pandas as pd    # for ancillary DataFrame wrappers


# functions

def demo():
    """
    DEMO:
    Demonstrates prediction using simple inputs.
    
    Returns
    -------
    prediction_df : DataFrame
        Predicted temperatures for New York and Sydney, plus additional stats.
    relevance_df : DataFrame
        The relevance of each observation supporting each prediction. 
    prediction_circumstances_df : DataFrame
        The attributes of New York and Sydney that inform predictions.
    actual_outcomes_df : DataFrame
        Actual average temperatures for New York and Sydney. 
    """
    # Note: latitudes are approximate. temperature data are for the year 2010, 
    # from Terrestrial Air Temperature, 1900-2017 Gridded Monthly Time Series,
    # v5.01, Kenji Matsuura and Cort J. Willmott.
    
    # Note: January = 0, July = 1
    
    prior_observations_df = pd.DataFrame(
        [
            [51.25, 0, 1.1],
            [35.75, 0, 6.3],
            [-22.75, 0, 27.9],
            [-34.75, 0, 24.8],
            [51.25, 1, 18.2],
            [35.75, 1, 28.0],
            [-22.75, 1, 21.1],
            [-34.75, 1, 10.0]
        ],
        columns = ['latitude', 'january_or_july', 'avg_temp_celcius'],
        index = ['London', 'Tokyo', 'Rio de Janiero', 'Buenos Aires',
                 'London', 'Tokyo', 'Rio de Janiero', 'Buenos Aires'
                 ]
    )
    
    to_predict_df = pd.DataFrame(
        [
            [40.75, 0, 0.1],
            [-33.75, 0, 23.4],
            [40.75, 1, 25.9],
            [-33.75, 1, 12.5]
        ],
        columns = ['latitude', 'january_or_july', 'avg_temp_celcius'],
        index = ['New York', 'Sydney', 'New York', 'Sydney']
    )
    
    # define attributes, outcomes, and prediction circumstances.
    # our goal is to predict New York and Sydney temperatures from the other data

    observed_attributes_df = prior_observations_df[['latitude', 'january_or_july']]
    observed_outcomes_df = prior_observations_df[['avg_temp_celcius']]
    prediction_circumstances_df = to_predict_df[['latitude', 'january_or_july']]
    actual_outcomes_df = to_predict_df[['avg_temp_celcius']]
    
    # calibration
    thresh = 0.75  # use top 25 percent most relevant observations (just two data points in this case)
    most = True
    pct_thresh = True
    pairwise_fits = True
    predict_binary_outcome = False

    prediction_df, obs_relevance = predict_df(observed_attributes_df,
                                              prediction_circumstances_df,
                                              observed_outcomes_df, 
                                              thresh,
                                              most,
                                              pct_thresh,
                                              pairwise_fits,
                                              predict_binary_outcome
                                              )
    
    # create dataframe of relevance
    relevance_df = pd.DataFrame(obs_relevance.get('relevance'),
         columns = ['New York, Jan', 'Sydney, Jan', 'New York, Jul', 'Sydney, Jul'],
         index = ['London, Jan', 'Tokyo, Jan', 'Rio de Janiero, Jan', 'Buenos Aires, Jan',
                  'London, Jul', 'Tokyo, Jul', 'Rio de Janiero, Jul', 'Buenos Aires, Jul'
                  ]
    )
    
    print("PREDICTIONS:\n\n",
          prediction_df,
          "\n\nPREDICTION CIRCUMSTANCES:\n\n",
          prediction_circumstances_df,
          "\n\nRELEVANCE:\n\n",
          relevance_df,
          "\n\nACTUAL OUTCOMES:\n\n",
          actual_outcomes_df
          )
    
    return prediction_df, relevance_df, prediction_circumstances_df, actual_outcomes_df



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
    stdev = np.power(spread(X, shortcut), 0.5); # square root of spread (variance)

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
    z_i = np.atleast_2d(np.divide(x_i - x_bar, stdev))
    
    # compute either a scalar or a matrix
    if K == 1:
        co_occur = 1; # trivial case by definition
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
    


def predict(X, x_t, y, thresh=0.5, most=True, pct_thresh=True,
            cov_inv=None, predict_binary_outcome=False):
    """
    PREDICT:
    Predicts an outcome based on a circumstance using partial sample regression.
    
    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows.
    x_t : ndarray [1-by-K]
        Row vector of attributes for the circumstance of prediction.
    y : ndarray [N-by-1]
        Column vector of outcomes for each observation. 
    thresh : float, optional (default=0.5)
        Threshold for determining relevance. Interpret as a raw value, or as 
        percent threshold if pct_thresh is True.
    most : bool, optional(default=True)
        Predicts from the most relevant if True, least relevant if False.
    pct_thresh : bool, optional(default=True)
        Interpret thresh as a percentile if True, raw value if False.
    cov_inv : ndarray [K-by-K], optional(default=None)
        Computation speed is faster when cov_inv is supplied. 
    predict_binary_outcome : bool(default=False)
        Transform prediction to be between 0 and 1, if y is binary. 
    
    Returns
    -------
    yhat : ndarray [1-by-1]
        The partial sample regression prediction.  
    pred_details: dict
        Contains detailed interim calculations, including: y, relevance, 
        include, lambda_sq, sim_it, info_i, info_t, fit (basic).
    """
    if x_t.ndim > 1:
        raise ValueError("Use predict_many for multiple prediction trials.")
    
    x_bar = average(X)
    if cov_inv is None:
        _, cov_inv = covariance(X, shortcut=True)  # favor efficiency here
    
    N, K = X.shape
    
    y_bar = average(y)
    
    # initialize
    sim_it = np.full([N,1], np.nan)
    info_i = np.full([N,1], np.nan)
    info_t = np.full([N,1], np.nan)

    # initialize
    rel = np.empty([N,1])
    rel[:] = np.nan
    # compute the relevance of every observation
    for i in range(N):
        rel[i], my_rel_details = relevance(X[i,:], x_t, x_bar, cov_inv)
        sim_it[i] = my_rel_details.get('sim_ij')
        info_i[i] = my_rel_details.get('info_i')
        info_t[i] = my_rel_details.get('info_j')
        
    # apply threshold to filter
    if pct_thresh is True:
        if most is True:
            include = rel >= np.percentile(rel, thresh*100)
        else:
            include = rel <= np.percentile(rel, thresh*100)
    else:
        if most is True:
            include = rel >= thresh
        else:
            include = rel <=thresh
    n = np.sum(include)
    
    full_var = np.sum(np.power(rel, 2)) / (N-1) # same as np.var (note that average(rel) always equals zero)
    part_var = np.sum(np.power(include * rel, 2)) / (n-1)
        
    lambda_sq = full_var / part_var
    if predict_binary_outcome is False: # this is traditional prediction
        yhat = y_bar + (lambda_sq / (n-1)) * np.matmul(
                np.transpose(include * rel), y-y_bar
                )
    else: # this is prediction from a binary outcome, interpreted as probability
        A = y_bar/(1-y_bar);
        B = 1/((y_bar*(1-y_bar)));
        C = A**2 - 1
        n1 = np.sum(y)
        mu1 = np.matmul(np.transpose(y), X) / n1
        yhat_forlogistic = np.log(A) + B * (lambda_sq / (n-1)) * np.matmul(
            np.transpose(include * rel), y-y_bar
            ) + C * informativeness(mu1, x_bar, cov_inv)
        yhat = (1 + np.exp(-yhat_forlogistic)) ** -1
    
    # store and pass detailed outputs
    pred_details = {
        'N': N,
        'n': n,
        'K': K,
        'y': y,
        'relevance': rel,
        'include': include,
        'lambda_sq': lambda_sq,
        'sim_it': sim_it,
        'info_i': info_i,
        'info_t': info_t
        }
    
    # compute fit the quick way - leave other fit details to the fit function
    f = fit(pred_details, shortcut=True, yhat=yhat)[0] # favor efficiency here
    pred_details['fit'] = f
    
    return yhat, pred_details



def predict_many(X, X_t, y, thresh=0.5, most=True, pct_thresh=True,
                 pairwise_fits=True, predict_binary_outcome=False):
    """
    PREDICT_MANY:
    Makes multiple predictions for a range of circumstances.
    
    Input relevance (rel) to avoid recalculating it, which is faster. 
    
    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows. Required if 
        cov_inv is not provided. 
    X_t : ndarray [P-by-K]
        Matrix of attributes for many circumstances (rows) of prediction.
    y : ndarray [N-by-1]
        Column vector of outcomes for each observation. 
    thresh : float, optional (default=0.5)
        Threshold for determining relevance. Interpret as a raw value, or as 
        percent threshold if pct_thresh is True. 
    most : bool, optional(default=True)
        Predicts from the most relevant if True, least relevant if False.
    pct_thresh : bool, optional(default=True)
        Interpret thresh as a percentile if True, raw value if False.
    pairwise_fits : bool, optional(default=True)
        Compute agreement and outlier_influence if True, but this runs slower.
    predict_binary_outcome : bool(default=False)
        Transform prediction to be between 0 and 1, if y is binary. 
    
    Returns
    -------
    yhats : ndarray [T-by-1]
        The partial sample regression predictions, in a column.  
    pred_many_details : dict
        Contains detailed interim calculations, including: fits, agreements,
        outlier_influences, reliability, reliability_agreement,
        reliability_outlier_influence.
    """
    N, K = X.shape
    T = X_t.shape[0]
    
    # preallocate
    yhats = np.full([T,1], np.nan)
    fits = np.full([T,1], np.nan)
    agreements = np.full([T,1], np.nan)
    outlier_influences = np.full([T,1], np.nan)
    precisions = np.full([T,1], np.nan)
    info_ts = np.full([T,1], np.nan)
    rel = np.full([N,T], np.nan)
    sim_it = np.full([N,T], np.nan)
    info_i = np.full([N,T], np.nan)
    info_t = np.full([N,T], np.nan)
    
    # compute covariance inverse once upfront (for greater efficiency)
    _, cov_inv = covariance(X, shortcut=True)
    
    for t in range(T):
        # use stored relevance when possible to avoid recalculating
        yhats[t], my_pred_details = predict(
            X, X_t[t,:], y, thresh, most, pct_thresh, 
            cov_inv, predict_binary_outcome
            )
    
        # store relevance data in arrays with N rows and T columns
        rel[:,t] = np.squeeze(my_pred_details.get('relevance'))
        sim_it[:,t] = np.squeeze(my_pred_details.get('sim_it'))
        info_i[:,t] = np.squeeze(my_pred_details.get('info_i'))
        info_t[:,t] = np.squeeze(my_pred_details.get('info_t'))
    
        # compute fits and reliability (weighted average fit across tasks)
        info_ts[t] = info_t[0,t] # this is the info_t for this prediction task
        if pairwise_fits is True:
            fits[t], my_fit_details = fit(my_pred_details, shortcut=False)
            agreements[t] = my_fit_details.get('agreement')
            outlier_influences[t] = my_fit_details.get('outlier_influence')
            precisions[t], _ = precision(my_pred_details, my_fit_details)
        else:
            fits[t], my_fit_details = fit(my_pred_details, True, yhats[t])
            agreements[t] = np.nan
            outlier_influences[t] = np.nan
            precisions[t] = np.nan
    
    if pairwise_fits is True:
        rely, rely_details = reliability(
            fits, info_ts, agreements, outlier_influences
            )
    else:
         rely, rely_details = reliability(fits, info_ts)   
    
    # store and pass detailed outputs
    pred_many_details = {
        'N': my_pred_details.get('N'), # constant across iterations
        'n': my_pred_details.get('n'), # constant across iterations
        'K': my_pred_details.get('K'), # constant across iterations
        'y': my_pred_details.get('y'), # constant across iterations
        'relevance': rel,
        'sim_it': sim_it,
        'info_i': info_i,
        'info_t': info_t, 
        'fits': fits,
        'agreements': agreements,
        'outlier_influences': outlier_influences,
        'precisions': precisions,
        'info_ts': info_ts, # this is info_t for each prediction task
        'reliability': rely,
        'reliability_agreement': rely_details.get('reliability_agreement'),
        'reliability_outlier_influence': rely_details.get('reliability_outlier_influence')
        }
    
    return yhats, pred_many_details



def predict_df(attributes_df, circumstances_df, outcomes_df,
               thresh=0.5, most=True, pct_thresh=True, pairwise_fits=True,
               predict_binary_outcome=False):
    """
    PREDICT_DF:
    Predictions from dataframes, with intuitively labeled output.
    
    Input relevance (rel) to avoid recalculating it, which is faster. 
    
    Parameters
    ----------
    attributes_df : DataFrame
        Attributes for each observation. 
    circumstances_df : DataFrame
        Circumstances of prediction (could be many).
    outcomes_df : DataFrame
        Outcomes for each observation.
    thresh : float, optional (default=0.5)
        Threshold for determining relevance. Interpret as a raw value, or as 
        percent threshold if pct_thresh is True. 
    most : bool, optional(default=True)
        Predicts from the most relevant if True, least relevant if False.
    pct_thresh : bool, optional(default=True)
        Interpret thresh as a percentile if True, raw value if False.
    pairwise_fits : bool, optional(default=True)
        Compute agreement and outlier_influence if True, but this runs slower.
    allstats : bool, optional(default=False)
        Compute and return a wide range of prediction statistics.
    
    Returns
    -------
    yhats : ndarray [P-by-1]
        The partial sample regression predictions, in a column.  
    pred_many_details: dict
        Contains detailed interim calculations, including: fits, agreements,
        outlier_influences, reliability, reliability_agreement,
        reliability_outlier_influence.
    """
    X = attributes_df.to_numpy()
    y = outcomes_df.to_numpy()
    X_t = circumstances_df.to_numpy()
    
    yhats, pred_many_details = predict_many(X, X_t, y, thresh, most, 
                                            pct_thresh, pairwise_fits,
                                            predict_binary_outcome
                                            )
    
    # package results as dataframe
    prediction_results = {'yhat': np.squeeze(yhats)}
    obs_relevance = {
        'relevance': pred_many_details.get('relevance'),
        'similarity': pred_many_details.get('sim_it'),
        'info_i': pred_many_details.get('info_i'),
        'info_t': pred_many_details.get('info_t')
        }

    prediction_results['fit'] = np.squeeze(pred_many_details.get('fits'))
    prediction_results['agreement'] = np.squeeze(pred_many_details.get('agreements'))
    prediction_results['outlier_influence'] = np.squeeze(pred_many_details.get('outlier_influences'))
    prediction_results['precision'] = np.squeeze(pred_many_details.get('precisions'))
    prediction_results['info_t'] = np.squeeze(pred_many_details.get('info_ts'))
    
    predictions_df = pd.DataFrame(prediction_results)
    
    return predictions_df, obs_relevance



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
        info_y = informativeness(yhat, y_bar, 1/spread_y)
        info_x_t = info_t[0]
        f = info_y / info_x_t
        agreement = np.nan
        outlier_influence = np.nan
        
    else:
    
        y_keep = np.transpose(np.atleast_2d(np.extract(include, y)))
        rel_keep = np.transpose(np.atleast_2d(np.extract(include, rel)))
        n = np.sum(include)
        
        # standardize y and relevance to z-scores
        z_y = (y_keep - average(y)) / standard_deviation(y)
        z_r = (rel_keep - average(rel)) / standard_deviation(rel) # note avg(rel)=0
        
        # initialize agreement and outlier_influence, to sum incrementally
        agreement = 0
        outlier_influence = 0
        
        # loop through all pairs
        for i in range(n):
            for j in range(n):
                to_add = (lambda_sq / (n-1))**2 * z_r[i] * z_r[j] * z_y[i] * z_y[j]
                if i is j:
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
    precision = n_minus_1 / (info_t * outlier_minus_agree)
    
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
    


def asymmetry(X, x_t, y, thresh=0.5, shortcut=True):
    """
    ASYMMETRY:
    Computes the asymmetry associated with a single prediction.
    
    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows.
    x_t : ndarray [1-by-K]
        Row vector of attributes for the circumstance of prediction.
    y : ndarray [N-by-1]
        Column vector of outcomes for each observation. 
    thresh : float, optional (default=0.5)
        Threshold for determining relevance. Interpret as a percent threshold.
    
    Returns
    -------
    asymmetry : ndarray [1-by-1]
        The asymmetry of the prediction. 
    asymmetry_details : dict
        Contains 'yhat_most', 'yhat_least', and 'asymmetry_scaled'
    """
    
    N, K = X.shape
    
    # specify high and low percentile thresholds (must be a percentile)
    if thresh >= 0.5:
        thresh_high = thresh
    else:
        thresh_high = 1-thresh
    thresh_low = 1-thresh_high
    
    # predict from same-size sets of most and least relevant observations
    yhat_most, details_most = predict(X, x_t, y, thresh_high, most=True, pct_thresh=True)
    yhat_least, details_least = predict(X, x_t, y, thresh_low, most=False, pct_thresh=True)
    
    # need informativeness for the 'shortcut' version, and return it regardless
    info_t = informativeness(x_t, X=X)
    
    if shortcut is True:
        sim_yhats = similarity(yhat_most, yhat_least, X=y)
        asymmetry = -sim_yhats / info_t
        
    else:
        avg_fit = 0.5 * (fit(details_most)[0] + fit(details_least)[0])
        cross_fit = crossfit(details_most, details_least)
        asymmetry = avg_fit - cross_fit
        
    # compute scaled version
    asymmetry_scaled = asymmetry * K
    
    # store and pass detailed outputs
    asymmetry_details = {
        'yhat_most': yhat_most,
        'yhat_least': yhat_least,
        'info_t': info_t,
        'asymmetry_scaled': asymmetry_scaled
        }
    
    return asymmetry, asymmetry_details
    


def crossfit(pred_details1, pred_details2):
    """
    CROSSFIT:
    Computes the cross-fit of a single prediction, which evaluates the 
    alignment between the most and least relevant subsets of observations.
    
    This calculation is only needed to illustrate the pairwise construction
    of asymmetry, so it does not have a "shortcut" option. The inputs come
    directly from the predict function. 
    
    Note: It is required that both subsamples must be of the same size. For
    the correct calculation of asymmetry, the subsamples must come from
    opposite extremes of relevance, but this is not enforced in this function.
    
    Parameters
    ----------
    pred_details1 : dict
        Calculation is based on the detailed output from the predict function. 
    pred_details2 : dict
        Compute fit pairwise if True, use ratio of informativeness if False.
    
    Returns
    -------
    cf : ndarray [1-by-1]
        The cross-fit associated with this individual prediction.
    """
    # unpack prediction details for the first subset
    y1 = pred_details1.get('y')
    rel1 = pred_details1.get('relevance')
    include1 = pred_details1.get('include')
    lambda_sq1 = pred_details1.get('lambda_sq')
    info_t1 = pred_details1.get('info_t')
    K1 = pred_details1.get('K')
    
    # and the second subset
    y2 = pred_details2.get('y')
    rel2 = pred_details2.get('relevance')
    include2 = pred_details2.get('include')
    lambda_sq2 = pred_details2.get('lambda_sq')
    info_t2 = pred_details2.get('info_t')
    K2 = pred_details2.get('K')
    
    # validate inputs
    if sum(np.absolute(y1 - y2)) != 0:
        raise ValueError("y vectors must be equal in both subsamples.")
    if sum(np.absolute(rel1 - rel2)) != 0:
        raise ValueError("Relevance vectors must be equal in both subsamples.")
    if sum(np.absolute(info_t1 - info_t2)):
        raise ValueError("info_t must be equal in both subsamples.")
    if K1 != K2:
        raise ValueError("K must be equal in both subsamples.")
    n1 = np.sum(include1)
    n2 = np.sum(include2)
    if n1 != n2:
        raise ValueError("Subsamples in the 'include' vectors must be of equal size.")
    
    # because these values must be equivalent, we can simply use the first one
    y = y1
    rel = rel1
    n = n1
    
    y_keep1 = np.transpose(np.atleast_2d(np.extract(include1, y)))
    rel_keep1 = np.transpose(np.atleast_2d(np.extract(include1, rel)))
    
    y_keep2 = np.transpose(np.atleast_2d(np.extract(include2, y)))
    rel_keep2 = np.transpose(np.atleast_2d(np.extract(include2, rel)))
    
    # standardize y and relevance to z-scores
    z_y1 = (y_keep1 - average(y)) / standard_deviation(y)
    z_r1 = (rel_keep1 - average(rel)) / standard_deviation(rel) # note avg(rel)=0

    z_y2 = (y_keep2 - average(y)) / standard_deviation(y)
    z_r2 = (rel_keep2 - average(rel)) / standard_deviation(rel) # note avg(rel)=0
    
    # initialize cross-fit variable to sum incrementally
    cf = 0
    
    # loop through all pairs
    for i in range(n):
        for j in range(n):
            to_add = (lambda_sq1**0.5 * lambda_sq2**0.5 / (n-1))**2 \
                * z_r1[i] * z_r2[j] * z_y1[i] * z_y2[j]
            cf = cf + to_add
    
    return cf
    


def asymmetry_many(X, X_t, y, thresh=0.5, shortcut=True):
    """
    ASYMMETRY_MANY:
    Computes the weighted average asymmetry for many predictions.
    
    Parameters
    ----------
    X : ndarray [N-by-K]
        Matrix of attributes in columns, for observations in rows. Required if 
        cov_inv is not provided. 
    x_t : ndarray [1-by-K]
        Row vector of attributes for the circumstance of prediction.
    y : ndarray [N-by-1]
        Column vector of outcomes for each observation. 
    thresh : float, optional (default=0.5)
        Threshold for determining relevance. Interpret as a raw value, or as 
        percent threshold if pct_thresh is True. 
    
    Returns
    -------
    asymmetry : ndarray [1-by-1]
        The asymmetry of the prediction. 
    asymmetry_many_details : dict
        Contains ...

    """
    N, K = X.shape
    T, _ = X_t.shape
    
    # preallocate
    asymmetries = np.full([T,1], np.nan)
    yhats_most = np.full([T,1], np.nan)
    yhats_least = np.full([T,1], np.nan)
    info_ts = np.full([T,1], np.nan)
    
    # loop through prediction circumstances and compute asymmetries
    for t in range(T):
        my_asymmetry, my_asymmetry_details = asymmetry(X, X_t[t,:], y, 
                                                       thresh, shortcut=True
                                                       )
        
        asymmetries[t] = my_asymmetry
        yhats_most[t] = my_asymmetry_details.get('yhat_most')
        yhats_least[t] = my_asymmetry_details.get('yhat_least')
        info_ts[t] = my_asymmetry_details.get('info_t')
    
    asymmetries_scaled = asymmetries * K
    
    avg_asymmetry = (1/(T-1)) * np.matmul(np.transpose(asymmetries), info_ts)
    
    # store and pass detailed outputs
    asymmetry_many_details = {
        'yhats_most': yhats_most,
        'yhats_least': yhats_least,
        'asymmetries_scaled': asymmetries_scaled
        }
    
    return avg_asymmetry, asymmetry_many_details
    

# Run the demo if this file is executed directly
if __name__ == '__main__':

    print(demo.__doc__)
    (prediction_df, relevance_df, prediction_circumstances_df, 
     actual_outcomes_df) = demo()
    
    
    