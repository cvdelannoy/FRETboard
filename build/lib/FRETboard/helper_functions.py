from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import warnings


def print_timestamp():
    dt = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
    return f'[{dt}]: '


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_var(a, window):
    return np.std(rolling_window(a, window), -1)


def rolling_cross_correlation(don, acc,  window):
    return np.array([np.convolve(a, b, mode='valid')[0]
                     for a, b in zip(rolling_window(acc, window), rolling_window(don, window))])
    # return np.correlate(rolling_window(don,window), rolling_window(acc,window))


def rolling_corr_coef(don, acc,  window):
    return np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rolling_window(acc, window), rolling_window(don, window))])


def parallel_subtract(data, eps):
    """
    subtract background from data presented as pd Series or pd dataframe object.

    """
    if type(data) == pd.Series:
        data = subtract_background_fun(data, eps)
        return data
    for tidx, tup in data.iterrows():
        tup_mod = subtract_background_fun(tup, eps)
        data.loc[tidx] = tup_mod.values
    return data


def subtract_background_fun(tup, eps):
    """
    Cluster points in i_don and i_acc using DBSCAN, recalculate derived features. Expects pd Series object!
    :param tup: pd Series object
    :param eps: integer, max distance in cluster parameter for DBSCAN (epsilon)
    :return: pd Series object
    """
    for nt in (('i_don_raw', 'i_don'), ('i_acc_raw', 'i_acc')):
        tup.at[nt[1]] = bg_filter_trace(tup.loc[nt[0]], eps)
    for feat, vec in zip(['E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient'],
                         get_derived_features(tup.loc['i_don'], tup.loc['i_acc'])):
        tup.at[feat] = vec
    tup.at['eps'] = eps
    return tup


def bg_filter_trace(tr, eps):
    if np.isnan(eps):
        return tr
    min_clust = 10
    pc10 = np.percentile(tr, 20)
    tr_pc10 = tr[tr < pc10]
    clust = DBSCAN(eps=eps, min_samples=min_clust).fit(tr_pc10.reshape(-1, 1)).labels_
    if np.sum(np.unique(clust) != -1) == 0:
        return tr - tr.min()
    med_list = np.array([np.median(tr_pc10[clust == lab]) for lab in np.unique(clust)])
    return tr - np.min(med_list)


def get_derived_features(i_don, i_acc):
    window = 9
    ss = (window - 1) // 2  # sequence shortening
    i_sum = np.sum((i_don, i_acc), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_FRET = np.divide(i_acc, np.sum((i_don, i_acc), axis=0))
    E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out

    correlation_coefficient = np.full_like(E_FRET, np.nan)
    correlation_coefficient[ss:-ss] = rolling_corr_coef(i_don, i_acc, window)
    E_FRET_sd = np.full_like(E_FRET, np.nan)
    E_FRET_sd[ss:-ss] = rolling_var(E_FRET, window)
    i_sum_sd = np.full_like(E_FRET, np.nan)
    i_sum_sd[ss:-ss] = rolling_var(i_sum, window)
    return E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient