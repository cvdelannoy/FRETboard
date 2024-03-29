import matplotlib
matplotlib.use('Agg')

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import threading
from scipy.stats import norm, mode
from scipy.linalg import logm
from math import sqrt

colnames = ['time',
            'f_dex_dem_raw', 'f_dex_aem_raw', 'f_dex_dem', 'f_dex_aem',
            'E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient']


colnames_alex = ['time',
                 'f_dex_dem_raw', 'f_dex_aem_raw', 'f_dex_dem', 'f_dex_aem',
                 'f_aex_dem_raw', 'f_aex_aem_raw', 'f_aex_dem', 'f_aex_aem',
                 'E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient']

colnames_w_labels = colnames + ['predicted']
colnames_alex_w_labels = colnames_alex + ['predicted']

def installThreadExcepthook():
    """
    Workaround for sys.excepthook thread bug
    From
http://spyced.blogspot.com/2007/06/workaround-for-sysexcepthook-bug.html

(https://sourceforge.net/tracker/?func=detail&atid=105470&aid=1230540&group_id=5470).
    Call once from __main__ before creating any threads.
    If using psyco, call psyco.cannotcompile(threading.Thread.run)
    since this replaces a new-style class method.
    """
    init_old = threading.Thread.__init__

    def init(self, *args, **kwargs):
        init_old(self, *args, **kwargs)
        run_old = self.run

        def run_with_except_hook(*args, **kw):
            try:
                run_old(*args, **kw)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init

def multi_joint_plot(col_x, col_y, col_k, df, scatter_alpha=.5, palette='Blues'):
    """
    seaborn joint plot for multiple data sets plotted separately
    adapted from: https://stackoverflow.com/questions/35920885/how-to-overlay-a-seaborn-jointplot-with-a-marginal-distribution-histogram-fr
    """

    def colored_scatter(x, y, c):
        def scatter(*args, **kwargs):
            args = (x, y)
            kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            kwargs['s'] = 1
            plt.scatter(*args, **kwargs)
        return scatter

    g = sns.JointGrid(x=col_x, y=col_y, data=df)
    unique_labels = df.loc[:, col_k].unique()
    unique_labels.sort()
    colors = sns.color_palette(palette, len(unique_labels))
    legends = []
    for ui, ul in enumerate(unique_labels):
        legends.append(ul+1)
        df_group = df.loc[df.loc[:, col_k] == ul, :]
        color = colors[ui]
        g.plot_joint(colored_scatter(df_group[col_x], df_group[col_y], color))
        try:
            sns.distplot(df_group[col_x].values, ax=g.ax_marg_x, color=color, hist=False)
            sns.distplot(df_group[col_y].values,ax=g.ax_marg_y, color=color, vertical=True, hist=False)
        except:
            print('Cannot plot distributions in scatterplot due to statsmodels bug')
    # # Do also global Hist
    # sns.distplot(df[col_x].values, ax=g.ax_marg_x, color='grey')
    # sns.distplot(df[col_y].values.ravel(), ax=g.ax_marg_y, color='grey', vertical=True)
    plt.legend(legends)

def print_timestamp():
    dt = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
    return f'[{dt}]: '

def numeric_timestamp():
    return int(datetime.now().strftime('%H%M%S%f'))

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
    """
    Calculate rolling window correlation coefficient between don and acc, over window of size window.
    NOTE: will return np.nan if either don or acc is filled with all the same values!
    """
    return np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rolling_window(acc, window), rolling_window(don, window))])


def parallel_subtract(data, eps, d, l, gamma):
    """
    subtract background from data presented as pd Series or pd dataframe object.

    """
    if type(data) == pd.Series:
        data = subtract_background_fun(data, eps, d, l, gamma)
        return data
    for tidx, tup in data.iterrows():
        tup_mod = subtract_background_fun(tup, eps, d, l, gamma)
        data.loc[tidx] = tup_mod.values
    return data


def subtract_background_fun(tup, eps, d, l, gamma):
    """
    Cluster points in i_don and i_acc using DBSCAN, recalculate derived features. Expects pd Series object!
    :param tup: pd Series object
    :param eps: integer, max distance in cluster parameter for DBSCAN (epsilon)
    :return: pd Series object
    """
    for nt in (('i_don_raw', 'i_don'), ('i_acc_raw', 'i_acc')):
        tup.at[nt[1]] = bg_filter_trace(tup.loc[nt[0]], eps)
    for feat, vec in zip(['E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient'],
                         get_derived_features(tup.loc['i_don'], tup.loc['i_acc'],
                                              tup.loc['f_acc_don_raw'], tup.loc['f_acc_acc_raw'],
                                              l=l, d=d, gamma=gamma)):
        tup.at[feat] = vec
    tup.at['eps'], tup.at['d'], tup.at['l'], tup.at['gamma'] = eps, d, l, gamma
    return tup

def remove_outliers(mat):
    bool_out = np.ones(mat.shape[1], dtype=bool)
    for seq in mat:
        mu = np.mean(seq)
        sd = np.std(seq)
        bool_out = np.logical_and(bool_out, np.logical_and(seq > mu - 2 * sd, seq < mu + 2 * sd))
    return mat[:, bool_out]

def remove_last_event(seq):
    event_found = False
    for i in range(len(seq)):
        if seq[-i] != 0:
            event_found = True
        elif event_found:
            return seq
        seq[-i] = 0

def series_to_array(df):
    """
    convert pd series containing np arrays into 1 numpy array
    """
    return np.concatenate([dv for dv in df.values if len(dv)])


def bg_filter_trace(tr, eps, f_aex=None):
    if np.isnan(eps):
        if f_aex is None:
            return np.copy(tr)
        return tr, f_aex
    min_clust = 10
    if f_aex is not None:
        tr_joined = np.concatenate((tr, f_aex))
    else:
        tr_joined = tr
    pc10 = np.percentile(tr_joined, 20)
    tr_pc10 = tr_joined[tr_joined < pc10]
    clust = DBSCAN(eps=eps, min_samples=min_clust).fit(tr_pc10.reshape(-1, 1)).labels_
    if np.sum(np.unique(clust) != -1) == 0:
        bg = np.min(tr_joined)
    else:
        med_list = np.array([np.median(tr_pc10[clust == lab]) for lab in np.unique(clust)])
        bg = np.min(med_list)
    if f_aex is None:
        return tr - bg
    return tr - bg, f_aex - bg


def df_empty(columns, dtypes, index=None):
    """
    Create empty dataframe with set data types. from: https://stackoverflow.com/questions/36462257/create-empty-dataframe-in-pandas-specifying-column-types
    """
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def get_derived_features(i_don, i_acc, f_acc_don_raw, f_acc_acc_raw, gamma=1.0, l=0.0, d=0.0):
    window = 5
    ss = (window - 1) // 2  # sequence shortening
    cross_correct = 0
    if len(f_acc_don_raw) and len(f_acc_acc_raw):
        cross_correct = l * f_acc_acc_raw + d * f_acc_don_raw
    i_sum = np.sum(( gamma * i_don, i_acc - cross_correct), axis=0)
    f_fret = i_acc - cross_correct
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_FRET = f_fret / (gamma * i_don + f_fret)
        # E_FRET = np.divide(i_acc - cross_correct, i_sum)
    E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out

    correlation_coefficient = np.full_like(E_FRET, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation_coefficient[ss:-ss] = rolling_corr_coef(i_don, i_acc, window)
    E_FRET_sd = np.full_like(E_FRET, np.nan)
    E_FRET_sd[ss:-ss] = rolling_var(E_FRET, window)
    i_sum_sd = np.full_like(E_FRET, np.nan)
    i_sum_sd[ss:-ss] = rolling_var(i_sum, window)
    return E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient

def get_tuple(fc, eps, l, d, gamma, traceswitch):
    """
    construct tuple from numpy file content array. Tuple measures [nb_features x len(trace)]
    """
    window = 5
    ss = (window - 1) // 2  # sequence shortening

    time = fc[0, :].astype(np.float64)
    f_dex_dem_raw = fc[1, :].astype(np.float64)
    f_dex_aem_raw = fc[2, :].astype(np.float64)
    if fc.shape[0] == 5:
        f_aex_dem_raw = fc[3, :].astype(np.float64)
        f_aex_aem_raw = fc[4, :].astype(np.float64)
        f_dex_dem, f_aex_dem = bg_filter_trace(f_dex_dem_raw, eps, f_aex_dem_raw)
        f_dex_aem, f_aex_aem = bg_filter_trace(f_dex_aem_raw, eps, f_aex_aem_raw)
        if traceswitch:
            f_dex_dem, f_aex_dem, f_dex_aem, f_aex_aem = f_aex_dem, f_dex_dem, f_aex_aem, f_dex_aem
        # f_aex_dem = bg_filter_trace(f_aex_dem_raw, eps)
        # f_aex_aem = bg_filter_trace(f_aex_aem_raw, eps)
        cross_correct = l * f_dex_dem + d * f_aex_aem
    else:
        f_dex_dem = bg_filter_trace(f_dex_dem_raw, eps)
        f_dex_aem = bg_filter_trace(f_dex_aem_raw, eps)
        cross_correct = 0
    f_fret = f_dex_aem - cross_correct
    i_sum = gamma * f_dex_dem + f_fret
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_FRET = f_fret / i_sum
    E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out

    correlation_coefficient = np.full_like(E_FRET, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation_coefficient[ss:-ss] = rolling_corr_coef(f_dex_dem, f_dex_aem, window)
    E_FRET_sd = np.full_like(E_FRET, np.nan)
    E_FRET_sd[ss:-ss] = rolling_var(E_FRET, window)
    i_sum_sd = np.full_like(E_FRET, np.nan)
    i_sum_sd[ss:-ss] = rolling_var(i_sum, window)
    if fc.shape[0] == 5:
        return np.vstack([time, f_dex_dem_raw, f_dex_aem_raw, f_dex_dem, f_dex_aem,
                          f_aex_dem_raw, f_aex_aem_raw, f_aex_dem, f_aex_aem,
                          E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient])
    else:
        dummy = np.zeros_like(f_dex_dem_raw)
        return np.vstack([time, f_dex_dem_raw, f_dex_aem_raw, f_dex_dem, f_dex_aem,
                          dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(),
                          E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient])


def split_trace_dict_on_source(td):
    origin_list = [t.split(':')[0] if len(t.split(':')) > 1 else 'txt' for t in td]
    out_dict = {f: {} for f in set(origin_list)}
    for o, t in zip(origin_list, td):
        out_dict[o][t] = td[t]
    return out_dict


def get_edge_labels(labels, buffer_size):
    """
    Encode transitions between differing states X and Y as strings of shape 'eX_Y'
    """
    edge_labels = np.zeros(labels.size, dtype='<U6')
    overhang_right = (buffer_size - 1) // 2
    overhang_left = (buffer_size - 1) - overhang_right
    oh_counter = 0
    cur_edge = ''
    cur_label = labels[0]
    for li, l in enumerate(labels):
        if l == cur_label:
            if oh_counter != 0:
                edge_labels[li] = cur_edge
                oh_counter -= 1
        else:
            cur_edge = f'e{cur_label}_{l}'
            edge_labels[li-overhang_left:li+1] = cur_edge
            cur_label = l
            oh_counter = overhang_right
    return edge_labels

def get_ssfret_dist(efret, idx=None):
    # DBSCAN filter
    print(f'{print_timestamp()}started a trace')
    efret = efret[np.invert(np.isnan(efret))]
    if len(efret > 30000):
        efret = np.random.choice(efret, size=30000)
    # clust = DBSCAN(eps=0.05).fit(efret.reshape(-1, 1))
    n_points = len(efret)
    clust = DBSCAN(eps=0.05).fit(efret.reshape(-1, 1))
    # clust = OPTICS(eps=0.05, cluster_method='dbscan').fit(efret.reshape(-1, 1))
    clust_mode = mode(clust.labels_)[0]
    efret_target = efret[clust.labels_ == clust_mode]

    # Curve fitting
    mu, sd = norm.fit(efret_target)
    srsd = sd / sqrt(len(efret_target))
    print(f'{print_timestamp()}finished a trace')
    if idx is None:
        return mu, sd, srsd, n_points
    return mu, sd, srsd, n_points, idx

def discrete2continuous(tm, framerate):
    """
    Convert discrete transition matrix such as output by HMMs to continous transition rates estimates
    """
    assert tm.ndim == 2
    assert tm.shape[0] == tm.shape[1]
    nb_states = tm.shape[0]
    rm = np.eye(nb_states) + framerate * logm(np.nan_to_num(tm, 0.0))
    rm[rm < 0] = 0.0  # correct small negative values
    rm[np.eye(nb_states, dtype=bool)] -= 1
    return rm
