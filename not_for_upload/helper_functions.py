import os
import warnings
import fnmatch
from datetime import datetime
import numpy as np
import shutil
import pathlib

from scipy.stats import norm, mode
from sklearn.cluster import DBSCAN, OPTICS
from math import sqrt


# --- I/O ---
def parse_input_path(location, pattern=None):
    """
    Take path, list of files or single file, Return list of files with path name concatenated.
    """
    if not isinstance(location, list):
        location = [location]
    all_files = []
    for loc in location:
        loc = os.path.abspath(loc)
        if os.path.isdir(loc):
            if loc[-1] != '/':
                loc += '/'
            for root, dirs, files in os.walk(loc):
                if pattern:
                    for f in fnmatch.filter(files, pattern):
                        all_files.append(os.path.join(root, f))
                else:
                    for f in files:
                        all_files.append(os.path.join(root, f))
        elif os.path.exists(loc):
            if pattern and fnmatch.fnmatch(loc, pattern):
                    all_files.append(loc)
    if not len(all_files):
        ValueError('Input file location(s) did not exist or did not contain any files.')
    return all_files

def parse_output_dir(out_dir, clean=False):
    out_dir = os.path.abspath(out_dir) + '/'
    if clean:
        shutil.rmtree(out_dir, ignore_errors=True)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def print_timestamp():
    dt = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
    return f'[{dt}]: '


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_var(a, window):
    return np.std(rolling_window(a,window), -1)


def rolling_cross_correlation(don, acc,  window):
    return np.array([np.convolve(a,b, mode='valid')[0] for a, b in zip(rolling_window(acc,window), rolling_window(don,window))])
    # return np.correlate(rolling_window(don,window), rolling_window(acc,window))


def rolling_corr_coef(don, acc,  window):
    return np.array([np.corrcoef(a,b)[0,1] for a, b in zip(rolling_window(acc,window), rolling_window(don,window))])


def condense_sequence(values, labels):
    """
    Take two numpy arrays, return list of tuples of continous stretches of labels in [label_column]
    :return: list of tuples of format (label, nb_elements, average)
    """
    if labels.ndim != labels.ndim != 1:
        raise ValueError(f'Dimensions of labels array ({labels.ndim}) and values array ({values.ndim}) are not 1')
    if len(labels) != len(values):
        raise ValueError(f'Values and labels arrays do not contain same number of elements ({len(values)} vs {len(labels)})')

    seq_condensed = [[labels[0], 0, []]]  # symbol, duration, E_FRET sum
    for s, i in zip(labels, values):
        if s == seq_condensed[-1][0]:
            seq_condensed[-1][1] += 1
            seq_condensed[-1][2].append(i)
        else:
            seq_condensed[-1][2] = np.nanmedian(seq_condensed[-1][2])
            seq_condensed.append([s, 1, [i]])
    seq_condensed[-1][2] = np.nanmedian(seq_condensed[-1][2])
    return seq_condensed


def get_ssfret_dist(efret, idx=None):
    # DBSCAN filter
    print(f'{print_timestamp()}started a trace')
    efret = efret[np.invert(np.isnan(efret))]
    if len(efret > 30000):
        efret = np.random.choice(efret, size=30000)
    # clust = DBSCAN(eps=0.05).fit(efret.reshape(-1, 1))
    n_points = len(efret)
    clust = OPTICS(eps=0.05, cluster_method='dbscan').fit(efret.reshape(-1, 1))
    clust_mode = mode(clust.labels_)[0]
    efret_target = efret[clust.labels_ == clust_mode]

    # Curve fitting
    mu, sd = norm.fit(efret_target)
    srsd = sd / sqrt(len(efret_target))
    print(f'{print_timestamp()}finished a trace')
    if idx is None:
        return mu, sd, srsd, n_points
    return mu, sd, srsd, n_points, idx