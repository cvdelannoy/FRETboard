import os
import warnings
import fnmatch
from datetime import datetime
import numpy as np


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
            all_files.extend(loc)
        else:
            warnings.warn('Given file/dir %s does not exist, skipping' % loc, RuntimeWarning)
    if not len(all_files):
        ValueError('Input file location(s) did not exist or did not contain any files.')
    return all_files

# def read_line(fc, full_set=False):
#     # window = 9
#     window = 15
#     ss = (window - 1) // 2  # sequence shortening
#     fc[fc <= 0] = np.finfo(np.float64).eps  # hacky, required to get rid of overzealous background subtraction
#     time = fc[0, :]
#     i_don = fc[1, :]
#     i_acc = fc[2, :]
#     i_sum = np.sum((i_don, i_acc), axis=0)
#     # i_sum = i_sum / i_sum.max()F
#     E_FRET = np.divide(i_acc, np.sum((i_don, i_acc), axis=0))
#     # sd_roll = rolling_var(E_FRET, window)
#     sd_roll = rolling_corr_coef(i_don, i_acc, window)
#     if full_set:
#         return (time[ss:-ss], i_don[ss:-ss], i_acc[ss:-ss],
#                 i_sum[ss:-ss], E_FRET[ss:-ss], sd_roll,
#                 np.array([], dtype=np.int64), np.array([], dtype=np.int64),  # labels, edge labels
#                 np.array([], dtype=np.int64),  # prediction
#                 np.array([], dtype=np.float64), False)  # logprob, is_labeled
#     return time[ss:-ss], i_don[ss:-ss], i_acc[ss:-ss], i_sum[ss:-ss], E_FRET[ss:-ss], sd_roll


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
