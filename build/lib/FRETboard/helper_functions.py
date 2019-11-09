from datetime import datetime
import numpy as np


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
