import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import DBSCAN, OPTICS
from joblib import Parallel, delayed

from FRETboard.helper_functions import subtract_background_fun, get_derived_features, bg_filter_trace, parallel_subtract


class MainTable(object):

    def __init__(self, data, eps):
        self.eps = eps
        self.data = data

    @property
    def data(self):
        """
        All traces
        """
        return self._data

    @property
    def data_clean(self):
        """
        traces without those predicted/marked junk, not currently update with bg subtraction
        """
        # return self._data.loc[np.invert(self._data.predicted_junk).astype(bool), :]
        return self._data.loc[np.logical_and(np.invert(self._data.predicted_junk).astype(bool),
                                             np.nan_to_num(self._data.eps) == np.nan_to_num(self.eps)), :]

    @property
    def is_junk(self):
        return np.logical_or(self._data.marked_junk, self._data.predicted_junk)

    @data.setter
    def data(self, dat_files):
        nb_files = len(dat_files)
        df_out = pd.DataFrame({
            'time': [np.array([], dtype=np.int64)] * nb_files,
            'i_don_raw': [np.array([], dtype=np.int64)] * nb_files,
            'i_acc_raw': [np.array([], dtype=np.int64)] * nb_files,
            'i_don': [np.array([], dtype=np.int64)] * nb_files,
            'i_acc': [np.array([], dtype=np.int64)] * nb_files,
            'i_sum': [np.array([], dtype=np.float64)] * nb_files,
            'i_sum_sd': [np.array([], dtype=np.float64)] * nb_files,
            'E_FRET': [np.array([], dtype=np.float64)] * nb_files,
            'E_FRET_sd': [np.array([], dtype=np.float64)] * nb_files,
            'correlation_coefficient': [np.array([], dtype=np.float64)] * nb_files,
            'labels': [np.array([], dtype=np.int64)] * nb_files,
            'edge_labels': [np.array([], dtype=np.int64)] * nb_files,
            'prediction': [np.array([], dtype=np.int64)] * nb_files},
            index=dat_files, dtype=object)
        df_out['eps'] = pd.Series([np.nan] * nb_files, dtype=int)
        df_out['logprob'] = pd.Series([np.nan] * nb_files, dtype=float)
        df_out['is_labeled'] = pd.Series([False] * nb_files, dtype=bool)
        df_out['is_predicted'] = pd.Series([False] * nb_files, dtype=bool)
        df_out['marked_junk'] = pd.Series([False] * nb_files, dtype=bool)
        df_out['predicted_junk'] = pd.Series([False] * nb_files, dtype=bool)
        self._data = df_out
        for dat_file in dat_files:
            try:
                fc = np.loadtxt(dat_file)
                self.add_tuple(fc.T, dat_file)
            except:
                print('File {} could not be read, skipping'.format(dat_file))
                df_out.drop([dat_file], inplace=True)

    def add_tuple(self, fc, fn):
        """
        Add a new tuple to the data table
        :param fc: file contents, np array of size [3 x len(sequence)], rows are time (s), don acc
        :param fn: file name, will be used as index
        :param eps: epsilon value for background subtraction, or None if no bg subtraction
        :return: None
        """
        time = fc[0, :].astype(np.float64)
        i_don_raw = fc[1, :].astype(np.float64)
        i_acc_raw = fc[2, :].astype(np.float64)
        if not np.isnan(self.eps):
            i_don = bg_filter_trace(i_don_raw, self.eps)
            i_acc = bg_filter_trace(i_acc_raw, self.eps)
        else:
            i_don = i_don_raw.copy()
            i_acc = i_acc_raw.copy()

        E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient = get_derived_features(i_don, i_acc)

        self._data.loc[fn] = (time,
                              i_don_raw, i_acc_raw,     # traces w/o background subtraction
                              i_don, i_acc,             # traces with background subtraction
                              i_sum, i_sum_sd,          # summed intensity, rolling window sd of summed intensity
                              E_FRET, E_FRET_sd,        # FRET efficiency, rolling window sd of E_FRET
                              correlation_coefficient,  # rolling window correlation coefficient i_don vs i_acc
                              [], [],                   # manual labels [object], manual edge labels [object]
                              [],                       # predicted labels [object]
                              self.eps,                      # DBSCAN filter epsilon value [int]
                              np.nan,                   # logprob of prediction [float]
                              False, False,             # is_labeled [bool], is_predicted [bool]
                              False, False)             # marked_junk [bool], predicted_junk [bool]

    def add_df_list(self, df_list):
        """
        Add a list of pd dfs to the current table at once
        """
        if type(df_list) != list:
            df_list = [df_list]
        self._data = pd.concat(df_list + [self.data])
        self._data.is_labeled = self._data.is_labeled.astype(bool)  # todo not elegant, track down where is_labeled stops being bool


    def del_tuple(self, idx):
        self._data.loc[idx, 'marked_junk'] = True
        # self._data.drop(idx, inplace=True)

    def set_value(self, example_idx, column, value, idx=None):
        """
        Set a value for a column in the current example in the original dataframe. Workaround to avoid 'chain indexing'.
        If cell contains a list/array, may provide index to define which values should be changed.
        """
        if idx is None:
            self.data.at[example_idx, column] = value
        else:
            self.data.at[example_idx, column][idx] = value

    # def subtract_background(self, eps, idx=None):
    # # todo: multiprocessing variant, do not delete!
    #
    #     threads=8
    #     if idx is None:
    #         idx = self._data.index
    #     if len(idx) == 1:
    #         self._data.loc[idx[0]] = parallel_subtract(self._data.loc[idx[0]], eps)
    #         return
    #     df_list = Parallel(n_jobs=threads)(delayed(parallel_subtract)(data, eps)
    #                                   for data in np.array_split(self.data.loc[idx], threads))
    #     for df in df_list:
    #         if type(df) == pd.Series:
    #             self._data.loc[idx[0]] = df
    #         else:
    #             self._data.update(df)

    def subtract_background(self, idx):
        self._data.loc[idx] = parallel_subtract(self._data.loc[idx], self.eps)

    # --- derived properties ---
    @property
    def accuracy(self):
        """
        Return array of per-trace accuracy values and mean accuracy over entire dataset
        :return:
        """
        if not any(self.data.is_labeled):
            return np.array([np.nan], dtype=float), np.nan
        labeled_data = self.data.loc[ np.logical_and(self.data.is_labeled, self.data.is_predicted), ('prediction', 'labels')]
        nb_correct = labeled_data.apply(lambda x: np.sum(np.equal(x.prediction, x.labels)), axis=1)
        nb_points = labeled_data.apply(lambda x: x.labels.size, axis=1)
        return nb_correct / nb_points * 100, nb_correct.sum() / nb_points.sum() * 100
