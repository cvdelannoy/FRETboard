import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import DBSCAN

from FRETboard.helper_functions import rolling_corr_coef, rolling_var

class MainTable(object):

    def __init__(self, data):
        self.data = data

    @property
    def data(self):
        """
        traces including those predicted/marked junk
        """
        return self._data

    @property
    def data_clean(self):
        """
        traces without those predicted/marked junk
        """
        return self._data.loc[np.invert(self._data.predicted_junk), :]

    @data.setter
    def data(self, dat_files):
        nb_files = len(dat_files)
        df_out = pd.DataFrame({
            'time': [np.array([], dtype=np.int64)] * nb_files,
            'i_don': [np.array([], dtype=np.int64)] * nb_files,
            'i_acc': [np.array([], dtype=np.int64)] * nb_files,
            'i_sum': [np.array([], dtype=np.float64)] * nb_files,
            'i_sum_sd': [np.array([], dtype=np.float64)] * nb_files,
            'E_FRET': [np.array([], dtype=np.float64)] * nb_files,
            'correlation_coefficient': [np.array([], dtype=np.float64)] * nb_files,
            'E_FRET_sd': [np.array([], dtype=np.float64)] * nb_files,
            'labels': [np.array([], dtype=np.int64)] * nb_files,
            'edge_labels': [np.array([], dtype=np.int64)] * nb_files,
            'prediction': [np.array([], dtype=np.int64)] * nb_files},
            index=dat_files, dtype=object)
        df_out['logprob'] = pd.Series([np.nan] * nb_files, dtype=float)
        df_out['is_labeled'] = pd.Series([False] * nb_files, dtype=bool)
        df_out['is_junk'] = pd.Series([False] * nb_files, dtype=bool)
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
        :return: None
        """
        window = 9
        ss = (window - 1) // 2  # sequence shortening
        time = fc[0, :].astype(np.float64)
        i_don = fc[1, :].astype(np.float64)
        i_acc = fc[2, :].astype(np.float64)

        # DBSCAN filter
        min_clust = int(fc.shape[1] * 0.02)
        for tr in (i_don, i_acc):
            clust = DBSCAN(eps=1.0, min_samples=min_clust).fit(tr.reshape(-1, 1))
            tr -= np.nanmin([np.mean(tr[clust.labels_ == lab]) for lab in np.unique(clust.labels_)]).astype(fc.dtype)

        i_sum = np.sum((i_don, i_acc), axis=0)
        # E_FRET = np.divide(i_acc, np.sum((i_don, i_acc), axis=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E_FRET = np.clip(np.divide(i_acc, np.sum((i_don, i_acc), axis=0)), a_min=0.0, a_max=1.0)
        E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out

        correlation_coefficient = np.full_like(E_FRET, np.nan)
        correlation_coefficient[ss:-ss] = rolling_corr_coef(i_don, i_acc, window)
        # correlation_coefficient = rolling_corr_coef(i_don, i_acc, window)
        E_FRET_sd = np.full_like(E_FRET, np.nan)
        E_FRET_sd[ss:-ss] = rolling_var(E_FRET, window)
        # E_FRET_sd = rolling_var(E_FRET, window)
        i_sum_sd = np.full_like(E_FRET, np.nan)
        i_sum_sd[ss:-ss] = rolling_var(i_sum, window)

        self._data.loc[fn] = (time, i_don, i_acc, i_sum, i_sum_sd, E_FRET, correlation_coefficient, E_FRET_sd,
                              [], [], [], np.nan, False, False, False)

        # self._data.loc[fn] = (time[ss:-ss], i_don[ss:-ss], i_acc[ss:-ss],
        #                       i_sum[ss:-ss], E_FRET[ss:-ss], correlation_coefficient, E_FRET_sd, [], [], [], np.nan,
        #                       False, False, False)

    def add_df_list(self, df_list):
        """
        Add a list of pd dfs to the current table at once
        """
        if type(df_list) != list:
            df_list = [df_list]
        self._data = pd.concat(df_list + [self.data])


    def del_tuple(self, idx):
        self._data.loc[idx, 'is_junk'] = True
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

    # --- derived features ---
    @property
    def accuracy(self):
        """
        Return array of per-trace accuracy values and mean accuracy over entire dataset
        :return:
        """
        if not any(self.data.is_labeled):
            return np.array([np.nan], dtype=float), np.nan
        is_predicted = self.data.prediction.apply(lambda x: len(x) != 0)
        labeled_data = self.data.loc[ np.logical_and(self.data.is_labeled, is_predicted), ('prediction', 'labels')]
        nb_correct = labeled_data.apply(lambda x: np.sum(np.equal(x.prediction, x.labels)), axis=1)
        nb_points = labeled_data.apply(lambda x: x.labels.size, axis=1)
        return nb_correct / nb_points * 100, nb_correct.sum() / nb_points.sum() * 100
