import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import DBSCAN, OPTICS
from joblib import Parallel, delayed

from FRETboard.helper_functions import rolling_corr_coef, rolling_var


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
        return self._data.loc[np.invert(self._data.predicted_junk).astype(bool), :]

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
        time = fc[0, :].astype(np.float64)
        i_don = fc[1, :].astype(np.float64)
        i_acc = fc[2, :].astype(np.float64)

        E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient = self.get_derived_features(i_don, i_acc)

        self._data.loc[fn] = (time, i_don, i_acc, i_don.copy(), i_acc.copy(), i_sum, i_sum_sd, E_FRET, correlation_coefficient, E_FRET_sd,
                              [], [], [], np.nan, False, False, False)

    def add_df_list(self, df_list):
        """
        Add a list of pd dfs to the current table at once
        """
        if type(df_list) != list:
            df_list = [df_list]
        self._data = pd.concat(df_list + [self.data])
        self._data.is_labeled = self._data.is_labeled.astype(bool)  # todo not elegant, track down where is_labeled stops being bool


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

    def subtract_background(self, eps, idx=None):
        threads=8
        if idx is None:
            idx = self._data.index
        if len(idx) == 1:
            self._data.update(self.parallel_subtract(self._data.loc[idx[0]], eps))
        df_list = Parallel(n_jobs=threads)(delayed(self.parallel_subtract)(data, eps)
                                      for data in np.array_split(self.data.loc[idx], threads))
        for df in df_list:
            self._data.update(df)

    @staticmethod
    def parallel_subtract(data, eps):
        if type(data) == pd.Series:
            cols = data.index
            data = data.to_frame().T
            data.columns = cols
        for tidx, tup in data.iterrows():
            for nt in (('i_don_raw', 'i_don'), ('i_acc_raw', 'i_acc')):
                min_clust = 10
                pc10 = np.percentile(tup.loc[nt[0]], 20)
                tr = tup.loc[nt[0]][tup.loc[nt[0]] < pc10]
                clust = DBSCAN(eps=eps, min_samples=min_clust).fit(tr.reshape(-1, 1)).labels_
                if len(np.unique(clust)) == 0:
                    data.at[tidx, nt[1]] = tup.loc[nt[0]]
                    continue
                med_list = np.array([np.median(tr[clust == lab]) for lab in np.unique(clust)])
                data.at[tidx, nt[1]] = tup.loc[nt[0]] - np.min(med_list)
            for feat, vec in zip (['E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient'],
                                  get_derived_features(data.loc[tidx, 'i_don'], data.loc[tidx, 'i_acc'])):
                data.at[tidx, feat] = vec
        return data

    def restore_background(self):
        for tidx, tup in self.data.iterrows():
            for nt in (('i_don_raw', 'i_don'), ('i_acc_raw', 'i_acc')):
                self.data.at[tidx, nt[1]] = tup.loc[nt[0]]
            for feat, vec in zip(['E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient'],
                                 self.get_derived_features(self.data.loc[tidx, 'i_don'], self.data.loc[tidx, 'i_acc'])):
                self.data.at[tidx, feat] = vec

    @staticmethod
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
