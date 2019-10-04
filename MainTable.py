import numpy as np
import pandas as pd

from helper_functions import rolling_corr_coef

class MainTable(object):

    def __init__(self, data):
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, dat_files):
        nb_files = len(dat_files)
        df_out = pd.DataFrame({
            'time': [np.array([], dtype=np.int64)] * nb_files,
            'i_don': [np.array([], dtype=np.int64)] * nb_files,
            'i_acc': [np.array([], dtype=np.int64)] * nb_files,
            'i_sum': [np.array([], dtype=np.float64)] * nb_files,
            'E_FRET': [np.array([], dtype=np.float64)] * nb_files,
            'sd_roll': [np.array([], dtype=np.float64)] * nb_files,
            'labels': [np.array([], dtype=np.int64)] * nb_files,
            'edge_labels': [np.array([], dtype=np.int64)] * nb_files,
            'prediction': [np.array([], dtype=np.int64)] * nb_files,
            'logprob': [np.array([], dtype=np.float64)] * nb_files},
            index=dat_files, dtype=object)
        df_out['is_labeled'] = pd.Series([False] * nb_files, dtype=bool)
        for dat_file in dat_files:
            try:
                fc = np.loadtxt(dat_file)
                self.add_tuple(fc.T, dat_file)
            except:
                print('File {} could not be read, skipping'.format(dat_file))
                df_out.drop([dat_file], inplace=True)
        self._data = df_out

    def add_tuple(self, fc, fn):
        """
        Add a new tuple to the data table
        :param fc: file contents
        :param fn: file name, will be used as index
        :return: None
        """
        # window = 9
        window = 15
        ss = (window - 1) // 2  # sequence shortening
        fc[fc <= 0] = np.finfo(np.float64).eps  # hacky, required to get rid of overzealous background subtraction
        time = fc[0, :]
        i_don = fc[1, :]
        i_acc = fc[2, :]
        i_sum = np.sum((i_don, i_acc), axis=0)
        E_FRET = np.divide(i_acc, np.sum((i_don, i_acc), axis=0))
        sd_roll = rolling_corr_coef(i_don, i_acc, window)
        self._data.loc[fn] = (time[ss:-ss], i_don[ss:-ss], i_acc[ss:-ss],
                              i_sum[ss:-ss], E_FRET[ss:-ss], sd_roll, [], [], [], [], False)

    def del_tuple(self, idx):
        self._data.drop(idx, inplace=True)

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
        if not any(self.data.is_labeled):
            return np.array([np.nan], dtype=float)
        labeled_data = self.data.loc[self.data.is_labeled, ('prediction', 'labels')]
        nb_correct = labeled_data.apply(lambda x: np.sum(np.equal(x.prediction, x.labels)), axis=1)
        nb_points = labeled_data.apply(lambda x: x.labels.size, axis=1)
        return nb_correct / nb_points * 100
