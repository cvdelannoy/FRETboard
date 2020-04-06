import os
import pandas as pd
import numpy as np
from os.path import basename
from FRETboard.helper_functions import colnames, colnames_alex, df_empty, get_tuple

class MainTable(object):

    def __init__(self, eps, l, d, gamma, alex, dat_list):
        self.trace_dict = {}
        self.label_dict = dict()
        self.eps, self.l, self.d, self.gamma, self.alex = eps, l, d, gamma, alex
        self.dat_list = dat_list
        self.init_table()

    def init_table(self):
        for fn in self.dat_list:
            fc = np.genfromtxt(fn, delimiter='\t')
            fc = get_tuple(fc.T, self.eps, self.l, self.d, self.gamma)
            df = pd.DataFrame(fc.T, columns=colnames_alex if self.alex else colnames)
            # df = pd.read_csv(fn, sep='\t', names=colnames_alex if self.alex else colnames)
            df.loc[:, 'predicted'] = 0.0
            self.trace_dict[basename(fn)] = df

        self.index_table = pd.DataFrame({'trace': list(self.trace_dict),
                                         'eps': self.eps, 'l': self.l, 'd': self.d, 'gamma':self.gamma,
                                         'data_timestamp':000, 'logprob': 0, 'mod_timestamp': 000}).set_index('trace')

        self.manual_table = pd.DataFrame({'trace': list(self.trace_dict),
                                          'is_labeled': False,
                                          'is_junk': False}).set_index('trace')

    def get_trace_dict(self):
        return self.trace_dict

    @property
    def accuracy(self):
        """
        Return array of per-trace accuracy values and mean accuracy over entire dataset
        :return:
        """
        if not len(self.label_dict):
            return np.array([np.nan], dtype=float), np.nan
        pred_dict = {tr: self.trace_dict[tr].predicted.to_numpy() for tr in self.trace_dict}
        nb_correct = np.array([np.sum(self.label_dict[idx] == pred_dict[idx])
                               for idx in self.label_dict if pred_dict[idx] is not None])
        nb_points = np.array([len(self.label_dict[idx]) for idx in self.label_dict
                              if pred_dict[idx] is not None])
        return nb_correct / nb_points * 100, nb_correct.sum() / nb_points.sum() * 100