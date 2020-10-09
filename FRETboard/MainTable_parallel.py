import numpy as np
import base64
import pandas as pd
from time import sleep
from multiprocessing import Process
from FRETboard.SafeH5 import SafeH5
from FRETboard.SafeHDFStore import SafeHDFStore
from FRETboard.FileParser import FileParser
from FRETboard.helper_functions import numeric_timestamp, colnames_w_labels, colnames_alex_w_labels, df_empty
from datetime import datetime


class MainTable(object):

    def __init__(self, framerate, eps, l, d, gamma, alex, h5_dir, main_process):
        self.main_process = main_process
        self.index_table = None  # contains names and meta data of traces in hdf store
        self.manual_table = None # contains info as entered by user: is_labeled, is_junk
        if h5_dir[-1] != '/': h5_dir += '/'
        self.traces_store_fn = h5_dir + 'traces_store.h5'
        self.predict_store_fn = h5_dir + 'predict_store_fn.h5'
        self.toparse_fn = h5_dir + 'to_parse.h5'
        self.label_dict = dict()
        self._eps, self._l, self._d, self._gamma, self._alex = eps, l, d, gamma, alex
        self.framerate = framerate
        # _ = self.init_table(framerate, alex)
        # self.file_parser_process = self.init_table(framerate, alex)

    @property
    def framerate(self):
        return self._framerate

    @property
    def eps(self):
        return self._eps

    @property
    def l(self):
        return self._l

    @property
    def d(self):
        return self._d

    @property
    def gamma(self):
        return self._gamma

    @property
    def alex(self):
        return self._alex

    @framerate.setter
    def framerate(self, framerate):
        self.data_timestamp = numeric_timestamp()
        with SafeH5(self.toparse_fn, 'a') as fh:
            fh.attrs['framerate'] = framerate
            fh.attrs['data_timestamp'] = self.data_timestamp
        self._framerate = framerate

    @alex.setter
    def alex(self, alex):
        self.data_timestamp = numeric_timestamp()
        with SafeH5(self.toparse_fn, 'a') as fh:
            fh.attrs['alex'] = alex
            fh.attrs['data_timestamp'] = self.data_timestamp
        self._alex = alex

    @eps.setter
    def eps(self, eps):
        self.data_timestamp = numeric_timestamp()
        with SafeH5(self.toparse_fn, 'a') as fh:
            fh.attrs['eps'] = eps
            fh.attrs['data_timestamp'] = self.data_timestamp
        self._eps = eps

    @l.setter
    def l(self, l):
        self.data_timestamp = numeric_timestamp()
        with SafeH5(self.toparse_fn, 'a') as fh:
            fh.attrs['l'] = l
            fh.attrs['data_timestamp'] = self.data_timestamp
        self._l = l

    @d.setter
    def d(self, d):
        self.data_timestamp = numeric_timestamp()
        with SafeH5(self.toparse_fn, 'a') as fh:
            fh.attrs['d'] = d
            fh.attrs['data_timestamp'] = self.data_timestamp
        self._d = d

    @gamma.setter
    def gamma(self, gamma):
        self.data_timestamp = numeric_timestamp()
        with SafeH5(self.toparse_fn, 'a') as fh:
            fh.attrs['gamma'] = gamma
            fh.attrs['data_timestamp'] = self.data_timestamp
        self._gamma = gamma

    def init_table(self):
        # Create index table
        self.index_table = pd.DataFrame(
            columns=[
                'trace', 'eps', 'l', 'd', 'gamma', 'data_timestamp','logprob', 'mod_timestamp'
            ]).set_index('trace')
        self.manual_table = df_empty(columns=['trace', 'is_labeled', 'is_junk'], dtypes=[str, np.bool, np.bool]).set_index('trace')
        with SafeHDFStore(self.traces_store_fn, 'a') as fh:
            fh.put('index_table', value=self.index_table, format='table', append=True)

        # make traces group
        with SafeH5(self.traces_store_fn, 'a') as fh:
            fh.create_group('traces')

        # hdf5 file for transfer to file parser
        self.data_timestamp = numeric_timestamp()
        with SafeH5(self.toparse_fn, 'w') as fh:
            (fh.attrs['data_timestamp'],
             fh.attrs['framerate'], fh.attrs['eps'],
             fh.attrs['l'], fh.attrs['d'],
             fh.attrs['gamma'], fh.attrs['alex']) = (self.data_timestamp, self.framerate, self.eps, self.l, self.d,
                                                     self.gamma, self.alex)

        # hdf5 file for transfer to predictor
        with SafeH5(self.predict_store_fn, 'w') as fh:
            pass
        fp_process = Process(target=FileParser, args=(self.toparse_fn, self.traces_store_fn, self.main_process),
                             name='file_parser')
        fp_process.start()
        return fp_process

    def get_trace(self, idx, await_labels=False):
        with SafeH5(self.traces_store_fn, 'r') as fh:
            tup = fh['/traces/'+idx][()]
        dummy = np.array([]) if await_labels else np.zeros(tup.shape[1])
        while True:
            with SafeH5(self.predict_store_fn, 'r') as fh:
                pred = fh.get('/' + idx, dummy)[()]
            if len(pred):
                break  # todo somehow signal that this example has to be classified fast
            # sleep(0.1)
        # if not self.manual_table.loc[idx, 'is_labeled']:
        #     self.label_dict[idx] = pred
        return pd.DataFrame(data=np.vstack((tup, pred)).T, columns=colnames_alex_w_labels)

    def get_trace_dict(self, labeled_only=False):
        out_dict = {}
        idx_list = list(self.label_dict) if labeled_only else self.index_table.index
        for idx in idx_list:
            out_dict[str(idx)] = self.get_trace(idx)
        return out_dict

    def add_tuple(self, content, fn):
        _, b64_contents = content.split(",", 1)  # remove the prefix that JS adds
        file_contents = base64.b64decode(b64_contents)
        with SafeH5(self.toparse_fn, 'a') as fh:
            fh[fn] = np.void(file_contents)

    def update_index(self):
        with SafeHDFStore(self.traces_store_fn, 'r') as fh:
            if 'index_table' in fh: self.index_table = fh.get('index_table')
        new_indices = [idx for idx in self.index_table.index if idx not in self.manual_table.index]
        if len(new_indices):
            new_df = pd.DataFrame({'trace': new_indices, 'is_labeled': False, 'is_junk': False}).set_index('trace')
            self.manual_table = pd.concat((self.manual_table, new_df))
        # invalidate accuracy cached property
        # self._invalidate_property('accuracy')

    # def push_index(self, idx, col, new):
    #     with SafeHDFStore(self.traces_store_fn, 'a') as fh:
    #         fh.loc[idx, col] = new
    #         self.index_table.loc[idx, col] = new

    @property
    def accuracy(self):
        """
        Return array of per-trace accuracy values and mean accuracy over entire dataset
        :return:
        """
        if not len(self.label_dict):
            return np.array([np.nan], dtype=float), np.nan
        try:
            with SafeH5(self.predict_store_fn, 'r') as fh:
                pred_dict = {idx: fh.get('/' + idx, None)[()] for idx in self.label_dict}
        except:
            cp=1
        nb_correct = np.array([np.sum(self.label_dict[idx] == pred_dict[idx])
                               for idx in self.label_dict if pred_dict[idx] is not None])
        nb_points = np.array([len(self.label_dict[idx]) for idx in self.label_dict if pred_dict[idx] is not None])
        return nb_correct / nb_points * 100, nb_correct.sum() / nb_points.sum() * 100

    # def _invalidate_property(self, prop):
    #     if prop in self.__dict__:
    #         del self.__dict__[prop]
