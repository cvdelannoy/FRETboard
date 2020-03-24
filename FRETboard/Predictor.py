from FRETboard.MainTable import MainTable
import os
import numpy as np
import pandas as pd
from time import sleep
from FRETboard.SafeHDFStore import SafeHDFStore
from FRETboard.SafeH5 import SafeH5
from FRETboard.helper_functions import colnames, colnames_alex
import pickle


class Predictor(object):
    def __init__(self, classifier, h5_dir, main_process):
        if h5_dir[-1] != '/': h5_dir += '/'
        self.h5_dir = h5_dir
        self.chunk_size = 5
        self.traces_store_fn = h5_dir + 'traces_store.h5'
        self.predict_store_fn = h5_dir + 'predict_store_fn.h5'
        self.main_process = main_process
        self.classifier = classifier
        self.pid = os.getpid()
        self.read_id = '/read_' + str(self.pid)
        self.write_id = 'write_' + str(self.pid)
        # self.run()

    def run(self):
        while self.main_process.is_alive:

            self.check_mod_update()
            if self.classifier.trained is None: continue

            # Check for data to predict
            with SafeHDFStore(self.traces_store_fn) as fh:
                index_table = fh.get('index_table')
            pred_idx = index_table.index[index_table.mod_timestamp != self.classifier.timestamp][:self.chunk_size]

            # predict
            state_seq_dict = {}
            for idx in pred_idx:
                trace_df = self.get_trace(idx)
                state_seq_dict[idx], index_table.loc[idx, 'logprob'] = self.classifier.predict(trace_df)
            index_table.mod_timestamp = self.classifier.timestamp

            # Save new predictions
            with SafeH5(self.predict_store_fn) as fh:
                for idx in state_seq_dict:
                    if idx in fh:
                        fh[idx][:] = state_seq_dict[idx]
                    else:
                        fh[idx] = state_seq_dict[idx]
            with SafeHDFStore(self.traces_store_fn) as fh:
                fh.remove('index_table', where='index in index_table.index') # todo check
                fh.append('index_table', index_table, append=True, data_columns=True)

    def check_mod_update(self):
        while True:  # only load if 1 mod file in folder
            mod_fn = [fn for fn in os.listdir(self.h5_dir) if fn.endswith('.mod')]
            if len(mod_fn) <= 1: break

        if len(mod_fn):
            if mod_fn[0][3:] != self.classifier.timestamp:
                while True:
                    try:
                        with open(f'{self.h5_dir}/{mod_fn[0]}', 'rb') as fh:
                            self.classifier = pickle.load(fh)
                    except:
                        continue
                    break
        return

    def get_trace(self, idx):
        with SafeH5(self.traces_store_fn, 'r') as fh:
            tup = fh['/traces/'+idx][()]
        cn = colnames if tup.shape[0] == 10 else colnames_alex
        return pd.DataFrame(data=tup.T, columns=cn)
