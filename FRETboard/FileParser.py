import os
import numpy as np
import warnings
import pandas as pd
import base64
import h5py
import psutil
from FRETboard.H5Walker import H5Walker
from FRETboard.helper_functions import get_tuple
from FRETboard.SafeH5 import SafeH5
from FRETboard.SafeHDFStore import SafeHDFStore

class FileParser(object):
    def __init__(self, toparse_fn, traces_store_fn, main_pid):
        self.nb_files = 0
        self.dat_dict = {}
        self.alex = None
        self.framerate = None
        self.l = None
        self.d = None
        self.gamma = None
        self.eps = None
        self.data_timestamp = None
        self.chunk_size = 5
        self.toparse_fn = toparse_fn
        self.traces_store_fn = traces_store_fn
        self.main_pid = main_pid
        self.main_loop()

    def main_loop(self):
        while psutil.pid_exists(self.main_pid):
            # Check for new traces to parse
            h5w = H5Walker()
            to_parse_dict = {}
            with SafeH5(self.toparse_fn, 'a') as fh:
                self.update_filter_params(fh)
                fh.visititems(h5w)
                trace_list = h5w.names
                for trace in trace_list:
                    to_parse_dict[trace] = fh[trace][()]
                    del fh[trace]
            if len(to_parse_dict):
                self.nb_files = len(to_parse_dict)
                for fn in to_parse_dict:
                    if fn.endswith('.traces'):
                        self.parse_trace_file(fn, to_parse_dict[fn])
                    elif fn.endswith('.dat'):
                        self.parse_dat_file(fn, to_parse_dict[fn])

            # Check for traces requiring update to data
            with SafeHDFStore(self.traces_store_fn, 'r') as fh:
                if 'index_table' in fh:
                    index_table = fh.get('index_table')
                else:
                    index_table = None
            if index_table is None: continue
            update_idx = index_table.loc[index_table.data_timestamp != self.data_timestamp, :].index
            if not len(update_idx): continue
            self.update_traces(update_idx)

    def update_traces(self, update_idx):
        chunk_limit = self.chunk_size - 1
        out_dict = {}
        for ii, idx in enumerate(update_idx):
            with SafeH5(self.traces_store_fn, 'r') as fh:
                trace_old = fh['/traces/' + idx][()]  # todo fn right?
            if self.alex:
                trace_new = get_tuple(trace_old[(0, 1, 2, 5, 6), :], self.eps, self.l, self.d, self.gamma)  # todo indexing works here?
            else:
                trace_new = get_tuple(trace_old[(0, 1, 2), :], self.eps, self.l, self.d, self.gamma)
            out_dict[idx] = trace_new
            if ii >=chunk_limit:
                self.write_away_traces(out_dict)
                chunk_limit += self.chunk_size
                out_dict = {}
        if len(out_dict): self.write_away_traces(out_dict)

    def parse_dat_file(self, fn, fc):
        self.nb_files -= 1
        file_contents = fc.tostring().decode('utf-8')
        file_contents = np.column_stack([np.fromstring(n, sep=' ') for n in file_contents.split('\n') if len(n)])
        if not len(file_contents): return
        self.dat_dict[fn] = get_tuple(file_contents, self.eps, self.l, self.d, self.gamma)
        if len(self.dat_dict) > self.chunk_size or self.nb_files == 0:
            self.write_away_traces(self.dat_dict)
            self.dat_dict = dict()


    def parse_trace_file(self, fn, file_contents):
        self.nb_files -= 1
        nb_frames, _, nb_traces = np.frombuffer(file_contents, dtype=np.int16, count=3)
        nb_colors = 4 if self.alex else 2
        nb_samples = nb_traces // nb_colors
        traces_vec = np.frombuffer(file_contents, dtype=np.int16)
        traces_vec = traces_vec[3:]
        nb_points_expected = nb_colors * nb_samples * nb_frames
        traces_vec = traces_vec[:nb_points_expected]
        file_contents = traces_vec.reshape((nb_colors, nb_samples, nb_frames), order='F')
        fn_clean = os.path.splitext(fn)[0]
        fn_list = [f'{fn_clean}_{it}.dat' for it in range(nb_samples)]
        sampling_freq = 1.0 / self.framerate
        chunk_lim = self.chunk_size - 1
        out_dict = {}
        for fi, f in enumerate(np.hsplit(file_contents, file_contents.shape[1])):
            f = f.squeeze()
            time = np.arange(f.shape[1]) * sampling_freq
            out_dict[fn_list[fi]] = get_tuple(np.row_stack((time, f)), self.eps, self.l, self.d, self.gamma)
            if fi >= chunk_lim:
                self.write_away_traces(out_dict)
                out_dict = {}
                chunk_lim += self.chunk_size
        if len(out_dict):
            self.write_away_traces(out_dict)

    def update_filter_params(self, fh):
        """
        Retrieve filter params from traces_store. Done at start of loop and  after each chunk is written away.
        """
        (self.data_timestamp, self.framerate,
         self.l, self.d, self.gamma,
         self.eps, self.alex) = (fh.attrs['data_timestamp'], fh.attrs['framerate'],
                                 fh.attrs['l'], fh.attrs['d'], fh.attrs['gamma'],
                                 fh.attrs['eps'], fh.attrs['alex'])

    def write_away_traces(self, out_dict):
        # note: mod_timestamp set to -1 to signify nan integer
        index_table = pd.DataFrame({'trace': list(out_dict),
                      'eps': self.eps, 'l': self.l, 'd': self.d, 'gamma': self.gamma, 'data_timestamp': self.data_timestamp,
                      'logprob': np.nan, 'mod_timestamp': -1}).set_index('trace')
        with SafeH5(self.traces_store_fn, 'a') as fh:
            for tk in out_dict:
                if 'traces/' + tk in fh:
                    del fh['traces/' + tk]
                fh['traces/' + tk] = out_dict[tk]
        with SafeHDFStore(self.traces_store_fn, 'a') as fh:
            fh.put('index_table', index_table, format='table', append=True, min_itemsize={'index': 50})
        with SafeH5(self.toparse_fn, 'r') as fh:
            self.update_filter_params(fh)
    #
    #
    # def get_tuple(self, fc):
    #     """
    #     construct tuple from numpy file content array. Tuple measures [nb_features x len(trace)]
    #     """
    #     return get_tuple(fc, self.eps, self.l, self.d, self.gamma)
    #
    #     window = 5
    #     ss = (window - 1) // 2  # sequence shortening
    #
    #     time = fc[0, :].astype(np.float64)
    #     f_dex_dem_raw = fc[1, :].astype(np.float64)
    #     f_dex_aem_raw = fc[2, :].astype(np.float64)
    #     f_dex_dem = bg_filter_trace(f_dex_dem_raw, self.eps)
    #     f_dex_aem = bg_filter_trace(f_dex_aem_raw, self.eps)
    #     cross_correct = 0
    #     if fc.shape[0] == 5:
    #         f_aex_dem_raw = fc[3, :].astype(np.float64)
    #         f_aex_aem_raw = fc[4, :].astype(np.float64)
    #         f_aex_dem = bg_filter_trace(f_aex_dem_raw, self.eps)
    #         f_aex_aem = bg_filter_trace(f_aex_aem_raw, self.eps)
    #         cross_correct = self.l * f_dex_dem + self.d * f_aex_aem
    #     f_fret = f_dex_aem - cross_correct
    #     i_sum = self.gamma * f_dex_dem + f_fret
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         E_FRET = f_fret / i_sum
    #     E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out
    #
    #     correlation_coefficient = np.full_like(E_FRET, np.nan)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         correlation_coefficient[ss:-ss] = rolling_corr_coef(f_dex_dem, f_dex_aem, window)
    #     E_FRET_sd = np.full_like(E_FRET, np.nan)
    #     E_FRET_sd[ss:-ss] = rolling_var(E_FRET, window)
    #     i_sum_sd = np.full_like(E_FRET, np.nan)
    #     i_sum_sd[ss:-ss] = rolling_var(i_sum, window)
    #     if fc.shape[0] == 5:
    #         return np.vstack([time, f_dex_dem_raw, f_dex_aem_raw, f_dex_dem, f_dex_aem,
    #                           f_aex_dem_raw, f_aex_aem_raw, f_aex_dem, f_aex_aem,
    #                           E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient])
    #     else:
    #         return np.vstack([time, f_dex_dem_raw, f_dex_aem_raw, f_dex_dem, f_dex_aem,
    #                           E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient])
    #
    #
    # def add_derived_features(self, tup):
    #     """
    #     Add to a given tuple:
    #     - E_FRET
    #     - E_FRETsd
    #     - i_sum
    #     - i_sum_sd
    #     - corr_coef
    #     :param tup:
    #     :return:
    #     """
    #     window = 5
    #     ss = (window - 1) // 2  # sequence shortening
    #     cross_correct = 0
    #     if len(tup) == 9:
    #         cross_correct = self.l * tup[8] + self.d * tup[7]
    #     i_sum = np.sum((self.gamma * tup[3], tup[4] - cross_correct), axis=0)  # gamma * i_don * + (i_acc - cross correct)
    #     f_fret = tup[4] - cross_correct
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         E_FRET = f_fret / (self.gamma * tup[3] + f_fret)  # E_FRET = f_fret / (self.gamma * i_don + f_fret)
    #         # E_FRET = np.divide(i_acc - cross_correct, i_sum)
    #     E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out
    #
    #     correlation_coefficient = np.full_like(E_FRET, np.nan)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         correlation_coefficient[ss:-ss] = rolling_corr_coef(tup[3], tup[4], window)
    #     E_FRET_sd = np.full_like(E_FRET, np.nan)
    #     E_FRET_sd[ss:-ss] = rolling_var(E_FRET, window)
    #     i_sum_sd = np.full_like(E_FRET, np.nan)
    #     i_sum_sd[ss:-ss] = rolling_var(i_sum, window)
    #     tup.extend([E_FRET, E_FRET_sd, i_sum, i_sum_sd, correlation_coefficient])
    #     return tup
