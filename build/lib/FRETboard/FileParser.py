import os, sys
from os.path import splitext
import numpy as np
from pathlib import Path
import warnings
import pandas as pd
import base64
import h5py
import io
from FRETboard.H5Walker import H5Walker
from FRETboard.helper_functions import get_tuple
from FRETboard.SafeH5 import SafeH5
from FRETboard.GracefulKiller import GracefulKiller
from FRETboard.SafeHDFStore import SafeHDFStore

class FileParser(object):
    def __init__(self, toparse_fn, traces_store_fn, main_pid):
        self.nb_files = 0
        self.dat_dict = {}
        self.alex = None
        self.traceswitch = None
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
        killer = GracefulKiller()
        while not killer.kill_now:
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
                    elif fn.endswith('.nc'):
                        self.parse_nc_file(fn, to_parse_dict[fn])
                    elif fn.endswith('.hdf5') or fn.endswith('h5'):
                        self.parse_photonhdf5(to_parse_dict[fn])

            # Check for traces requiring update to data todo why was this necessary again?
            with SafeHDFStore(self.traces_store_fn, 'r') as fh:
                if 'index_table' in fh:
                    index_table = fh.get('index_table')
                else:
                    index_table = None
            if index_table is None: continue
            update_idx = index_table.loc[index_table.data_timestamp != self.data_timestamp, :].index
            if not len(update_idx): continue
            self.update_traces(update_idx)
        sys.exit(0)

    def update_traces(self, update_idx):
        chunk_limit = self.chunk_size - 1
        out_dict = {}
        for ii, idx in enumerate(update_idx):
            with SafeH5(self.traces_store_fn, 'r') as fh:
                trace_old = fh[str(Path('/traces/') / idx)][()]  # todo fn right?
            if self.alex:
                trace_new = get_tuple(trace_old[(0, 1, 2, 5, 6), :], self.eps, self.l, self.d, self.gamma, self.traceswitch)  # todo indexing works here?
            else:
                trace_new = get_tuple(trace_old[(0, 1, 2), :], self.eps, self.l, self.d, self.gamma, self.traceswitch)
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
        self.dat_dict[fn] = get_tuple(file_contents, self.eps, self.l, self.d, self.gamma, self.traceswitch)
        if len(self.dat_dict) > self.chunk_size or self.nb_files == 0:
            self.write_away_traces(self.dat_dict)
            self.dat_dict = dict()

    def parse_nc_file(self, fn, fc):
        fn_clean = Path(fn).stem
        with io.BytesIO(fc) as fh:
            with h5py.File(fh) as h5f:
                intensity_array = np.array(h5f['intensity'])
                time = np.array(h5f['time'])
        chunk_lim = self.chunk_size = 1
        out_dict = {}
        for fi, f in enumerate(np.vsplit(intensity_array, intensity_array.shape[0])):
            fi_str = str(fi).rjust(6, '0')
            out_dict[f'{fn}:trace_{fi_str}'] = get_tuple(np.row_stack((time, f.squeeze())),
                                                         self.eps, self.l, self.d, self.gamma, False)
            if fi >= chunk_lim:
                self.write_away_traces(out_dict, fc, fn)
                out_dict = {}
                chunk_lim += self.chunk_size
        if len(out_dict):
            self.write_away_traces(out_dict, fc, fn)

    def parse_trace_file(self, fn, fc):
        self.nb_files -= 1
        # nb_frames, _, nb_traces = np.frombuffer(file_contents, dtype=np.int16, count=3)
        traces_vec = np.frombuffer(fc, dtype=np.int16)
        nb_frames, nb_traces, traces_vec = traces_vec[0], traces_vec[2], traces_vec[3:]
        if self.alex:
            nb_frames = nb_frames // 2
            nb_colors = 4
        else:
            nb_colors = 2
        nb_samples = nb_traces // 2
        nb_points_expected = nb_colors * nb_samples * nb_frames
        traces_vec = traces_vec[:nb_points_expected]
        sampling_freq = 1.0 / self.framerate
        if self.alex:
            Data = traces_vec.reshape((nb_traces, nb_frames * 2), order='F')  # CL: Direct copy of procedure in matlab script
            GRem = Data[np.arange(0, nb_traces, 2), :]
            REem = Data[np.arange(0, nb_traces, 2) + 1, :]
            GRexGRem = GRem[:, np.arange(0, GRem.shape[1], 2)]
            REexGRem = GRem[:, np.arange(0, GRem.shape[1], 2) + 1]
            GRexREem = REem[:, np.arange(0, REem.shape[1], 2)]
            REexREem = REem[:, np.arange(0, REem.shape[1], 2) + 1]
            file_contents = np.stack((GRexGRem, GRexREem, REexGRem, REexREem), axis=0)
            sampling_freq *= 2
        else:
            file_contents = traces_vec.reshape((nb_colors, nb_samples, nb_frames), order='F')
        fn_clean = os.path.splitext(fn)[0]
        fn_list = [f'{fn_clean}_{it}.dat' for it in range(nb_samples)]

        chunk_lim = self.chunk_size - 1
        out_dict = {}
        for fi, f in enumerate(np.hsplit(file_contents, file_contents.shape[1])):
            f = f.squeeze()
            time = np.arange(f.shape[1]) * sampling_freq
            out_dict[fn_list[fi]] = get_tuple(np.row_stack((time, f)), self.eps, self.l, self.d, self.gamma, self.traceswitch)
            if fi >= chunk_lim:
                self.write_away_traces(out_dict)
                out_dict = {}
                chunk_lim += self.chunk_size
        if len(out_dict):
            self.write_away_traces(out_dict)

    def parse_photonhdf5(self, file_contents):
        out_dict = {}
        chunk_lim = self.chunk_size - 1
        with io.BytesIO(file_contents) as fh:
            h5f = h5py.File(fh, mode='r')

            fn_base = splitext(h5f['identity']['filename'][()].decode())[0]

            # Collect keys for spot groups
            pd_list = [fn for fn in h5f.keys() if 'photon_data' in fn]

            # pixel ID to detector specification dict
            for fi, pdn in enumerate(pd_list):
                pd = h5f[pdn]
                ds = pd['measurement_specs']['detectors_specs']
                if not('spectral_ch1' in ds and 'spectral_ch2' in ds):
                    continue

                raw_data = pd['timestamps'][()]
                detector = pd['detectors'][()]

                # split out raw data per detector, write away
                don = raw_data[detector == ds['spectral_ch1']]
                acc = raw_data[detector == ds['spectral_ch2']]
                seqlen = min(len(don), len(acc))
                f = np.row_stack((don[:seqlen], acc[:seqlen]))
                time = np.arange(seqlen) * pd['timestamps_specs']['timestamps_unit']
                ft = np.row_stack((time, f))

                out_dict[f'{fn_base}_{pdn}'] = get_tuple(ft, self.eps, self.l, self.d, self.gamma, self.traceswitch)
                if fi >= chunk_lim:
                    self.write_away_traces(out_dict, file_contents)
                    out_dict = {}
                    chunk_lim += self.chunk_size

            if len(out_dict):
                self.write_away_traces(out_dict, file_contents)


    def update_filter_params(self, fh):
        """
        Retrieve filter params from traces_store. Done at start of loop and  after each chunk is written away.
        """
        (self.data_timestamp, self.framerate,
         self.l, self.d, self.gamma,
         self.eps, self.alex, self.traceswitch) = (fh.attrs['data_timestamp'], fh.attrs['framerate'],
                                 fh.attrs['l'], fh.attrs['d'], fh.attrs['gamma'],
                                 fh.attrs['eps'], fh.attrs['alex'], fh.attrs['traceswitch'])

    def write_away_traces(self, out_dict, fc=None, fn=None):
        # note: mod_timestamp set to -1 to signify nan integer
        index_table = pd.DataFrame({'trace': list(out_dict),
                      'eps': self.eps, 'l': self.l, 'd': self.d, 'gamma': self.gamma, 'data_timestamp': self.data_timestamp,
                      'logprob': np.nan, 'mod_timestamp': -1}).set_index('trace')
        with SafeH5(self.traces_store_fn, 'a') as fh:
            # store raw data
            if fn is not None and fc is not None:
                if 'raw/' + fn not in fh:
                    fh['raw/' + fn] = fc
            # store traces in FRETboard readable format
            for tk in out_dict:
                if 'traces/' + tk in fh:
                    del fh['traces/' + tk]
                fh['traces/' + tk] = out_dict[tk]
        with SafeHDFStore(self.traces_store_fn, 'a') as fh:
            if 'index_table' in fh:
                fh.remove('index_table', where='index in index_table.index')
            fh.put('index_table', value=index_table, format='table', append=True, min_itemsize={'index': 50})
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
