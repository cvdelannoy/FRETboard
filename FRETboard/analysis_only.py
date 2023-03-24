import argparse, os, sys, pickle, shutil, h5py, importlib

import multiprocessing as mp
import numpy as np
import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime

sys.path.append(str(Path(__file__).parents[1].resolve()))

from FRETboard.io_functions import parse_output_path, parse_input_path
from FRETboard.Gui import algo_dict, algo_inv_dict
from FRETboard.helper_functions import get_tuple, colnames, colnames_alex
from FRETboard.MainTable_dummy import MainTable

def classify_parallel(npy_list, mod_fn, out_dir):
    with open(mod_fn, 'rb') as fh:
        mod = pickle.load(fh)
    for npy_fn in npy_list:
        tup = np.load(npy_fn)
        trace_df = pd.DataFrame(data=tup.T, columns=colnames_alex)
        pred, logprob = mod.predict(trace_df)
        trace_df.loc[:, 'predicted'] = pred
        trace_df.to_csv(f'{out_dir}{str(Path(npy_fn).stem)}.csv', sep='\t', na_rep='NA', index=False)

class FRETboardAnalyzer(object):
    def __init__(self, model_fn, nb_cores, l=0.0, d=0.0, gamma=1.0, eps=None):
        self.l = l
        self.d = d
        self.gamma = gamma
        self.eps = eps

        self.model_fn = model_fn
        self.nb_cores = nb_cores



    @property
    def model_fn(self):
        return self._model_fn

    @model_fn.setter
    def model_fn(self, fn):
        with open(fn) as fh: model_txt = fh.read()
        self.algo_name = algo_inv_dict.get(model_txt.split('\n')[0], 'custom')
        classifier_class = importlib.import_module('FRETboard.algorithms.' + algo_dict.get(self.algo_name, self.algo_name)).Classifier
        self._classifier = classifier_class(nb_states=2, buffer=1, supervision_influence=None, features=None,  # Values should not matter; replaced in next line
                                            data=MainTable(self.eps, self.l, self.d, self.gamma, False, False))  # todo: no data table --> no get_states etc. --> no html report!   # todo
        self._classifier.load_params(model_txt)
        self._model_fn = fn

    def predict(self, traces, out_dir):
        # write away classifier
        mod_fn = f'{out_dir}mod.pkl'
        dat_dir = parse_output_path(out_dir + 'dats_classified')
        with open(mod_fn, 'wb') as fh:
            pickle.dump(self._classifier, fh, protocol=pickle.HIGHEST_PROTOCOL)

        nc_list = []
        with TemporaryDirectory() as td:
            # 1. write away traces in unified format
            for tr in traces:
                if tr.endswith('.dat'):
                    self.parse_dat_file(tr, td)
                elif tr.endswith('.nc'):
                    self.parse_nc_file(tr, td)
                    nc_list.append(tr)
                elif tr.endswith('.traces'):
                    self.parse_trace_file(tr, td)
                else:
                    raise ValueError(f'Cannot (yet) parse file with extension {os.path.splitext(tr)[1]}')

            # 2. Process in parallel
            npy_list = parse_input_path(td)
            p_list = [mp.Process(target=classify_parallel, args=(sub_list, mod_fn, dat_dir))
                      for sub_list in np.array_split(npy_list, self.nb_cores)]
            for p in p_list:
                p.start()
            while True:
                running = any(p.is_alive() for p in p_list)
                if not running:
                    break

        # Join files in nc format if necessary
        if len(nc_list):
            nc_path = parse_output_path(f'{out_dir}nc_classified')
            for nc_fn in nc_list:
                nc_stem = Path(nc_fn).stem
                nc_fn_out = f'{nc_path}{nc_stem}.nc'
                nc_dat_list = parse_input_path(dat_dir, pattern=f'*{nc_stem}:*')
                nc_dat_list.sort()
                shutil.copyfile(nc_fn, nc_fn_out)
                nc_dat_tuples = [pd.read_csv(ncd, sep='\t').predicted.to_numpy().astype(int) for ncd in nc_dat_list]
                with h5py.File(nc_fn_out, 'r+') as h5f:
                    h5f['FRETboard_classification'] = nc_dat_tuples
                    try:
                        h5f['FRETboard_classification'].dims[0].attach_scale(h5f['molecule'])
                        h5f['FRETboard_classification'].dims[1].attach_scale(h5f['frame'])
                    except:
                        print(f'Warning: could not attach scales for nc file {nc_stem}. This may happen e.g.'
                              f'if you delete traces. nc output was still generated but reading may fail '
                              f'for some applications.')
                    h5f['FRETboard_classification'].attrs['datetime'] = str(datetime.now())
                    h5f['FRETboard_classification'].attrs['classifier'] = self.algo_name
                    h5f['FRETboard_classification'].attrs['nb_classes'] = self._classifier.nb_states

    def parse_dat_file(self, fn, td):
        fn_clean = Path(fn).stem
        with open(fn) as fh:
            file_contents = fh.read()
        file_contents = np.column_stack([np.fromstring(n, sep=' ') for n in file_contents.split('\n') if len(n)])
        if not len(file_contents): return
        tup = get_tuple(file_contents, self.eps, self.l, self.d, self.gamma, False)
        np.save(f'{td}/{fn_clean}.npy', tup, allow_pickle=False)

    def parse_nc_file(self, fn, td):
        fn_clean = Path(fn).stem
        with h5py.File(fn) as h5f:
                intensity_array = np.array(h5f['intensity'])
                time = np.array(h5f['time'])
        for fi, f in enumerate(np.vsplit(intensity_array, intensity_array.shape[0])):
            tup = get_tuple(np.row_stack((time, f.squeeze())),
                            self.eps, self.l, self.d, self.gamma, False)
            np.save(f'{td}/{fn_clean}:trace_{fi}.npy', tup, allow_pickle=False)

    def parse_trace_file(self, fn, td):
        # todo ALEX implementation
        # todo option for framerate (?)
        framerate = 10
        with open(fn, 'rb') as fh:
            fc = fh.read()
        traces_vec = np.frombuffer(fc, dtype=np.int16)
        nb_frames, nb_traces, traces_vec = traces_vec[0], traces_vec[2], traces_vec[3:]
        nb_colors = 2
        # if self.alex:
        #     nb_frames = nb_frames // 2
        #     nb_colors = 4
        # else:
        #     nb_colors = 2
        nb_samples = nb_traces // 2
        nb_points_expected = nb_colors * nb_samples * nb_frames
        traces_vec = traces_vec[:nb_points_expected]
        sampling_freq = 1.0 / framerate

        file_contents = traces_vec.reshape((nb_colors, nb_samples, nb_frames), order='F')
        # if self.alex:
        #     Data = traces_vec.reshape((nb_traces, nb_frames * 2), order='F')  # CL: Direct copy of procedure in matlab script
        #     GRem = Data[np.arange(0, nb_traces, 2), :]
        #     REem = Data[np.arange(0, nb_traces, 2) + 1, :]
        #     GRexGRem = GRem[:, np.arange(0, GRem.shape[1], 2)]
        #     REexGRem = GRem[:, np.arange(0, GRem.shape[1], 2) + 1]
        #     GRexREem = REem[:, np.arange(0, REem.shape[1], 2)]
        #     REexREem = REem[:, np.arange(0, REem.shape[1], 2) + 1]
        #     file_contents = np.stack((GRexGRem, GRexREem, REexGRem, REexREem), axis=0)
        #     sampling_freq *= 2
        # else:
        #     file_contents = traces_vec.reshape((nb_colors, nb_samples, nb_frames), order='F')
        fn_clean = Path(fn).stem
        for fi, f in enumerate(np.hsplit(file_contents, file_contents.shape[1])):
            f = f.squeeze()
            time = np.arange(f.shape[1]) * sampling_freq
            tup = get_tuple(np.row_stack((time, f)), self.eps, self.l, self.d, self.gamma, False)  # todo traceswitch param?
            np.save(f'{td}/{fn_clean}:trace_{fi}.npy', tup, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GUI-less high-speed prediction of traces using a '
                                                 'FRETboard model.')
    parser.add_argument('--traces', type=str, nargs='+', required=True,
                        help='(Directory of) traces to be classified')
    parser.add_argument('--model', type=str, required=True,
                        help='FRETboard model, generated using the GUI')
    parser.add_argument('--cores', type=int, default=4,
                        help='Maximum number of cores to engage at once [default:4]')
    parser.add_argument('--out-dir', type=str, required=True)
    # --- optional args ---
    parser.add_argument('-l', type=float, default=0.0,
                        help='l parameter for leakage correction [default: 0.0]')
    parser.add_argument('-d', type=float, default=0.0,
                        help='d parameter for direct excitation correct [default: 0.0]')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma correction factor [default: 1.0]')
    parser.add_argument('--eps', type=float, default=None,
                        help='epsilon parameter for DBSCAN base level detection, leave unspecified to not '
                             'perform base level detection.')
    args = parser.parse_args()
    eps = args.eps if args.eps is not None else np.nan
    out_dir = parse_output_path(args.out_dir)
    analyzer = FRETboardAnalyzer(args.model, args.cores, args.l, args.d, args.gamma, eps)
    analyzer.predict(args.traces, out_dir)
