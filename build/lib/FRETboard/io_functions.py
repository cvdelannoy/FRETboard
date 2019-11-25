import os
import fnmatch
import warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from FRETboard.MainTable import MainTable


def parse_input_path(location, pattern=None):
    """
    Take path, list of files or single file, Return list of files with path name concatenated.
    """
    if not isinstance(location, list):
        location = [location]
    all_files = []
    for loc in location:
        loc = os.path.abspath(loc)
        if os.path.isdir(loc):
            if loc[-1] != '/':
                loc += '/'
            for root, dirs, files in os.walk(loc):
                if pattern:
                    for f in fnmatch.filter(files, pattern):
                        all_files.append(os.path.join(root, f))
                else:
                    for f in files:
                        all_files.append(os.path.join(root, f))
        elif os.path.exists(loc):
            all_files.extend(loc)
        else:
            warnings.warn('Given file/dir %s does not exist, skipping' % loc, RuntimeWarning)
    if not len(all_files):
        ValueError('Input file location(s) did not exist or did not contain any files.')
    return all_files


def parallel_fn(f_array, fn_list, dt):
    for fi, f in enumerate(np.hsplit(f_array, f_array.shape[1])):
        f = f.squeeze()
        dt.add_tuple(np.row_stack((np.arange(f.shape[1]), f)), fn_list[fi])
    return dt.data


def parse_trace_file(file_contents, fn, threads, pool):
    """
    Take contents extracted from .trace binary file, return list of [threads] MainTable objects
    """
    nb_colors = 2
    nb_frames, _, nb_traces = np.frombuffer(file_contents, dtype=np.int16, count=3)
    traces_vec = np.frombuffer(file_contents, dtype=np.int16)
    traces_vec = traces_vec[3:]
    nb_points_expected = nb_colors * (nb_traces // nb_colors) * nb_frames
    traces_vec = traces_vec[:nb_points_expected]
    file_contents = traces_vec.reshape((nb_colors, nb_traces // nb_colors, nb_frames), order='F')
    fn_clean = os.path.splitext(fn)[0]

    file_chunks = np.array_split(file_contents, threads, axis=1)
    fn_list = [f'{fn_clean}_{it}.dat' for it in range(file_contents.shape[1])]
    fn_chunks = np.array_split(fn_list, threads)
    df_list = pool(delayed(parallel_fn)(fc, fnc, MainTable([]))
                   for fc, fnc in zip(file_chunks, fn_chunks))
    return df_list
