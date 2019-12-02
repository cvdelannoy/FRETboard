import argparse
import os, fnmatch, warnings

from os.path import basename, splitext, abspath
from shutil import rmtree
import pathlib
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 400


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



def plot_trace(data_dicts, nb_classes):
    """
    Plot traces, provided as one dict per plot
    :return:
    """
    line_cols=['green', 'red', 'brown', 'black']
    fig = plt.figure()
    if type(data_dicts) == dict:
        data_dicts = [data_dicts]
    nb_plots = len(data_dicts)
    plot_dict = dict()
    for pid, cdd in enumerate(data_dicts):
        if cdd['data'].ndim < 2:
            cdd['data'] = np.expand_dims(cdd['data'], -1)
        length_series = cdd['data'].shape[0]
        if pid == 0:
            # plot_dict[0] = plt.figure(dpi=600)
            plot_dict[0] = plt.subplot2grid((nb_plots, 1), (pid, 0))
        else:
            plot_dict[pid] = plt.subplot2grid((nb_plots, 1), (pid, 0), sharex=plot_dict[0])
        plot_dict[pid].set(ylabel=cdd['ylabel'], xlabel=cdd['xlabel'])
        plot_dict[pid].set_xlim(0, length_series)
        for n in range(cdd['data'].shape[1]):
            plot_dict[pid].plot(cdd['data'][:, n], linewidth=0.5, color=line_cols[n % len(line_cols)])
        if cdd.get('label') is not None:
            if cdd['label'].ndim < 2:
                cdd['label'] = np.expand_dims(cdd['label'], -1)
            nb_colbars = cdd['label'].shape[1]
            for n in range(nb_colbars):
                if n == 0:
                    cb_vrange = [axc * 0.1 for axc in plot_dict[pid].get_ylim()]
                    colbar_width = cb_vrange[1] - cb_vrange[0]
                else:
                    cb_top = cb_vrange[n]
                    cb_vrange = [cb_top, cb_top + colbar_width]
                plot_dict[pid].pcolorfast((0, length_series), cb_vrange, cdd['label'][:, n].reshape(1,-1),
                                          vmin=0, vmax=nb_classes, cmap='RdYlGn', alpha=0.3)
    plt.tight_layout()
    return plt


def parse_output_dir(out_dir, clean=False):
    out_dir = abspath(out_dir) + '/'
    if clean:
        rmtree(out_dir, ignore_errors=True)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir

parser = argparse.ArgumentParser(description='Compare FRETboard output to labels of kinSoft traces.')
parser.add_argument('--fb', type=str, required=True, help='FRETboard results dir')
parser.add_argument('--ks', type=str, required=True, help='kinsoft dir')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
args = parser.parse_args()

fb_files = parse_input_path(args.fb, pattern='*.dat')
ks_files = parse_input_path(args.ks, pattern='state_time_*')
pat_ks_nb = re.compile('(?<=state_time_)[0-9]+')
ks_dict = {re.search(pat_ks_nb, ks).group(0): ks for ks in ks_files}
outdir = parse_output_dir(args.outdir, clean=True)

acc_list = []
total_pts = 0
correct_pts = 0
for fb in fb_files:
    dat_df = pd.read_csv(fb, sep='\t')

    # load kinsoft labels
    ks = ks_dict[re.search('[0-9]+(?=.dat)', basename(fb)).group(0)]
    ks_df = pd.read_csv(ks, sep='\t')
    ks_df.loc[:,'t_end (s)'] = ks_df.loc[:,'t_start (s)'] + ks_df.loc[:, 't_dwell (s)']
    ks_label = [ks_df.loc[ks_df[ks_df.loc[:, 't_start (s)'] <= t].index.max(), '%state'] for t in dat_df.time]
    dat_df.loc[:, 'ks_label'] = ks_label

    # set classes same
    cl_dict = {}
    for cl in np.unique(ks_label):
        pred_cl = dat_df.loc[dat_df.ks_label == cl, 'predicted'].mode()[0]
        cl_dict[cl] = pred_cl
    dat_df.loc[:, 'ks_label'] = dat_df.ks_label.apply(lambda x: cl_dict[x])

    # Record accuracy
    correct = (dat_df.predicted == dat_df.ks_label).sum()
    acc_list.append(correct / dat_df.shape[0])
    total_pts += dat_df.shape[0]
    correct_pts += correct

    plot_dict = {
        'ylabel': 'I',
        'xlabel': 'time',
        'data': dat_df.loc[:, ['i_don', 'i_acc']].to_numpy(),
        'label': dat_df.loc[:, ['predicted', 'ks_label']].to_numpy()
    }

    nb_classes = pd.unique(dat_df.loc[:, 'predicted']).size
    plot_obj = plot_trace(plot_dict, nb_classes)
    plot_obj.savefig(f'{outdir}{splitext(basename(fb))[0]}.png')
    plt.close()

plt.hist(acc_list, density=False)
plt.xlabel('Accuracy')
plt.ylabel('# traces')
plt.title(f'Per-trace accuracy for {basename(args.ks)}')
plt.text(0.1, 0.9, f'Overall acc: {correct_pts / total_pts :.2f}')
print(f'Accuracy: {correct_pts / total_pts}')
plt.savefig(f'{outdir}accuracy_hist.png')



