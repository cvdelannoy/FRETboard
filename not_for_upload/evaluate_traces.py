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
import seaborn as sns

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

def condense_seq(labels, values):
    seq_condensed = [[labels[0], 0, 0, []]]  # label, start, duration, E_FRET
    for it, (lab, val) in enumerate(zip(labels, values)):
        if lab == seq_condensed[-1][0]:
            seq_condensed[-1][2] += 1
            seq_condensed[-1][3].append(val)
        else:
            seq_condensed[-1][3] = np.nanmean(seq_condensed[-1][3])
            seq_condensed.append([lab, it, 1, [val]])
    seq_condensed[-1][3] = np.nanmedian(seq_condensed[-1][3])
    return seq_condensed

def get_transitions(labels):
    out_list = []
    cur_label = labels[0]
    count = 0
    for label in labels:
        if label == cur_label:
            count += 1
        else:
            out_list.append((str(cur_label) + str(label), count))
            cur_label = label
            count = 1
    return out_list

def condense_df(seq, manual):
    """
    Take a pd df of labels and Efret values, turn into df w/ 1 row per event
    """
    seq_condensed = condense_seq(seq.loc[:, manual], seq.loc[:, 'E_FRET'])
    # seq_condensed = [[seq.iloc[0][manual], 0, 0, []]]  # symbol, start, duration, E_FRET
    # for ti, tup in seq.iterrows():
    #     if tup[manual] == seq_condensed[-1][0]:
    #         seq_condensed[-1][2] += 1
    #         seq_condensed[-1][3].append(tup.E_FRET)
    #     else:
    #         seq_condensed[-1][3] = np.nanmean(seq_condensed[-1][3])
    #         seq_condensed.append([tup[manual], ti, 1, [tup.E_FRET]])
    # seq_condensed[-1][3] = np.nanmedian(seq_condensed[-1][3])
    out_df = pd.DataFrame(seq_condensed, columns=['label', 'start', 'duration', 'E_FRET'])
    out_df.loc[:, 'end'] = out_df.start + out_df.duration
    return out_df



def plot_trace(data_dicts, nb_classes):
    """
    Plot traces, provided as one dict per plot
    :return:
    """
    line_cols=['green', 'red', 'brown', 'black']
    fig = plt.figure(figsize=(15, 5))
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
                                          vmin=0, vmax=nb_classes, cmap='RdYlGn', alpha=0.5)
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
parser.add_argument('--manual', type=str, required=True, help='manual dir')
parser.add_argument('--manual-type', type=str, default='matlab', choices=['matlab', 'fretboard', 'kinsoft'],
                    help='Format of manual label files')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
parser.add_argument('--remove-last-event', action='store_true',
                    help='special for Mikes approach: remove last non-ground state event')
parser.add_argument('--categories', type=str, nargs='+', default=[''],
                    help='In plots, split files and stats by strings that are to be recognized in path names')
parser.add_argument('--tr-files', type=str, required=False, nargs='+',
                    help='Transition rate statistics files (1 per category) as returned by FRETboard')
args = parser.parse_args()

fb_files = parse_input_path(args.fb, pattern='*.dat')
manual_files = parse_input_path(args.manual, pattern='*.dat')
manual_dict = {basename(mf): mf for mf in manual_files}
pat_ks_nb = re.compile('(?<=state_time_)[0-9]+')
outdir = parse_output_dir(args.outdir, clean=True)
summary_dir = parse_output_dir(outdir+'summary_stats/', clean=True)

tracestats_df = pd.DataFrame(index=fb_files, columns=['nb_events', 'nb_events_predicted', 'mean_coverage'])
eventstats_list = []
transition_df = pd.DataFrame(columns=['nb_samples', 'nb_transitions'])
framerate_list = []


load_fun_dict = {'matlab': lambda fn: pd.read_csv(fn, sep='\t', names=['time', 'i_don', 'i_acc', 'label']),
                 'fretboard': lambda fn: pd.read_csv(fn, sep='\t', header=0)}

loader_fun = load_fun_dict[args.manual_type]
acc_list = []
total_pts = 0
correct_pts = 0
for fb in fb_files:
    # try:
        cat = [cat for cat in args.categories if cat in fb]
        if not len(cat): continue
        elif len(cat) > 1: raise ValueError(f'trace {fb} falls in multiple categories, redefine categories')
        cat = cat[0]
        fb_base = basename(fb)
        if fb_base not in manual_dict: continue
        dat_df = pd.read_csv(fb, sep='\t')

        # load manual labels
        manual_df = loader_fun(manual_dict[fb_base])
        # manual_df = pd.read_csv(manual_dict[fb_base], sep='\t', names=['time', 'i_don', 'i_acc', 'label'])
        dat_df.loc[:, 'manual'] = manual_df.label.astype(int)

        if args.remove_last_event:
            ground_state = dat_df.predicted.min()
            nb_rows = len(dat_df) - 1
            ground_bool = dat_df.predicted == ground_state
            state_found = False
            for gi, gb in enumerate(ground_bool[::-1]):
                if not gb:
                    state_found = True
                    dat_df.loc[nb_rows-gi, 'predicted'] = ground_state.astype(int)
                elif state_found:
                    break
            dat_df.predicted = dat_df.predicted.astype(int)

        # set classes same
        # cl_dict = {}
        # for cl in np.unique(dat_df.manual):
        #     pred_cl = dat_df.loc[dat_df.manual == cl, 'predicted'].mode()[0]
        #     cl_dict[cl] = pred_cl

        cl_dict = {man: pred for man, pred in zip(np.sort(dat_df.manual.unique()), np.sort(dat_df.predicted.unique()))}
        dat_df.loc[:, 'manual'] = dat_df.manual.apply(lambda x: cl_dict[x])

        # Add EFRET
        i_sum = np.sum((dat_df.i_don, dat_df.i_acc), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E_FRET = np.divide(dat_df.i_acc, i_sum)
        E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out
        dat_df.loc[:, "E_FRET"] = E_FRET

        # Update transitions data
        transition_list = get_transitions(dat_df.manual)
        for trl in transition_list:
            if trl[0] in transition_df.index:
                transition_df.loc[trl[0], 'nb_samples'] += trl[1]
                transition_df.loc[trl[0], 'nb_transitions'] += 1
            else:
                transition_df.loc[trl[0], 'nb_samples'] = trl[1]
                transition_df.loc[trl[0], 'nb_transitions'] = 1
        framerate_list.append(1 / (dat_df.time[1:].to_numpy() - dat_df.time[:-1].to_numpy()).mean())

        # make condensed df
        condensed_df = condense_df(dat_df, 'manual')
        condensed_df.loc[:, 'category'] = cat
        condensed_df = condensed_df.loc[condensed_df.label != 1]  # remove ground state 'events'
        condensed_pred_df = condense_df(dat_df, 'predicted')
        for ti, tup in condensed_df.iterrows():
            idx_range = np.arange(tup.start, tup.end)
            overlaps = condensed_pred_df.apply(lambda x: np.sum(np.in1d(np.arange(x.start, x.end), idx_range)), axis=1)
            pred_event = condensed_pred_df.iloc[overlaps.idxmax()]
            condensed_df.loc[ti, 'predicted'] = int(pred_event.label)
            condensed_df.loc[ti, 'pred_duration'] = int(overlaps.max())
        if len(condensed_df):
            condensed_df.loc[:, 'coverage'] = (condensed_df.pred_duration) / condensed_df.duration * 100.0
            eventstats_list.append(condensed_df)

        # Record accuracy
        correct = (dat_df.predicted == dat_df.manual).sum()
        acc_list.append(correct / dat_df.shape[0])
        total_pts += dat_df.shape[0]
        correct_pts += correct

        # plot trace
        plot_dict = {
            'ylabel': 'I',
            'xlabel': 'time (s)',
            'data': dat_df.loc[:, ['i_don', 'i_acc']].to_numpy(),
            'label': dat_df.loc[:, ['predicted', 'manual']].to_numpy()
        }
        nb_classes = pd.unique(dat_df.loc[:, 'predicted']).size
        plot_obj = plot_trace(plot_dict, nb_classes)
        plot_obj.savefig(f'{outdir}{splitext(basename(fb))[0]}.png')
        plt.close()
    # except:
    #     continue

# plot transition rates
if args.tr_files:
    if any(np.array(framerate_list) - framerate_list[0] > 0.1):
        raise ValueError('Framerate not constant for all samples')

    # Ground truth from data
    framerate = np.mean(framerate_list)
    transition_df.loc[:, 'rate'] = transition_df.nb_transitions / transition_df.nb_samples * framerate

    # Loaded from FRETboard
    tr_df_list = []
    for trf in args.tr_files:
        trf_base = basename(trf)
        cat_list = [cat for cat in args.categories if cat in trf_base]
        if len(cat_list) > 1: raise ValueError(f'Multiple categories match to transition rate file {trf_base}')
        elif len(cat_list) == 0: continue
        cat = cat_list[0]
        tr_cur_df = pd.read_csv(trf, names=['transition', 'rate', 'low_bound', 'high_bound'], skiprows=1)
        tr_cur_df.loc[:, 'category'] = cat
        tr_df_list.append(tr_cur_df)
    tr_df = pd.concat(tr_df_list)
    tr_df.loc[:, 'transition'] = tr_df.transition.astype(str)
    transition_df.loc[:, 'category'] = 'actual'
    transition_df.loc[:, 'low_bound'] = transition_df.rate
    transition_df.loc[:, 'high_bound'] = transition_df.rate
    transition_df = transition_df.rename_axis('transition').reset_index()
    transition_df.drop(['nb_samples', 'nb_transitions'], axis=1, inplace=True)
    transition_df = pd.concat((transition_df, tr_df), sort=True)

    # derive values for CI bars
    yerr_list = []
    for tr, sdf in transition_df.groupby(['category']):
        yerr_list.append(np.expand_dims(np.vstack((sdf.rate - sdf.low_bound, sdf.high_bound - sdf.rate)), 0))
    yerr = np.vstack(yerr_list)
    transition_df.pivot(index='transition', columns='category', values='rate').plot(kind='bar', yerr=yerr)
    plt.savefig(f'{summary_dir}/transition_rate.svg')
    plt.clf()

# plot event coverage
eventstats_df = pd.concat(eventstats_list)
eventstats_df.loc[:, 'coverage'] = eventstats_df.pred_duration / eventstats_df.duration * 100.0
sns.violinplot(x='category', y='coverage', data=eventstats_df.loc[eventstats_df.label == eventstats_df.predicted])
plt.gcf().axes[0].xaxis.label.set_visible(False)
plt.ylabel('Event coverage (%)')
plt.savefig(f'{summary_dir}/coverage_violin.svg')
eventstats_df.to_csv(f'{summary_dir}/eventstats.tsv', sep='\t')

plt.hist(acc_list, density=False)
plt.xlabel('Accuracy')
plt.ylabel('# traces')
plt.title(f'Per-trace accuracy for {basename(args.manual)}')
plt.text(0.1, 0.9, f'Overall acc: {correct_pts / total_pts :.2f}')
print(f'Accuracy: {correct_pts / total_pts}')
plt.savefig(f'{outdir}accuracy_hist.png')



