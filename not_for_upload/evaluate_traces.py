import argparse
import os, fnmatch, sys
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
from copy import copy
from os.path import basename, splitext, abspath
from shutil import rmtree
import pathlib
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import pickle
from itertools import permutations
import warnings
from scipy.linalg import logm
from scipy.stats import ttest_ind, ks_2samp, shapiro
from statsmodels.stats.weightstats import ttost_ind
from matplotlib.ticker import MaxNLocator
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
from plotting_functions import plot_transition_bubble

mpl.rcParams['figure.dpi'] = 400

colmap = LinearSegmentedColormap.from_list('custom_blues', ['#FFFFFF', '#084594'])

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


def get_transitions(labels, target_states):
    out_list = []
    cur_label = labels[0]
    for label in labels:
        if label != cur_label:
            out_list.append((cur_label, label))
            cur_label = label
    out_list = [ol for ol in out_list if ol[0] in target_states and ol[1] in target_states]
    return out_list


def condense_df(seq, manual):
    """
    Take a pd df of labels and Efret values, turn into df w/ 1 row per event
    """
    seq_condensed = condense_seq(seq.loc[:, manual], seq.loc[:, 'E_FRET'])
    out_df = pd.DataFrame(seq_condensed, columns=['label', 'start', 'duration', 'E_FRET'])
    out_df.loc[:, 'end'] = out_df.start + out_df.duration
    return out_df

def matlab_load_fun(fn):
    df = pd.read_csv(fn, sep='\t', names=['time', 'i_don', 'i_acc', 'label'])
    df.label += 1  # matlab classification is 0-based
    return df

def fretboard_load_fun(fn):
    df = pd.read_csv(fn, sep='\t', header=0)
    return df


def plot_trace(data_dicts, nb_classes, time):
    """
    Plot traces, provided as one dict per plot
    :return:
    """
    line_cols=['green', 'red', 'brown', 'black']
    fig = plt.figure(figsize=(20, 5))
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
        plot_dict[pid].set_xlim(0, time.max())
        for n in range(cdd['data'].shape[1]):
            plot_dict[pid].plot(time, cdd['data'][:, n], linewidth=0.5, color=line_cols[n % len(line_cols)])
        if cdd.get('label') is not None:
            if cdd['label'].ndim < 2:
                cdd['label'] = np.expand_dims(cdd['label'], -1)
            nb_colbars = cdd['label'].shape[1]
            for n in range(nb_colbars):
                if n == 0:
                    bw = (plot_dict[pid].get_ylim()[1] - plot_dict[pid].get_ylim()[0]) * 0.1
                    spacing = bw * 0.1
                    cb_vrange = [cdd['data'].min() - bw, cdd['data'].min()]
                    # cb_vrange = [axc * 0.1 for axc in plot_dict[pid].get_ylim()]
                    colbar_width = cb_vrange[1] - cb_vrange[0]
                else:
                    cb_bottom = cb_vrange[0]
                    cb_vrange = [cb_bottom - spacing, cb_bottom - colbar_width - spacing]
                # plot_dict[pid].pcolorfast(np.vstack((cdd['label'][:, n].reshape(-1), time)).T, vmin=1, vmax=nb_classes, cmap=colmap, alpha=1)
                plot_dict[pid].pcolorfast((0, time.max()), cb_vrange, cdd['label'][:, n].reshape(1,-1),
                                          vmin=1, vmax=nb_classes, cmap=colmap, alpha=1)
    plt.tight_layout()
    return plt

def parse_output_dir(out_dir, clean=False):
    out_dir = abspath(out_dir) + '/'
    if clean:
        rmtree(out_dir, ignore_errors=True)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir

def discrete2continuous(tm, framerate):
    """
    Convert discrete transition matrix such as output by HMMs to continous transition rates estimates
    """
    assert tm.ndim == 2
    assert tm.shape[0] == tm.shape[1]
    nb_states = tm.shape[0]
    rm = np.eye(nb_states) + framerate * logm(tm)
    rm[rm < 0] = 0.0  # correct small negative values
    rm[np.eye(nb_states, dtype=bool)] -= 1
    return rm


parser = argparse.ArgumentParser(description='Compare FRETboard output to labels of kinSoft traces.')
parser.add_argument('--fb', type=str, required=True, help='FRETboard results dir')
parser.add_argument('--manual', type=str, required=True, help='manual dir')
parser.add_argument('--target-states', type=int, required=True, nargs='+',
                    help='States (as denoted in manually labeled files) to take into account.')
parser.add_argument('--manual-type', type=str, default='matlab', choices=['matlab', 'fretboard', 'kinsoft'],
                    help='Format of manual label files')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
parser.add_argument('--remove-last-event', action='store_true',
                    help='special for Mikes approach: remove last non-ground state event')
parser.add_argument('--categories', type=str, nargs='+', default=[''],
                    help='In plots, split files and stats by strings that are to be recognized in path names')
# parser.add_argument('--tr-files', type=str, required=False, nargs='+',
#                     help='Transition rate statistics files (1 per category) as returned by FRETboard')
args = parser.parse_args()


fb_files = parse_input_path(args.fb, pattern='*.dat')
manual_files = parse_input_path(args.manual, pattern='*.dat')
manual_dict = {basename(mf): mf for mf in manual_files}
pat_ks_nb = re.compile('(?<=state_time_)[0-9]+')
outdir = parse_output_dir(args.outdir, clean=True)
summary_dir = parse_output_dir(outdir+'summary_stats/', clean=True)
trace_dir = parse_output_dir(outdir+'trace_plots/', clean=True)
trace_csv_dir = parse_output_dir(outdir+'trace_csvs/', clean=True)
trace_dir_dict = {cat: parse_output_dir(trace_dir + cat, clean=True) for cat in args.categories}
trace_csv_dir_dict = {cat: parse_output_dir(trace_csv_dir + cat, clean=True) for cat in args.categories}

tracestats_df = pd.DataFrame(index=fb_files, columns=['nb_events', 'nb_events_predicted', 'mean_coverage'])
eventstats_list = []
eventstats_pred_list = []
target_states_w_ground = copy(args.target_states)
if 1 not in target_states_w_ground: target_states_w_ground += [1]
midx = pd.MultiIndex.from_tuples(list(permutations(target_states_w_ground, 2)), names=['from', 'to'])
transition_df = pd.DataFrame(0, columns=['nb_samples', 'nb_transitions'], index=midx)
framerate_list = []
confusion_list = []
tdp_list = []

loader_fun = dict(matlab=matlab_load_fun, fretboard=fretboard_load_fun)[args.manual_type]

nb_classes = len(target_states_w_ground)
acc_list = []
acc_df = pd.DataFrame(0, columns=['correct', 'total'], index=args.categories)
total_pts = 0
correct_pts = 0
max_state = 0
plt.rcParams.update({'font.size': 30})  # large text for trace plots
efret_dict = {cat:{st: [] for st in target_states_w_ground} for cat in args.categories}
efret_pred_dict = {cat:{st: [] for st in target_states_w_ground} for cat in args.categories}
for fb in fb_files:
        cat = [cat for cat in args.categories if cat in fb]
        if not len(cat): continue
        elif len(cat) > 1: raise ValueError(f'trace {fb} falls in multiple categories, redefine categories')
        cat = cat[0]
        fb_base = basename(fb)
        if fb_base not in manual_dict: continue  # skip if no ground truth file available
        dat_df = pd.read_csv(fb, sep='\t')
        if not dat_df.label.isnull().all(): continue  # skip if read was used as labeled example

        # load manual labels
        manual_df = loader_fun(manual_dict[fb_base])
        dat_df.loc[:, 'manual'] = manual_df.label.astype(int)

        # dat_df = dat_df.loc[np.in1d(dat_df.manual, target_states_w_ground), :]

        # Store E_FRET per predicted & ground truth state
        present_states_manual = [lab for lab in dat_df.manual.unique() if lab in target_states_w_ground]
        present_states_pred = [lab for lab in dat_df.predicted.unique() if lab in target_states_w_ground]
        for lab in present_states_manual:
            efret_dict[cat][lab].append(dat_df.query(f'manual=={lab}').E_FRET)
        for lab in present_states_pred:
            efret_pred_dict[cat][lab].append(dat_df.query(f'predicted=={lab}').E_FRET)

        # Update transitions data
        transition_list = get_transitions(dat_df.manual, target_states_w_ground)
        if len(transition_list):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for trl in transition_list:
                    transition_df.loc[trl, 'nb_transitions'] += 1
                for ul in present_states_manual:
                    transition_df.loc[(ul,), 'nb_samples'] = transition_df.loc[(ul,), 'nb_samples'].to_numpy()[0] + np.sum(dat_df.manual == ul)
        framerate_list.append(1 / (dat_df.time[1:].to_numpy() - dat_df.time[:-1].to_numpy()).mean())

        # make condensed df
        condensed_df = condense_df(dat_df, 'manual')
        condensed_df.loc[:, 'category'] = cat
        condensed_df.loc[:, 'pred_idx'] = np.nan
        condensed_df.loc[:, 'abs_overlap'] = np.nan
        condensed_df = condensed_df.loc[np.in1d(condensed_df.label, args.target_states)].reset_index().drop('index', axis=1)
        condensed_pred_df = condense_df(dat_df, 'predicted')
        condensed_pred_df.loc[:, 'category'] = cat
        tdp_list.append(pd.DataFrame({'departure_efret': condensed_pred_df.E_FRET.iloc[:-1].to_numpy(),
                               'arrival_efret': condensed_pred_df.E_FRET.iloc[1:].to_numpy(), 'category': cat}))

        # Compare actual events vs predicted events
        # condensed_pred_df.loc[np.invert(np.in1d(condensed_pred_df.label, args.target_states)), 'label'] = None
        verified_predicted_events = []
        confusion_idx = pd.MultiIndex.from_product([args.categories, args.target_states])
        confusion_df = pd.DataFrame(columns=['tp', 'fp', 'fn'], index=confusion_idx, data=0)

        # Loop over ground truth events find tp and fn
        for ti, tup in condensed_df.iterrows():
            manual_range = np.arange(tup.start, tup.end)
            overlaps = condensed_pred_df.apply(lambda x: np.sum(np.in1d(np.arange(x.start, x.end), manual_range)), axis=1)
            pred_idx = overlaps.idxmax()
            abs_overlap = overlaps.max()
            pred_event = condensed_pred_df.iloc[pred_idx]
            condensed_df.loc[ti, 'predicted'] = int(pred_event.label)
            condensed_df.loc[ti, 'pred_duration'] = pred_event.duration
            condensed_df.loc[ti, 'abs_overlap'] = overlaps.max()
            # Collect classifier performance measures. Taking into account:
            # - 1 predicted event cannot be counted doubly if covering two actual events; take the longest

            if pred_idx in condensed_df.pred_idx.values:
                # The predicted event was matched to a previous actual event
                confusion_df.loc[(cat, tup.label), 'fn'] += 1
                # Check which has largest absolute overlap and leave that one in
                ppidx = condensed_df.index[condensed_df.pred_idx.values == pred_idx][0]
                if ti > 0 and condensed_df.loc[ti, 'abs_overlap'] > condensed_df.loc[ppidx, 'abs_overlap']:
                    condensed_df.drop(ppidx, inplace=True)
                else:
                    condensed_df.drop(ti, inplace=True)
                continue
            else:
                condensed_df.loc[ti,'pred_idx'] = pred_idx
            if pred_event.label == tup.label:
                confusion_df.loc[(cat, tup.label), 'tp'] += 1
            else:
                confusion_df.loc[(cat, tup.label), 'fn'] += 1
                condensed_df.drop(ti, inplace=True)
        confusion_list.append(confusion_df)

        # Loop over predicted events, record fp
        for ti, tup in condensed_pred_df.iterrows():
            if ti in condensed_df.pred_idx.values: continue
            if tup.label in args.target_states: confusion_df.loc[(cat, tup.label), 'fp'] += 1

        # Calculate coverage
        if len(condensed_df):
            condensed_df.loc[:, 'coverage'] = (condensed_df.pred_duration) / condensed_df.duration * 100.0
            eventstats_list.append(condensed_df)

        # Store predicted events df
        eventstats_pred_list.append(condensed_pred_df)

        # Record accuracy
        dat_df_target_only = dat_df.loc[np.in1d(dat_df.manual, args.target_states), :]
        if not len(dat_df_target_only): continue
        correct = len(dat_df_target_only.query('predicted == manual'))
        acc_list.append(correct / len(dat_df_target_only))
        total_pts += len(dat_df_target_only)
        correct_pts += correct
        acc_df.loc[cat, 'correct'] = acc_df.loc[cat, 'correct'] + correct
        acc_df.loc[cat, 'total'] = acc_df.loc[cat, 'total'] + len(dat_df_target_only)

        # plot trace
        plot_dict = {
            'ylabel': 'I',
            'xlabel': 'time (s)',
            'data': dat_df.loc[:, ['f_dex_dem', 'f_dex_aem']].to_numpy(),
            'label': dat_df.loc[:, ['predicted', 'manual']].to_numpy()
        }
        plot_obj = plot_trace(plot_dict, dat_df.manual.max(), dat_df.loc[:, 'time'])
        plot_obj.savefig(f'{trace_dir_dict[cat]}{splitext(basename(fb))[0]}.svg')
        dat_df.to_csv(f'{trace_csv_dir_dict[cat]}{splitext(basename(fb))[0]}.csv', sep='\t', header=True, index=True)
        plt.close()
transition_df.index = [f'{str(idx[0])}_{str(idx[1])}' for idx in transition_df.index.to_flat_index()]
plt.rcParams.update({'font.size': 15})  # smaller text for summary plots
acc_df.loc[:, 'accuracy'] = acc_df.correct / acc_df.total
acc_df.to_csv(f'{summary_dir}/accuracy_per_category.tsv', sep='\t')

# Plot transition density plots
efret_means = {}  # note actual label-based!
for cat in args.categories:
    efret_means[cat] = {}
    for lab in efret_dict[cat]:
        if len(efret_dict[cat][lab]):
            efret_means[cat][lab] = np.nanmean(np.concatenate(efret_dict[cat][lab]))
with open(f'{summary_dir}/means_dict.pkl', 'wb') as fh:
    pickle.dump(efret_means, fh, protocol=pickle.HIGHEST_PROTOCOL)

tdp_df = pd.concat(tdp_list)
for cat, df in tdp_df.groupby('category'):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for lab in efret_means[cat]:
        plt.axvline(efret_means[cat][lab], color='black', ls='--')
        plt.axhline(efret_means[cat][lab], color='black', ls='--')
    sns.kdeplot(df.departure_efret, df.arrival_efret, shade=True, cmap="coolwarm", ax=ax)
    # ax.collections[0].set_color('#3b4cc0')
    ax.set_facecolor('#4961d2')
    ax.set_xlabel('$E_{PR}$ before')
    ax.set_ylabel('$E_{PR}$ after')
    plt.savefig(f'{summary_dir}/{cat}_tdp.svg')
    df.to_csv(f'{summary_dir}/{cat}_tdp.tsv', header=True, index=True, sep='\t')
    with open(f'{summary_dir}/{cat}_means_dict.pkl', 'wb') as fh:
        pickle.dump(efret_means[cat], fh, protocol=pickle.HIGHEST_PROTOCOL)
    plt.clf()

# Plot precision/recall
confusion_df = reduce(lambda x, y: x.add(y, fill_value=0), confusion_list)
confusion_df.loc[:, 'precision'] = confusion_df.tp / (confusion_df.tp + confusion_df.fp)
confusion_df.loc[:, 'recall'] = confusion_df.tp / (confusion_df.tp + confusion_df.fn)
confusion_df = confusion_df.rename_axis(['category', 'state']).reset_index()
confusion_df.sort_values(['category', 'state'], inplace=True)
confusion_df.to_csv(f'{summary_dir}/confusion.tsv', header=True, index=True, sep='\t')
pr = sns.scatterplot(x='recall', y='precision', style='state', hue='category',
                     markers=('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'),
                     data=confusion_df)
# pr.legend_.remove()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
lgd = plt.gca().legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # plt.gca().get_legend().remove()
plt.savefig(f'{summary_dir}/precision_recall.svg', bbox_extra_artists=(lgd, ), bbox_inches='tight')
plt.savefig(f'{summary_dir}/precision_recall.svg', bbox_inches='tight')
plt.clf()

# plot transition rates
tr_files = parse_input_path(args.fb, pattern='*FRETboard_data_transition_rates.csv')
if tr_files:
    if any(np.array(framerate_list) - framerate_list[0] > 0.1):
        raise ValueError('Framerate not constant for all samples')

    # Ground truth from data
    framerate = np.mean(framerate_list)
    transition_df.loc[:, 'rate'] = 0
    tb = transition_df.nb_transitions != 0

    # correct discrete --> continuous
    tm = np.zeros((nb_classes, nb_classes))
    for ti, tup in transition_df.iterrows():
        s1, s2 = [int(tii) - 1 for tii in ti.split('_')]
        tm[s1, s2] = tup.nb_transitions / tup.nb_samples
    for i in range(nb_classes):
        tm[i,i] = 1 - np.sum(tm[i,:])

    tm_rate = discrete2continuous(tm, framerate)
    for tdi in transition_df.index:
        i1, i2 = [int(ii) for ii in tdi.split('_')]
        transition_df.loc[tdi, 'rate'] = tm_rate[int(i1) - 1, int(i2) - 1]

    # Loaded from FRETboard
    tr_df_list = []
    for trf in tr_files:
        trf_base = basename(trf)
        cat_list = [cat for cat in args.categories if cat in trf]
        if len(cat_list) > 1: raise ValueError(f'Multiple categories match to transition rate file {trf_base}')
        elif len(cat_list) == 0: continue
        cat = cat_list[0]
        tr_cur_df = pd.read_csv(trf, names=['transition', 'rate', 'low_bound', 'high_bound'], skiprows=1)
        tr_cur_df.loc[:, 'category'] = cat
        tr_df_list.append(tr_cur_df)
    tr_df = pd.concat(tr_df_list)
    target_states_str = np.array(target_states_w_ground, dtype=str)
    tr_df = tr_df.loc[tr_df.transition.apply(lambda x: np.all(np.in1d(list(x.split('_')), target_states_str))), :]

    tr_df.loc[:, 'transition'] = tr_df.transition.astype(str)
    tr_df.sort_values(['category', 'transition'], inplace=True)

    transition_df.loc[:, 'category'] = 'actual'
    transition_df.loc[:, 'low_bound'] = transition_df.rate
    transition_df.loc[:, 'high_bound'] = transition_df.rate
    transition_df = transition_df.rename_axis('transition').reset_index()
    transition_df.drop(['nb_samples', 'nb_transitions'], axis=1, inplace=True)
    transition_df.sort_values(['category', 'transition'], inplace=True)
    transition_df = pd.concat((tr_df, transition_df), sort=True)

    # derive values for CI bars
    yerr_list = []

    for tr, sdf in transition_df.groupby(['category'], sort=False):
        yerr_list.append(np.expand_dims(np.vstack((sdf.rate - sdf.low_bound, sdf.high_bound - sdf.rate)), 0))
    yerr = np.vstack(yerr_list)
    transition_piv_df = transition_df.pivot(index='transition', columns='category', values='rate')
    # move 'actual' column to last
    colnames = list(transition_piv_df.columns)
    colnames.remove('actual'); colnames += ['actual']
    transition_piv_df = transition_piv_df.loc[:, colnames]
    transition_piv_df.plot(kind='bar', yerr=yerr, legend=False)
    # lgd = plt.gca().legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.savefig(f'{summary_dir}/transition_rate.svg', bbox_extra_artists=(lgd, ), bbox_inches='tight')
    transition_piv_df.to_csv(f'{summary_dir}/transitions.tsv', sep='\t', index=True, header=True)
    np.save(f'{summary_dir}/transitions.tsv.npy', yerr)
    plt.savefig(f'{summary_dir}/transition_rate.svg', bbox_inches='tight')
    plt.clf()

    # tr bubble plot
    transition_piv_df.index = pd.MultiIndex.from_tuples([ [int(it) for it in idx.split('_')]
                                                          for idx in transition_piv_df.index], names=['from', 'to'])
    transition_piv_df.reset_index(inplace=True)

    for cat in args.categories:
        fig = plot_transition_bubble(transition_piv_df, cat)
        fig.savefig(f'{summary_dir}/transition_bubbleplot_{cat}.svg')
        plt.close()

# plot correct E_FRET histograms
eventstats_df = pd.concat(eventstats_list)
eventstats_pred_df = pd.concat(eventstats_pred_list)
eventstats_pred_df = copy(eventstats_pred_df.loc[np.in1d(eventstats_pred_df.label, args.target_states), :])
for cat in args.categories:
    ttest_df = pd.DataFrame(np.nan, columns=['ttost_p', 'nb_events_pred', 'nb_events_true'],
                            index=pd.MultiIndex.from_product((args.categories, args.target_states), names=['category', 'state']))

    for ts in args.target_states:
        efret_pred_dict[cat][ts] = np.concatenate(efret_pred_dict[cat][ts])
        efret_pred_dict[cat][ts] = efret_pred_dict[cat][ts][np.invert(np.isnan(efret_pred_dict[cat][ts]))]
        efret_dict[cat][ts] = np.concatenate(efret_dict[cat][ts])
        efret_dict[cat][ts] = efret_dict[cat][ts][np.invert(np.isnan(efret_dict[cat][ts]))]
        epdf = efret_pred_dict[cat][ts]
        edf = efret_dict[cat][ts]
        epdf[epdf < 0] = 0.0
        edf[edf < 0] = 0.0

        col = colmap(ts / max(args.target_states))
        sns.kdeplot(epdf, color=col, legend=False, bw=0.1)
        sns.kdeplot(edf, color=col, linestyle='--', legend=False, bw=0.1)
        ttest_df.loc[(cat, ts), ('nb_events_pred', 'nb_events_true')] = len(epdf), len(edf)
        if len(edf) and len(epdf):
            ttest_df.loc[(cat, ts), 'ttost_p'] = ttost_ind(epdf, edf, low=-0.025, upp=0.025, usevar='unequal')[0]
    ttest_df.to_csv(f'{summary_dir}/ttest_results_{cat}.tsv', sep='\t')
    plt.xlim(0.0, 1.2)
    plt.xlabel('$E_{FRET}$')
    plt.ylabel('count')
    plt.savefig(f'{summary_dir}/event_counts_kde_{cat}.svg')
    eventstats_pred_df.loc[eventstats_pred_df.category == cat, :].drop('category', axis=1).to_csv(
        f'{summary_dir}/event_counts_kde_{cat}.tsv', sep='\t', header=True, index=False)
    plt.clf()
eventstats_df.to_csv(f'{summary_dir}/event_counts_kde_true.tsv', sep='\t', header=True, index=False)

with open(f'{summary_dir}/efret_pred_dict.pkl', 'wb') as fh:
    pickle.dump(efret_pred_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{summary_dir}/efret_dict.pkl', 'wb') as fh:
    pickle.dump(efret_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)


# plot event coverage
eventstats_df.loc[:, 'coverage'] = eventstats_df.pred_duration / eventstats_df.duration * 100.0
eventstats_df.sort_values(['category'], inplace=True)
sns.violinplot(x='category', y='coverage', data=eventstats_df.loc[eventstats_df.label == eventstats_df.predicted])
ax = plt.gcf().axes[0]
ax.xaxis.label.set_visible(False)
plt.ylabel('Event coverage (%)')
plt.savefig(f'{summary_dir}/coverage_violin_full.svg')
eventstats_df.to_csv(f'{summary_dir}/eventstats.tsv', sep='\t')
plt.clf()

# event coverage with equalized axes
sns.violinplot(x='category', y='coverage', data=eventstats_df.loc[eventstats_df.label == eventstats_df.predicted])
ax = plt.gcf().axes[0]
ax.xaxis.label.set_visible(False)
# ax.yaxis.label.set_visible(False)
plt.ylabel('Event coverage (%)')
ax.set(xticklabels=[])
ax.set(ylim=(0, 300))
plt.savefig(f'{summary_dir}/coverage_violin.svg')
plt.clf()

# plot accuracy per individual trace
plt.hist(acc_list, density=False)
plt.xlabel('Accuracy')
plt.ylabel('# traces')
plt.title(f'Per-trace accuracy for {basename(args.manual)}')
plt.text(0.1, 0.9, f'Overall acc: {correct_pts / total_pts :.2f}')
print(f'Accuracy: {correct_pts / total_pts}')
plt.savefig(f'{summary_dir}accuracy_hist.png')
