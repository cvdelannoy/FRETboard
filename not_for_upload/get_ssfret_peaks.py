import os
import sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.insert(1, f'{__location__}/..')
import argparse
from not_for_upload.helper_functions import parse_input_path, parse_output_dir, condense_sequence, get_ssfret_dist
from FRETboard.MainTable import MainTable
from FRETboard.io_functions import parse_trace_file
from itertools import chain
import importlib
import yaml

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from joblib import Parallel, delayed

def plot_efret_hist(efret, label=None):
    sns.distplot(efret, bins=50, label=label)
    plt.xlabel('E_FRET')
    plt.ylabel('count')
    plt.legend()

def plot_sr_dists(mu, sd, label=None):
    support = np.linspace(0.0, 1.0, 1000)
    kernel = norm(mu, sd).pdf(support)
    sns.lineplot(support, kernel, label=label)
    plt.xlabel('E_FRET')
    plt.ylabel('density')
    plt.legend()

parser = argparse.ArgumentParser(description='extract ssFRET peaks from TIRF data.')
parser.add_argument('--indir', type=str, required=True, nargs='+',
                    help='Input directories/files')
parser.add_argument('--outdir', type=str, required=True,
                    help='output directory')
parser.add_argument('--label', type=int, required=True,
                    help='Number of the class for which ssFRET estimate should be returned.')
parser.add_argument('--model', type=str, required=True,
                    help='Model file produced by FRETboard.')
parser.add_argument('--analysis-type', type=str, nargs='+', choices=['samples', 'traces', 'events'],
                    default=['samples'],
                    help='return one ssFRET value for all files (samples), one per trace (traces) or one per '
                         'uninterrupted event (events).')
parser.add_argument('--groups', nargs='+', type=str,
                    help='regex patterns used to classify different groups of traces based on name,'
                         'plotted with different colors')
parser.add_argument('--threads', default=1, type=int,
                    help='Number of threads to use for DBSCAN filter')
args = parser.parse_args()

outdir = parse_output_dir(args.outdir, clean=False)
label_0based = args.label - 1

# load data
dat_list = parse_input_path(args.indir, pattern='*.dat')
traces_list = parse_input_path(args.indir, pattern='*.traces')
main_table = MainTable(dat_list)

if len(traces_list):
    table_list = []
    for trace_fn in traces_list:
        with open(trace_fn, 'rb') as fh: trace_content = fh.read()
        table_list.extend(parse_trace_file(trace_content, trace_fn, args.threads))
    main_table.add_df_list(table_list)

# load model
with open(f'{__location__}/../FRETboard/algorithms.yml', 'r') as fh: algo_dict = yaml.safe_load(fh)
with open(args.model, 'r') as fh: model_txt = fh.read()
model_list = model_txt.split('\nSTART_NEW_SECTION\n')
model_name = model_list[0]
features = model_list[2].split('\n')
misc_dict = yaml.load(model_list[-1], Loader=yaml.SafeLoader)
classifier = importlib.import_module('FRETboard.algorithms.'+model_name).Classifier(nb_states=misc_dict['nb_states'],
                                                                                    buffer=misc_dict['buffer'],
                                                                                    features=features,
                                                                                    data=main_table,
                                                                                    nb_threads=args.threads)
classifier.load_params(model_txt)

# classify samples
pred, logprob = classifier.predict(main_table.data.index)

if 'samples' in args.analysis_type:
    # Return one value for histogram of all samples in all traces
    out_df = pd.DataFrame(index=args.groups, columns=['mu', 'sd', 'srsd', 'n_samples'])
    for group in args.groups:
        group_bool = main_table.data.index.str.contains(group)
        efret_mat = np.concatenate(main_table.data.loc[group_bool, 'E_FRET'].to_numpy())
        label_mat = np.concatenate([p for pi, p in enumerate(pred) if group_bool[pi]])
        efret = efret_mat[label_mat == label_0based]
        efret = efret[np.invert(np.isnan(efret))]
        out_df.loc[group] = get_ssfret_dist(efret)
        plot_efret_hist(efret, label=group)
    plt.savefig(outdir + 'samples_hist.png', dpi=400)

    # superresolution distributions
    plt.clf()
    for gr, f in out_df.iterrows(): plot_sr_dists(f.mu, f.srsd, label=gr)
    plt.savefig(outdir + 'samples_sr_dists.png', dpi=400)
    out_df.to_csv(outdir + 'samples_stats.csv')

if 'events' in args.analysis_type:
    # Average E_FRET over events, return one value for histogram of events of all traces
    plt.clf()
    out_df = pd.DataFrame(index=args.groups, columns=['mu', 'sd', 'srsd', 'n_samples'])
    for group in args.groups:
        group_bool = main_table.data.index.str.contains(group)
        pred_group = [p for pi, p in enumerate(pred) if group_bool[pi]]
        events = [condense_sequence(val, lab) for val, lab in zip(main_table.data.loc[group_bool, 'E_FRET'], pred_group)]
        events = list(chain.from_iterable(events))
        efret = np.array([ev[2] for ev in events if ev[0] == label_0based])
        out_df.loc[group] = get_ssfret_dist(efret)
        plot_efret_hist(efret, label=group)
    plt.savefig(outdir + 'events_hist.png', dpi=400)

    # superresolution distributions
    plt.clf()
    for gr, f in out_df.iterrows(): plot_sr_dists(f.mu, f.srsd, label=gr)
    plt.savefig(outdir + 'events_sr_dists.png', dpi=400)
    out_df.to_csv(outdir + 'events_stats.csv')

if 'traces' in args.analysis_type:
    # Return one value per trace, based on sample histogram
    for ii, i in enumerate(main_table.data.index):
        main_table.data.at[i, "prediction"] = pred[ii]
    filtered_df = main_table.data.loc[main_table.data.prediction.apply(lambda x: np.any(x == label_0based)), :]
    dist_list = Parallel(n_jobs=args.threads)(delayed(get_ssfret_dist)(x.E_FRET[x.prediction == label_0based], xi)
                                  for xi, x in filtered_df.iterrows())
    dist_df = pd.DataFrame.from_dict({idx: [mu, sd, srsd, n_points] for mu, sd, srsd, n_points, idx in dist_list},
                                     orient='index', columns=['mu', 'sd', 'srsd', 'n_points'])
    # plotting
    plt.clf()
    support = np.linspace(0.0, 1.0, 200)
    df_list = []
    # plot_df = pd.DataFrame(columns=['E_FRET', 'density', 'group', 'trace'], index=dist_df.index)
    for group in args.groups:
        for fn, f in dist_df.loc[dist_df.index.str.contains(group)].iterrows():
            density = norm(f.mu, f.srsd).pdf(support)
            df_list.append(pd.DataFrame({'E_FRET': support, 'density': density,
                                         'group': group, 'trace': fn}))
    plot_df = pd.concat(df_list)
    sns.lineplot(x='E_FRET', y='density', hue='group', units='trace', estimator=None, data=plot_df)
    plt.legend()
    plt.savefig(outdir + 'traces_sr_dists.png', dpi=400)

    dist_df.to_csv(outdir + 'traces_stats.csv')

    # heatmap plot
    # dist_mat = pd.DataFrame(columns=dist_df.index, index=dist_df.index)
    # dist_mat.loc[:, :] = 0.5
    # for f1, f2 in combinations(dist_df.index, 2):
    #     dmu = abs(dist_df.loc[f1, 'mu'] - dist_df.loc[f2, 'mu'])
    #     dsd = sqrt(dist_df.loc[f1, 'srsd'] ** 2 + dist_df.loc[f2, 'srsd'] ** 2)
    #     p = norm(dmu, dsd).cdf(0)
    #     dist_mat.loc[f1, f2] = p
    #     dist_mat.loc[f2, f1] = p
    # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    # sns.heatmap(data=dist_mat.astype(float), vmin=0.0, vmax=0.1, square=True,
    #             xticklabels=False, yticklabels=False, cbar_kws={'label': '$P_{(equal)}$'}, ax=axes[0])
    # bool_mat = (dist_mat > 0.05)
    # bool_mata = bool_mat.astype(float)
    # sns.heatmap(data=bool_mat, square=True, cmap='Greys', cbar=False,
    #             xticklabels=False, yticklabels=False, ax=axes[1])
