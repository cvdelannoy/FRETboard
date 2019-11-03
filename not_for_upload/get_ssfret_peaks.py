import os
import sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.insert(1, f'{__location__}/..')
import argparse
from not_for_upload.helper_functions import parse_input_path, condense_sequence, get_ssfret_dist
from FRETboard.MainTable import MainTable
from itertools import chain, combinations
import importlib
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
from scipy.stats import norm, mode
from sklearn.cluster import DBSCAN

def plot_efret_hist(efret):
    plt.hist(efret, bins=20)
    plt.xlabel('E_FRET')
    plt.ylabel('count')

parser = argparse.ArgumentParser(description='extract ssFRET peaks from TIRF data.')
parser.add_argument('--indir', type=str, required=True, nargs='+',
                    help='Input directories/files')
parser.add_argument('--label', type=int, required=True,
                    help='Number of the class for which ssFRET estimate should be returned.')
parser.add_argument('--model', type=str, required=True,
                    help='Model file produced by FRETboard.')
parser.add_argument('--analysis-type', type=str, choices=['samples', 'traces', 'events'], default='samples',
                    help='return one ssFRET value for all files (samples), one per trace (traces) or one per '
                         'uninterrupted event (events).')
parser.add_argument('--plot', type=str,
                    help='Plot histogram of samples/events, save to file')
args = parser.parse_args()

# load data
dat_list = parse_input_path(args.indir, pattern='*.dat')
traces_list = parse_input_path(args.indir, pattern='*.traces')
main_table = MainTable(dat_list)

# todo implement traces file parsing

# load model
with open(f'{__location__}/../FRETboard/algorithms.yml', 'r') as fh: algo_dict = yaml.safe_load(fh)
with open(args.model, 'r') as fh: model_txt = fh.read()
model_list = model_txt.split('\nSTART_NEW_SECTION\n')
model_name = model_list[0]
misc_dict = yaml.load(model_list[-1], Loader=yaml.SafeLoader)
classifier = importlib.import_module('FRETboard.algorithms.'+model_name).Classifier(nb_states=misc_dict['nb_states'],
                                                                                    buffer=misc_dict['buffer'],
                                                                                    data=main_table)
classifier.load_params(model_txt)
pred, logprob = classifier.predict(main_table.data.index)

# todo implement for per-trace and per-event as well
if args.analysis_type == 'samples':
    # Return one value for histogram of all samples in all traces
    efret_mat = np.concatenate(main_table.data.E_FRET.to_numpy())
    label_mat = np.concatenate(pred)
    efret = efret_mat[label_mat == args.label]
    mu, srsd = get_ssfret_dist(efret)
    if args.plot: plot_efret_hist(efret); plt.savefig(args.plot, dpi=400)
    print(f'mu: {mu}\nsrsd: {srsd}')
elif args.analysis_type == 'events':
    # Average E_FRET over events, return one value for histogram of events of all traces
    events = [condense_sequence(val, lab) for val, lab in zip(main_table.data.E_FRET, pred)]
    events = list(chain.from_iterable(events))
    # todo: here events tuples still contain event duration. Do we need some lower cutoff in event duration?
    efret = np.array([ev[2] for ev in events if ev[0] == args.label])
    mu, srsd = get_ssfret_dist(efret)
    if args.plot: plot_efret_hist(efret); plt.savefig(args.plot, dpi=400)
    print(f'mu: {mu}\nsrsd: {srsd}')
elif args.analysis_type == 'traces':
    # Return one value per trace, based on sample histogram
    main_table.data.loc[:, "prediction"] = pred
    dist_series = main_table.data.apply(lambda x: get_ssfret_dist(x.E_FRET[x.prediction == args.label]), axis=1)
    dist_df = pd.DataFrame({'mu': dist_series.apply(lambda x: x[0]),
                            'srsd': dist_series.apply(lambda x: x[1])}, index=dist_series.index)
    if args.plot:
        dist_mat = pd.DataFrame(columns=dist_df.index, index=dist_df.index)
        dist_mat.loc[:, :] = 0.5
        for f1, f2 in combinations(dist_df.index, 2):
            dmu = abs(dist_df.loc[f1, 'mu'] - dist_df.loc[f2, 'mu'])
            dsd = sqrt(dist_df.loc[f1, 'mu'] ** 2 + dist_df.loc[f2, 'mu'] ** 2)
            p = norm(dmu, dsd).cdf(0)
            dist_mat.loc[f1, f2] = p
            dist_mat.loc[f2, f1] = p
        fig, axes = plt.subplots(1,2, figsize=(20, 10))
        sns.heatmap(data=dist_mat.astype(float), vmin=0.05, vmax=0.5, square=True,
                    xticklabels=False, yticklabels=False, cbar_kws={'label': '$P_{(equal)}$'}, ax=axes[0])
        bool_mat = (dist_mat > 0.05)
        bool_mata = bool_mat.astype(float)
        sns.heatmap(data=bool_mat, square=True, cmap='Greys', cbar=False,
                    xticklabels=False, yticklabels=False, ax=axes[1])
        fig.savefig(args.plot, dpi=400)
    print(dist_df.to_string())
