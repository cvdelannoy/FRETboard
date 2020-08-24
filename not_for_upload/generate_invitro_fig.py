import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import re
import os, pathlib
import numpy as np
import seaborn as sns
import matplotlib as mpl
import pickle
from math import ceil
from plotting_functions import plot_transition_bar, plot_kde_from_vec, plot_publication_trace

font = {
    # 'family' : 'normal',
        'size'   : 22}

mpl.rc('font', **font)


parser = argparse.ArgumentParser(description='Generate paper figure for comparison simulated results')
parser.add_argument('--eval-dirs', nargs='+', type=str,required=True)
parser.add_argument('--cat-names', nargs='+', type=str, required=True)
parser.add_argument('--out-svg', type=str, required=True)
parser.add_argument('--example-traces-tsv', type=str, required=True)

args = parser.parse_args()

kde_tsv = []
for ev in args.eval_dirs:
    kde_tsv.append(ev + '/summary_stats/' + [fn for fn in os.listdir(ev+'/summary_stats/')  if re.search('_kde_.+tsv', fn)][0])


confusion_tsv = [ev +'/summary_stats/confusion.tsv' for ev in args.eval_dirs]
eventstats_tsv = [ev +'/summary_stats/eventstats.tsv' for ev in args.eval_dirs]
transitions_tsv = [ev +'/summary_stats/transitions.tsv' for ev in args.eval_dirs]
efret_pkl = [ev +'/summary_stats/efret_dict.pkl' for ev in args.eval_dirs]
efret_pred_pkl = [ev +'/summary_stats/efret_pred_dict.pkl' for ev in args.eval_dirs]
target_states_dict = {ev: pd.read_csv(ev+'/summary_stats/confusion.tsv', index_col=0, header=0, sep='\t').state.to_list() for ev in args.eval_dirs}

# assert len(confusion_tsv) == len(eventstats_tsv) == len(transitions_tsv) == len(cat_names)

nb_plots = len(confusion_tsv)
fig = plt.figure(constrained_layout=False, figsize=(48/2.54, 40/2.54))
gs = gridspec.GridSpec(9, nb_plots, figure=fig, wspace=0.2, hspace=0.30)


cat_str = '|'.join(args.cat_names)

# confusion_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in confusion_tsv}
eventstats_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in eventstats_tsv}
transitions_dict = {re.search(cat_str, fn).group(0): [pd.read_csv(fn, sep='\t', header=0, index_col='transition'),
                                                      np.load(fn+'.npy')] for fn in transitions_tsv}
kde_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in kde_tsv}

all_transition_rates = np.concatenate([td[0].to_numpy().reshape(-1) for td in transitions_dict.values()])
max_transition_rate = all_transition_rates.max()

efret_dict = {}
for pkl in efret_pkl:
    with open(pkl, 'rb') as fh: efret_dict[re.search(cat_str, pkl).group(0)] = pickle.load(fh)['']
efret_pred_dict = {}
for pkl in efret_pred_pkl:
        with open(pkl, 'rb') as fh: efret_pred_dict[re.search(cat_str, pkl).group(0)] = pickle.load(fh)['']

examples_df = pd.read_csv(args.example_traces_tsv, sep='\t', header=0, index_col=0)
examples_df.loc[:, 'fn'] = str(pathlib.Path(args.example_traces_tsv).parent.absolute()) + '/' + examples_df.fn

plot_types = ['trace', 'kde', 'transition']
nb_cats = len(args.cat_names)
for cidx, cat in enumerate(args.cat_names):
    # plot
    if cidx == 0:
        ax_dict ={
            'trace': [fig.add_subplot(gs[ii, 0]) for ii in range(3)],
            'kde': fig.add_subplot(gs[3:6, 0]),
            'transition': fig.add_subplot(gs[6:9, 0])
        }
        first_ax_dict = ax_dict
    else:
        ax_dict = {
            'trace': [fig.add_subplot(gs[ii, cidx],
                                      # sharey=first_ax_dict['trace'][ii]
                                      ) for ii in range(3)],
            'kde': fig.add_subplot(gs[3:6, cidx], sharey=first_ax_dict['kde']),
            'transition': fig.add_subplot(gs[6:9, cidx], sharey=first_ax_dict['transition'])
        }
    target_states = target_states_dict[[ts for ts in target_states_dict if cat in ts][0]]
    fn, start_pos, duration = examples_df.loc[cat]
    plot_publication_trace(fn, start_pos, duration, len(target_states), ax_dict['trace'])
    plot_kde_from_vec(efret_pred_dict[cat], efret_dict[cat], target_states, ax_dict['kde'])
    plot_transition_bar(transitions_dict[cat][0], transitions_dict[cat][1], ax_dict['transition'])

    # Add axis labels if left-most plot
    if cidx == 0:
        ax_dict['transition'].set_ylabel('Transitions ($s^{-1}$)')
        ax_dict['transition'].set_ylim(0, ceil(max_transition_rate / 0.5) * 0.5)
        ax_dict['kde'].set_ylabel('Count density')
    if cidx ==  (nb_cats - 1) // 2:
        ax_dict['transition'].set_xlabel('Transition')
        ax_dict['kde'].set_xlabel('$E_{FRET}$')
fig.savefig(args.out_svg)
