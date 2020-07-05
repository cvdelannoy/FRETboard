import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import re
import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
from plotting_functions import plot_transition_bar, plot_kde, plot_transition_bubble

font = {'family' : 'normal',
        'size'   : 22}

mpl.rc('font', **font)


parser = argparse.ArgumentParser(description='Generate paper figure for comparison simulated results')
parser.add_argument('--eval-dirs', nargs='+', type=str,required=True)
parser.add_argument('--target-states', nargs='+', type=int, required=True)
parser.add_argument('--cat-names', nargs='+', type=str, required=True)
parser.add_argument('--out-svg', type=str, required=True)

args = parser.parse_args()

kde_tsv = []
for ev in args.eval_dirs:
    kde_tsv.append(ev + '/summary_stats/' + [fn for fn in os.listdir(ev+'/summary_stats/')  if re.search('_kde_.+\.tsv', fn)][0])


confusion_tsv = [ev +'/summary_stats/confusion.tsv' for ev in args.eval_dirs]
eventstats_tsv = [ev +'/summary_stats/eventstats.tsv' for ev in args.eval_dirs]
transitions_tsv = [ev +'/summary_stats/transitions.tsv' for ev in args.eval_dirs]

# assert len(confusion_tsv) == len(eventstats_tsv) == len(transitions_tsv) == len(cat_names)

nb_plots = len(confusion_tsv)
fig = plt.figure(constrained_layout=False, figsize=(48/2.54, 40/2.54))
gs = gridspec.GridSpec(3, nb_plots, figure=fig, wspace=0.2, hspace=0.30)
cat_str = '|'.join(args.cat_names)

# confusion_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in confusion_tsv}
eventstats_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in eventstats_tsv}
transitions_dict = {re.search(cat_str, fn).group(0): [pd.read_csv(fn, sep='\t', header=0, index_col='transition'),
                                                      np.load(fn+'.npy')] for fn in transitions_tsv}
kde_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in kde_tsv}

plot_types = ['kde', 'transition']
nb_cats = len(args.cat_names)
for cidx, cat in enumerate(args.cat_names):

    # plot
    if cidx == 0:
        ax_dict = {plot_name: fig.add_subplot(gs[idx, cidx]) for idx, plot_name in enumerate(plot_types)}
        first_ax_dict = ax_dict
    else:
        ax_dict = {plot_name: fig.add_subplot(gs[idx, cidx], sharey=first_ax_dict[plot_name]) for
                   idx, plot_name in enumerate(plot_types)}
    plot_kde(kde_dict[cat], eventstats_dict[cat], args.target_states, ax_dict['kde'])
    plot_transition_bar(transitions_dict[cat][0], transitions_dict[cat][1], ax_dict['transition'])
    # if '10_state' in cat:
    #     plot_transition_bubble(transitions_dict[cat][0], transitions_dict[cat][0].columns[0], ax_dict['transition'])
    # else:
    #     plot_transition_bar(transitions_dict[cat][0], transitions_dict[cat][1], ax_dict['transition'])


    # transitions_dict[cat][0].plot(kind='bar', yerr=transitions_dict[cat][1], legend=False, ax=ax_dict['transition'])

    # Add axis labels if left-most plot
    if cidx == 0:
        ax_dict['transition'].set_ylabel('Transitions ($s^{-1}$)')
        ax_dict['kde'].set_ylabel('Count density')
    if cidx ==  nb_cats // 2:
        ax_dict['transition'].set_xlabel('Transition')
        ax_dict['kde'].set_xlabel('$E_{FRET}$')
# plt.tight_layout()
fig.savefig(args.out_svg)
