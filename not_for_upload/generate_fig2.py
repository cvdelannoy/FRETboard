import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib as mpl
from plotting_functions import plot_recall_precision, plot_coverage_violin, plot_transition_bar


parser = argparse.ArgumentParser(description='Generate second paper figure')
parser.add_argument('--eval-dirs', nargs='+', type=str,required=True)
# parser.add_argument('--confusion-tsv', nargs='+', type=str, required=True)
# parser.add_argument('--eventstats-tsv', nargs='+', type=str, required=True)
# parser.add_argument('--transitions-tsv', nargs='+', type=str, required=True)
parser.add_argument('--cat-names', nargs='+', type=str, required=True)
parser.add_argument('--out-svg', type=str, required=True)

args = parser.parse_args()

confusion_tsv = [ev +'/summary_stats/confusion.tsv' for ev in args.eval_dirs]
eventstats_tsv = [ev +'/summary_stats/eventstats.tsv' for ev in args.eval_dirs]
transitions_tsv = [ev +'/summary_stats/transitions.tsv' for ev in args.eval_dirs]

# assert len(confusion_tsv) == len(eventstats_tsv) == len(transitions_tsv) == len(cat_names)

nb_plots = len(confusion_tsv)
fig = plt.figure(constrained_layout=False, figsize=(16/2.54 * 4, 40/2.54))
gs = gridspec.GridSpec(3, nb_plots, figure=fig, wspace=0.2, hspace=0.13)
cat_str = '|'.join(args.cat_names)

confusion_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in confusion_tsv}
eventstats_dict = {re.search(cat_str, fn).group(0): pd.read_csv(fn, sep='\t', header=0) for fn in eventstats_tsv}
transitions_dict = {re.search(cat_str, fn).group(0): [pd.read_csv(fn, sep='\t', header=0, index_col='transition'),
                                                      np.load(fn+'.npy')] for fn in transitions_tsv}
plot_types = ['recall_precision', 'coverage', 'transition']
for cidx, cat in enumerate(args.cat_names):

    # plot
    if cidx == 0:
        ax_dict = {plot_name: fig.add_subplot(gs[idx, cidx]) for idx, plot_name in enumerate(plot_types)}
        first_ax_dict = ax_dict
    else:
        ax_dict = {plot_name: fig.add_subplot(gs[idx, cidx], sharey=first_ax_dict[plot_name]) for idx, plot_name in enumerate(plot_types)}
    plot_recall_precision(confusion_dict[cat], ax_dict['recall_precision'])
    plot_coverage_violin(eventstats_dict[cat], ax_dict['coverage'])
    plot_transition_bar(transitions_dict[cat][0], transitions_dict[cat][1], ax_dict['transition'])
    # transitions_dict[cat][0].plot(kind='bar', yerr=transitions_dict[cat][1], legend=False, ax=ax_dict['transition'])

    # Add axis labels if left-most plot
    if cidx == 0:
        ax_dict['recall_precision'].set_ylabel('precision')
        ax_dict['coverage'].set_ylabel('Event coverage (%)')
        ax_dict['transition'].set_ylabel('transitions ($s^{-1}$)')
plt.tight_layout()
fig.savefig(args.out_svg)