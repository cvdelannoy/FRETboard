import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from plotting_functions import plot_trace

# pointers taken from https://stackoverflow.com/questions/31452451/importing-an-svg-file-into-a-matplotlib-figure

parser = argparse.ArgumentParser(description='Generate first paper figure')
# parser.add_argument('--flowchart-svg', type=str, required=True)
parser.add_argument('--trace-tsv', type=str, required=True)
# parser.add_argument('--schematic-svg', type=str, required=True)
parser.add_argument('--kde-tsv', type=str, required=True)
parser.add_argument('--eventstats-tsv', type=str, required=True)
parser.add_argument('--target-states', type=int, nargs='+', required=True)
parser.add_argument('--out-svg', type=str, required=True)
parser.add_argument('--second-trace', type=str, default='E_FRET')

args = parser.parse_args()
nb_states = len(args.target_states)

# states_colors = ['#377eb8', '#fc8d62']
states_colors = ['#ff7f00', '#377eb8'][:len(args.target_states)] # orange blue (http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=5)
font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    'size'   : 25}

mpl.rc('font', **font)

#  --- Make plotted part figure ---

# traces
traces_fig = plt.figure(constrained_layout=True, figsize=(16/2.54 * 4, 14.4/2.54))
traces_gs = gridspec.GridSpec(2, 4, figure=traces_fig,
                              hspace=0.1,
                              bottom=0.2
                              # left=0.055
                              )
ax_ts1 = traces_fig.add_subplot(traces_gs[0, :3])
ax_ts2 = traces_fig.add_subplot(traces_gs[1, :3], #sharey=ax_ts1, #sharex=ax_ts1,
                                )
kde_gs = gridspec.GridSpec(2, 4, figure=traces_fig, wspace=0.5, bottom=0.2)
ax_kde = traces_fig.add_subplot(kde_gs[:, 3])

plot_dicts = []
dat_df = pd.read_csv(args.trace_tsv, header=0, sep='\t')
dat_df.loc[:, 'i_sum'] = dat_df.i_don + dat_df.i_acc
if args.second_trace == 'E_FRET':
    ylabel='$E_{FRET}$'
    second_data = dat_df.loc[:, ['E_FRET']].to_numpy()
    ylim = [0.0, 1.1]
elif args.second_trace == 'I_sum':
    ylabel = '$I_{sum}$'
    second_data = dat_df.loc[:, 'i_don'] + dat_df.loc[:, 'i_acc']
    ylim = []
plot_dicts.append({
    'ylabel': ylabel,
    'xlabel': 'time (s)',
    'data': second_data,
    # 'label': dat_df.loc[:, 'predicted'].to_numpy(),
    'colors': ['blue'],
    'ylim': ylim
})
plot_dicts.append({
    'ylabel': 'I',
    'xlabel': 'time (s)',
    'data': dat_df.loc[:, ['i_don', 'i_acc', 'i_sum']].to_numpy(),
    'label': dat_df.loc[:, 'predicted'].to_numpy(),
    'colors': ['green', 'red', 'grey']
})
plot_trace(plot_dicts, nb_states + 1, dat_df.time, [ax_ts1, ax_ts2], states_colors=states_colors)

# kde
kde_df = pd.read_csv(args.kde_tsv, sep='\t', header=0)
eventstats_df = pd.read_csv(args.eventstats_tsv, sep='\t', header=0)
for tsi, ts in enumerate(args.target_states):
    sns.kdeplot(kde_df.loc[kde_df.label == ts, 'E_FRET'], ax=ax_kde, color=states_colors[tsi], legend=False)
    sns.kdeplot(eventstats_df.loc[eventstats_df.label == ts, 'E_FRET'], ax=ax_kde, color=states_colors[tsi], legend=False, linestyle='--')
ax_kde.set_xlabel('$E_{FRET}$')
ax_kde.set_ylabel('count density')
ax_kde.set_xlim(0,1)
# plt.draw()
# xt_labs = ax_kde.get_xticklabels()
# xt_labs[0] = ''
# ax_kde.set(xticklabels=xt_labs)
plt.tight_layout()
traces_fig.savefig(args.out_svg, format='svg')
