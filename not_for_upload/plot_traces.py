import argparse, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import basename, splitext

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
from plotting_functions import plot_trace


parser = argparse.ArgumentParser(description='Plot traces in paper quality, with matching domains and ranges.')
parser.add_argument('--tsv', type=str, required=True)
parser.add_argument('--out-svg', type=str, required=True)
parser.add_argument('--duration', type=float, default=np.inf, help='Duration of traces to retain in s [default: keep all]')
parser.add_argument('--start-pos', type=float, default=0.0, help='Optionally, note for each trace how long [default: 0.0] ')
parser.add_argument('--nb-states', type=int, default=-1, help='Optionally, set number of states to be labeled [default: determine from file]')

args = parser.parse_args()

dat_df = pd.read_csv(args.tsv, header=0, sep='\t')

# Retain indicated stretch
dat_df = dat_df.loc[dat_df.time >= args.start_pos, :]
dat_df.time = dat_df.time - dat_df.time.iloc[0]
dat_df = dat_df.loc[dat_df.time <= args.duration, :]

# get number of states
if args.nb_states == -1:
    nb_states = dat_df.predicted.max()
elif args.nb_states > 0:
    nb_states = args.nb_states
else:
    raise ValueError(f'{args.nb_states} is not a valid number of states.')

# states_colors = ['#ff7f00', '#377eb8'][:nb_states] # orange blue (http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=5)
states_colors = ['#a3a3a3', '#373737'][:nb_states]  # grey shades

# Collect plot info
plot_dicts = []

# don/acc traces
plot_dicts.append({
    'ylabel': '$I$',
    'xlabel': 'time ($s$)',
    'data': dat_df.loc[:, ['f_dex_dem', 'f_dex_aem']].to_numpy(),
    'label': dat_df.loc[:, 'predicted'].to_numpy(),
    'colors': ['green', 'magenta']
})

# summed intensity
plot_dicts.append({
    'ylabel': '$I_{sum}$',
    'xlabel': 'time (s)',
    'data': dat_df.loc[:, ['i_sum']].to_numpy(),
    'label': dat_df.loc[:, 'predicted'].to_numpy(),
    'colors': ['black']
})

# E_FRET
plot_dicts.append({
    'ylabel': '$E_{PR}$',
    'xlabel': 'time (s)',
    'data': dat_df.loc[:, ['E_FRET']].to_numpy(),
    # 'label': dat_df.loc[:, 'predicted'].to_numpy(),
    'colors': ['blue'],
    'label': dat_df.loc[:, 'predicted'].to_numpy(),
    'ylim': [0.0, 1.1]
})

# Define figure and axes
fig = plt.figure(constrained_layout=True, figsize=(16/2.54 * 2, 14.4/2.54))
traces_gs = gridspec.GridSpec(3, 4, figure=fig,
                              hspace=0.1,
                              bottom=0.2
                              # left=0.055
                              )
axes = [fig.add_subplot(traces_gs[i,:]) for i in range(3)]

plot_trace(plot_dicts, nb_classes=nb_states + 1, time=dat_df.time, axes=axes, states_colors=states_colors)
fig.savefig(args.out_svg, format='svg')
