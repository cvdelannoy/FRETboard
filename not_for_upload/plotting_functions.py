import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from pylab import cm
from matplotlib.ticker import MaxNLocator
cmap = cm.get_cmap('Set1', 9)
cmap_list = np.array([rgb2hex(cmap(i)[:3]) for i in range(9)])



def plot_tdp(df, efret_means, ax):
    # Plot transition density plots
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for lab in efret_means:
        plt.axvline(efret_means[lab], color='black', ls='--')
        plt.axhline(efret_means[lab], color='black', ls='--')
    sns.kdeplot(df.departure_efret, df.arrival_efret, shade=True, cmap="coolwarm", ax=ax)
    ax.set_facecolor('#4961d2')
    ax.set_xlabel('$E_{PR}$ before')
    ax.set_ylabel('$E_{PR}$ after')
    plt.xticks(np.arange(0, 1.00001, 0.5))
    plt.yticks(np.arange(0, 1.00001, 0.5))


def plot_trace(data_dicts, nb_classes, time, axes=None, states_colors=None):
    """
    Plot traces, provided as one dict per plot
    :return:
    """
    # line_cols=['green', 'red', 'brown', 'black']
    # fig = plt.figure(figsize=(20, 5))
    if type(data_dicts) == dict:
        data_dicts = [data_dicts]
    assert len(axes) == len(data_dicts)
    if states_colors:
        # assert len(states_colors) == len(data_dicts)
        states_colors = ['#FFFFFF'] + states_colors
    else:
        states_colors = ['#FFFFFF', '#084594']
    nb_plots = len(data_dicts)
    if axes:
        plot_dict = {ai: ax for ai, ax in enumerate(axes)}
    else:
        plot_dict = {0: plt.subplot2grid((nb_plots, 1), (0, 0))}
        for pid in range(1, len(data_dicts)):
            plot_dict[pid] = plt.subplot2grid((nb_plots, 1), (0, 0), sharex=plot_dict[0]
                                              )
    for pid, cdd in enumerate(data_dicts):
        if cdd['data'].ndim < 2:
            cdd['data'] = np.expand_dims(cdd['data'], -1)

        if cdd.get('label') is not None:
            vrange = cdd.get('ylim', (cdd['data'].min(), cdd['data'].max()))
            plot_dict[pid].pcolorfast((0, time.max()), vrange, cdd['label'].reshape(1,-1),
                                      vmin=1, vmax=nb_classes,
                                      cmap=LinearSegmentedColormap.from_list('custom', states_colors),
                                      alpha=1)
        if pid == len(data_dicts)-1:
            plot_dict[pid].set(ylabel=cdd['ylabel'], xlabel=cdd['xlabel'])
        else:
            plot_dict[pid].set(ylabel=cdd['ylabel'])
            plot_dict[pid].axes.get_xaxis().set_ticks([])
        plot_dict[pid].set_xlim(0, time.max())
        for n in range(cdd['data'].shape[1]):
            plot_dict[pid].plot(time, cdd['data'][:, n], linewidth=0.5, color=cdd['colors'][n])
            if cdd.get('ylim', False):
                plot_dict[pid].set_ylim(cdd['ylim'])
        # if cdd.get('label') is not None:
        #     if cdd['label'].ndim < 2:
        #         cdd['label'] = np.expand_dims(cdd['label'], -1)
        #     nb_colbars = cdd['label'].shape[1]
        #     for n in range(nb_colbars):
        #         if n == 0:
        #             bw = (plot_dict[pid].get_ylim()[1] - plot_dict[pid].get_ylim()[0]) * 0.3
        #             spacing = bw * 0.1
        #             cb_vrange = [cdd['data'].min() - bw, cdd['data'].min()]
        #             colbar_width = cb_vrange[1] - cb_vrange[0]
        #         else:
        #             cb_bottom = cb_vrange[0]
        #             cb_vrange = [cb_bottom - spacing, cb_bottom - colbar_width - spacing]
        #         plot_dict[pid].pcolorfast((0, time.max()), cb_vrange, cdd['label'][:, n].reshape(1,-1),
        #                                   vmin=1, vmax=nb_classes,
        #                                   cmap=LinearSegmentedColormap.from_list('custom', states_colors),
        #                                   alpha=1)
    plt.tight_layout()

def plot_recall_precision(confusion_df, ax):
    pr = sns.scatterplot(x='recall', y='precision', style='state', hue='category',
                         markers=('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'),
                         data=confusion_df, ax=ax, s=100)
    pr.legend_.remove()
    # categories = confusion_df.category.sort_values().to_numpy()
    # for _, tup in confusion_df.iterrows():
    #     ax.text(float(tup.recall + 0.001), float(tup.precision + 0.001), str(tup.state), color=str(cmap_list[np.argwhere(categories == tup.category)[0,0]]))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('')

def plot_coverage_violin(eventstats_df, ax):
    edf = eventstats_df.loc[eventstats_df.label == eventstats_df.predicted]
    sns.violinplot(x='category', y='coverage', data=edf, ax=ax)
    ax.xaxis.label.set_visible(False)
    ax.axhline(100, ls='--', color='black')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    if edf.coverage.max() > 400:
        ax.set_ylim(0, min(edf.coverage.max(), 400))


def plot_transition_bar(transition_df, yerr, ax):
    transition_df.index = [ind.replace('_', '') for ind in transition_df.index]
    if transition_df.shape[1] <= 2:
        transition_df.plot(kind='bar', yerr=yerr, legend=False, color='grey', ax=ax)
    else:
        transition_df.plot(kind='bar', yerr=yerr, legend=False, ax=ax)

    # dot the actual values bar outline
    nb_transitions, nb_groups = transition_df.shape
    nb_bars_offset = nb_transitions * (nb_groups - 1)
    for i in range(nb_transitions):
        ax.patches[nb_bars_offset + i].set_linestyle('dashed')
        ax.patches[nb_bars_offset + i].set_color('white')
        ax.patches[nb_bars_offset + i].set_edgecolor('black')

    ax.xaxis.set_tick_params(rotation='default')
    ax.set_ylim(bottom=0)

def plot_transition_bubble(tdf, cat):
    pts_scaling = 7500
    fig = plt.figure()

    # grid
    plt.rc('grid', linestyle='-', color='#dddddd')
    plt.grid(True, zorder=0)

    # bubbles
    plt.scatter(x=tdf.loc[:, 'from'], y=tdf.loc[:, 'to'],
                s=tdf.loc[:, cat] * pts_scaling, edgecolors='none', c='grey', zorder=3)
    plt.scatter(x=tdf.loc[:, 'from'], y=tdf.loc[:, 'to'],
                s=tdf.loc[:, 'actual'] * pts_scaling, linestyle='dashed', edgecolors='black',
                c='', zorder=3)

    # Clean up axes
    max_state = tdf.to.max()
    fig.gca().set_ylim(bottom=0.5, top=max_state + 0.5)
    fig.gca().set_xlim(left=0.5, right=max_state + 0.5)
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yticks([n for n in range(1, max_state+1)])
    plt.xticks([n for n in range(1, max_state + 1)])
    fig.gca().set_xlabel('From state')
    fig.gca().set_ylabel('To state')
    fig.gca().set_aspect('equal', adjustable='box')

    # make legend, from: https://blogs.oii.ox.ac.uk/bright/2014/08/12/point-size-legends-in-matplotlib-and-basemap-plots/
    max_marker_size = tdf.loc[:, [cat, 'actual']].max().max()
    labels_raw = np.arange(0.05, 0.5, 0.05)
    labels = [0.05] + [labels_raw[li].round(2) for li in range(1, len(labels_raw)) if
                       labels_raw[li - 1] <= max_marker_size]
    lab_obj_list = [plt.scatter([], [], s=l * pts_scaling, edgecolors='none',
                                color='grey') for l in labels]
    fig.gca().legend(lab_obj_list, labels, ncol=4, frameon=False, scatterpoints=1,
                     loc='upper center', bbox_to_anchor=(0.5, -0.20), )
    plt.tight_layout()
    return fig

def plot_kde(kde_df, eventstats_df, target_states, ax):
    states_colors = sns.color_palette(None, 10)
    # states_colors = ['#ff7f00', '#377eb8'][:len(target_states)]  # orange blue (http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=5)

    for tsi, ts in enumerate(target_states):
        sns.kdeplot(kde_df.loc[kde_df.label == ts, 'E_FRET'], ax=ax, color=states_colors[tsi], legend=False,
                    shade=True, linestyle='', alpha=1)
        sns.kdeplot(eventstats_df.loc[eventstats_df.label == ts, 'E_FRET'], ax=ax, color='black',
                    legend=False, linestyle='--')
    # ax.set_xlabel('$E_{FRET}$')
    ax.set_ylabel('')
    ax.set_xlim(0, 1)
