import numpy as np
from cached_property import cached_property
import itertools
import os
import pathlib
import pickle
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource

class FretReport(object):
    def __init__(self, gui_obj):
        self.gui_obj = gui_obj
        self.save_path = self.gui_obj.save_path.value
        self.hmm_obj = self.gui_obj.hmm_obj

        self.draw_Efret_duration_plot()
        self.transition_density_plot()
        self.save_dats()
        self.save_hmm()

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, sp):
        pathlib.Path(sp).mkdir(parents=True, exist_ok=True)
        if sp[-1] != '/':
            sp += '/'
        pathlib.Path(sp+'dat_files/').mkdir(exist_ok=True)
        pathlib.Path(sp + 'plots/').mkdir(exist_ok=True)
        self._save_path = sp

    @cached_property
    def out_labels(self):
        return self.hmm_obj.data.apply(
            lambda x: x.labels if x.labels.size is not 0 else x.prediction, axis=1)

    @cached_property
    def condensed_seq_df(self):
        seq_df = pd.DataFrame({'i_fret': self.hmm_obj.data.i_fret,
                               'out_labels': self.out_labels},
                              index=self.hmm_obj.data.index)
        return seq_df.apply(lambda x: self.condense_sequence(x), axis=1)


    @cached_property
    def event_df(self):
        seq_mat = self.condensed_seq_df.values
        seq_mat = np.array(list(itertools.chain.from_iterable(seq_mat)))
        event_df = pd.DataFrame({'state': seq_mat[:, 0].astype(int), 'i_fret': seq_mat[:, 2] / seq_mat[:, 1],
                                 'duration': seq_mat[:, 1].astype(int)})
        return event_df

    @cached_property
    def transition_df(self):
        cdf = self.condensed_seq_df
        before = []
        after = []
        for r in self.condensed_seq_df:
            r_array = np.array(r)
            i_fret_means = r_array[:, 2] / r_array[:, 1]
            before.extend(i_fret_means[:-1])
            after.extend(i_fret_means[1:])
        return pd.DataFrame({'i_fret_before': before, 'i_fret_after': after})

    @staticmethod
    def condense_sequence(seq):
        seq_condensed = [[seq.out_labels[0], 0, 0]]  # symbol, duration, I_fret sum
        for s, i in zip(seq.out_labels, seq.i_fret):
            if s == seq_condensed[-1][0]:
                seq_condensed[-1][1] += 1
                seq_condensed[-1][2] += i
            else:
                seq_condensed.append([s, 1, i])
        return seq_condensed

    def save_dats(self):
        dat_path = self.save_path + 'dat_files/'
        for idx in self.hmm_obj.data.index:
            fc = np.loadtxt(idx)
            fc = np.hstack((fc, np.expand_dims(self.out_labels.loc[idx], axis=1)))
            dat_fn = dat_path + os.path.basename(idx)
            np.savetxt(dat_fn, fc, fmt='%s')

    def save_hmm(self):
        hmm_fn = self.save_path + 'hmm.pkl'
        if os.path.isfile(hmm_fn):
            os.remove(hmm_fn)
        with open(hmm_fn, 'wb') as fh:
            pickle.dump(self.hmm_obj, fh, pickle.HIGHEST_PROTOCOL)

    # plotting functions
    def draw_Efret_duration_plot(self):
        cds = ColumnDataSource(ColumnDataSource.from_df(self.event_df))
        ed_scatter = figure(plot_width=500, plot_height=500, title='Fret efficiency vs event duration')
        ed_scatter.background_fill_color = '#a6a6a6'
        ed_scatter.grid.visible = False
        ed_scatter.xaxis.axis_label = 'duration (# measurements)'
        ed_scatter.yaxis.axis_label = 'FRET intensity'
        ed_scatter.scatter(x='duration', y='i_fret', color={'field': 'state', 'transform': self.gui_obj.col_mapper},
                           source=cds)
        output_file(self.save_path+'plots/efret_duration_scatter.html')
        save(ed_scatter)

    def transition_density_plot(self):
        # todo: add states grid
        cds = ColumnDataSource(ColumnDataSource.from_df(self.transition_df))
        tdp_hex = figure(plot_width=500, plot_height=500, #x_range=[0, 1], y_range=[0, 1],
                         background_fill_color='#440154', title='Transition density plot')
        tdp_hex.grid.visible = False
        tdp_hex.xaxis.axis_label = 'I FRET before transition'
        tdp_hex.yaxis.axis_label = 'I FRET after transition'
        # tdp_hex.hexbin(x='i_fret_before', y='i_fret_after', size=0.01, source=cds) # note: hexbin doesn't recongize source?!
        tdp_hex.hexbin(x=self.transition_df['i_fret_before'], y=self.transition_df['i_fret_after'], size=0.01)
        output_file(self.save_path + 'plots/transition_density_plot.html')
        save(tdp_hex)

    # def hmm_params(self):
    #     self.hmm_obj

    # todo: accuracy/posterior probability estimates: hist + text
    # todo: gauss curves per model
    # todo: all hmm params: per-state mean/stdev table + transition matrix + P_start
