import os
import numpy as np
from cached_property import cached_property
import itertools
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PreText
from bokeh.models.widgets import DataTable, TableColumn, Div
from bokeh.layouts import column, row
from bokeh.embed import file_html, components
from bokeh.resources import CDN
from jinja2 import Template


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class FretReport(object):
    def __init__(self, gui_obj):
        self.gui_obj = gui_obj
        self.hmm_obj = self.gui_obj.hmm_obj

    @cached_property
    def out_labels(self):
        return self.hmm_obj.data.apply(
            lambda x: x.labels if x.labels.size is not 0 else x.prediction, axis=1)

    @cached_property
    def condensed_seq_df(self):
        seq_df = pd.DataFrame({'E_FRET': self.hmm_obj.data.E_FRET,
                               'out_labels': self.out_labels},
                              index=self.hmm_obj.data.index)
        return seq_df.apply(lambda x: self.condense_sequence(x), axis=1)

    @cached_property
    def event_df(self):
        seq_mat = self.condensed_seq_df.values
        seq_mat = np.array(list(itertools.chain.from_iterable(seq_mat)))
        event_df = pd.DataFrame({'state': seq_mat[:, 0].astype(int), 'E_FRET': seq_mat[:, 2],
                                 'duration': seq_mat[:, 1].astype(int)})
        return event_df

    @cached_property
    def transition_df(self):
        before = []
        after = []
        for r in self.condensed_seq_df:
            r_array = np.array(r)
            before.extend(r_array[:-1, 2])
            after.extend(r_array[1:, 2])
        return pd.DataFrame({'E_FRET_before': before, 'E_FRET_after': after})

    @ cached_property
    def k_off(self):
        for seq in self.condensed_seq_df:
            pass

    @staticmethod
    def condense_sequence(seq):
        """
        Take a pd df of labels and Efret values, turn into list of tuples (label, nb_elements, average )
        :param seq:
        :return:
        """
        seq_condensed = [[seq.out_labels[0], 0, 0]]  # symbol, duration, E_FRET sum
        for s, i in zip(seq.out_labels, seq.E_FRET):
            if s == seq_condensed[-1][0]:
                seq_condensed[-1][1] += 1
                seq_condensed[-1][2] += i
            else:
                seq_condensed[-1][2] = seq_condensed[-1][2] / seq_condensed[-1][1]
                seq_condensed.append([s, 1, i])
            seq_condensed[-1][2] = seq_condensed[-1][2] / seq_condensed[-1][1]
        return seq_condensed

    def construct_html_report(self):
        ed_scatter = self.draw_Efret_duration_plot()
        tdp_hex = self.draw_transition_density_plot()
        tm_table, em_table = self.make_hmm_param_table()
        # page = column(row(ed_scatter, tdp_hex), hmm_table)
        # page = column(hmm_table, row(column(ed_scatter), column(tdp_hex)))
        with open(f'{__location__}/templates/report_template.html', 'r') as fh:
            template = Template(fh.read())
        # # return file_html(page, CDN, template=f'{__location__}/templates/report_template.html')
        esh, esd = components(ed_scatter)
        thh, thd, = components(tdp_hex)
        # tabh, tabd = components(hmm_table)
        return template.render(ed_scatter_script=esh, ed_scatter_div=esd,
                               tdp_hex_script=thh, tdp_hex_div=thd,
                               # hmm_table_script=tabh, hmm_table_div=tabd,
                               tm_table=tm_table, em_table=em_table)
        # return file_html(page, CDN, 'FRET report')

    # plotting functions
    def draw_Efret_duration_plot(self):
        cds = ColumnDataSource(ColumnDataSource.from_df(self.event_df))
        ed_scatter = figure(plot_width=500, plot_height=500, title='Fret efficiency vs event duration')
        ed_scatter.background_fill_color = '#a6a6a6'
        ed_scatter.grid.visible = False
        ed_scatter.xaxis.axis_label = 'duration (# measurements)'
        ed_scatter.yaxis.axis_label = 'FRET intensity'
        ed_scatter.scatter(x='duration', y='E_FRET', color={'field': 'state', 'transform': self.gui_obj.col_mapper},
                           source=cds)
        return ed_scatter

    def draw_transition_density_plot(self):
        # todo: add states grid
        tdp_hex = figure(plot_width=500, plot_height=500,
                         background_fill_color='#440154', title='Transition density plot')
        tdp_hex.grid.visible = False
        tdp_hex.xaxis.axis_label = 'E FRET before transition'
        tdp_hex.yaxis.axis_label = 'E FRET after transition'
        tdp_hex.hexbin(x=self.transition_df['E_FRET_before'], y=self.transition_df['E_FRET_after'], size=0.01)
        return tdp_hex

    def make_hmm_param_table(self):

        # Transition matrix
        nb_states = self.hmm_obj.nb_states
        state_list = [str(nb+1) for nb in range(self.hmm_obj.nb_states)]
        ci_vecs = self.hmm_obj.confidence_intervals
        tm1 = np.char.array(self.hmm_obj.trained_hmm.transmat_.round(3).astype(str))
        tm_newline = np.tile(np.char.array('\n'), (nb_states, nb_states))
        tm2 = np.char.array(ci_vecs[:, :, 0].round(3).astype(str))
        tm_sep = np.tile(np.char.array(' - '), (nb_states, nb_states))
        tm3 = np.char.array(ci_vecs[:, :, 1].round(3).astype(str))
        tm_str = tm1 + tm_newline + tm2 + tm_sep + tm3
        tm = pd.DataFrame(index=state_list, columns=state_list,data=tm_str)
        # tm_vecs = np.split(self.hmm_obj.trained_hmm.transmat_, indices_or_sections=self.hmm_obj.nb_states, axis=0)
        # tm_dict = {str(i + 1): np.round(tmi, 3).squeeze().tolist() for i, tmi, in enumerate(tm_vecs)}
        # tm = pd.DataFrame(tm_dict, index=state_list)
        tm_obj = tm.to_html(border=0)
        # tm_obj = Div(text=tm.to_html(border=0), width=50, height=500)

        # Emissions table
        nb_features = len(self.hmm_obj.feature_list)
        mu_vecs = np.split(self.hmm_obj.trained_hmm.means_.round(3), nb_features, axis=1)
        mu_vecs = [m.squeeze() for m in mu_vecs]
        cv_vecs = np.split(self.hmm_obj.trained_hmm.covars_.round(3), nb_features, axis=1)
        cv_vecs = [cv.squeeze()[:,i] for i, cv in enumerate(cv_vecs)]

        em_cols = [f'mu {fn}' for fn in self.hmm_obj.feature_list] + [f'sd {fn}' for fn in self.hmm_obj.feature_list]
        em_dict = {cn: mc.tolist() for cn, mc in zip(em_cols, np.concatenate((mu_vecs, cv_vecs)))}

        em_df = pd.DataFrame(em_dict, index=state_list)
        em_obj = em_df.to_html(border=0)
        # em_obj = Div(text=em_df.to_html(border=0), width=50, height=500)
        return tm_obj, em_obj
        # return row(column(em_obj), column(tm_obj))
        # todo: accuracy/posterior probability estimates: hist + text
        # todo: gauss curves per model
        # todo: all hmm params: per-state mean/stdev table + transition matrix + P_start

    def make_k_table(self):
        state_name_list = ['State ' + str(sn + 1) for sn in range(self.hmm_obj.nb_states)]

