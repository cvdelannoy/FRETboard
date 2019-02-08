import numpy as np
from cached_property import cached_property
import itertools
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PreText
from bokeh.models.widgets import DataTable, TableColumn, Div
from bokeh.layouts import column, row
from bokeh.embed import file_html
from bokeh.resources import CDN


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
        seq_df = pd.DataFrame({'i_fret': self.hmm_obj.data.i_fret,
                               'out_labels': self.out_labels},
                              index=self.hmm_obj.data.index)
        return seq_df.apply(lambda x: self.condense_sequence(x), axis=1)

    @cached_property
    def event_df(self):
        seq_mat = self.condensed_seq_df.values
        seq_mat = np.array(list(itertools.chain.from_iterable(seq_mat)))
        event_df = pd.DataFrame({'state': seq_mat[:, 0].astype(int), 'i_fret': seq_mat[:, 2],
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
        return pd.DataFrame({'i_fret_before': before, 'i_fret_after': after})

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
        seq_condensed = [[seq.out_labels[0], 0, 0]]  # symbol, duration, I_fret sum
        for s, i in zip(seq.out_labels, seq.i_fret):
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
        hmm_table = self.make_hmm_param_table()
        # page = column(row(ed_scatter, tdp_hex), hmm_table)
        page = column(hmm_table, row(ed_scatter, tdp_hex))
        return file_html(page, CDN, 'FRET report')

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
        return ed_scatter

    def draw_transition_density_plot(self):
        # todo: add states grid
        tdp_hex = figure(plot_width=500, plot_height=500,
                         background_fill_color='#440154', title='Transition density plot')
        tdp_hex.grid.visible = False
        tdp_hex.xaxis.axis_label = 'E FRET before transition'
        tdp_hex.yaxis.axis_label = 'E FRET after transition'
        tdp_hex.hexbin(x=self.transition_df['i_fret_before'], y=self.transition_df['i_fret_after'], size=0.01)
        return tdp_hex

    def make_hmm_param_table(self):

        state_name_list = ['State ' + str(sn + 1) for sn in range(self.hmm_obj.nb_states)]

        # emisisons table
        columns = [TableColumn(field='state', title='state')]
        cds_dict = {'state': state_name_list}
        for fi, feat in enumerate(self.hmm_obj.feature_list):
            columns.append(TableColumn(field=str(feat) + ' mu', title=str(feat) + ' mu'))
            columns.append(TableColumn(field=str(feat) + ' sd', title=str(feat) + ' sd'))
            cds_dict[str(feat) + ' mu'] = self.hmm_obj.trained_hmm.means_[:, fi].round(3)
            cds_dict[str(feat) + ' sd'] = self.hmm_obj.trained_hmm.covars_[:, fi, fi].round(3)

        # cds_dict = {}
        # header = []
        # for fi, feat in enumerate(self.hmm_obj.feature_list):
        #     header.extend(f'<th>{str(feat)} mu</th> <th>{str(feat)} sd</th>')
        #     cds_dict[str(feat) + ' mu'] = self.hmm_obj.trained_hmm.means_[:, fi].round(3)
        #     cds_dict[str(feat) + ' sd'] = self.hmm_obj.trained_hmm.covars_[:, fi, fi].round(3)
        # em_data_table = f"<table>{header}"
        # for k in cds_dict

        em_data_table = DataTable(source=ColumnDataSource(cds_dict), columns=columns)
        em_obj = column(PreText(text='Emission probabilities'), em_data_table)

        # transition matrix
        tm = self.hmm_obj.trained_hmm.transmat_.round(3)
        tm_columns = [TableColumn(field='state', title='')]
        tm_dict = {'state': state_name_list}
        for si, sn in enumerate(state_name_list):
            tm_columns.append(TableColumn(field=sn, title=''))
            tm_dict[sn] = tm[:,si].tolist()
        tm_table = DataTable(source=ColumnDataSource(tm_dict), columns=tm_columns)
        tm_obj = column(PreText(text='Transition table'), tm_table)
        return row(em_obj, tm_obj)
        # todo: accuracy/posterior probability estimates: hist + text
        # todo: gauss curves per model
        # todo: all hmm params: per-state mean/stdev table + transition matrix + P_start

    def make_k_table(self):
        state_name_list = ['State ' + str(sn + 1) for sn in range(self.hmm_obj.nb_states)]

