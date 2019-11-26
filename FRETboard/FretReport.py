import os
import io
import numpy as np

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt


from cached_property import cached_property
import itertools
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from jinja2 import Template
from tabulate import tabulate
from FRETboard.helper_functions import print_timestamp

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class FretReport(object):
    def __init__(self, gui):
        self.gui = gui
        self.classifier = self.gui.classifier
        self.data = self.gui.data

    @cached_property
    def out_labels(self):
        return self.data.data.apply(
            lambda x: x.labels if len(x.labels) is not 0 else x.prediction, axis=1)

    @cached_property
    def condensed_seq_df(self):
        seq_df = pd.DataFrame({'E_FRET': self.data.data.E_FRET,
                               'out_labels': self.out_labels},
                              index=self.data.data.index)
        return seq_df.apply(lambda x: self.condense_sequence(x), axis=1)

    @cached_property
    def event_df(self):
        seq_mat = self.condensed_seq_df.values
        seq_mat = np.array(list(itertools.chain.from_iterable(seq_mat)))
        event_df = pd.DataFrame({'state': seq_mat[:, 0] / seq_mat[:,0].max(), 'E_FRET': seq_mat[:, 2],
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

    # @ cached_property
    # def k_off(self):
    #     for seq in self.condensed_seq_df:
    #         pass

    @staticmethod
    def condense_sequence(seq):
        """
        Take a pd df of labels and Efret values, turn into list of tuples (label, nb_elements, average )
        :param seq:
        :return:
        """
        seq_condensed = [[seq.out_labels[0], 0, []]]  # symbol, duration, E_FRET sum
        for s, i in zip(seq.out_labels, seq.E_FRET):
            if s == seq_condensed[-1][0]:
                seq_condensed[-1][1] += 1
                seq_condensed[-1][2].append(i)
            else:
                seq_condensed[-1][2] = np.nanmedian(seq_condensed[-1][2])
                seq_condensed.append([s, 1, [i]])
        seq_condensed[-1][2] = np.nanmedian(seq_condensed[-1][2])
        return seq_condensed

    def construct_html_report(self):
        ed_scatter = self.draw_Efret_duration_plot()
        tdp = self.draw_transition_density_plot()
        efret_hist = self.draw_efret_histograms()
        tm_table, em_table = self.get_param_tables()
        # kinetic_table = self.get_stats_tables()
        with open(f'{__location__}/templates/report_template.html', 'r') as fh:
            template = Template(fh.read())
        esh, esd = components(ed_scatter)
        # thh, thd = components(tdp_hex)
        return template.render(ed_scatter_script=esh, ed_scatter_div=esd,
                               # tdp_hex_script=thh,
                               transition_density_plot=tdp,
                               efret_hist=efret_hist,
                               tm_table_div=tm_table,
                               em_table_div=em_table,
                               date=print_timestamp(),
                               model_params=self.model_params())

    # --- plotting functions ---

    def model_params(self):
        nb_labeled = self.data.data.is_labeled.sum()
        pct_labeled = nb_labeled / len(self.data.data) * 100
        out_str = f"""
        Algorithm: {self.gui.algo_select.value} <br/>
        Number of states: {self.classifier.nb_states} <br/>
        Supervision influence: {self.gui.supervision_slider.value} <br/>
        Buffer size: {self.gui.buffer_slider.value} <br/>
        Classified traces: {nb_labeled} ({round(pct_labeled, 1)}%) <br/>
        Training accuracy: {round(self.data.accuracy[1], 1)}%<br/>
        """
        return out_str

    def draw_Efret_duration_plot(self):
        cds = ColumnDataSource(ColumnDataSource.from_df(self.event_df))
        ed_scatter = figure(plot_width=500, plot_height=500, title='')
        ed_scatter.background_fill_color = '#a6a6a6'
        ed_scatter.grid.visible = False
        ed_scatter.xaxis.axis_label = 'duration (# measurements)'
        ed_scatter.yaxis.axis_label = 'E FRET'
        ed_scatter.scatter(x='duration', y='E_FRET', color={'field': 'state', 'transform': self.gui.col_mapper},
                           source=cds)
        return ed_scatter

    def draw_transition_density_plot(self):
        fig = plt.figure()
        ax = sns.kdeplot(self.transition_df.E_FRET_before, self.transition_df.E_FRET_after, shade=True, cmap="coolwarm", ax=fig.gca())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # plt.ylim(0, 1)

        mu_list = self.classifier.get_states_mu('E_FRET')
        sd_list = self.classifier.get_states_sd('E_FRET')
        for mi, mm in enumerate(mu_list[:-1]):
            ratio = sd_list[mi] / (sd_list[mi] + sd_list[mi + 1])
            dmu = mu_list[mi + 1] - mm
            lin = mm + dmu * ratio
            plt.axvline(lin, color='black')
            plt.axhline(lin, color='black')

        ax.set_aspect('equal')
        ax.set_xlabel('$E_{FRET}$ before')
        ax.set_ylabel('$E_{FRET}$ after')

        f = io.StringIO()
        plt.savefig(f, format='svg')
        return f.getvalue()

    # def draw_transition_density_plot(self):
    #     tdp_hex = figure(plot_width=500, plot_height=500,
    #                      background_fill_color='#440154',
    #                      x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    #     tdp_hex.grid.visible = False
    #     tdp_hex.xaxis.axis_label = 'E FRET before transition'
    #     tdp_hex.yaxis.axis_label = 'E FRET after transition'
    #     tdp_hex.hexbin(x=self.transition_df['E_FRET_before'], y=self.transition_df['E_FRET_after'], size=0.01)
    #     return tdp_hex

    def get_stats_tables(self):
        # time spent in each state
        time_dict = {}
        tst = []
        mean_times = self.data.data_clean.time.apply( lambda x: (x[-1] - x[0]) / len(x))
        for state in range(self.classifier.nb_states):
            tst.append(mean_times * self.out_labels.apply(lambda x: np.sum(x == state)))
            time_dict[state] = np.sum(mean_times * self.out_labels.apply(lambda x: np.sum(x == state)))

    def draw_efret_histograms(self):
        fig = plt.figure()
        ax = fig.gca()
        efret_vec = np.concatenate(self.data.data_clean.E_FRET)
        label_vec = np.concatenate(self.out_labels)
        unique_labels = np.unique(label_vec)
        colors = sns.color_palette('Blues', len(unique_labels))
        for li, lab in enumerate(unique_labels):
            cur_vec = efret_vec[label_vec == lab]
            cur_vec = cur_vec[~np.isnan(cur_vec)]
            ax = sns.distplot(cur_vec, kde=False, bins=100, color=colors[li], ax=ax)

        ax.set_xlabel('$E_{FRET}$')
        ax.set_ylabel('count')

        f = io.StringIO()
        plt.savefig(f, format='svg')
        return f.getvalue()


    def get_param_tables(self):
        # Transitions table
        nb_states = self.classifier.nb_states
        state_list = [str(nb+1) for nb in range(self.classifier.nb_states)]
        state_list_bold = [f'<b>{s}</b>' for s in state_list]
        ci_vecs = self.classifier.confidence_intervals
        tm_trained = self.classifier.get_tm(self.classifier.trained).to_numpy()

        tm1 = np.char.array(tm_trained.round(3).astype(str))
        tm_newline = np.tile(np.char.array('<br/>'), (nb_states, nb_states))
        tm_sm = np.tile(np.char.array('<small>'), (nb_states, nb_states))
        tm_sme = np.tile(np.char.array('</small>'), (nb_states, nb_states))
        tm2 = np.char.array(ci_vecs[:, :, 0].round(3).astype(str))
        tm_sep = np.tile(np.char.array(' - '), (nb_states, nb_states))
        tm3 = np.char.array(ci_vecs[:, :, 1].round(3).astype(str))
        tm_str = tm1 + tm_newline + tm_sm + tm2 + tm_sep + tm3 + tm_sme
        tm_obj = tabulate(tm_str, tablefmt='html',
                          headers=state_list, showindex=state_list_bold,
                          numalign='center', stralign='center')

        # Emissions table
        nb_features = len(self.classifier.feature_list)
        mu_vecs = [np.array(self.classifier.get_states_mu(feature)).round(3) for feature in self.classifier.feature_list]
        cv_vecs = [np.array(self.classifier.get_states_sd(feature)).round(3) for feature in self.classifier.feature_list]

        em_cols = [f'mu {fn}' for fn in self.classifier.feature_list] + \
                  [f'sd {fn}' for fn in self.classifier.feature_list]
        em_dict = {cn: mc.tolist() for cn, mc in zip(em_cols, np.concatenate((mu_vecs, cv_vecs)))}
        em_obj = tabulate(em_dict, tablefmt='html',
                          headers=em_cols, showindex=state_list_bold,
                          numalign='center', stralign='center')
        return tm_obj, em_obj
        # todo: accuracy/posterior probability estimates: hist + text
        # todo: gauss curves per model

    # def make_k_table(self):
    #     state_name_list = ['State ' + str(sn + 1) for sn in range(self.classifier.nb_states)]
