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
from jinja2 import Template
from tabulate import tabulate
from FRETboard.helper_functions import print_timestamp, multi_joint_plot

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class FretReport(object):
    def __init__(self, gui):
        self.gui = gui
        self.classifier = self.gui.classifier
        self.data = self.gui.data

    @cached_property
    def tr_dict(self):
        return self.data.get_trace_dict()

    @cached_property
    def out_labels(self):
        """
        Return manual label if available, else predicted label
        """
        return [self.tr_dict[tr].predicted.to_numpy().astype(int) if tr not in self.data.label_dict
                else self.data.label_dict[tr].astype(int) for tr in self.tr_dict]

    @cached_property
    def condensed_seq_df(self):
        seq_df = pd.DataFrame({'E_FRET': [self.tr_dict[tr].E_FRET.to_numpy() for tr in self.tr_dict],
                               'out_labels': self.out_labels},
                              index=list(self.tr_dict))
        return seq_df.apply(lambda x: self.condense_sequence(x), axis=1)

    @cached_property
    def event_df(self):
        seq_mat = self.condensed_seq_df.values
        seq_mat = np.array(list(itertools.chain.from_iterable(seq_mat)))
        event_df = pd.DataFrame({'state': [int(i) for i in seq_mat[:, 0]],
                                 'state_pct': seq_mat[:, 0] / seq_mat[:, 0].max(),
                                 'E_FRET': seq_mat[:, 2],
                                 'duration': seq_mat[:, 1].astype(int)})
        return event_df

    @cached_property
    def outlier_free_event_df(self):
        df_list = []
        for st, df in self.event_df.groupby('state'):
            if st not in self.gui.saveme_checkboxes.active: continue
            ef_sd3 = df.E_FRET.std() * 3
            ef_lb, ef_hb = df.E_FRET.mean() - ef_sd3, df.E_FRET.mean() + ef_sd3
            ef_bool = np.logical_and(df.E_FRET > ef_lb, df.E_FRET < ef_hb)
            dur_sd3 = df.duration.std() * 3
            dur_lb, dur_hb = df.duration.mean() - dur_sd3, df.duration.mean() + dur_sd3
            dur_bool = np.logical_and(df.duration > dur_lb, df.duration < dur_hb)
            keep_bool = np.logical_and(ef_bool, dur_bool)
            df_list.append(df.loc[keep_bool, :])
        return pd.concat(df_list)

    @cached_property
    def transition_df(self):
        active_states = self.gui.saveme_checkboxes.active
        before = []
        after = []
        state_before = []
        state_after = []
        for r in self.condensed_seq_df:
            r_array = np.array(r)
            r_array = r_array[np.logical_and(np.in1d(r_array[:, 0], active_states),
                                             np.in1d(r_array[:, 1], active_states)), :]
            before.extend(r_array[:-1, 2])
            after.extend(r_array[1:, 2])
            state_before.extend(r_array[:-1, 0])
            state_after.extend(r_array[1:, 0])
        return pd.DataFrame({'E_FRET_before': before, 'E_FRET_after': after,
                             'state_before': state_before, 'state_after': state_after})

    @cached_property
    def out_label_vec(self):
        'Labels returned as one long numpy vector'
        return np.concatenate([np.stack(list(tup), axis=-1) for tup in self.out_labels], 0)

    @cached_property
    def out_label_vec_not_last(self):
        'Labels returned as one long numpy vector, omitting the last label in each sequence'
        return np.concatenate([np.stack(list(tup)[:-1], axis=-1) for tup in self.out_labels], 0)

    @cached_property
    def efret_vec(self):
        'efret values returned as one long numpy vector'
        return np.concatenate([self.tr_dict[tr].E_FRET.to_numpy() for tr in self.tr_dict], 0)
        # return np.concatenate([np.stack(list(tup), axis=-1) for tup in self.data.data_clean.E_FRET.to_numpy()], 0)

    @cached_property
    def data_states_mu(self):
        return [np.mean(self.efret_vec[self.out_label_vec==yn]) for yn in np.arange(self.classifier.nb_states)]

    @cached_property
    def data_states_sd(self):
        return [np.std(self.efret_vec[self.out_label_vec == yn]) for yn in np.arange(self.classifier.nb_states)]

    @cached_property
    def data_efret_stats(self):
        active_states = list(self.gui.saveme_checkboxes.active)
        state_list = [str(nb + 1) for nb in active_states]
        state_list_bold = [f'<b>{s}</b>' for s in state_list]
        table = tabulate({'mu': [mu for mi, mu in enumerate(self.data_states_mu) if mi in active_states],
                          'sd': [sd for si, sd in enumerate(self.data_states_sd) if si in active_states]},
                         tablefmt='html', headers=['mean E_FRET', 'sd E_FRET'], showindex=state_list_bold,
                         numalign='center', stralign='center')
        return table

    @cached_property
    def frame_rate(self):
        return 1 / np.concatenate([self.tr_dict[tr].time.iloc[1:].to_numpy() -
                                   self.tr_dict[tr].time.iloc[:-1].to_numpy() for tr in self.tr_dict]).mean()

    def get_data_tm(self):
        tm_vec, ci_vecs = self.classifier.get_data_tm(self.tr_dict, self.out_labels,
                                                      self.gui.bootstrap_size_spinner.value)
        asi = self.gui.saveme_checkboxes.active
        tm_vec = tm_vec.take(asi, axis=0).take(asi, axis=1)
        ci_vecs = ci_vecs.take(asi, axis=0).take(asi, axis=1)

        # make df for csv file
        state_list = [str(nb + 1) for nb in asi]
        transition_list = ['_'.join(it) for it in itertools.permutations(state_list, 2)]
        msk = np.invert(np.eye(len(asi), dtype=bool))
        csv_df = pd.DataFrame({'rate': tm_vec[msk],
                               'low_bound': ci_vecs[:, :, 0][msk],
                               'high_bound': ci_vecs[:, :, 1][msk]}, index=transition_list)
        return self.transition_np_to_html(tm_vec, ci_vecs, state_list), csv_df.to_csv().replace('\n', '\\n')

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
                if np.isnan(seq_condensed[-1][2]): seq_condensed.pop()
                seq_condensed.append([s, 1, [i]])
        seq_condensed[-1][2] = np.nanmedian(seq_condensed[-1][2])
        if np.isnan(seq_condensed[-1][2]): seq_condensed.pop()
        return seq_condensed

    def construct_html_report(self):
        ed_scatter = self.draw_Efret_duration_plot()
        tdp = self.draw_transition_density_plot()
        efret_hist = self.draw_efret_histograms()
        dwelltime_hist, dwelltime_df = self.draw_dwelltime_histogram()
        # tm_table, em_table, tm_str = self.get_param_tables()
        data_tm, data_tm_csv = self.get_data_tm()

        with open(f'{__location__}/templates/report_template.html', 'r') as fh:
            template = Template(fh.read())
        # thh, thd = components(tdp_hex)
        return template.render(ed_scatter=ed_scatter,
                               transition_density_plot=tdp,
                               efret_hist=efret_hist,
                               dwelltime_hist=dwelltime_hist,
                               dwelltime_csv=dwelltime_df,
                               data_efret_stats=self.data_efret_stats,
                               data_tm_div=data_tm,
                               # tm_table_div=tm_table,
                               # em_table_div=em_table,
                               date=print_timestamp(),
                               # transition_csv=tm_str,
                               data_tm_csv=data_tm_csv,
                               model_params=self.model_params())

    # --- plotting functions ---

    def model_params(self):
        nb_labeled = self.data.manual_table.is_labeled.sum()
        pct_labeled = nb_labeled / len(self.data.index_table) * 100
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
        multi_joint_plot('duration', 'E_FRET', 'state', df=self.outlier_free_event_df)
        f = io.StringIO()
        plt.savefig(f, format='svg')
        plt.clf()
        return f.getvalue()

        # cds = ColumnDataSource(ColumnDataSource.from_df(self.event_df))
        # ed_scatter = figure(plot_width=500, plot_height=500, title='')
        # ed_scatter.background_fill_color = '#a6a6a6'
        # ed_scatter.grid.visible = False
        # ed_scatter.xaxis.axis_label = 'duration (# measurements)'
        # ed_scatter.yaxis.axis_label = 'E FRET'
        # ed_scatter.scatter(x='duration', y='E_FRET', color={'field': 'state_pct', 'transform': self.gui.col_mapper},
        #                    source=cds)
        # return ed_scatter

    def draw_transition_density_plot(self):
        fig = plt.figure()
        if len(self.transition_df) < 3: return "Not enough events"
        ax = sns.kdeplot(self.transition_df.E_FRET_before, self.transition_df.E_FRET_after, shade=True, cmap="coolwarm", ax=fig.gca())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # plt.ylim(0, 1)

        if 'E_FRET' in self.classifier.feature_list:
            mu_list = self.classifier.get_mus('E_FRET')
            # sd_list = self.classifier.get_states_sd('E_FRET')
            for m in mu_list:
                if np.isnan(m): continue
                plt.axvline(m, color='black', ls='--')
                plt.axhline(m, color='black', ls='--')
            # for mi, mm in enumerate(mu_list[:-1]):
            #     ratio = sd_list[mi] / (sd_list[mi] + sd_list[mi + 1])
            #     dmu = mu_list[mi + 1] - mm
            #     lin = mm + dmu * ratio
            #     plt.axvline(lin, color='black')
            #     plt.axhline(lin, color='black')

        ax.set_aspect('equal')
        ax.set_xlabel('$E_{FRET}$ before')
        ax.set_ylabel('$E_{FRET}$ after')

        f = io.StringIO()
        plt.savefig(f, format='svg')
        plt.clf()
        return f.getvalue()

    def draw_efret_histograms(self):
        fig = plt.figure()
        ax = fig.gca()
        unique_labels = np.unique(self.out_label_vec)
        colors = sns.color_palette('Blues', len(unique_labels))
        for li, lab in enumerate(unique_labels):
            if lab not in self.gui.saveme_checkboxes.active: continue
            cur_vec = self.efret_vec[self.out_label_vec == lab]
            cur_vec = cur_vec[~np.isnan(cur_vec)]
            ax = sns.distplot(cur_vec, kde=False, bins=100, color=colors[li], ax=ax)

        ax.set_xlabel('$E_{FRET}$')
        ax.set_ylabel('count')

        f = io.StringIO()
        plt.savefig(f, format='svg')
        plt.clf()
        return f.getvalue()

    def draw_dwelltime_histogram(self):
        #construct plot
        fig = plt.figure()
        ax = fig.gca()
        unique_labels = np.unique(self.out_label_vec)
        colors = sns.color_palette('Blues', len(unique_labels))
        for li, lab in enumerate(unique_labels):
            if lab not in self.gui.saveme_checkboxes.active: continue
            ax = sns.distplot(self.event_df.query(f'state == {lab}').duration / self.frame_rate,
                              kde=True, bins=100, color=colors[li], ax=ax)

        ax.set_xlabel('dwell time (s)')
        ax.set_ylabel('count')

        f = io.StringIO()
        plt.savefig(f, format='svg')
        plt.clf()

        # construct csv
        edf = self.event_df.drop(['state_pct'], axis=1).sort_values(['state'])
        edf.state = edf.state + 1
        return f.getvalue(), edf.to_csv(index=False).replace('\n', '\\n')

    # def get_param_tables(self):
    #
    #     # Transitions table
    #     ci_vecs = self.classifier.get_confidence_intervals(data_dict)
    #     tm_trained = self.classifier.get_tm(self.classifier.trained).to_numpy()
    #     tm_obj = self.transition_np_to_html(tm_trained, ci_vecs)
    #
    #     nb_states = self.classifier.nb_states
    #     state_list = [str(nb + 1) for nb in range(self.classifier.nb_states)]
    #     state_list_bold = [f'<b>{s}</b>' for s in state_list]
    #     transition_list = [''.join(it) for it in itertools.permutations(state_list, 2)]
    #     msk = np.invert(np.eye(nb_states, dtype=bool))
    #     csv_df = pd.DataFrame({'rate': tm_trained[msk],
    #                           'low_bound': ci_vecs[:,:,0][msk],
    #                           'high_bound': ci_vecs[:,:,1][msk]}, index=transition_list)
    #     csv_str = csv_df.to_csv().replace('\n', '\\n')
    #
    #     # Emissions table
    #     mu_vecs = [np.array(self.classifier.get_states_mu(feature)).round(3) for feature in self.classifier.feature_list]
    #     cv_vecs = [np.array(self.classifier.get_states_sd(feature)).round(3) for feature in self.classifier.feature_list]
    #     em_cols = [f'mu {fn}' for fn in self.classifier.feature_list] + \
    #               [f'sd {fn}' for fn in self.classifier.feature_list]
    #     em_dict = {cn: mc.tolist() for cn, mc in zip(em_cols, np.concatenate((mu_vecs, cv_vecs)))}
    #     em_obj = tabulate(em_dict, tablefmt='html',
    #                       headers=em_cols, showindex=state_list_bold,
    #                       numalign='center', stralign='center')
    #     return tm_obj, em_obj, csv_str
    #     # todo: accuracy/posterior probability estimates: hist + text
    #     # todo: gauss curves per model

    def transition_np_to_html(self, tm_trained, ci_vecs, state_list):
        # nb_states = self.classifier.nb_states
        nb_states = len(state_list)
        # state_list = [str(nb + 1) for nb in range(self.classifier.nb_states)]
        state_list_bold = [f'<b>{s}</b>' for s in state_list]
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
        return tm_obj
