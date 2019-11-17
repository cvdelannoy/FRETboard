import os
import tempfile
import shutil
import base64
import numpy as np
import pandas as pd
import yaml
import importlib
# from tornado import gen
# from threading import Thread

from cached_property import cached_property
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import row, column, widgetbox
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LinearColorMapper#, CustomJS
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Slider, Select, Button, PreText, RadioGroup, Div, CheckboxButtonGroup, CheckboxGroup, Spinner
from bokeh.models.widgets.panels import Panel, Tabs
from tornado.ioloop import IOLoop

from FRETboard.io_functions import parse_trace_file
from FRETboard.helper_functions import print_timestamp
from FRETboard.FretReport import FretReport
from FRETboard.MainTable import MainTable

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
line_opts = dict(line_width=1)
rect_opts = dict(width=1.01, alpha=1, line_alpha=0)
with open(f'{__location__}/algorithms.yml', 'r') as fh: algo_dict = yaml.safe_load(fh)
white_blue_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
# pastel_colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
# col_mapper = LinearColorMapper(palette=white_blue_colors, low=0, high=1)
# diverging_colors = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
colors = white_blue_colors

with open(f'{__location__}/js_widgets/upload.js', 'r') as fh: upload_js = fh.read()
with open(f'{__location__}/js_widgets/upload_model.js', 'r') as fh: upload_model_js = fh.read()
with open(f'{__location__}/js_widgets/download_datzip.js', 'r') as fh: download_datzip_js = fh.read()
with open(f'{__location__}/js_widgets/download_report.js', 'r') as fh: download_report_js = fh.read()
with open(f'{__location__}/js_widgets/download_model.js', 'r') as fh: download_csv_js = fh.read()


class Gui(object):
    def __init__(self, nb_states=3, data=[]):
        self.version = '0.0.3'
        self.cur_example_idx = None
        self.nb_threads = 8
        self.feature_list = ['E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient', 'i_don', 'i_acc']

        self.data = MainTable(data)

        # widgets
        self.algo_select = Select(title='Algorithm:', value=list(algo_dict)[0], options=list(algo_dict))
        self.example_select = Select(title='Current example', value='None', options=['None'])
        self.num_states_slider = Slider(title='Number of states', value=nb_states, start=2, end=10, step=1)
        self.sel_state_slider = Slider(title='Change selection to state', value=1, start=1,
                                       end=self.num_states_slider.value, step=1)
        self.bg_button = Button(label='Subtract background')
        self.bg_test_button = Button(label='Test')
        self.bg_eps = Spinner(value=9, step=1)
        self.supervision_slider = Slider(title='Influence supervision', value=1.0, start=0.0, end=1.0, step=0.01)
        self.buffer_slider = Slider(title='Buffer', value=3, start=0, end=20, step=1)
        self.notification = PreText(text='', width=1000, height=15)
        self.acc_text = PreText(text='N/A')
        self.posterior_text = PreText(text='N/A')
        self.mc_text = PreText(text='0%')
        self.report_holder = PreText(text='', css_classes=['hidden'])  # hidden holder to generate js callbacks
        self.datzip_holder = PreText(text='', css_classes=['hidden'])  # hidden holder to generate js callbacks

        self.features_checkboxes = CheckboxGroup(labels=[''] * len(self.feature_list), active=[0, 1, 2, 3, 4])
        self.state_radio = RadioGroup(labels=[''] * len(self.feature_list), active=0)

        # Classifier object
        self.classifier_class = importlib.import_module(
            'FRETboard.algorithms.' + algo_dict[self.algo_select.value]).Classifier
        self.classifier = self.classifier_class(nb_states=nb_states, data=self.data, gui=self,
                                                features=[feat for fi, feat in enumerate(self.feature_list)
                                                          if fi in self.features_checkboxes.active])

        # ColumnDataSources
        self.source = ColumnDataSource(data=dict(i_don=[], i_acc=[], E_FRET=[],
                                                 correlation_coefficient=[], E_FRET_sd=[], i_sum=[], time=[],
                                                 rect_height=[], rect_mid=[],
                                                 i_sum_height=[], i_sum_mid=[],
                                                 labels=[], labels_pct=[]))
        ahb = np.arange(start=0, stop=100, step=5)
        self.new_source = ColumnDataSource({'file_contents': [], 'file_name': []})
        self.accuracy_source = ColumnDataSource(data=dict(lb=ahb[:-1], rb=ahb[1:], accuracy_counts=np.repeat(0, 19)))
        self.logprob_source = ColumnDataSource(data=dict(lb=ahb[:-1], rb=ahb[1:],
                                                         accuracy_counts=np.repeat(0, 19), logprob_counts=np.repeat(0, 19)))
        self.state_source = ColumnDataSource(
            data=dict(xs=[np.arange(0, 1, 0.01)] * self.num_states_slider.value,
                      ys=[np.zeros(100, dtype=float)] * self.num_states_slider.value,
                      color=self.curve_colors))
        self.loaded_model_source = ColumnDataSource(data=dict(file_contents=[]))
        self.classifier_source = ColumnDataSource(data=dict(params=[]))
        self.html_source = ColumnDataSource(data=dict(html_text=[]))
        self.datzip_source = ColumnDataSource(data=dict(datzip=[]))
        self.scroll_state_source = ColumnDataSource(data=dict(new_state=[False]))
        self.model_loaded = False
        self.new_tot = ColumnDataSource(data=dict(value=[0]))
        self.new_cur = 0

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        self._classifier = classifier
        # if classifier.data.shape[0] == 0:
        #     return
        # self.example_select.options = self.data.data.index.tolist()
        # if self.cur_example_idx is None:
        #     self.update_example(None, None, self.example_select.options[0])

    @property
    def nb_examples(self):
        return self.data.data.shape[0]

    @property
    def curve_colors(self):
        return [colors[n] for n in (np.linspace(0, len(colors)-1, self.num_states_slider.value)).astype(int)]

    @property
    def col_mapper(self):
        return LinearColorMapper(palette=colors, low=0, high=0.99)

    @cached_property
    def E_FRET(self):
        return self.data.data.loc[self.cur_example_idx].E_FRET

    @cached_property
    def E_FRET_sd(self):
        return self.data.data.loc[self.cur_example_idx].E_FRET_sd

    @cached_property
    def correlation_coefficient(self):
        return self.data.data.loc[self.cur_example_idx].correlation_coefficient

    @cached_property
    def i_don(self):
        return self.data.data.loc[self.cur_example_idx].i_don

    @cached_property
    def i_acc(self):
        return self.data.data.loc[self.cur_example_idx].i_acc

    @cached_property
    def time(self):
        return self.data.data.loc[self.cur_example_idx].time

    @cached_property
    def i_sum(self):
        return self.data.data.loc[self.cur_example_idx].i_sum

    def get_buffer_state_matrix(self):
        ns = self.num_states_slider.value
        bsm = np.zeros((ns, ns), dtype=int)
        ns_buffered = ns + ns * (ns - 1) // 2
        state_array = np.arange(ns, ns_buffered)
        bsm[np.triu_indices(3, 1)] = state_array
        bsm[np.tril_indices(3, -1)] = state_array
        return bsm

    def invalidate_cached_properties(self):
        for k in ['E_FRET', 'correlation_coefficient', 'i_sum', 'tp', 'ts', 'ts_don', 'ts_acc', 'i_don', 'i_acc', 'time',
                  'accuracy_hist', 'E_FRET_sd']:
            if k in self.__dict__:
                del self.__dict__[k]

    def predict_safe(self):
        """
        If more than nb_threads samples exist, predict nb_threads randomly chosen ones to cut down on running time.
        """
        if len(self.data.data) > self.nb_threads:  # predict [nb_threads] traces as indicator of logprob
            idx = self.data.data.index[self.data.data.is_labeled].to_numpy()
            if len(idx) < self.nb_threads:
                # todo: suddenly is_labeled is not bool anymore??
                idx = np.union1d(idx, np.random.choice(self.data.data.index[np.invert(self.data.data.is_labeled.astype(bool))],
                                                       self.nb_threads - len(idx), replace=True))
        else:
            idx = self.data.data.index
        print(f'{print_timestamp()}starting predictions')
        self.predict_all(idx=idx)

    def predict_all(self, idx=None):
        """
        Rerun prediction for all of idx, remove other predictions.
        """
        if idx is None: idx = self.data.data.index
        pred_list, logprob_list = self.classifier.predict(idx)
        pred_series = pd.Series([[]]*len(self.data.data), index=self.data.data.index, dtype=object)
        pred_series.loc[idx] = pred_list
        self.data.data.logprob = np.nan
        self.data.data.prediction = pred_series
        self.data.data.loc[idx, 'logprob'] = logprob_list

    def train_and_update(self):
        self.classifier.train(supervision_influence=self.supervision_slider.value)
        self.predict_safe()
        self.classifier_source.data = dict(params=[self.classifier.get_params()])

    def load_params(self, attr, old, new):
        raw_contents = self.loaded_model_source.data['file_contents'][0]
        # remove the prefix that JS adds
        _, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents).decode('utf-8')
        # file_contents = base64.b64decode(b64_contents).decode('utf-8').split('\n')[1:-1]
        self.classifier.load_params(file_contents)
        self.num_states_slider.value = self.classifier.nb_states
        self.model_loaded = True
        self.notification.text = f'{print_timestamp()}Model loaded'
        if self.data.data.shape[0] != 0:
            self.data.data.is_labeled = False
            self.predict_safe()
            self._redraw_all()

    def update_data(self, attr, old, new):
        raw_contents = self.new_source.data['file_contents'][0]
        _, b64_contents = raw_contents.split(",", 1)  # remove the prefix that JS adds
        file_contents = base64.b64decode(b64_contents)
        fn = self.new_source.data['file_name'][0]
        nb_colors = 2
        # Process .trace files
        if '.traces' in fn:
            df_list = parse_trace_file(file_contents, fn, self.nb_threads)
            # self.doc.add_next_tick_callback(self.update_notification(f'{print_timestamp()}Adding to list'))
            print(f'{print_timestamp()}Adding to list')
            self.data.add_df_list(df_list)
            print(f'{print_timestamp()}Done')

        # Process .dat files
        elif '.dat' in fn:
            # self.doc.add_next_tick_callback(self.update_notification(f'{print_timestamp()}Adding to list'))
            file_contents = base64.b64decode(b64_contents).decode('utf-8')
            file_contents = np.column_stack([np.fromstring(n, sep=' ') for n in file_contents.split('\n') if len(n)])
            if len(file_contents):
                self.data.add_tuple(file_contents, self.new_source.data['file_name'][0])
        self.new_cur += 1

        # Reload/retrain model
        if self.new_tot.data['value'][0] == self.new_cur:
            self.new_cur = 0
            if len(self.data.data):
                self.example_select.options = self.data.data.index.tolist()
                self.showme_checkboxes.active = list(range(self.classifier.nb_states))
            if self.model_loaded:
                self.notification.text = f'{print_timestamp()} Classifying loaded traces using current model...'
                self.predict_safe()
            else:
                self.notification.text = f'{print_timestamp()} Training initial model, on loaded traces...'
                self.train_and_update()
                self.model_loaded = True
            if self.cur_example_idx is None:
                self.example_select.value = self.example_select.options[0]
                # self.update_example(None, None, self.example_select.options[0])
            else:
                self._redraw_all()
            self.notification.text = f'{print_timestamp()} Done'

    def get_edge_labels(self, labels):
        """
        Encode transitions between differing states as 1 and other data points as 0
        """
        edge_labels = np.zeros(labels.size, dtype='<U3')
        overhang_right = (self.buffer_slider.value - 1) // 2
        overhang_left = (self.buffer_slider.value - 1) - overhang_right
        oh_counter = 0
        cur_edge = ''
        cur_label = labels[0]
        for li, l in enumerate(labels):
            if l == cur_label:
                if oh_counter != 0:
                    edge_labels[li] = cur_edge
                    oh_counter -= 1
            else:
                cur_edge = f'e{cur_label}{l}'
                edge_labels[li-overhang_left:li+1] = cur_edge
                cur_label = l
                oh_counter = overhang_right
        return edge_labels

    def update_example(self, attr, old, new):
        """
        Update the example currently on the screen.
        """
        if old == new:
            return
        if not self.data.data.loc[new, 'is_labeled']:
            if not len(self.data.data.loc[new, 'prediction']):
                pred, logprob = self.classifier.predict([new])
                self.data.data.at[new, 'prediction'] = pred[0]
                self.data.data.loc[new, 'logprob'] = logprob[0]
            self.data.set_value(new, 'labels', self.data.data.loc[new, 'prediction'].copy())
            self.data.set_value(new, 'edge_labels', self.get_edge_labels(self.data.data.loc[new, 'labels']))
            self.data.set_value(new, 'is_labeled', True)
        self.cur_example_idx = new
        if self.data.data.loc[new, 'is_junk']:
            self.notification.text = f'{print_timestamp()} Warning: {new} was marked junk!'
        elif self.data.data.loc[new, 'predicted_junk']:
            self.notification.text = f'{print_timestamp()} Warning: {new} is predicted junk!'
        self._redraw_all()

    def update_example_retrain(self):
        """
        Assume current example is labeled correctly, retrain and display new random example
        """
        self.train_and_update()
        if all(self.data.data.is_labeled):
            self.notification.text = f'{print_timestamp()}All examples have already been manually classified'
        else:
            sm_check = self.data.data.prediction.apply(lambda x:
                                                       True if len(x) == 0
                                                               or any(np.in1d(self.showme_checkboxes.active, x))
                                                       else False)
            valid_bool = np.logical_and(sm_check, np.invert(self.data.data.is_labeled.astype(bool)))
            if not any(valid_bool):
                self.notification.text = f'{print_timestamp()} No new traces with states of interest left'
                return
            new_example_idx = np.random.choice(self.data.data.loc[valid_bool, 'logprob'].index)
            self.example_select.value = new_example_idx

    def _redraw_all(self):
        self.invalidate_cached_properties()
        nb_samples = self.i_don.size
        all_ts = np.concatenate((self.i_don, self.i_acc))
        rect_mid = (all_ts.min() + all_ts.max()) / 2
        rect_height = np.abs(all_ts.min()) + np.abs(all_ts.max())
        self.source.data = dict(i_don=self.i_don, i_acc=self.i_acc, time=np.arange(nb_samples),
                                E_FRET=self.E_FRET, correlation_coefficient=self.correlation_coefficient, i_sum=self.i_sum,
                                E_FRET_sd=self.E_FRET_sd,
                                rect_height=np.repeat(rect_height, nb_samples),
                                rect_mid=np.repeat(rect_mid, nb_samples),
                                i_sum_height=np.repeat(self.i_sum.max(), nb_samples),
                                i_sum_mid=np.repeat(self.i_sum.mean(), nb_samples),
                                labels=self.data.data.loc[self.cur_example_idx, 'labels'],
                                labels_pct=self.data.data.loc[self.cur_example_idx, 'labels'] * 1.0 / self.num_states_slider.value)
        self.update_accuracy_hist()
        self.update_logprob_hist()
        self.update_state_curves()
        self.update_stats_text()

    def generate_report(self):
        self.predict_all()
        self.html_source.data['html_text'] = [FretReport(self).construct_html_report()]
        self.report_holder.text += ' '

    def update_classification(self, attr, old, new):
        if len(new):
            self.source.selected.indices = []
            patch = {'labels': [(i, self.sel_state_slider.value - 1) for i in new],
                     'labels_pct': [(i, (self.sel_state_slider.value - 1) * 1.0 / self.num_states_slider.value) for i in new]}
            self.source.patch(patch)
            self.update_accuracy_hist()
            self.update_stats_text()

            # update data in main table
            self.data.set_value(self.cur_example_idx, 'labels', self.source.data['labels'])
            self.data.set_value(self.cur_example_idx, 'edge_labels', self.get_edge_labels(self.source.data['labels']))

    def subtract_background(self):
        self.data.subtract_background(self.bg_eps.value)
        self._redraw_all()

    def subtract_test(self):
        self.data.subtract_background(self.bg_eps.value, [self.cur_example_idx])
        self._redraw_all()

    def restore_background(self):
        self.data.restore_background()
        self._redraw_all()

    def update_accuracy_hist(self):
        acc_counts = np.histogram(self.data.accuracy[0], bins=np.linspace(5, 100, num=20))[0]
        self.accuracy_source.data['accuracy_counts'] = acc_counts

    def update_logprob_hist(self):
        counts, edges = np.histogram(self.data.data.logprob.loc[self.data.data.logprob.notna()], bins=20)
        self.logprob_source.data = {'logprob_counts': counts, 'lb': edges[:-1], 'rb': edges[1:]}

    def update_stats_text(self):
        pct_labeled = round(self.data.data.is_labeled.sum() / self.data.data.is_labeled.size * 100.0, 1)
        if pct_labeled == 0:
            acc = 'N/A'
            post = 'N/A'
        else:
            acc = round(self.data.accuracy[1], 1)
            post = round(self.data.data.logprob.mean(), 1)
        self.acc_text.text = f'{acc}'
        self.posterior_text.text = f'{post}'
        self.mc_text.text = f'{pct_labeled}'

    def update_feature_list(self, attr, old, new):
        if len(new) == 0: return
        if old == new: return
        self.classifier.feature_list = [feat for fi, feat in enumerate(self.feature_list) if fi in new]
        self.train_and_update()
        self.predict_safe()
        self.data.data.loc[self.cur_example_idx, 'is_labeled'] = False
        self.update_example(None, None, self.cur_example_idx)

    def update_state_curves(self):
        feature = self.feature_list[self.state_radio.active]
        if self.classifier.trained is None: return
        if not feature in self.classifier.feature_list:
            self.state_source.data = {
                'xs': [np.arange(0, 1, 0.01)] * self.num_states_slider.value,
                'ys': [np.zeros(100, dtype=float)] * self.num_states_slider.value,
                'color': self.curve_colors}
            return
        mus = self.classifier.get_states_mu(feature)
        sds = self.classifier.get_states_sd(feature)
        x_high = max([mu + sd * 3 for mu, sd in zip(mus, sds)])
        x_low = min([mu - sd * 3 for mu, sd in zip(mus, sds)])
        xs = [np.arange(x_low, x_high, (x_high - x_low) / 100)] * self.num_states_slider.value
        ys = [1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-(xs[0] - mu) ** 2 / (2 * sd ** 2))
              for mu, sd in zip(mus, sds)]
        self.state_source.data = {'ys': ys, 'xs': xs, 'color': self.curve_colors}

    def update_algo(self, attr, old, new):
        if old == new:
            return
        self.classifier_class = importlib.import_module('FRETboard.algorithms.'+algo_dict[new]).Classifier
        self.classifier = self.classifier_class(nb_states=self.num_states_slider.value, data=self.data, gui=self,
                                                features=self.feature_list)
        if len(self.data.data):
            self.train_and_update()
            self.predict_safe()
            self.data.data.loc[self.cur_example_idx, 'is_labeled'] = False
            self.update_example(None, None, self.cur_example_idx)

    def update_num_states(self, attr, old, new):
        if new != self.classifier.nb_states:
            self.data.data.loc[self.data.data.index, 'labels'] = [ [[]] * len(self.data.data)]
            self.data.data.loc[self.data.data.index, 'edge_labels'] = [[[]] * len(self.data.data)]
            self.data.data.is_labeled = False
            self.classifier = self.classifier_class(nb_states=new, data=self.data, gui=self, features=self.feature_list)

            # Update widget: show-me checkboxes
            showme_idx = list(range(new))
            showme_states = [str(n) for n in range(1,new+1)]
            self.showme_checkboxes.labels = showme_states
            self.showme_checkboxes.active = showme_idx
            self.saveme_checkboxes.labels = showme_states
            self.saveme_checkboxes.active = showme_idx

            # Update widget: selected state slider
            self.sel_state_slider.end = new
            if self.sel_state_slider.value > new: self.sel_state_slider.value = new

            # retraining is too heavy for longer traces, setting current example to lowest state instead
            blank_labels = [(i, 0) for i in range(len(self.data.data.loc[self.cur_example_idx, 'i_don']))]
            patch = {'labels': blank_labels,
                     'labels_pct': blank_labels}
            self.source.patch(patch)
            self.source.selected.indices = []
            self.update_accuracy_hist()
            self.update_stats_text()

            # update data in main table
            self.data.set_value(self.cur_example_idx, 'labels', self.source.data['labels'])
            self.data.set_value(self.cur_example_idx, 'edge_labels', self.get_edge_labels(self.source.data['labels']))
            self.data.set_value(self.cur_example_idx, 'is_labeled', True)

            # # retraining classifier
            # if self.data.data.shape[0] != 0:
            #     self.invalidate_cachedhttps://github.com/akdel/locality-sensitive-hashing/blob/master/LSH/period_based.py_properties()
            #     self.train_and_update()
            #     self.update_state_curves()
            #     self.update_example(None, '', self.cur_example_idx)

    def export_data(self):
        self.fretReport = FretReport(self)

    def generate_dats(self):
        tfh = tempfile.TemporaryDirectory()
        self.predict_all()
        for fn, tup in self.data.data.iterrows():
            sm_test = tup.labels if len(tup.labels) else tup.prediction
            sm_bool = [True for sm in self.saveme_checkboxes.active if sm in sm_test]
            if not any(sm_bool): continue
            labels = tup.labels + 1 if len(tup.labels) != 0 else [None] * len(tup.time)
            out_df = pd.DataFrame(dict(time=tup.time, i_don=tup.i_don, i_acc=tup.i_acc,
                                       label=labels, predicted=tup.prediction + 1))
            out_df.to_csv(f'{tfh.name}/{fn}', sep='\t', na_rep='NA', index=False)
        zip_dir = tempfile.TemporaryDirectory()
        zip_fn = shutil.make_archive(f'{zip_dir.name}/dat_files', 'zip', tfh.name)
        with open(zip_fn, 'rb') as f:
            self.datzip_source.data['datzip'] = [base64.b64encode(f.read()).decode('ascii')]
        tfh.cleanup()
        zip_dir.cleanup()
        self.datzip_holder.text += ' '

    def del_trace(self):
        self.data.del_tuple(self.cur_example_idx)
        nonjunk_bool = np.logical_and(self.data.data.is_labeled, np.invert(self.data.data.is_junk))
        # if any(nonjunk_bool):  # Cant predict without positive examples
        #     df = self.data.data.loc[self.data.data.is_labeled]
        #     x = np.stack(df.E_FRET.to_numpy())
        #     predicted_junk = lsh_classify(x, self.data.data.is_junk, x, bits=32)
        #     self.data.data.predicted_junk = np.logical_or(self.data.data.junk, predicted_junk)
        #     # self.example_select.options.remove(self.cur_example_idx)
        self.update_example_retrain()

    def update_showme(self, attr, old, new):
        if old == new: return
        if self.data.data.shape[0] == 0: return
        valid_bool = self.data.data.apply(lambda x: any(i in new for i in x.prediction), axis=1)
        if any(valid_bool):
            valid_idx = list(self.data.data.index[valid_bool])
            if self.cur_example_idx not in valid_idx:
                new_example_idx = np.random.choice(valid_idx)
                self.update_example = new_example_idx
        else:
            self.notification.text = f'''\n{print_timestamp()}No valid (unclassified) traces to display for classes {', '.join([str(ts + 1) for ts in new])}'''

    # @gen.coroutine
    # def update_notification(self, text):
    #     self.notification.text = text

    def make_document(self, doc):
        # --- Define widgets ---
        ff_title = Div(text=f"""<font size=15><b>FRETboard</b></font><br>v.{self.version}<hr>""",
                       width=280, height=90)

        # --- 1. Load ---
        load_button = Button(label='Data')
        load_button.callback = CustomJS(args=dict(file_source=self.new_source, new_counter=self.new_tot), code=upload_js)

        load_model_button = Button(label='Model')
        load_model_button.callback = CustomJS(args=dict(file_source=self.loaded_model_source),
                                              code=upload_model_js)


        # --- 2. Teach ---
        del_trace_button = Button(label='Delete', button_type='danger')
        del_trace_button.on_click(self.del_trace)

        showme_states = [str(n + 1) for n in range(self.num_states_slider.value)]
        showme_idx = list(range(self.num_states_slider.value))

        self.showme_checkboxes = CheckboxButtonGroup(labels=showme_states, active=showme_idx)
        self.showme_checkboxes.on_change('active', self.update_showme)
        showme_col = column(Div(text='Show traces with states:', width=300, height=16),
                            self.showme_checkboxes,
                            height=80, width=300)

        example_button = Button(label='Train', button_type='success')

        # --- 3. Save ---
        save_model_button = Button(label='Model')
        save_model_button.callback = CustomJS(args=dict(file_source=self.classifier_source),
                                              code=download_csv_js)
        save_data_button = Button(label='Data')
        save_data_button.on_click(self.generate_dats)

        report_button = Button(label='Report')
        report_button.on_click(self.generate_report)

        self.saveme_checkboxes = CheckboxButtonGroup(labels=showme_states, active=showme_idx)
        saveme_col = column(Div(text='Save traces with states:', width=300, height=16),
                            self.saveme_checkboxes,
                            height=80, width=300)

        # --- Define plots ---

        # Main timeseries
        ts = figure(tools='xbox_select,save,xwheel_zoom,xwheel_pan,pan', plot_width=1075, plot_height=275, active_drag='xbox_select')
        ts.rect('time', 'rect_mid', height='rect_height', fill_color={'field': 'labels_pct',
                                                                      'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts.line('time', 'i_don', color='#4daf4a', source=self.source, **line_opts)
        ts.line('time', 'i_acc', color='#e41a1c', source=self.source, **line_opts)
        ts_panel = Panel(child=ts, title='Traces')

        # E_FRET series
        ts_efret = figure(tools='xbox_select,save,xwheel_zoom,xwheel_pan,pan', plot_width=1075, plot_height=275,
                          active_drag='xbox_select', x_range=ts.x_range, y_range=[0,1])  #  todo: add tooltips=[('$index')]
        ts_efret.rect('time', 0.5, height=1.0, fill_color={'field': 'labels_pct',
                                                           'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts_efret.line('time', 'E_FRET', color='#1f78b4', source=self.source, **line_opts)
        ts_efret.line('time', 'E_FRET_sd', color='#a6cee3', source=self.source, **line_opts)
        efret_panel = Panel(child=ts_efret, title='E_FRET & sd')

        # correlation coeff series
        ts_corr = figure(tools='xbox_select,save,xwheel_zoom,xwheel_pan,pan', plot_width=1075, plot_height=275,
                          active_drag='xbox_select', x_range=ts.x_range)  # todo: add tooltips=[('$index')]
        ts_corr.rect('time', 0.0, height=2.0, fill_color={'field': 'labels_pct',
                                                          'transform': self.col_mapper},
                      source=self.source, **rect_opts)
        ts_corr.line('time', 'correlation_coefficient', color='#b2df8a', source=self.source, **line_opts)
        corr_panel = Panel(child=ts_corr, title='Correlation coefficient')

        # i_sum series
        ts_i_sum = figure(tools='xbox_select,save,xwheel_zoom,xwheel_pan,pan', plot_width=1075, plot_height=275,
                          active_drag='xbox_select', x_range=ts.x_range)
        ts_i_sum.rect('time', 'i_sum_mid', height='i_sum_height', fill_color={'field': 'labels_pct',
                                                                              'transform': self.col_mapper},
                      source=self.source, **rect_opts)
        ts_i_sum.line('time', 'i_sum', color='#1f78b4', source=self.source, **line_opts)
        i_sum_panel = Panel(child=ts_i_sum, title='I sum')

        tabs = Tabs(tabs=[ts_panel, efret_panel, corr_panel, i_sum_panel])

        # accuracy histogram
        acc_hist = figure(toolbar_location=None, plot_width=275, plot_height=275, x_range=[0, 100],
                          title='Training accuracy')
        acc_hist.grid.visible = False
        acc_hist.xaxis.axis_label = '%'
        acc_hist.yaxis.axis_label = 'count'
        acc_hist.quad(bottom=0, left='lb', right='rb', top='accuracy_counts', color='white',
                      line_color='#084594', source=self.accuracy_source)

        # posteriors histogram
        logprob_hist = figure(toolbar_location=None, plot_width=275, plot_height=275, title='Posteriors')
        logprob_hist.grid.visible = False
        logprob_hist.xaxis.axis_label = 'log(P)'
        logprob_hist.yaxis.axis_label = 'count'
        logprob_hist.quad(bottom=0, left='lb', right='rb', top='logprob_counts', color='white',
                          line_color='#084594', source=self.logprob_source)

        # state property curves
        state_curves = figure(toolbar_location=None, plot_width=275, plot_height=275, title='States')
        state_curves.background_fill_color = '#a6a6a6'
        state_curves.grid.visible = False

        state_curves.xaxis.axis_label = 'Feature value'
        state_curves.yaxis.axis_label = 'Density'
        state_curves.multi_line('xs', 'ys', line_color='color', source=self.state_source)

        # Stats in text
        stats_text = column( row(Div(text='Accuracy (%): ', width=140, height=18), self.acc_text, height=18),
                             row(Div(text='Mean log-posterior: ', width=140, height=18), self.posterior_text, height=18),
                             row(Div(text='Manually classified (%): ', width=140, height=18), self.mc_text, height=18))

        # todo: Average accuracy and posterior

        # --- Define update behavior ---
        self.algo_select.on_change('value', self.update_algo)
        self.source.selected.on_change('indices', self.update_classification)  # for manual selection on trace
        self.new_source.on_change('data', self.update_data)
        self.loaded_model_source.on_change('data', self.load_params)
        self.example_select.on_change('value', self.update_example)
        example_button.on_click(self.update_example_retrain)
        self.bg_button.on_click(self.subtract_background)
        self.bg_test_button.on_click(self.subtract_test)
        self.num_states_slider.on_change('value', self.update_num_states)
        self.state_radio.on_change('active', lambda attr, old, new: self.update_state_curves())
        self.features_checkboxes.on_change('active', self.update_feature_list)

        # hidden holders to generate saves
        self.report_holder.js_on_change('text', CustomJS(args=dict(file_source=self.html_source),
                                                         code=download_report_js))
        self.datzip_holder.js_on_change('text', CustomJS(args=dict(file_source=self.datzip_source),
                                                         code=download_datzip_js))

        # --- Build layout ---
        state_block = row(state_curves,
                          Div(text='<br />'.join(self.feature_list), margin=[32, 5, 5, 5]),
                          column(Div(text='View'), self.state_radio, width=40),
                          column(Div(text='Active'), self.features_checkboxes, width=40),
                          )
        widgets = column(ff_title,
                         Div(text="<font size=4>1. Load</font>", width=280, height=15),
                         self.algo_select,
                         row(widgetbox(load_model_button, width=150), widgetbox(load_button, width=150),
                             width=300),
                         row(widgetbox(self.bg_button, width=150), widgetbox(self.bg_eps, width=75),
                             widgetbox(self.bg_test_button, width=75), width=300),
                         Div(text="<font size=4>2. Teach</font>", width=280, height=15),
                         self.num_states_slider,
                         self.sel_state_slider,
                         self.supervision_slider,
                         self.buffer_slider,
                         showme_col,
                         row(widgetbox(del_trace_button, width=150), widgetbox(example_button, width=150)),
                         Div(text="<font size=4>3. Save</font>", width=280, height=15),
                         saveme_col,
                         row(widgetbox(report_button, width=100), widgetbox(save_data_button, width=100), widgetbox(save_model_button, width=100)),
                         width=300)
        hists = row(acc_hist, logprob_hist, state_block)
        graphs = column(self.example_select,
                        tabs,
                        hists,
                        stats_text,
                        Div(text='Notifications: ', width=100, height=15),
                        self.notification,
                        self.report_holder, self.datzip_holder)
        layout = row(widgets, graphs)
        doc.add_root(layout)
        doc.title = f'FRETboard v. {self.version}'
        self.doc = doc

    def start_gui(self, port=0):
        apps = {'/': Application(FunctionHandler(self.make_document))}
        server = Server(apps, port=port, websocket_max_message_size=100000000)
        server.show('/')
        loop = IOLoop.current()
        loop.start()

    def start_server(self):
        return self.make_document(curdoc())
