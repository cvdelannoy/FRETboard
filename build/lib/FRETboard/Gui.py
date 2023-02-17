import io
import os
import sys
import tempfile
import shutil
import base64
import numpy as np
import pandas as pd
import yaml
import importlib
# import imp
import h5py
import traceback
from multiprocessing import Process
import threading
from threading import Thread, Event
import pickle
from sklearn import linear_model
from jinja2 import Template
from datetime import datetime

from cached_property import cached_property
from bokeh.client import session, websocket
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import row, column
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LinearColorMapper, Column
from bokeh.models.tools import BoxSelectTool, WheelZoomTool, WheelPanTool, SaveTool, PanTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Slider, Select, Button, PreText, RadioGroup, Div, CheckboxButtonGroup, CheckboxGroup, Spinner, Toggle
from bokeh.models.widgets.panels import Panel, Tabs
from tornado.ioloop import IOLoop
from tornado import gen

from pathlib import Path
__location__ = Path(__file__).parent.resolve()
sys.path.append(str(__location__ / '..'))

from FRETboard.SafeH5 import SafeH5
from FRETboard.Predictor import Predictor
from FRETboard.helper_functions import print_timestamp, series_to_array, installThreadExcepthook, get_tuple
from FRETboard.FretReport import FretReport
from FRETboard.MainTable_parallel import MainTable
from FRETboard.OneshotHmm import OneshotHmm

from FRETboard.helper_functions import get_tuple, colnames_alex, colnames, get_ssfret_dist, split_trace_dict_on_source


line_opts = dict(line_width=1)
rect_opts = dict(alpha=1, line_alpha=0)
with open(__location__ / 'algorithms.yml', 'r') as fh: algo_dict = yaml.safe_load(fh)
algo_dict['custom'] = 'custom'
algo_inv_dict = {algo_dict[k]: k for k in algo_dict}
white_blue_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
# pastel_colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
# col_mapper = LinearColorMapper(palette=white_blue_colors, low=0, high=1)
# diverging_colors = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
colors = white_blue_colors

with open(__location__ / 'js_widgets/upload.js', 'r') as fh: upload_js = fh.read()
with open(__location__ / 'js_widgets/upload_model.js', 'r') as fh: upload_model_js = fh.read()
with open(__location__ / 'js_widgets/download_datzip.js', 'r') as fh: download_datzip_js = fh.read()
with open(__location__ / 'js_widgets/download_report.js', 'r') as fh: download_report_js = fh.read()
with open(__location__ / 'js_widgets/download_ssfret.js', 'r') as fh: download_ssfret_js = fh.read()
with open(__location__ / 'js_widgets/download_model.js', 'r') as fh: download_csv_js = fh.read()
with open(__location__ / 'js_widgets/upload_custom_script.js', 'r') as fh: upload_custom_script_js = fh.read()

with open(__location__ / 'templates/index.html', 'r') as fh: template = Template(fh.read())

class Gui(object):
    def __init__(self, nb_processes=3, nb_states=2, allow_custom_scripts=False, data=[]):
        installThreadExcepthook()
        self.version = '0.0.3'
        self.cur_trace_idx = None
        self.nb_processes = nb_processes
        self.allow_custom_scripts = allow_custom_scripts
        self.feature_list = ['E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd',
                             'correlation_coefficient', 'f_dex_dem', 'f_dex_aem',
                             'f_aex_dem', 'f_aex_aem']
        self.h5_dir = Path(tempfile.mkdtemp())
        self.data = MainTable(10.0, np.nan, 0.0, 0.0, 1.0, 0, self.h5_dir, os.getpid())
        self.file_parser_process = self.data.init_table()
        self.app_is_up = True

        # --- Widgets ---

        # 1. Load
        self.example_collector = pd.DataFrame(columns=['example'])
        self.algo_select = Select(title='Algorithm:', value=list(algo_dict)[0], options=list(algo_dict))
        self.num_states_slider = Slider(title='Number of states', value=nb_states, start=2, end=11, step=1,
                                        name='num_states_slider')
        self.bg_checkbox = CheckboxGroup(labels=[''], active=[])

        # 2. Train
        self.sel_state_slider = Slider(title='Change selection to state  (num keys)', value=1, start=1,
                                       end=self.num_states_slider.value, step=1, name='sel_state_slider')
        self.sel_state = Div(text='1', name='sel_state')
        self.guess_toggle = Toggle(label='guess trace')
        self.del_trace_button = Button(label='Delete (Q)', button_type='danger', name='delete_button')
        # 3. Save


        # Graph area
        self.example_select = Select(title='Current example', value='None', options=['None'])
        self.notification = PreText(text='', width=1000, height=15)
        self.acc_text = PreText(text='N/A')
        self.posterior_text = PreText(text='N/A')
        self.mc_text = PreText(text='0%')
        self.report_holder = PreText(text='', css_classes=['hidden'])  # hidden holder to generate js callbacks
        self.datzip_holder = PreText(text='', css_classes=['hidden'])
        self.ssfret_holder = PreText(text='', css_classes=['hidden'])
        self.custom_script_holder = PreText(text='', css_classes=['hidden'])
        self.keystroke_holder = PreText(text='', css_classes=['hidden'], name='keystroke_holder')
        self.features_checkboxes = CheckboxGroup(labels=[''] * len(self.feature_list), active=[0, 2, 3, 4])
        self.state_radio = RadioGroup(labels=[''] * len(self.feature_list), active=0)

        # Settings
        self.eps_spinner = Spinner(value=15, step=1)
        self.bg_button = Button(label='Apply')
        self.framerate_spinner = Spinner(value=10, step=1)
        self.remove_last_checkbox = CheckboxGroup(labels=[''], active=[])
        self.supervision_slider = Slider(title='Influence supervision', value=1.0, start=0.0, end=1.0, step=0.01)
        self.buffer_slider = Slider(title='Buffer', value=1, start=1, end=20, step=1)
        self.bootstrap_size_spinner = Spinner(value=100, step=1)
        self.alex_checkbox = CheckboxGroup(labels=[''], active=[])
        self.traceswitch_checkbox = CheckboxGroup(labels=[''], active=[])
        self.gamma_factor_spinner = Spinner(value=1.0, step=0.001)
        self.l_spinner = Spinner(value=0.0, step=0.001)
        self.d_spinner = Spinner(value=0.0, step=0.001)
        self.d_only_state_spinner = Spinner(value=1, step=1)
        self.a_only_state_spinner = Spinner(value=2, step=1)
        self.alex_corr_button = Button(label='Apply')
        self.alex_estimate_button = Button(label='Estimate parameters')


        # Classifier object
        self.classifier_class = self.algo_select.value
        self.classifier = self.classifier_class(nb_states=nb_states, data=self.data, buffer=self.buffer_slider.value,
                                                supervision_influence=self.supervision_slider.value,
                                                features=[feat for fi, feat in enumerate(self.feature_list)
                                                          if fi in self.features_checkboxes.active])

        # ColumnDataSources
        self.source = ColumnDataSource(data=dict(f_dex_dem=[], f_dex_aem=[], E_FRET=[],
                                                 correlation_coefficient=[], E_FRET_sd=[], i_sum=[], time=[],
                                                 rect_height=[], rect_height_half=[], rect_mid=[], rect_mid_up=[], rect_mid_down=[], rect_width=[],
                                                 i_sum_height=[], i_sum_mid=[],
                                                 f_aex_dem=[], f_aex_aem=[],
                                                 labels=[], labels_pct=[], prediction_pct=[]))
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
        self.custom_script_source = ColumnDataSource(data=dict(file_contents=[]))
        self.classifier_source = ColumnDataSource(data=dict(params=[]))
        self.html_source = ColumnDataSource(data=dict(html_text=[]))
        self.ssfret_source = ColumnDataSource(data=dict(ssfret_text=[]))
        self.datzip_source = ColumnDataSource(data=dict(datzip=[]))
        self.scroll_state_source = ColumnDataSource(data=dict(new_state=[False]))

        self.new_tot = ColumnDataSource(data=dict(value=[0]))
        self.new_cur = 0

        self.model_loaded = False
        self.features_changed = True
        self.train_trigger_activated = False
        self.total_redraw_activated = False
        self.partial_redraw_activated = False
        self.fn_buffer = []

    @property
    def nb_processes(self):
        return self._nb_processes

    @nb_processes.setter
    def nb_processes(self, nb):
        if nb < 3: raise ValueError('Need at least 3 processes to function')
        self._nb_processes = nb

    @property
    def alex(self):
        return len(self.alex_checkbox.active) == 1

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        self._classifier = classifier

    @property
    def classifier_class(self):
        return self._classifier_class

    @classifier_class.setter
    def classifier_class(self, class_name):
        self._classifier_class = importlib.import_module('FRETboard.algorithms.' + algo_dict.get(class_name, class_name)).Classifier

    @property
    def nb_examples(self):
        return self.data.index_table.shape[0]

    @property
    def curve_colors(self):
        return [colors[n] for n in (np.linspace(0, len(colors)-1, self.num_states_slider.value)).astype(int)]

    @property
    def col_mapper(self):
        return LinearColorMapper(palette=colors, low=0, high=0.99)

    # --- trigger functions ---
    # Required to enable communication between threads

    def notify(self, text):
        self.notification_buffer = f'{print_timestamp()}{text}\n'
        self.doc.add_next_tick_callback(self.push_notification)

    def notify_exception(self, e, tb_str, section):
        self.notify(f'''Exception of type {str(e.__class__)} encountered during {section}! 
Please register an issue at github.com/cvdelannoy/FRETboard/issues, include the data on which it occurred if
possible, and the error message below
--- start error message ---
{tb_str}''')

    def train_trigger(self):
        if not self.train_trigger_activated:
            self.notify('Start training...')
            self.train_trigger_activated = True
            self.doc.add_next_tick_callback(self.train)

    def new_example_trigger(self):
        self.doc.add_next_tick_callback(self.update_example_fun)

    def update_example_fun(self):
        self.example_select.value = self.cur_trace_idx

    def redraw_trigger(self):
        if not self.total_redraw_activated:
            self.total_redraw_activated = True
            self.doc.add_next_tick_callback(self._redraw_all)

    def redraw_info_trigger(self):
        if not self.partial_redraw_activated:
            self.partial_redraw_activated = True
            self.doc.add_next_tick_callback(self._redraw_info)

    def append_fn(self, fn_list):
        self.fn_buffer.extend(fn_list)
        self.doc.add_next_tick_callback(self.push_fn)

    def push_fn(self):
        self.example_select.options = self.example_select.options + self.fn_buffer
        self.fn_buffer = []

    def push_notification(self):
        self.notification.text = self.notification_buffer + self.notification.text

    # --- asynchronous functions ---

    def train(self):
        try:
            data_dict = self.data.get_trace_dict()
            self.classifier.train(data_dict=data_dict, supervision_influence=self.supervision_slider.value)
            self.model_loaded = True
            self.classifier_source.data = dict(params=[self.classifier.get_params()])
            with open(self.h5_dir / f'{self.classifier.timestamp}.mod', 'wb') as fh:
                pickle.dump(self.classifier, fh, pickle.HIGHEST_PROTOCOL)
            # with SafeH5(f'{self.data.predict_store_fn}', 'w') as fh:
            #     for idx in fh: del fh[idx]
            self.notify('Finished training')
            self.features_changed = False
            if self.cur_trace_idx is not None:
                self.current_example.loc[:, 'predicted'], _ = self.classifier.predict(self.current_example)
                self.redraw_trigger()
        except Exception as e:
            self.notify_exception(e, traceback.format_exc(), 'training')
        finally:
            self.train_trigger_activated = False

    def update_example_list(self):
        to_add = [idx for idx in self.data.index_table.index
                  if idx not in self.example_select.options
                  and idx not in self.fn_buffer]
        if len(to_add):
            self.append_fn(to_add)

    def new_example(self):
        if all(self.data.manual_table.is_labeled):
            self.notify('All examples have already been manually classified')
        else:
            # Collect indices of eligible samples
            cur_index_table = self.data.index_table.copy()  # make local copy in case it changes in size in mean time
            valid_idx = cur_index_table.index[np.logical_and(
                cur_index_table.mod_timestamp == self.classifier.timestamp,
                cur_index_table.data_timestamp == self.data.data_timestamp
            )]

            manual_invalid_idx = self.data.manual_table.index[np.logical_or(
                self.data.manual_table.is_labeled,  # todo potential problem: unequal df sizes
                self.data.manual_table.is_junk
            )]
            valid_idx = [vi for vi in valid_idx if vi not in manual_invalid_idx]

            if not len(valid_idx):
                self.notify('Just a moment, classifying traces...')
                return
            # if np.all(np.isnan(self.data.index_table.loc[valid_idx, 'logprob'])):
            #     self.notify('No valid unpredicted traces left. Try to train before choosing next example.')
            #     return
            new_example_idx = self.data.index_table.loc[valid_idx, 'logprob'].idxmin()
            self.example_select.value = new_example_idx

    def load_params(self, attr, old, new):
        raw_contents = self.loaded_model_source.data['file_contents'][0]
        # remove the prefix that JS adds
        _, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents).decode('utf-8')
        self.algo_select.value = algo_inv_dict.get(file_contents.split('\n')[0], 'custom')
        self.classifier.load_params(file_contents)
        self.num_states_slider.value = self.classifier.nb_states
        self.saveme_checkboxes.labels = [str(n) for n in range(1, self.classifier.nb_states + 1)]
        self.saveme_checkboxes.active = list(range(self.classifier.nb_states))
        if np.isnan(self.data.eps):
            self.bg_checkbox.active = []
        else:
            self.eps_spinner.value = self.data.eps
            self.bg_checkbox.active = [0]
        self.l_spinner.value, self.d_spinner.value, self.gamma_factor_spinner.value = self.data.l, self.data.d, self.data.gamma

        self.model_loaded = True
        self.classifier.timestamp = int(datetime.now().strftime('%H%M%S%f'))
        with open(self.h5_dir / f'{self.classifier.timestamp}.mod', 'wb') as fh:
            pickle.dump(self.classifier, fh, pickle.HIGHEST_PROTOCOL)
        self.notify('Model loaded')
        if self.data.index_table.shape[0] != 0:
            self.data.manual_table.is_labeled = False
            self.redraw_trigger()

    def buffer_data(self, attr, old, new):
        # self.example_collector.loc[self.new_source.data['file_name'][0], 'example'] = self.new_source.data['file_contents'][0]
        self.data.add_tuple(self.new_source.data['file_contents'][0], self.new_source.data['file_name'][0])

    def load_example_data(self):
        example_df = pd.read_pickle(__location__ / 'example_data.pkl')
        for _, tup in example_df.iterrows():
            self.data.add_tuple(tup.example, tup.name)  # todo .name or .index?

    def get_edge_labels(self, labels):
        """
        Encode transitions between differing states X and Y as strings of shape 'eX_Y'
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
                cur_edge = f'e{cur_label}_{l}'
                edge_labels[li-overhang_left:li+1] = cur_edge
                cur_label = l
                oh_counter = overhang_right
        return edge_labels

    def update_buffer(self, attr, old, new):
        if old == new: return
        self.buffer_updated = True  # todo redo current buffer labels or take out?

    def update_example(self, attr, old, new):
        """
        Update the example currently on the screen.
        """
        if old == new:
            return
        self.cur_trace_idx = new
        if old == 'None':
            new_list = self.example_select.options
            new_list.remove('None')
            self.example_select.options = new_list
        self.current_example = self.data.get_trace(new, await_labels=True)
        self.data.label_dict[new] = self.current_example.predicted.to_numpy(copy=True)
        self.data.manual_table.loc[new, 'is_labeled'] = True

        # warnings
        if self.data.manual_table.loc[new, 'is_junk']: self.notify(f'Warning: {new} was marked junk!')
        self.redraw_trigger()

    def _redraw_all(self):
        self.total_redraw_activated = False
        if self.cur_trace_idx is None: return
        nb_samples = len(self.current_example)
        all_ts = self.current_example.loc[:, ('f_dex_dem', 'f_dex_aem')].to_numpy()
        ts_range = all_ts.max() - all_ts.min()
        rect_mid = (all_ts.max() + all_ts.min()) / 2
        rect_mid_up = all_ts.min() + ts_range * 0.75
        rect_mid_down = all_ts.min() + ts_range * 0.25
        rect_height = np.abs(all_ts.max()) + np.abs(all_ts.min())
        rect_height_half = rect_height / 2
        rect_width = np.abs(np.subtract(*self.current_example.loc[:1, 'time'])) * 1.01
        self.source.data = dict(f_dex_dem=self.current_example.f_dex_dem, f_dex_aem=self.current_example.f_dex_aem, time=self.current_example.time,
                                E_FRET=self.current_example.E_FRET, correlation_coefficient=self.current_example.correlation_coefficient, i_sum=self.current_example.i_sum,
                                E_FRET_sd=self.current_example.E_FRET_sd,
                                rect_height=np.repeat(rect_height, nb_samples),
                                rect_height_half=np.repeat(rect_height_half, nb_samples),
                                rect_mid=np.repeat(rect_mid, nb_samples),
                                rect_mid_up=np.repeat(rect_mid_up, nb_samples),
                                rect_mid_down=np.repeat(rect_mid_down, nb_samples),
                                rect_width=np.repeat(rect_width, nb_samples),
                                i_sum_height=np.repeat(self.current_example.i_sum.max(), nb_samples),
                                i_sum_mid=np.repeat(self.current_example.i_sum.mean(), nb_samples),
                                f_aex_dem=self.current_example.f_aex_dem, f_aex_aem=self.current_example.f_aex_aem,
                                labels=self.data.label_dict[self.cur_trace_idx],
                                labels_pct=self.data.label_dict[self.cur_trace_idx] / (self.num_states_slider.value - 1),
                                prediction_pct=self.current_example.predicted / (self.num_states_slider.value - 1))
        self.x_range.start = self.current_example.time.min(); self.x_range.end = self.current_example.time.max()
        self.y_range.start = all_ts.min(); self.y_range.end = all_ts.max()
        self.update_accuracy_hist()
        self.update_logprob_hist()
        self.update_state_curves()
        self.update_stats_text()

    def _redraw_info(self):
        self.update_accuracy_hist()
        self.update_logprob_hist()
        self.update_state_curves()
        self.update_stats_text()
        self.partial_redraw_activated = False

    def generate_report(self):
        if not np.all(self.data.index_table.mod_timestamp == self.classifier.timestamp):
            self.notify('Please wait for prediction to finish before generating a report')
            return
        self.notify('Generating report, this may take a while...')
        try:
            self.doc.add_next_tick_callback(self.generate_report_fun)
        except Exception as e:
            self.notify_exception(e, traceback.format_exc(), 'report generation')

    def generate_report_fun(self):
        self.html_source.data['html_text'] = [FretReport(self).construct_html_report()]
        self.report_holder.text += ' '

    def update_classification(self, attr, old, new):
        new.sort()
        if len(new):
            self.source.selected.indices = []
            if self.guess_toggle.active:
                new_labels = self.guess_labels(new)
                patch = {'labels': [(i, nl) for i, nl in enumerate(new_labels)],
                         'labels_pct': [(i, nl) for i, nl in enumerate(new_labels / (self.num_states_slider.value - 1))]}
                self.guess_toggle.active = False
            else:
                patch = {'labels': [(i, self.sel_state_slider.value - 1) for i in new],
                         'labels_pct': [(i, (self.sel_state_slider.value - 1) * 1.0 / (self.num_states_slider.value - 1))
                                        for i in new]}
                # update data in main table
                self.data.label_dict[self.cur_trace_idx][new] = self.sel_state_slider.value - 1

            self.source.patch(patch)

            self.update_accuracy_hist()
            self.update_stats_text()


    def revert_manual_labels(self):
        patch = {'labels': [(i, v) for i, v in enumerate(self.current_example.predicted)],
                 'labels_pct': [(i, v) for i, v in enumerate(self.current_example.predicted * 1.0 / (self.num_states_slider.value - 1))]}
        self.source.patch(patch)
        self.data.label_dict[self.cur_trace_idx] = self.current_example.predicted.to_numpy(copy=True)

        # update data in main table
        self.update_accuracy_hist()
        self.update_stats_text()

    def update_eps(self):
        old_eps = self.data.eps
        if not len(self.bg_checkbox.active):
            self.data.eps = np.nan
        else:
            self.data.eps = float(self.eps_spinner.value)
        if old_eps != self.data.eps and self.cur_trace_idx is not None:
            self.refilter_current_example()
            self.redraw_trigger()

    def update_crosstalk(self):
        if (self.data.l != self.l_spinner.value
                or self.data.d != self.d_spinner.value
                or self.data.gamma != self.gamma_factor_spinner.value):
            self.data.l = self.l_spinner.value
            self.data.d = self.d_spinner.value
            self.data.gamma = self.gamma_factor_spinner.value
            self.refilter_current_example()
            self.redraw_trigger()

    def refilter_current_example(self):
        if self.cur_trace_idx is None: return
        if self.alex:
            cur_array = self.current_example.loc[:, ('time', 'f_dex_dem_raw', 'f_dex_aem_raw', 'f_aex_dem_raw', 'f_aex_aem_raw')].to_numpy(copy=True).T
        else:
            cur_array = self.current_example.loc[:, ('time', 'f_dex_dem_raw', 'f_dex_aem_raw')].to_numpy(copy=True).T
        out_array = get_tuple(cur_array, self.data.eps, self.data.l, self.data.d, self.data.gamma, self.data.traceswitch)
        self.current_example.loc[:, colnames_alex] = out_array.T

    def update_accuracy_hist(self):
        acc_counts = np.histogram(self.data.accuracy[0], bins=np.linspace(5, 100, num=20))[0]
        self.accuracy_source.data['accuracy_counts'] = acc_counts

    def update_logprob_hist(self):
        counts, edges = np.histogram(self.data.index_table.loc[self.data.index_table.logprob.notna(), 'logprob'], bins=20)
        self.logprob_source.data = {'logprob_counts': counts, 'lb': edges[:-1], 'rb': edges[1:]}

    def update_stats_text(self):
        pct_labeled = round(self.data.manual_table.is_labeled.sum() /
                            max(self.data.manual_table.is_labeled.size, 1) * 100.0, 1)
        if pct_labeled == 0:
            acc = 'N/A'
            post = 'N/A'
        else:
            acc = round(self.data.accuracy[1], 1)
            post = round(self.data.index_table.logprob.mean(), 1)
        self.acc_text.text = f'{acc}'
        self.posterior_text.text = f'{post}'
        self.mc_text.text = f'{pct_labeled}'

    def update_feature_list(self, attr, old, new):
        if len(new) == 0: return
        if old == new: return
        self.classifier.feature_list = [feat for fi, feat in enumerate(self.feature_list) if fi in new]
        self.features_changed = True

    def update_alex_checkbox(self, attr, old, new):
        if len(self.alex_checkbox.active):
            self.data.alex = 1
        else:
            self.data.alex = 0

    def update_traceswitch_checkbox(self, attr, old, new):
        if len(self.traceswitch_checkbox.active):
            self.data.traceswitch = 1
        else:
            self.data.traceswitch = 0
        if old != new:
            self.refilter_current_example()
            self.redraw_trigger()

    def update_state_curves(self):
        feature = self.feature_list[self.state_radio.active]
        if self.classifier.trained is None: return
        if self.features_changed or feature not in self.classifier.feature_list:
            self.state_source.data = {
                'xs': [np.arange(0, 1, 0.01)] * self.num_states_slider.value,
                'ys': [np.zeros(100, dtype=float)] * self.num_states_slider.value,
                'color': self.curve_colors}
            return
        mus = self.classifier.get_mus(feature)
        if any([mu is None for mu in mus]): return
        sds = [sd if sd > 0.1 else 0.1 for sd in self.classifier.get_sds(feature)]
        x_high = max([mu + sd * 3 for mu, sd in zip(mus, sds)])
        x_low = min([mu - sd * 3 for mu, sd in zip(mus, sds)])
        try:
            xs = [np.arange(x_low, x_high, (x_high - x_low) / 100)] * len(sds)
        except:
            return
        sdps = [-(xs[0] - mu) ** 2 / (2 * sd ** 2) for mu, sd in zip (mus, sds)]
        ys = []
        for sd, sdp in zip(sds, sdps):
            if any(sdp < -708):
                ys.append(np.zeros(len(sdp)))
            else:
                ys.append(1 / (sd * np.sqrt(2 * np.pi)) * np.exp(sdp))
        curve_colors = [colors[n] for n in (np.linspace(0, len(colors) - 1, len(ys))).astype(int)]
        self.state_source.data = {'ys': ys, 'xs': xs, 'color': curve_colors}

    def update_algo(self, attr, old, new):
        if old == new and new != 'custom':
            return
        algo = algo_dict.get(new, new)
        if algo == 'custom':
            if not self.allow_custom_scripts:
                self.notify('Custom scripts are disabled by default to limit the risk of code injection! Consult the '
                            'manual if you would like to turn it on.')
                self.algo_select.value = old
                return
            self.custom_script_holder.text += ' '
            return
        else:
            self.classifier_class = algo
        self.reload_classifier()

    def reload_classifier(self):
        cur_timestamp = self.classifier.timestamp
        cur_features = self.classifier.feature_list
        self.classifier = self.classifier_class(nb_states=self.num_states_slider.value, data=self.data,
                                                supervision_influence=self.supervision_slider.value,
                                                features=cur_features, buffer=self.buffer_slider.value)
        self.classifier.timestamp = cur_timestamp

    def load_custom_script(self, attr, old, new):
        raw_contents = new['file_contents'][0]
        # remove the prefix that JS adds
        _, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents).decode('utf-8')
        custom_fn = tempfile.NamedTemporaryFile(delete=False, suffix='.py').name
        with open(custom_fn, 'w') as fh: fh.write(file_contents)
        self._classifier_class = importlib.import_module(os.path.splitext(os.path.basename(custom_fn))[0], custom_fn).Classifier
        self.reload_classifier()

    def update_num_states(self, attr, old, new):
        if new != self.classifier.nb_states:

            # update classifier
            self.classifier.nb_states = new

            # Update widget: show-me checkboxes
            showme_idx = list(range(new))
            showme_states = [str(n) for n in range(1, new + 1)]
            # self.showme_checkboxes.labels = showme_states
            # self.showme_checkboxes.active = showme_idx
            self.saveme_checkboxes.labels = showme_states
            self.saveme_checkboxes.active = showme_idx

            self.alex_fret_checkboxes.labels = showme_states
            self.alex_fret_checkboxes.active = []

            # Update widget: selected state slider
            self.sel_state_slider.end = new
            if self.sel_state_slider.value > new: self.sel_state_slider.value = new

            # Reset all manual labels
            self.data.label_dict = dict()
            if len(self.data.manual_table):
                self.data.manual_table.is_labeled = False
                self.data.manual_table.loc[self.cur_trace_idx, 'is_labeled'] = True
                self.data.label_dict[self.cur_trace_idx] = np.zeros(len(self.current_example))
                blank_labels = [(i, 0) for i in range(len(self.current_example))]
                patch = {'labels': blank_labels,
                         'labels_pct': blank_labels}
                self.source.patch(patch)

    def export_data(self):
        self.fretReport = FretReport(self)

    def prediction_complete(self):
        timestamp_bool = np.logical_and(self.data.index_table.data_timestamp == self.data.data_timestamp,
                                        self.data.index_table.mod_timestamp == self.classifier.timestamp)
        return np.all(timestamp_bool)

    def generate_sspeaks(self):
        if not self.prediction_complete():
            self.notify('Please wait for prediction to finish before downloading labeled data...')
            return
        trace_dict = self.data.get_trace_dict()
        ssdf = pd.DataFrame(columns=['mu', 'sd', 'srsd', 'n_points'])
        self.notify('Generating FRET X peaks, please wait...')
        for fn in trace_dict:
            sm_bool = [True for sm in self.saveme_checkboxes.active if sm in trace_dict[fn].predicted]
            if not any(sm_bool): continue

            for lab in trace_dict[fn].predicted.unique():
                if lab == 0: continue
                efret = trace_dict[fn].loc[trace_dict[fn].predicted == lab,'E_FRET']
                # db_labels = DBSCAN(eps=0.005).fit_predict(efret.reshape(-1, 1))  # additional dbscan filter
                # efret = efret[db_labels == mode(db_labels)[0][0]]
                ssdf.loc[f'{fn}_{lab}'] = get_ssfret_dist(efret)
        self.ssfret_source.data['ssfret_text'] = [ssdf.to_csv(header=True, index=True, sep='\t')]
        self.ssfret_holder.text += ' '

    def write_nc(self, td, raw_fn, tfh):
        fc = self.data.get_raw(raw_fn)
        with io.BytesIO(fc) as fh:
            with h5py.File(fh, 'w') as h5f:
                h5f['FRETboard_classification'] = [td[tdi].predicted.to_numpy() for tdi in td]
            _ = fh.seek(0)
            with open(f'{tfh.name}/{raw_fn}', 'wb') as fh_out:
                fh_out.write(fh.read())

    def write_dats(self, td, tfh):
        for fn in td:
            if fn in self.data.label_dict:
                labels = self.data.label_dict[fn].astype(int) + 1
                sm_test = self.data.label_dict[fn]
            else:
                labels = [None] * len(td[fn])
                sm_test = td[fn].predicted
            sm_bool = [True for sm in self.saveme_checkboxes.active if sm in sm_test]
            if not any(sm_bool): continue
            td[fn].loc[:, 'label'] = labels
            td[fn].loc[:, 'predicted'] = td[fn].predicted.astype(int) + 1  # change to 1-based integers todo: set to integer upstream someday
            td[fn].to_csv(f'{tfh.name}/{fn}', sep='\t', na_rep='NA', index=False)

    def generate_dats(self):
        if not self.prediction_complete():
            self.notify('Please wait for prediction to finish before downloading labeled data...')
            return
        trace_dict = self.data.get_trace_dict()
        split_trace_dict = split_trace_dict_on_source(trace_dict)   # sort trace_dict keys on raw source
        tfh = tempfile.TemporaryDirectory()
        for raw_fn in split_trace_dict:
            if raw_fn.endswith('.nc'):
                self.write_nc(split_trace_dict[raw_fn], raw_fn, tfh)
            elif raw_fn == 'txt':
                self.write_dats(split_trace_dict[raw_fn], tfh)
            else:
                raise ValueError(f'exporting traces not supported for file {raw_fn}')  # should not happen
        zip_dir = tempfile.TemporaryDirectory()
        zip_fn = shutil.make_archive(f'{zip_dir.name}/dat_files', 'zip', tfh.name)
        with open(zip_fn, 'rb') as f:
            self.datzip_source.data['datzip'] = [base64.b64encode(f.read()).decode('ascii')]
        tfh.cleanup()
        zip_dir.cleanup()
        self.datzip_holder.text += ' '

    def guess_trigger(self, is_active):
        if is_active:
            self.notify('Select correctly labeled area...')

    def guess_labels(self, label_idx):
        ohmm = OneshotHmm(self.num_states_slider.value, self.current_example.loc[label_idx,
                                                                                 self.feature_list
                                                                                 # [f for i, f in enumerate(self.feature_list) if i in self.features_checkboxes.active],
        ],
                          self.data.label_dict[self.cur_trace_idx][label_idx])
        return ohmm.predict(self.current_example.loc[:, self.feature_list])


    def del_trace(self):
        # self.data.del_tuple(self.cur_trace_idx)
        self.data.manual_table.loc[self.cur_trace_idx,  'is_junk'] = True
        if self.cur_trace_idx in self.data.label_dict:
            del self.data.label_dict[self.cur_trace_idx]
        self.new_example()

    # def update_showme(self, attr, old, new):
    #     if old == new: return
    #     if not len(self.data.index_table): return
    #     valid_bool = self.data.data.apply(lambda x: any(i in new for i in x.prediction), axis=1)
    #     if any(valid_bool):
    #         valid_idx = list(self.data.data.index[valid_bool])
    #         if self.cur_trace_idx not in valid_idx:
    #             new_example_idx = np.random.choice(valid_idx)
    #             self.example_select.value = new_example_idx
    #     else:
    #         classes_not_found = ', '.join([str(ts + 1) for ts in new])
    #         self.notify(f'No valid (unclassified) traces to display for classes {classes_not_found}')

    def process_keystroke(self, attr, old, new):
        if not len(new): return
        self.keystroke_holder.text = ''
        if new == 'q': self.del_trace()
        elif new == 'w': self.train_trigger()
        elif new == 'e': self.new_example()

    def collect_samples_for_state(self, trace_dict, num_state):
        ldf_list = []
        for idx in trace_dict:
            ldf_list.append(trace_dict[idx].loc[self.data.label_dict[idx] == num_state - 1,
                                                ('f_aex_dem_raw', 'f_dex_dem_raw')])
        return pd.concat(ldf_list)

    def update_framerate(self, attr, old, new):
        if old == new: return
        self.data.framerate = new

    def estimate_crosstalk_params(self):

        # Input checks
        if not self.data.alex:
            self.notify('Cannot estimate l/d/gamma parameters without ALEX data!')
            return
        trace_dict = self.data.get_trace_dict(labeled_only=True)
        if self.d_only_state_spinner.value == self.a_only_state_spinner.value:
            self.notify('One state cannot be Donor-only and Acceptor-only simultaneously!')
            return
        if self.d_only_state_spinner.value -1 in self.alex_fret_checkboxes.active or \
                self.a_only_state_spinner.value - 1 in self.alex_fret_checkboxes.active:
            self.notify('D/A-only states cannot simulateously be a FRET state!')

        # Retrieve D and A-only state measurements
        ddf = self.collect_samples_for_state(trace_dict, self.d_only_state_spinner.value - 1)
        adf = self.collect_samples_for_state(trace_dict, self.a_only_state_spinner.value - 1)
        fret_list = [self.collect_samples_for_state(trace_dict, fs) for fs in self.alex_fret_checkboxes.active]
        if not len(ddf):
            self.notify('No points in labeled examples belonging to dedicated D-only state')
            return
        if not len(adf):
            self.notify('No points in labeled examples belonging to dedicated D-only state')
            return
        total_df = pd.concat(trace_dict.values())

        # leakage
        l = (ddf.f_aex_dem / ddf.f_dex_dem).mean()

        # direct excitation
        d = (adf.f_dex_aem / adf.f_aex_aem).mean()

        # gamma
        if len(self.alex_fret_checkboxes.active) < 2:
            self.notify('Skipping gamma estimation as it requires at least two FRET states.')
            gamma = 1.0
        else:
            total_df.loc[:, 'f_fret'] = total_df.f_dex_aem - l * total_df.f_dex_dem - d * total_df.f_aex_aem
            total_df.loc[:, 'f_sum'] = total_df.f_dex_dem + total_df.f_fret
            e_pr = (total_df.f_fret / (total_df.f_fret + total_df.f_dex_dem)).to_numpy()
            s_inv = ((total_df.f_sum + total_df.f_aex_aem)/ total_df.f_sum).to_numpy()
            s_mean = s_inv.mean(); s_sd = s_inv.std()
            e_mean = e_pr.mean(); e_sd = e_pr.std()
            sb = np.logical_and(np.logical_and(s_inv > s_mean - 2 * s_sd, s_inv < s_mean + 2 * s_sd),
                                np.logical_and(e_pr > e_mean - 2 * e_sd, e_pr < e_mean + 2 * e_sd))
            lm = linear_model.LinearRegression().fit(X=e_pr[sb].reshape(-1, 1), y=s_inv[sb].reshape(-1, 1))
            omega, sigma = lm.coef_[0, 0], lm.intercept_[0]
            gamma = (omega - 1) / (omega + sigma - 1)

        self.l_spinner.value = l
        self.d_spinner.value = d
        self.gamma_factor_spinner.value = gamma
        self.update_crosstalk()

    def make_document(self, doc):
        # --- Define widgets ---
        ff_title = Div(text=f"""<font size=15><b>FRETboard</b></font><br>v.{self.version}<hr>""",
                       width=280, height=90)

        # --- 1. Load ---
        load_button = Button(label='Data')
        load_button.js_on_click(CustomJS(args=dict(file_source=self.new_source, new_counter=self.new_tot), code=upload_js))

        load_model_button = Button(label='Model')
        load_model_button.js_on_click(CustomJS(args=dict(file_source=self.loaded_model_source), code=upload_model_js))
        revert_labels_button = Button(label='Undo changes')
        revert_labels_button.on_click(self.revert_manual_labels)

        example_data_button = Button(label='Example')
        example_data_button.on_click(self.load_example_data)


        # --- 2. Teach ---

        self.del_trace_button.on_click(self.del_trace)

        showme_states = [str(n + 1) for n in range(self.num_states_slider.value)]
        showme_idx = list(range(self.num_states_slider.value))

        # self.showme_checkboxes = CheckboxButtonGroup(labels=showme_states, active=showme_idx)
        # self.showme_checkboxes.on_change('active', self.update_showme)
        # showme_col = column(Div(text='Show traces with states:', width=300, height=16),
        #                     self.showme_checkboxes,
        #                     height=80, width=300)

        train_button = Button(label='Train (W)', button_type='warning')
        new_example_button = Button(label='New (E)', button_type='success')

        # --- 3. Save ---
        save_model_button = Button(label='Model')
        save_model_button.js_on_click(CustomJS(args=dict(file_source=self.classifier_source), code=download_csv_js))
        # save_model_button._callback = CustomJS(args=dict(file_source=self.classifier_source),
        #                                       code=download_csv_js)
        save_data_button = Button(label='Data')
        save_data_button.on_click(self.generate_dats)

        report_button = Button(label='Report')
        report_button.on_click(self.generate_report)

        ssfret_button = Button(label='Get FRET X table')
        ssfret_button.on_click(self.generate_sspeaks)

        self.saveme_checkboxes = CheckboxButtonGroup(labels=showme_states, active=showme_idx)
        saveme_col = column(Div(text='Save traces with states:', width=300, height=16),
                            self.saveme_checkboxes,
                            height=80, width=300)

        # --- Define plots ---

        # timeseries tools
        self.xbox_select = BoxSelectTool(dimensions='width')
        self.xwheel_zoom = WheelZoomTool(dimensions='width')
        self.xwheel_pan = WheelPanTool(dimension='width')
        self.pan = PanTool()
        tool_list = [self.xbox_select, self.xwheel_zoom, self.xwheel_pan, self.pan]

        # Main timeseries
        ts = figure(tools=tool_list, plot_width=1075, plot_height=275,
                    active_drag=self.xbox_select, name='ts', active_scroll=self.xwheel_zoom,
                    x_axis_label='Time (s)', y_axis_label='Intensity')
        ts.rect(x='time', y='rect_mid', width='rect_width', height='rect_height', fill_color={'field': 'labels_pct',
                                                                      'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts.line('time', 'f_dex_dem', color='#4daf4a', source=self.source, **line_opts)
        ts.line('time', 'f_dex_aem', color='#e41a1c', source=self.source, **line_opts)
        ts_panel = Panel(child=ts, title='Traces')

        self.x_range = ts.x_range
        self.y_range = ts.y_range
        self.ts_toolbar = ts.toolbar

        # E_FRET series
        ts_efret = figure(tools=tool_list, plot_width=1075, plot_height=275,
                          active_drag=self.xbox_select, x_range=ts.x_range, y_range=[0,1], name='ts',
                          x_axis_label='Time (s)', y_axis_label='FRET efficiency')  #  todo: add tooltips=[('$index')]
        ts_efret.rect('time', 0.5, height=1.0, width='rect_width', fill_color={'field': 'labels_pct',
                                                           'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts_efret.line('time', 'E_FRET', color='#1f78b4', source=self.source, **line_opts)
        ts_efret.line('time', 'E_FRET_sd', color='#a6cee3', source=self.source, **line_opts)
        ts_efret.toolbar = self.ts_toolbar
        efret_panel = Panel(child=ts_efret, title='E_FRET & sd')

        # correlation coeff series
        ts_corr = figure(tools=tool_list, plot_width=1075, plot_height=275,
                         active_drag=self.xbox_select, x_range=ts.x_range,
                         x_axis_label='Time (s)', y_axis_label='Corr. coef.')  # todo: add tooltips=[('$index')]
        ts_corr.toolbar = self.ts_toolbar
        ts_corr.rect('time', 0.0, height=2.0, width='rect_width', fill_color={'field': 'labels_pct',
                                                          'transform': self.col_mapper},
                      source=self.source, **rect_opts)
        ts_corr.line('time', 'correlation_coefficient', color='#b2df8a', source=self.source, **line_opts)
        corr_panel = Panel(child=ts_corr, title='Correlation coefficient')

        # i_sum series
        ts_i_sum = figure(tools=tool_list, plot_width=1075, plot_height=275,
                          active_drag=self.xbox_select, x_range=ts.x_range,
                          x_axis_label='Time (s)', y_axis_label='Intensity')
        ts_i_sum.rect('time', 'i_sum_mid', width='rect_width', height='i_sum_height', fill_color={'field': 'labels_pct',
                                                                              'transform': self.col_mapper},
                      source=self.source, **rect_opts)
        ts_i_sum.line('time', 'i_sum', color='#1f78b4', source=self.source, **line_opts)
        ts_i_sum.toolbar = self.ts_toolbar
        i_sum_panel = Panel(child=ts_i_sum, title='I sum')

        # ALEX series
        ts_alex = figure(tools=tool_list, plot_width=1075, plot_height=275,
                          active_drag=self.xbox_select, x_range=ts.x_range,
                          x_axis_label='Time (s)', y_axis_label='Intensity')
        ts_alex.rect(x='time', y='rect_mid', width='rect_width', height='rect_height', fill_color={'field': 'labels_pct',
                                                                                              'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts_alex.line('time', 'f_dex_dem', color='#4daf4a', source=self.source, **line_opts)
        ts_alex.line('time', 'f_dex_aem', color='#e41a1c', source=self.source, **line_opts)
        ts_alex.line('time', 'f_aex_dem', color='#b6edb4', source=self.source, **line_opts)
        ts_alex.line('time', 'f_aex_aem', color='#f0a5a6', source=self.source, **line_opts)
        alex_panel = Panel(child=ts_alex, title='ALEX')

        # manual and predicted timeseries
        # old tools: 'xbox_select,save,xwheel_zoom,xwheel_pan,pan'
        ts_manual = figure(tools=tool_list, plot_width=1075, plot_height=275,
                           active_drag=self.xbox_select, x_range=ts.x_range,
                           x_axis_label='Time (s)', y_axis_label='Intensity')
        ts_manual.rect('time', 'rect_mid_up', width='rect_width', height='rect_height_half', fill_color={'field': 'labels_pct',
                                                                         'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts_manual.rect('time', 'rect_mid_down', width='rect_width', height='rect_height_half', fill_color={'field': 'prediction_pct',
                                                                           'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts_manual.line('time', 'f_dex_dem', color='#4daf4a', source=self.source, **line_opts)
        ts_manual.line('time', 'f_dex_aem', color='#e41a1c', source=self.source, **line_opts)
        ts_manual.toolbar = self.ts_toolbar
        pred_vs_manual_panel = Panel(child=Column(column(ts_manual,revert_labels_button)), title='Predicted')

        # Additional settings panel
        self.alex_fret_checkboxes = CheckboxButtonGroup(labels=showme_states, active=[])
        alex_fret_col = column(Div(text='FRET states (for gamma estimation):', width=300, height=16),
                            self.alex_fret_checkboxes,
                            height=80, width=300)

        settings_panel = Panel(child=Column(row(
            column(
                Div(text='<b>Data format options</b>', height=15, width=200),
                row(Div(text='Frame rate .trace files (Hz): ', height=15, width=200),Column(self.framerate_spinner, width=75)),

                Div(text='<b>Filtering options</b>', height=15, width=200),
                row(Div(text='DBSCAN filter epsilon: ', height=15, width=200), Column(self.eps_spinner, width=75), Column(self.bg_button, width=65), width=500),
                Div(text='<b>ALEX (experimental!)</b>', height=15, width=200),
                row(Div(text='Load ALEX traces: ', height=15, width=130), Column(self.alex_checkbox, width=30),
                    Div(text='Switch order lasers: ', height=15, width=130), Column(self.traceswitch_checkbox, width=30), width=500),
                alex_fret_col,
                row(Div(text='<i>D</i>-only state: ', height=15, width=80),
                    Column(self.d_only_state_spinner, width=75),
                    Div(text='<i>A</i>-only state: ', height=15, width=80),
                    Column(self.a_only_state_spinner, width=75),
                    self.alex_estimate_button, width=200),
                row(
                    Div(text='gamma: ', height=15, width=80), Column(self.gamma_factor_spinner, width=75),
                    Div(text='<i>l</i>: ', height=15, width=35), Column(self.l_spinner, width=75),
                    Div(text='<i>d</i>: ', height=15, width=35), Column(self.d_spinner, width=75),
                    Column(self.alex_corr_button, height=15, width=30)),
                width=500),
            column(Div(text=' ', height=15, width=250), width=75),
            column(
                Div(text='<b>Training options</b>', height=15, width=200),
                self.supervision_slider,
                self.buffer_slider,
                row(Div(text='CI bootstrap iterations: ', height=15, width=200),
                    Column(self.bootstrap_size_spinner, width=100)),
                Div(text='<b>Miscellaneous</b>', height=15, width=200),
                row(Div(text='Remove last event before analysis: ', height=15, width=250), self.remove_last_checkbox, width=500),

                   ssfret_button,
                   self.keystroke_holder,
                width=500)
            , width=1075)), title='Settings')
        tabs = Tabs(tabs=[ts_panel, efret_panel, corr_panel, i_sum_panel, pred_vs_manual_panel, alex_panel, settings_panel])

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
        stats_text = column( row(Div(text='Accuracy (%): ', width=150, height=18), self.acc_text, height=18),
                             row(Div(text='Mean log-path probability: ', width=150, height=18), self.posterior_text, height=18),
                             row(Div(text='Manually classified (%): ', width=150, height=18), self.mc_text, height=18))

        # --- Define update behavior ---
        self.algo_select.on_change('value', self.update_algo)
        self.source.selected.on_change('indices', self.update_classification)  # for manual selection on trace
        # self.source.on_change('data', lambda attr,old, new: self.redraw_info_trigger())
        self.new_source.on_change('data', self.buffer_data)
        self.loaded_model_source.on_change('data', self.load_params)
        self.custom_script_source.on_change('data', self.load_custom_script)
        self.example_select.on_change('value', self.update_example)
        train_button.on_click(self.train_trigger)
        self.guess_toggle.on_click(self.guess_trigger)
        new_example_button.on_click(self.new_example)
        self.bg_checkbox.on_change('active', lambda attr, old, new: self.update_eps())
        self.bg_button.on_click(self.update_eps)
        self.alex_checkbox.on_change('active', self.update_alex_checkbox)
        self.traceswitch_checkbox.on_change('active', self.update_traceswitch_checkbox)
        self.framerate_spinner.on_change('value', self.update_framerate)
        self.alex_estimate_button.on_click(self.estimate_crosstalk_params)
        self.num_states_slider.on_change('value', self.update_num_states)
        self.alex_corr_button.on_click(self.update_crosstalk)
        self.state_radio.on_change('active', lambda attr, old, new: self.update_state_curves())
        self.features_checkboxes.on_change('active', self.update_feature_list)
        self.buffer_slider.on_change('value', self.update_buffer)

        # hidden holders
        self.report_holder.js_on_change('text', CustomJS(args=dict(file_source=self.html_source),
                                                         code=download_report_js))
        self.ssfret_holder.js_on_change('text', CustomJS(args=dict(file_source=self.ssfret_source),
                                                         code=download_ssfret_js))
        self.datzip_holder.js_on_change('text', CustomJS(args=dict(file_source=self.datzip_source),
                                                         code=download_datzip_js))
        self.custom_script_holder.js_on_change('text', CustomJS(args=dict(file_source=self.custom_script_source),
                                                             code=upload_custom_script_js))
        self.keystroke_holder.on_change('text', self.process_keystroke)

        # --- Build layout ---
        state_block = row(state_curves,
                          Div(text='<br />'.join(self.feature_list), margin=[32, 5, 5, 5]),
                          column(Div(text='View'), self.state_radio, width=40),
                          column(Div(text='Active'), self.features_checkboxes, width=40),
                          )
        widgets = column(ff_title,
                         Div(text="<font size=4>1. Load</font>", width=280, height=15),
                         self.algo_select,
                         self.num_states_slider,
                         row(Column(load_model_button, width=150), Column(load_button, width=150),
                             width=300),
                         row(Div(text="DBSCAN background subtraction ", width=250, height=15),
                             Column(self.bg_checkbox, width=25), width=300),
                         Div(text="<font size=4>2. Teach</font>", width=280, height=15),
                         # row(Div(text='Change selection to state: '),
                         #     self.sel_state, width=300, height=15),
                         self.sel_state_slider,
                         # showme_col,
                         row(Column(self.del_trace_button, width=100), Column(train_button, width=100), Column(new_example_button, width=100)),
                         self.guess_toggle,
                         Div(text="<font size=4>3. Save</font>", width=280, height=15),
                         saveme_col,
                         row(Column(report_button, width=100), Column(save_data_button, width=100), Column(save_model_button, width=100)),
                         Div(text='', height=30),
                         Column(example_data_button, width=100),
                         width=300)
        hists = row(acc_hist, logprob_hist, state_block)
        graphs = column(self.example_select,
                        tabs,
                        hists,
                        stats_text,
                        Div(text='Notifications: ', width=100, height=15),
                        self.notification,
                        self.report_holder, self.datzip_holder, self.ssfret_holder, self.custom_script_holder)
        layout = row(widgets, graphs)
        doc.add_root(layout)
        doc.title = f'FRETboard v. {self.version}'
        doc.on_session_destroyed(self.shutdown_gracefully)
        doc.template = template
        self.doc = doc

    def create_gui(self):
        self.make_document(curdoc())

    def start_predictor(self):
        pred = Predictor(self.classifier, h5_dir=self.h5_dir, main_process=self.main_thread)
        pred.run()

    def loop_update(self):
        # while self.main_thread.is_alive():
        while self.app_is_up and self.main_thread.is_alive():
            self.data.update_index()
            if len(self.data.index_table) and self.model_loaded:
                self.update_example_list()
                self.redraw_info_trigger()
            if not self.model_loaded and len(self.data.index_table):
                self.train_trigger()
            if self.cur_trace_idx is None:
                if np.any(self.data.index_table.mod_timestamp == self.classifier.timestamp):
                    self.cur_trace_idx = self.data.index_table.index[self.data.index_table.mod_timestamp == self.classifier.timestamp][0]
                    self.new_example_trigger()
            Event().wait(1.0)
            # if self.server.get_sessions()[0].connection_count < 1:
        self.shutdown_gracefully(None)
        sys.exit(0)

    def start_threads(self):
        # Start background threads for prediction
        self.main_thread = threading.main_thread()
        self.loop_thread = Thread(target=self.loop_update, name='loop')
        self.loop_thread.start()
        self.pred_processes = []
        for p in range(self.nb_processes - 2):
            pred = Predictor(self.classifier, h5_dir=self.h5_dir)
            pred_process = Process(target=pred.run, name=f'predictor_{p}')
            pred_process.start()
            self.pred_processes.append(pred_process)

    def shutdown_gracefully(self, session_context):
        self.app_is_up = False
        for pp in self.pred_processes:
            pp.terminate()
            pp.join()
        self.file_parser_process.terminate()
        self.file_parser_process.join()
        # self.loop.stop()

        # sys.exit(0)

    def start_ioloop(self, port=4237):
        app = Application(FunctionHandler(self.make_document))
        apps = {'/': app}
        server = Server(apps, port=port, websocket_max_message_size=100000000)
        server.show('/')
        self.loop = IOLoop.current()
        self.loop.start()
