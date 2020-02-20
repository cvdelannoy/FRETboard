import os
import sys
import tempfile
import shutil
import base64
import numpy as np
import pandas as pd
import yaml
import importlib
import traceback
from multiprocessing import Process
# from tornado import gen
import threading
from threading import Thread, Event
import asyncio
from time import sleep
from joblib import Parallel
from sklearn import linear_model

from cached_property import cached_property
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import row, column, widgetbox
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.models.tools import BoxSelectTool, WheelZoomTool, WheelPanTool, SaveTool, PanTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Slider, Select, Button, PreText, RadioGroup, Div, CheckboxButtonGroup, CheckboxGroup, Spinner, TextInput
from bokeh.models.widgets.panels import Panel, Tabs
from tornado.ioloop import IOLoop
from tornado import gen

from FRETboard.io_functions import parse_trace_file
from FRETboard.helper_functions import print_timestamp, series_to_array
from FRETboard.FretReport import FretReport
from FRETboard.MainTable import MainTable

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
line_opts = dict(line_width=1)
rect_opts = dict(alpha=1, line_alpha=0)
with open(f'{__location__}/algorithms.yml', 'r') as fh: algo_dict = yaml.safe_load(fh)
algo_dict['custom'] = 'custom'
algo_inv_dict = {algo_dict[k]: k for k in algo_dict}
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


def installThreadExcepthook():
    """
    Workaround for sys.excepthook thread bug
    From
http://spyced.blogspot.com/2007/06/workaround-for-sysexcepthook-bug.html

(https://sourceforge.net/tracker/?func=detail&atid=105470&aid=1230540&group_id=5470).
    Call once from __main__ before creating any threads.
    If using psyco, call psyco.cannotcompile(threading.Thread.run)
    since this replaces a new-style class method.
    """
    init_old = threading.Thread.__init__

    def init(self, *args, **kwargs):
        init_old(self, *args, **kwargs)
        run_old = self.run

        def run_with_except_hook(*args, **kw):
            try:
                run_old(*args, **kw)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init

class Gui(object):
    def __init__(self, nb_states=3, data=[]):
        installThreadExcepthook()
        # sys.excepthook = self.error_logging
        self.version = '0.0.3'
        self.cur_example_idx = None
        self.nb_threads = 8
        self.feature_list = ['E_FRET', 'E_FRET_sd', 'i_sum', 'i_sum_sd', 'correlation_coefficient', 'i_don', 'i_acc']
        self.data = MainTable(data, np.nan, 0.0, 0.0, 1.0)

        # Buffer stores and triggers (for concurrency)
        self.notification_buffer = ''
        self.model_buffer = ''
        self.classifier_source_buffer = None
        self.fn_buffer = []
        self.new_data_buffer = []
        self.params_changed = False
        self.loading_in_progress = False
        self.training_in_progress = False
        self.prediction_in_progress = False
        self.eps_change_in_progress = False
        self.redraw_activated = False
        self.buffer_updated = False
        self.user_notified_of_training = False
        self.predict_error_count = 0


        # --- Widgets ---

        # 1. Load
        self.algo_select = Select(title='Algorithm:', value=list(algo_dict)[0], options=list(algo_dict))
        self.num_states_slider = Slider(title='Number of states', value=nb_states, start=2, end=10, step=1,
                                        name='num_states_slider')
        self.bg_checkbox = CheckboxGroup(labels=[''], active=[])

        # 2. Train
        self.sel_state_slider = Slider(title='Change selection to state  (num keys)', value=1, start=1,
                                       end=self.num_states_slider.value, step=1, name='sel_state_slider')
        self.sel_state = Div(text='1', name='sel_state')
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
        self.keystroke_holder = PreText(text='', css_classes=['hidden'], name='keystroke_holder')
        self.features_checkboxes = CheckboxGroup(labels=[''] * len(self.feature_list), active=[0, 1, 2, 3, 4])
        self.state_radio = RadioGroup(labels=[''] * len(self.feature_list), active=0)

        # Settings
        self.eps_spinner = Spinner(value=15, step=1)
        self.bg_button = Button(label='Apply')
        self.framerate_spinner = Spinner(value=10, step=1)
        self.remove_last_checkbox = CheckboxGroup(labels=[''], active=[])
        self.supervision_slider = Slider(title='Influence supervision', value=1.0, start=0.0, end=1.0, step=0.01)
        self.buffer_slider = Slider(title='Buffer', value=3, start=0, end=20, step=1)
        self.bootstrap_size_spinner = Spinner(value=10, step=1)
        self.alex_checkbox = CheckboxGroup(labels=[''], active=[])
        self.gamma_factor_spinner = Spinner(value=1.0, step=0.001)
        self.l_spinner = Spinner(value=0.0, step=0.001)
        self.d_spinner = Spinner(value=0.0, step=0.001)
        self.d_only_state_spinner = Spinner(value=1, step=1)
        self.alex_corr_button = Button(label='Apply')
        self.alex_estimate_button = Button(label='Estimate parameters')


        # Classifier object
        self.classifier_class = self.algo_select.value
        self.classifier = self.classifier_class(nb_states=nb_states, data=self.data, gui=self,
                                                features=[feat for fi, feat in enumerate(self.feature_list)
                                                          if fi in self.features_checkboxes.active])

        # ColumnDataSources
        self.source = ColumnDataSource(data=dict(i_don=[], i_acc=[], E_FRET=[],
                                                 correlation_coefficient=[], E_FRET_sd=[], i_sum=[], time=[],
                                                 rect_height=[], rect_height_half=[], rect_mid=[], rect_mid_up=[], rect_mid_down=[], rect_width=[],
                                                 i_sum_height=[], i_sum_mid=[],
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
        self.classifier_source = ColumnDataSource(data=dict(params=[]))
        self.html_source = ColumnDataSource(data=dict(html_text=[]))
        self.datzip_source = ColumnDataSource(data=dict(datzip=[]))
        self.scroll_state_source = ColumnDataSource(data=dict(new_state=[False]))
        self.model_loaded = False
        self.new_tot = ColumnDataSource(data=dict(value=[0]))
        self.new_cur = 0

        # temp dirs
        self.data_load_dir = tempfile.TemporaryDirectory()
        self.predict_dir = tempfile.TemporaryDirectory()


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
    def classifier_class(self):
        return self._classifier_class

    @classifier_class.setter
    def classifier_class(self, class_name):
        self._classifier_class = importlib.import_module('FRETboard.algorithms.' + algo_dict.get(class_name, class_name)).Classifier


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
        self.notify('Start training...')
        self.doc.add_next_tick_callback(self.train)

    def new_example_trigger(self):
        self.doc.add_next_tick_callback(self.update_example_fun)

    def update_example_fun(self):
        self.example_select.value = self.cur_example_idx

    def redraw_trigger(self):
        if not self.redraw_activated:
            self.redraw_activated = True
            self.doc.add_next_tick_callback(self._redraw_all)

    def redraw_info_trigger(self):
        self.doc.add_next_tick_callback(self._redraw_info)

    def append_fn(self, fn):
        self.fn_buffer.append(fn)
        self.doc.add_next_tick_callback(self.push_fn)

    def push_fn(self):
        self.example_select.options = self.example_select.options + [self.fn_buffer.pop()]

    def push_notification(self):
        self.notification.text = self.notification_buffer + self.notification.text

    # --- asynchronous functions ---
    def update_data(self):
        """
        Update MainTable with new provided examples
        """
        while self.main_thread.is_alive():
            if not len(self.new_data_buffer):
                Event().wait(0.01)
                continue
            self.loading_in_progress = True
            if self.new_cur == 0:
                self.notify('Loading data in background...')
            raw_contents, fn = self.new_data_buffer.pop()
            _, b64_contents = raw_contents.split(",", 1)  # remove the prefix that JS adds
            file_contents = base64.b64decode(b64_contents)
            if '.traces' in fn:

                self.notify(f'processing {fn}...')

                nb_colors = 2
                nb_frames, _, nb_traces = np.frombuffer(file_contents, dtype=np.int16, count=3)
                nb_samples = nb_traces // nb_colors
                traces_vec = np.frombuffer(file_contents, dtype=np.int16)
                traces_vec = traces_vec[3:]
                nb_points_expected = nb_colors * nb_samples * nb_frames
                traces_vec = traces_vec[:nb_points_expected]
                file_contents = traces_vec.reshape((nb_colors, nb_samples, nb_frames), order='F')
                fn_clean = os.path.splitext(fn)[0]

                fn_list = [f'{fn_clean}_{it}.dat' for it in range(nb_samples)]
                print_thres = 10
                sampling_freq = 1.0 / self.framerate_spinner.value
                for fi, f in enumerate(np.hsplit(file_contents, file_contents.shape[1])):
                    f = f.squeeze()
                    time = np.arange(f.shape[1]) * sampling_freq
                    self.data.add_tuple(np.row_stack((time, f)), fn_list[fi])
                    self.append_fn(fn_list[fi])
                    pct_done = int(fi / nb_samples * 100)
                    if pct_done > print_thres:
                        self.notify(f'{fi} ({pct_done} %) traces from {fn_clean} loaded')
                        print_thres += 10
                    if not self.model_loaded and not self.training_in_progress:
                        self.train_trigger()
                    elif self.cur_example_idx is None:
                        self.doc.add_next_tick_callback(self._redraw_all)
                        self.cur_example_idx = f'{fn_clean}_0.dat'
                self.notify('Done')
                # df_list = parse_trace_file(file_contents, fn, self.nb_threads, self.parallel_pool)
                # self.data.add_df_list(df_list)

            # Process .dat files
            elif '.dat' in fn:
                # self.doc.add_next_tick_callback(self.update_notification(f'{print_timestamp()}Adding to list'))
                file_contents = base64.b64decode(b64_contents).decode('utf-8')
                file_contents = np.column_stack([np.fromstring(n, sep=' ') for n in file_contents.split('\n') if len(n)])
                if len(file_contents):
                    self.data.add_tuple(file_contents, fn,
                                        gamma=self.gamma_factor_spinner.value, l=self.l_spinner.value, d=self.d_spinner.value)
                    self.append_fn(fn)
                    # Already start training if no previous model exists and more than 100 traces are loaded
                    if not self.model_loaded and not self.training_in_progress:
                        self.train_trigger()
                    elif self.cur_example_idx is None:
                        self.cur_example_idx = fn
                        self.new_example_trigger()
            self.new_cur += 1

            if self.new_tot.data['value'][0] == self.new_cur:
                self.new_cur = 0
                self.notify(f'All data loaded')
                self.loading_in_progress = False

    def subtract_background(self):
        """
        Subtract background continuously, when:
        - Data is available for bg subtraction
        :return:
        """
        while self.main_thread.is_alive():
            if not len(self.data.data): continue
            idx_list = self.data.data.index
            eps_bool = np.nan_to_num(self.data.data.loc[idx_list, 'eps']) != np.nan_to_num(self.data.eps)
            l_bool = self.data.data.loc[idx_list, 'l'] != self.data.l
            d_bool = self.data.data.loc[idx_list, 'd'] != self.data.d
            gamma_bool = self.data.data.loc[idx_list, 'gamma'] != self.data.gamma
            sb_indices = idx_list[np.logical_or.reduce((eps_bool, l_bool, d_bool, gamma_bool))]
            if not len(sb_indices):
                Event().wait(0.01)
                continue
            if self.cur_example_idx in sb_indices:
                idx = self.cur_example_idx
            else:
                idx = sb_indices[0]
            self.data.subtract_background(idx)
            if idx == self.cur_example_idx:
                self.redraw_trigger()

    def predict(self):
        while self.main_thread.is_alive():
            try:
                pred_success = self.pred_fun()
                if not pred_success:
                    self.prediction_in_progress = False
                    Event().wait(timeout=0.01)
            except Exception as e:
                self.prediction_in_progress = False
                self.notify_exception(e, traceback.format_exc(), 'prediction')
                self.predict_error_count += 1
                if self.predict_error_count > 10:
                    return
                continue

    def pred_fun(self):
        """
        predict valid examples continuously, when:
        - Model is loaded
        - Not in training
        - to-be predicted examples are present:
            - Is not marked as junk (in self.data_clean)
            - Has proper background subtraction applied to it (in self.data_clean)
            - has not been predicted yet: is_predicted is False
        """
        if not self.model_loaded:
            return False
        if self.params_changed or self.eps_change_in_progress:
            if not self.user_notified_of_training:
                self.notify('Parameters or classification changed, pausing prediction...')
                self.user_notified_of_training = True
            return False
        if self.training_in_progress:
            return False
        if self.cur_example_idx is None:
            return False
        self.prediction_in_progress = True
        is_predicted_series = self.data.data_clean.is_predicted.copy().astype(bool)
        if not len(is_predicted_series):
            return False
        not_predicted_indices = is_predicted_series.index[np.invert(is_predicted_series)]
        if not len(not_predicted_indices):
            self.prediction_in_progress = False
            return False
        if self.cur_example_idx in not_predicted_indices:
            idx = self.cur_example_idx
        else:
            idx = not_predicted_indices[0]
        pred_list, logprob = self.classifier.predict(idx)
        if len(self.remove_last_checkbox.active):
            event_found = False
            for i in range(len(pred_list)):
                if pred_list[-i] != 0:
                    event_found = True
                elif event_found:
                    break
                pred_list[-i] = 0
        self.data.data.at[idx, 'prediction'] = np.array(pred_list)
        self.data.data.loc[idx, 'logprob'] = logprob
        self.data.data.loc[idx, 'is_predicted'] = True
        if idx == self.cur_example_idx:
            if not len(self.data.data.at[self.cur_example_idx, 'labels']):
                self.data.data.at[self.cur_example_idx, 'labels'] = np.array(pred_list)
            self.redraw_trigger()
        else:
            self.redraw_info_trigger()
        self.prediction_in_progress = False
        return True

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

    def train(self):
        try:
            # Ensure training does not clash with prediction
            self.training_in_progress = True
            while self.prediction_in_progress:
                Event().wait(0.01)
            if self.buffer_updated:
                for idx in self.data.data.loc[self.data.data.is_labeled].index:
                    self.data.set_value(idx, 'edge_labels', self.get_edge_labels(self.data.data.loc[idx, 'labels']))
            self.classifier.train(supervision_influence=self.supervision_slider.value)
            self.data.data.is_predicted = False
            self.model_loaded = True
            self.params_changed = False
            self.training_in_progress = False
            if self.cur_example_idx is None:
                self.example_select.value = self.data.data_clean.index[0]
            else:
                self.redraw_trigger()
            self.classifier_source.data = dict(params=[self.classifier.get_params()])
            self.notify('Finished training')
        except Exception as e:
            self.notify_exception(e, traceback.format_exc(), 'training')

    def new_example(self):
        if all(self.data.data.is_labeled):
            self.notify('All examples have already been manually classified')
        else:
            sidx = self.data.data.index.copy()
            sm_check = self.data.data.loc[sidx, 'prediction'].apply(lambda x:
                                                                    True if len(x) == 0 or any(
                                                                        np.in1d(self.showme_checkboxes.active, x))
                                                                    else False)
            valid_idx = self.data.data.index[np.logical_and(sm_check, np.invert(self.data.data.loc[sidx, 'is_labeled']))]
            if not len(valid_idx):
                self.notify('No new traces with states of interest left')
                return
            # new_example_idx = np.random.choice(self.data.data.loc[sidx].loc[valid_bool].index)
            if np.all(np.isnan(self.data.data.loc[valid_idx, 'logprob'])):
                self.notify('No valid unpredicted traces left. Try to train before choosing next example.')
                return
            new_example_idx = self.data.data.loc[valid_idx, 'logprob'].idxmin()
            self.example_select.value = new_example_idx

    def load_params(self, attr, old, new):
        self.params_changed = True
        raw_contents = self.loaded_model_source.data['file_contents'][0]
        # remove the prefix that JS adds
        _, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents).decode('utf-8')
        # file_contents = base64.b64decode(b64_contents).decode('utf-8').split('\n')[1:-1]
        self.algo_select.value = algo_inv_dict.get(file_contents.split('\n')[0], 'custom')
        self.classifier.load_params(file_contents)
        self.num_states_slider.value = self.classifier.nb_states
        if np.isnan(self.data.eps):
            self.bg_checkbox.active = []
        else:
            self.eps_spinner.value = self.data.eps
            self.bg_checkbox.active = [0]
        self.model_loaded = True
        self.notify('Model loaded')
        self.params_changed = False
        if self.data.data.shape[0] != 0:
            self.data.data.is_labeled = False
            self.redraw_trigger()
            # self._redraw_all()

    def buffer_data(self, attr, old, new):
        self.new_data_buffer.append((self.new_source.data['file_contents'][0], self.new_source.data['file_name'][0]))


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

    def update_buffer(self, attr, old, new):
        if old == new: return
        self.buffer_updated = True

    def update_example(self, attr, old, new):
        """
        Update the example currently on the screen.
        """
        if old == new:
            return
        self.cur_example_idx = new
        if old == 'None':
            new_list = self.example_select.options
            new_list.remove('None')
            self.example_select.options = new_list
        if not self.data.data.loc[new, 'is_labeled']:
            if not len(self.data.data.loc[new, 'prediction']) and not self.params_changed:
                # case 1: example never predicted before, and prediction is still in progress --> wait, next prediction will be of self.cur_example_idx
                self.notify('Predicting current example...')
                while not self.data.data.loc[new, 'is_predicted']:
                    Event().wait(0.01)
            elif not len(self.data.data.loc[new, 'prediction']) and self.params_changed:
                # case 2: example never predicted before, params changed thus prediction halted --> fill in with zeros for now
                self.data.set_value(new, 'prediction', np.zeros(self.data.data.loc[new, 'i_don'].shape[0], dtype=np.int64))
                self.data.data.loc[new, 'is_predicted'] = True
            self.data.set_value(new, 'labels', self.data.data.loc[new, 'prediction'].copy())
            self.data.set_value(new, 'edge_labels', self.get_edge_labels(self.data.data.loc[new, 'labels']))
            self.data.set_value(new, 'is_labeled', True)
        if self.data.data.loc[new, 'marked_junk']:
            self.notify(f'Warning: {new} was marked junk!')
        elif self.data.data.loc[new, 'predicted_junk']:
            self.notify(f'Warning: {new} is predicted junk!')
        self.redraw_trigger()
        # self._redraw_all()

    def _redraw_all(self):
        self.redraw_activated = False
        self.invalidate_cached_properties()
        if not self.data.data.loc[self.cur_example_idx, 'is_labeled']:
            while not self.data.data.loc[self.cur_example_idx, 'is_predicted']: Event().wait(0.01)
            self.data.data.at[self.cur_example_idx, 'labels'] = self.data.data.loc[self.cur_example_idx, 'prediction']
            self.data.data.loc[self.cur_example_idx, 'is_labeled'] = True
        nb_samples = self.i_don.size
        all_ts = np.concatenate((self.i_don, self.i_acc))  # todo: i_acc shape != i_don shape???
        ts_range = all_ts.max() - all_ts.min()
        rect_mid = (all_ts.max() + all_ts.min()) / 2
        rect_mid_up = all_ts.min() + ts_range * 0.75
        rect_mid_down = all_ts.min() + ts_range * 0.25
        rect_height = np.abs(all_ts.max()) + np.abs(all_ts.min())
        rect_height_half = rect_height / 2
        rect_width = (self.time[1] - self.time[0]) * 1.01
        try:
            self.source.data = dict(i_don=self.i_don, i_acc=self.i_acc, time=self.time,
                                    E_FRET=self.E_FRET, correlation_coefficient=self.correlation_coefficient, i_sum=self.i_sum,
                                    E_FRET_sd=self.E_FRET_sd,
                                    rect_height=np.repeat(rect_height, nb_samples),
                                    rect_height_half=np.repeat(rect_height_half, nb_samples),
                                    rect_mid=np.repeat(rect_mid, nb_samples),
                                    rect_mid_up=np.repeat(rect_mid_up, nb_samples),
                                    rect_mid_down=np.repeat(rect_mid_down, nb_samples),
                                    rect_width=np.repeat(rect_width, nb_samples),
                                    i_sum_height=np.repeat(self.i_sum.max(), nb_samples),
                                    i_sum_mid=np.repeat(self.i_sum.mean(), nb_samples),
                                    labels=self.data.data.loc[self.cur_example_idx, 'labels'],
                                    labels_pct=self.data.data.loc[self.cur_example_idx, 'labels'] / (self.num_states_slider.value - 1),
                                    prediction_pct=self.data.data.loc[self.cur_example_idx, 'prediction'] / (self.num_states_slider.value - 1))
        except:
            cp=1
            raise
        self.x_range.start = 0; self.x_range.end = self.time.max()
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

    def generate_report(self):
        if self.loading_in_progress:
            self.notify('Please wait for data loading to finish before generating a report')
            return
        if not np.all(self.data.data_clean.is_predicted):
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
        if len(new):
            self.params_changed = True
            self.source.selected.indices = []
            patch = {'labels': [(i, self.sel_state_slider.value - 1) for i in new],
                     'labels_pct': [(i, (self.sel_state_slider.value - 1) * 1.0 / (self.num_states_slider.value - 1))
                                    for i in new]}
            # patch = {'labels': [(i, int(self.sel_state.text) - 1) for i in new],
            #          'labels_pct': [(i, (int(self.sel_state.text) - 1) * 1.0 / (self.num_states_slider.value - 1)) for i in new]}
            if not self.data.data.loc[self.cur_example_idx, 'is_labeled']:
                self.data.data.at[self.cur_example_idx, 'labels'] = self.source.data['labels']
                self.data.data.loc[self.cur_example_idx, 'is_labeled'] = True
            self.source.patch(patch)
            self.update_accuracy_hist()
            self.update_stats_text()

            # update data in main table
            self.data.set_value(self.cur_example_idx, 'labels', self.source.data['labels'])
            self.data.set_value(self.cur_example_idx, 'edge_labels', self.get_edge_labels(self.source.data['labels']))

    def revert_manual_labels(self):
        patch = {'labels': [(i, v) for i, v in enumerate(self.data.data.loc[self.cur_example_idx, 'prediction']) ],
                 'labels_pct': [(i, v) for i, v in enumerate(self.data.data.loc[self.cur_example_idx, 'prediction'] * 1.0 / (self.num_states_slider.value - 1))]}
        self.source.patch(patch)
        self.update_accuracy_hist()
        self.update_stats_text()

        # update data in main table
        self.data.set_value(self.cur_example_idx, 'labels', self.source.data['labels'])
        self.data.set_value(self.cur_example_idx, 'edge_labels', self.get_edge_labels(self.source.data['labels']))

    def update_eps(self):
        self.eps_change_in_progress = True
        while self.prediction_in_progress:
            Event().wait(0.01)
        if not len(self.bg_checkbox.active):
            self.data.eps = np.nan
        else:
            self.data.eps = self.eps_spinner.value
        self.eps_change_in_progress = False

    def update_crosstalk(self):
        self.eps_change_in_progress = True
        while self.prediction_in_progress:
            Event().wait(0.01)
        self.data.l = self.l_spinner.value
        self.data.d = self.d_spinner.value
        self.data.gamma = self.gamma_factor_spinner.value
        self.eps_change_in_progress = False

    def update_accuracy_hist(self):
        acc_counts = np.histogram(self.data.accuracy[0], bins=np.linspace(5, 100, num=20))[0]
        self.accuracy_source.data['accuracy_counts'] = acc_counts

    def update_logprob_hist(self):
        counts, edges = np.histogram(self.data.data.logprob.loc[self.data.data.logprob.notna()], bins=20)
        self.logprob_source.data = {'logprob_counts': counts, 'lb': edges[:-1], 'rb': edges[1:]}

    def update_stats_text(self):
        pct_labeled = round(self.data.data_clean.is_labeled.sum() /
                            max(self.data.data_clean.is_labeled.size, 1) * 100.0, 1)
        if pct_labeled == 0:
            acc = 'N/A'
            post = 'N/A'
        else:
            acc = round(self.data.accuracy[1], 1)
            post = round(self.data.data_clean.logprob.mean(), 1)
        self.acc_text.text = f'{acc}'
        self.posterior_text.text = f'{post}'
        self.mc_text.text = f'{pct_labeled}'

    def update_feature_list(self, attr, old, new):
        if len(new) == 0: return
        if old == new: return
        self.params_changed = True
        while self.prediction_in_progress: Event().wait(0.01)
        self.classifier.feature_list = [feat for fi, feat in enumerate(self.feature_list) if fi in new]
        if len(self.data.data):
            self.data.data.is_predicted = False
            # self.train_trigger()
            # self.update_example(None, None, self.cur_example_idx)

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
        sdps = [-(xs[0] - mu) ** 2 / (2 * sd ** 2) for mu, sd in zip (mus, sds)]
        ys = []
        for sd, sdp in zip(sds, sdps):
            if any(sdp < -708):
                ys.append(np.zeros(len(sdp)))
            else:
                ys.append(1 / (sd * np.sqrt(2 * np.pi)) * np.exp(sdp))
        self.state_source.data = {'ys': ys, 'xs': xs, 'color': self.curve_colors}

    def update_algo(self, attr, old, new):
        if old == new:
            return
        algo = algo_dict.get(new, new)
        self.params_changed = True
        self.classifier_class = algo
        self.classifier = self.classifier_class(nb_states=self.num_states_slider.value, data=self.data, gui=self,
                                                features=self.feature_list)
        if len(self.data.data):
            # self.train_trigger()
            self.update_example(None, None, self.cur_example_idx)

    def update_num_states(self, attr, old, new):
        if new != self.classifier.nb_states:
            self.params_changed = True
            while self.prediction_in_progress: Event().wait(0.01)
            self.classifier = self.classifier_class(nb_states=new, data=self.data, gui=self, features=self.feature_list)

            # Update widget: show-me checkboxes
            showme_idx = list(range(new))
            showme_states = [str(n) for n in range(1, new + 1)]
            self.showme_checkboxes.labels = showme_states
            self.showme_checkboxes.active = showme_idx
            self.saveme_checkboxes.labels = showme_states
            self.saveme_checkboxes.active = showme_idx

            # Update widget: selected state slider
            self.sel_state_slider.end = new
            # if int(self.sel_state.text) > new: self.sel_state.text = str(new)
            if self.sel_state_slider.value > new: self.sel_state_slider.value = new

            if len(self.data.data):
                self.data.data.is_labeled = False
                self.data.data.labels = [[]] * len(self.data.data)
                blank_labels = [(i, 0) for i in range(len(self.data.data.loc[self.cur_example_idx, 'i_don']))]
                patch = {'labels': blank_labels,
                         'labels_pct': blank_labels}
                self.source.patch(patch)

            # self.train_trigger()

            # # retraining is too heavy for longer traces, setting current example to lowest state instead
            # blank_labels = [(i, 0) for i in range(len(self.data.data.loc[self.cur_example_idx, 'i_don']))]
            # patch = {'labels': blank_labels,
            #          'labels_pct': blank_labels}
            # self.source.patch(patch)
            # self.source.selected.indices = []
            # self.update_accuracy_hist()
            # self.update_stats_text()
            #
            # # update data in main table
            # self.data.set_value(self.cur_example_idx, 'labels', self.source.data['labels'])
            # self.data.set_value(self.cur_example_idx, 'edge_labels', self.get_edge_labels(self.source.data['labels']))
            # self.data.set_value(self.cur_example_idx, 'is_labeled', True)

    def export_data(self):
        self.fretReport = FretReport(self)

    def generate_dats(self):
        if (len(self.data.data) - self.data.is_junk.sum() != len(self.data.data_clean)):
            self.notify('Please wait for prediction to finish before downloading labeled data...')
            return
        tfh = tempfile.TemporaryDirectory()
        for fn, tup in self.data.data_clean.iterrows():
            # try:
                sm_test = tup.labels if len(tup.labels) else tup.prediction
                sm_bool = [True for sm in self.saveme_checkboxes.active if sm in sm_test]
                if not any(sm_bool): continue
                labels = tup.labels + 1 if len(tup.labels) != 0 else [None] * len(tup.time)
                out_df = pd.DataFrame(dict(time=tup.time, i_don=tup.i_don, i_acc=tup.i_acc,
                                           label=labels, predicted=tup.prediction + 1))
                out_df.to_csv(f'{tfh.name}/{fn}', sep='\t', na_rep='NA', index=False)
            # except:
            #     cp=1
            #     pass
        zip_dir = tempfile.TemporaryDirectory()
        zip_fn = shutil.make_archive(f'{zip_dir.name}/dat_files', 'zip', tfh.name)
        with open(zip_fn, 'rb') as f:
            self.datzip_source.data['datzip'] = [base64.b64encode(f.read()).decode('ascii')]
        tfh.cleanup()
        zip_dir.cleanup()
        self.datzip_holder.text += ' '

    def del_trace(self):
        self.data.del_tuple(self.cur_example_idx)
        self.new_example()
        # nonjunk_bool = np.logical_and(self.data.data.is_labeled, np.invert(self.data.data.is_junk))
        # if any(nonjunk_bool):  # Cant predict without positive examples
        #     df = self.data.data.loc[self.data.data.is_labeled]
        #     x = np.stack(df.E_FRET.to_numpy())
        #     predicted_junk = lsh_classify(x, self.data.data.is_junk, x, bits=32)
        #     self.data.data.predicted_junk = np.logical_or(self.data.data.junk, predicted_junk)
        #     # self.example_select.options.remove(self.cur_example_idx)

    def update_showme(self, attr, old, new):
        if old == new: return
        if self.data.data.shape[0] == 0: return
        valid_bool = self.data.data.apply(lambda x: any(i in new for i in x.prediction), axis=1)
        if any(valid_bool):
            valid_idx = list(self.data.data.index[valid_bool])
            if self.cur_example_idx not in valid_idx:
                new_example_idx = np.random.choice(valid_idx)
                self.example_select.value = new_example_idx
        else:
            classes_not_found = ', '.join([str(ts + 1) for ts in new])
            self.notify(f'No valid (unclassified) traces to display for classes {classes_not_found}')

    def process_keystroke(self, attr, old, new):
        if not len(new): return
        self.keystroke_holder.text = ''
        if new == 'q': self.del_trace()
        elif new == 'w': self.train_trigger()
        elif new == 'e': self.new_example()

    def estimate_crosstalk_params(self):

        labeled_data = self.data.data.loc[self.data.data.labels.apply(lambda x: len(x) != 0), :]

        # leakage
        l_df = labeled_data.apply(lambda x: x.i_acc_raw[x.labels == self.d_only_state_spinner.value - 1] /
                                       x.i_don_raw[x.labels == self.d_only_state_spinner.value - 1], axis=1)
        l = series_to_array(l_df).mean()

        # direct excitation
        d_df = labeled_data.apply(lambda x: (x.i_acc_raw - l * x.i_don_raw) / x.f_acc_acc_raw, axis=1)
        d_array = series_to_array(d_df)
        d_array = d_array[np.invert(np.isinf(d_array))]
        d = d_array.mean()

        # gamma
        d_idx = self.data.data_clean.index
        f_fret = self.data.data.loc[d_idx].apply(lambda x: x.i_acc_raw - l * x.i_don_raw - d * x.f_acc_acc_raw, axis=1)
        gamma_df = pd.DataFrame({'f_fret': f_fret,
                                 'f_dem_dex': self.data.data.loc[d_idx, 'i_don_raw'],
                                 'f_aem_aex': self.data.data.loc[d_idx, 'f_acc_acc_raw']})
        gamma_df.loc[:, 'e_pr'] = gamma_df.apply(lambda x: x.f_fret / (x.f_fret + x.f_dem_dex), axis=1)
        gamma_df.loc[:, 'f_sum'] = gamma_df.apply(lambda x: x.f_fret - x.f_dem_dex, axis=1)
        gamma_df.loc[:, 'stoichiometry'] = gamma_df.apply(lambda x: x.f_sum  / (x.f_sum + x.f_aem_aex), axis=1)

        s = series_to_array(gamma_df.stoichiometry)
        e_pr = series_to_array(gamma_df.e_pr)
        s_mean = s.mean(); s_sd = s.std(); e_mean = e_pr.mean(); e_sd = e_pr.std()
        sb = np.logical_and(np.logical_and(s > s_mean - 2 * s_sd, s < s_mean + 2 * s_sd),
                            np.logical_and(e_pr > e_mean - 2 * e_sd, e_pr < e_mean + 2 * e_sd))
        lm = linear_model.LinearRegression().fit(X=e_pr[sb].reshape(-1, 1), y=s[sb].reshape(-1, 1))
        gamma, beta = lm.coef_[0,0], lm.intercept_[0]

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
        load_button.callback = CustomJS(args=dict(file_source=self.new_source, new_counter=self.new_tot), code=upload_js)

        load_model_button = Button(label='Model')
        load_model_button.callback = CustomJS(args=dict(file_source=self.loaded_model_source),
                                              code=upload_model_js)
        revert_labels_button = Button(label='Undo changes')
        revert_labels_button.on_click(self.revert_manual_labels)


        # --- 2. Teach ---

        self.del_trace_button.on_click(self.del_trace)

        showme_states = [str(n + 1) for n in range(self.num_states_slider.value)]
        showme_idx = list(range(self.num_states_slider.value))

        self.showme_checkboxes = CheckboxButtonGroup(labels=showme_states, active=showme_idx)
        self.showme_checkboxes.on_change('active', self.update_showme)
        showme_col = column(Div(text='Show traces with states:', width=300, height=16),
                            self.showme_checkboxes,
                            height=80, width=300)

        train_button = Button(label='Train (W)', button_type='warning')
        new_example_button = Button(label='New (E)', button_type='success')

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

        # timeseries tools
        self.xbox_select = BoxSelectTool(dimensions='width')
        # self.save = SaveTool()
        self.xwheel_zoom = WheelZoomTool(dimensions='width')
        self.xwheel_pan = WheelPanTool(dimension='width')
        self.pan = PanTool()
        tool_list = [self.xbox_select, self.xwheel_zoom, self.xwheel_pan, self.pan]

        # Main timeseries
        ts = figure(tools=tool_list, plot_width=1075, plot_height=275,
                    active_drag=self.xbox_select, name='ts',
                    x_axis_label='Time (s)', y_axis_label='Intensity')
        ts.rect(x='time', y='rect_mid', width='rect_width', height='rect_height', fill_color={'field': 'labels_pct',
                                                                      'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts.line('time', 'i_don', color='#4daf4a', source=self.source, **line_opts)
        ts.line('time', 'i_acc', color='#e41a1c', source=self.source, **line_opts)
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
        ts_manual.line('time', 'i_don', color='#4daf4a', source=self.source, **line_opts)
        ts_manual.line('time', 'i_acc', color='#e41a1c', source=self.source, **line_opts)
        ts_manual.toolbar = self.ts_toolbar
        pred_vs_manual_panel = Panel(child=widgetbox(column(ts_manual,revert_labels_button)), title='Predicted')

        # Additional settings panel
        settings_panel = Panel(child=widgetbox(row(
            column(
                Div(text='<b>Data format options</b>', height=15, width=200),
                row(Div(text='Load ALEX traces: ', height=15, width=200), self.alex_checkbox, width=500),
                row(Div(text='Frame rate .trace files (Hz): ', height=15, width=200),widgetbox(self.framerate_spinner, width=75)),

                Div(text='<b>Filtering options</b>', height=15, width=200),
                row(Div(text='DBSCAN filter epsilon: ', height=15, width=200), widgetbox(self.eps_spinner, width=75), widgetbox(self.bg_button, width=65), width=500),
                Div(text='<b>ALEX-based corrections</b>', height=15, width=200),
                row(
                    Div(text='gamma: ', height=15, width=80), widgetbox(self.gamma_factor_spinner, width=75),
                    Div(text='<i>l</i>: ', height=15, width=35), widgetbox(self.l_spinner, width=75),
                    Div(text='<i>r</i>: ', height=15, width=35), widgetbox(self.d_spinner, width=75),
                    widgetbox(self.alex_corr_button, height=15, width=30)),
                row(Div(text='<i>D</i>-only state: ', height=15, width=80),
                    widgetbox(self.d_only_state_spinner, width=75),
                    self.alex_estimate_button),
                width=500),
            column(Div(text=' ', height=15, width=250), width=75),
            column(
                Div(text='<b>Training options</b>', height=15, width=200),
                self.supervision_slider,
                self.buffer_slider,
                row(Div(text='Remove last event before analysis: ', height=15, width=250), self.remove_last_checkbox, width=500),
                row(Div(text='CI bootstrap iterations: ', height=15, width=200), widgetbox(self.bootstrap_size_spinner, width=100)),
                   Div(text='note: model is retrained every iteration, keep low for slow models!'),
                   self.keystroke_holder,
                width=500)
            , width=1075)), title='Settings')
        tabs = Tabs(tabs=[ts_panel, efret_panel, corr_panel, i_sum_panel, pred_vs_manual_panel, settings_panel])

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
                             row(Div(text='Mean log-posterior: ', width=150, height=18), self.posterior_text, height=18),
                             row(Div(text='Manually classified (%): ', width=150, height=18), self.mc_text, height=18))

        # --- Define update behavior ---
        self.algo_select.on_change('value', self.update_algo)
        self.source.selected.on_change('indices', self.update_classification)  # for manual selection on trace
        self.new_source.on_change('data', self.buffer_data)
        self.loaded_model_source.on_change('data', self.load_params)
        self.example_select.on_change('value', self.update_example)
        train_button.on_click(self.train_trigger)
        new_example_button.on_click(self.new_example)
        self.bg_checkbox.on_change('active', lambda attr, old, new: self.update_eps())
        self.bg_button.on_click(self.update_eps)
        self.alex_estimate_button.on_click(self.estimate_crosstalk_params)
        # self.bg_button.on_click(self.subtract_background)
        # self.bg_test_button.on_click(self.subtract_test)
        self.num_states_slider.on_change('value', self.update_num_states)
        self.alex_corr_button.on_click(self.update_crosstalk)
        self.state_radio.on_change('active', lambda attr, old, new: self.update_state_curves())
        self.features_checkboxes.on_change('active', self.update_feature_list)
        self.buffer_slider.on_change('value', self.update_buffer)

        # hidden holders
        self.report_holder.js_on_change('text', CustomJS(args=dict(file_source=self.html_source),
                                                         code=download_report_js))
        self.datzip_holder.js_on_change('text', CustomJS(args=dict(file_source=self.datzip_source),
                                                         code=download_datzip_js))
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
                         row(widgetbox(load_model_button, width=150), widgetbox(load_button, width=150),
                             width=300),
                         row(Div(text="DBSCAN background subtraction ", width=250, height=15),
                             widgetbox(self.bg_checkbox, width=25), width=300),
                         Div(text="<font size=4>2. Teach</font>", width=280, height=15),
                         # row(Div(text='Change selection to state: '),
                         #     self.sel_state, width=300, height=15),
                         self.sel_state_slider,
                         showme_col,
                         row(widgetbox(self.del_trace_button, width=100), widgetbox(train_button, width=100), widgetbox(new_example_button, width=100)),
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

    def create_gui(self):
        self.make_document(curdoc())

    def start_threads(self):
        # Start background threads for prediction and data loading
        self.main_thread = threading.main_thread()
        self.data_load_thread = Thread(target=self.update_data, name='data_loading')
        self.predict_thread = Thread(target=self.predict, name='predict')
        self.bg_subtraction_thread = Thread(target=self.subtract_background, name='Redo subtraction')
        self.data_load_thread.start()
        self.predict_thread.start()
        self.bg_subtraction_thread.start()

    def start_ioloop(self, port=0):
        apps = {'/': Application(FunctionHandler(self.make_document))}
        server = Server(apps, port=port, websocket_max_message_size=100000000)
        server.show('/')
        self.loop = IOLoop.current()
        self.loop.start()
