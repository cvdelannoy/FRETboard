import os
import numpy as np
from math import ceil
from cached_property import cached_property
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

from bokeh.layouts import row, column
from bokeh import palettes
from bokeh.plotting import figure, curdoc
from bokeh.models import BoxSelectTool, ColumnDataSource, LinearColorMapper
from bokeh.models.widgets import Slider, Select, Button, TextInput, PreText, RadioGroup
from tornado.ioloop import IOLoop

from FretReport import FretReport

# rects_opts = dict(cmap='Blues', color_index='level', line_width=0.000001, colorbar=True)
# curve_opts = dict(height=275, width=1000, xaxis=None, color='black', line_width=1)
# curve_opts_don = dict(height=275, width=1000, xaxis=None, color='green', line_width=1)
# curve_opts_acc = dict(height=275, width=1000, xaxis=None, tools=['xbox_select'], color='red', line_width=1)

line_opts = dict(line_width=1)
rect_opts = dict(width=1.01, alpha=1, line_alpha=0)
white_blue_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
pastel_colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
# col_mapper = LinearColorMapper(palette=white_blue_colors, low=0, high=1)
diverging_colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd']
colors = white_blue_colors

class Gui(object):
    def __init__(self, hmm_obj):
        self.hmm_obj = hmm_obj
        self.example_list = hmm_obj.data.index
        self.cur_example_idx = self.example_list[0]
        self.example_select = Select(title='current_example', value=self.cur_example_idx, options=self.example_list.tolist())

        #widgets
        self.num_states_slider = Slider(title='Number of states', value=self.hmm_obj.nb_states, start=2, end=6, step=1)
        self.save_path = TextInput(title='Save directory', value='')
        self.save_path_warning = PreText(text='', width=300)
        self.state_radio = RadioGroup(labels=self.hmm_obj.feature_list, active=0)

        # ColumnDataSources
        self.source = ColumnDataSource(data=dict(i_don=[], i_acc=[], time=[],
                                                 rect_height=[], rect_mid=[],
                                                 labels=[]))
        ahb = np.arange(start=0, stop=100, step=5)
        self.accuracy_source = ColumnDataSource(data=dict(lb=ahb[:-1], rb=ahb[1:], accuracy_counts=np.repeat(0, 19)))
        self.logprob_source = ColumnDataSource(data=dict(lb=ahb[:-1], rb=ahb[1:], accuracy_counts=np.repeat(0, 19)))
        self.state_source = ColumnDataSource(
            data=dict(xs=[np.arange(0, 1, 0.01)] * self.num_states_slider.value,
                      ys=[np.zeros(100, dtype=float)] * self.num_states_slider.value,
                      color=self.curve_colors))

    @property
    def nb_examples(self):
        return self.hmm_obj.data.shape[0]

    # @property
    # def nb_events_choices(self):
    #     return list(range(self.hmm_obj.nb_states))

    @cached_property
    def curve_colors(self):
        # ceil(len(colors) // self.num_states_slider.value)
        return [colors[n] for n in (np.linspace(0, len(colors)-1, self.num_states_slider.value)).astype(int)]

    @property
    def col_mapper(self):
        return LinearColorMapper(palette=self.curve_colors, low=0, high=self.num_states_slider.value - 1)

    @cached_property
    def sel_state_slider(self):
        return Slider(title='change selection to state', value=1, start=1, end=self.num_states_slider.value, step=1)

    @cached_property
    def i_fret(self):
        return self.hmm_obj.data.loc[self.cur_example_idx].i_fret

    @cached_property
    def i_don(self):
        return self.hmm_obj.data.loc[self.cur_example_idx].i_don

    @cached_property
    def i_acc(self):
        return self.hmm_obj.data.loc[self.cur_example_idx].i_acc

    @cached_property
    def i_acc(self):
        return self.hmm_obj.data.loc[self.cur_example_idx].i_acc

    def set_value(self, column, value, idx=None):
        """
        Set a value for a column in the current example in the original dataframe. Workaround to avoid 'chain indexing'.
        If cell contains a list/array, may provide index to define which values should be changed.
        """
        if idx is None:
            self.hmm_obj.data.at[self.cur_example_idx, column] = value
        else:
            self.hmm_obj.data.at[self.cur_example_idx, column][idx] = value

    def invalidate_cached_properties(self):
        for k in ['i_fret', 'tp', 'ts', 'ts_don', 'ts_acc', 'i_don', 'i_acc',
                  'accuracy_hist']:
            if k in self.__dict__:
                del self.__dict__[k]

    def update_example(self, attr, old, new):
        if old == new:
            return
        if old is not None:
            self.set_value('labels', self.source.data['labels'])
        self.hmm_obj.train()
        self.cur_example_idx = new
        self._redraw_all()

    def update_example_logprob(self):
        self.hmm_obj.train()
        self.cur_example_idx = self.hmm_obj.data.loc[np.invert(self.hmm_obj.data.is_labeled), 'logprob'].idxmin()
        self.example_select.value = self.cur_example_idx
        self._redraw_all()

    def _redraw_all(self):
        self.invalidate_cached_properties()
        y_height = np.max(np.concatenate((self.i_acc, self.i_don)))
        nb_samples = self.i_don.size
        if not self.hmm_obj.data.loc[self.cur_example_idx].is_labeled:
            self.set_value('labels', self.hmm_obj.data.loc[self.cur_example_idx].prediction.copy())
            self.set_value('is_labeled', True)
        self.source.data = dict(i_don=self.i_don, i_acc=self.i_acc, time=np.arange(nb_samples),
                                rect_height=np.repeat(y_height, nb_samples), rect_mid=np.repeat(0, nb_samples),
                                labels=self.hmm_obj.data.loc[self.cur_example_idx, 'labels'])

        self.update_accuracy_hist()
        self.update_logprob_hist()
        self.update_state_curves()

    def update_classification(self, attr, old, new):
        index = self.source.selected.indices
        if index:
            patch = {'labels': [(i, self.sel_state_slider.value - 1) for i in index]}
            self.source.patch(patch)
            self.source.selected.indices = []
            self.update_accuracy_hist()

    def update_accuracy_hist(self):
        acc_counts = np.histogram(self.hmm_obj.accuracy, bins=np.linspace(5, 100, num=20))[0]
        self.accuracy_source.data['accuracy_counts'] = acc_counts

    def update_logprob_hist(self):
        counts, edges = np.histogram(self.hmm_obj.data.logprob, bins='auto')
        self.logprob_source.data = {'logprob_counts': counts, 'lb': edges[:-1], 'rb': edges[1:]}

    def update_state_curves(self):
        fidx = self.state_radio.active
        # feature_idx = [i for i, n in enumerate(self.hmm_obj.feature_list) if n == feature_name]
        mus = self.hmm_obj.trained_hmm.means_[:, fidx]
        sds = self.hmm_obj.trained_hmm.covars_[:, fidx, fidx]

        x_high = max([mu + sd * 3 for mu, sd in zip(mus, sds)])
        x_low = min([mu - sd * 3 for mu, sd in zip(mus, sds)])
        xs = [np.arange(x_low, x_high, (x_high - x_low) / 100)] * self.num_states_slider.value
        ys = [1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-(xs[0] - mu) ** 2 / (2 * sd ** 2))
              for mu, sd in zip(mus, sds)]
        self.state_source.data = {'ys': ys, 'xs': xs, 'color': self.curve_colors}

    def check_save_path(self, attr, old, new):
        if os.path.isdir(new) and len(os.listdir(new)) != 0:
            self.save_path_warning.text = 'NOTE: directory exists and is not\nempty, contents may be overwritten!'
        else:
            self.save_path_warning.text = ''

    def update_num_states(self, attr, old, new):
        if new != self.hmm_obj.nb_states:
            self.hmm_obj.nb_states = new
            self.hmm_obj.train()
            self.invalidate_cached_properties()
            self.update_example(None, '', self.cur_example_idx)

    def export_data(self):
        self.fretReport = FretReport(self)

    def make_document(self, doc):
        # --- Define widgets ---
        example_button = Button(label='Next example', button_type='success')
        # todo: remove_example_button
        export_button = Button(label='Export results', button_type='success')

        # --- Define plots ---
        # Main timeseries
        ts = figure(tools='xbox_select,save', plot_width=1000, plot_height=275, active_drag='xbox_select')
        ts.rect('time', 'rect_mid', height='rect_height', fill_color={'field': 'labels', 'transform': self.col_mapper},
                source=self.source, **rect_opts)
        ts.line('time', 'i_don', color='#4daf4a', source=self.source, **line_opts)
        ts.line('time', 'i_acc', color='#e41a1c', source=self.source, **line_opts)

        # accuracy histogram
        acc_hist = figure(toolbar_location=None, plot_width=275, plot_height=275, x_range=[0, 100],
                          title='Accuracy')
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

        # todo: Average accuracy and posterior

        # --- Define update behavior ---
        self.example_select.on_change('value', self.update_example)
        example_button.on_click(self.update_example_logprob)
        export_button.on_click(self.export_data)
        self.num_states_slider.on_change('value', self.update_num_states)
        self.source.on_change('selected', self.update_classification)
        self.save_path.on_change('value', self.check_save_path)
        self.state_radio.on_change('active', lambda attr, old, new: self.update_state_curves())

        # Initialize for first display
        self.update_example(None, '', self.cur_example_idx)

        # --- Build layout ---
        state_block = row(state_curves, self.state_radio)
        widgets = column(self.sel_state_slider, self.num_states_slider,
                         example_button, self.save_path, export_button, self.save_path_warning)
        hists = row(acc_hist, logprob_hist, state_block)
        graphs = column(self.example_select, ts, hists)
        layout = row(widgets, graphs)
        # layout = column(ts, acc_hist, widgets)
        doc.add_root(layout)
        doc.title = "FRET with bokeh test"

    def start_gui(self):
        apps = {'/': Application(FunctionHandler(self.make_document))}
        server = Server(apps, port=0)
        server.show('/')
        loop = IOLoop.current()
        loop.start()
