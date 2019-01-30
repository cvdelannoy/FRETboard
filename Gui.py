import numpy as np
from cached_property import cached_property
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import base64

from bokeh.layouts import row, column, widgetbox
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LinearColorMapper, CustomJS
from bokeh.models.widgets import Slider, Select, Button, PreText, RadioGroup
from tornado.ioloop import IOLoop
from hmmlearn import hmm

from FretReport import FretReport

line_opts = dict(line_width=1)
rect_opts = dict(width=1.01, alpha=1, line_alpha=0)
white_blue_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
pastel_colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
# col_mapper = LinearColorMapper(palette=white_blue_colors, low=0, high=1)
diverging_colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd']
colors = white_blue_colors

upload_impl = """
function read_file(filename, idx) {
    var reader = new FileReader();
    reader.onload = (function(i){
        return function(event){
            var b64string = event.target.result;
            file_source.data = {'file_contents' : [b64string], 'file_name': [input.files[i].name]};
            file_source.trigger("change");
        };
    })(idx);
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.multiple=true
input.onchange = function(){
    if (window.FileReader) {
        for (var i = 0; i < input.files.length; i++){
            read_file(input.files[i], i);
        }
    } else {
        alert('FileReader is not supported in this browser');
    }
}
input.click();
"""

upload_model_impl = """
function read_file(filename) {
    var reader = new FileReader();
    reader.onload = load_handler;
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function load_handler(event) {
    var b64string = event.target.result;
    file_source.data = {'file_contents' : [b64string]};
    file_source.trigger("change");
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.onchange = function(){
    if (window.FileReader) {
        read_file(input.files[0]);
    } else {
        alert('FileReader is not supported in this browser');
    }
}
input.click();
"""


download_csv_impl = """
function table_to_csv(file_source) {
    const columns = Object.keys(file_source.data)
    const nrows = file_source.get_length()
    const lines = [columns.join(',')]

    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < columns.length; j++) {
            const column = columns[j]
            row.push(file_source.data[column][i].toString())
        }
        lines.push(row.join(','))
    }
    return lines.join('\\n').concat('\\n')
}


const filename = 'data_result.csv'
filetext = table_to_csv(file_source)
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}"""

download_report_impl = """
const filename = 'FRETfinder_report.html'
const blob = new Blob([file_source.data.html_text], { type: 'text;charset=utf-8;' })

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}"""

param_state_dict = {18: 2, 30: 3, 44: 4, 60: 5, 78: 6, 98: 7}

class Gui(object):
    def __init__(self, hmm_obj):
        self.hmm_obj = hmm_obj
        # self.example_list = hmm_obj.data.index
        self.cur_example_idx = None

        #widgets
        self.example_select = Select(title='current_example', value='None', options=['None'])
        self.num_states_slider = Slider(title='Number of states', value=self.hmm_obj.nb_states, start=2, end=6, step=1)
        self.sel_state_slider = Slider(title='change selection to state', value=1, start=1,
                                       end=self.num_states_slider.value, step=1)
        self.notification = PreText(text='', width=300)
        self.stats_text = PreText(text='Accuracy: N/A\n'
                                       'Manually classified: 0%\n')
        self.state_radio = RadioGroup(labels=self.hmm_obj.feature_list, active=0)
        self.holder = PreText(text='', css_classes=['hidden'])  # hidden holder to generate js callbacks

        # ColumnDataSources
        self.source = ColumnDataSource(data=dict(i_don=[], i_acc=[], time=[],
                                                 rect_height=[], rect_mid=[],
                                                 labels=[], labels_pct=[]))
        ahb = np.arange(start=0, stop=100, step=5)
        self.new_source = ColumnDataSource({'file_contents':[], 'file_name':[]})
        self.accuracy_source = ColumnDataSource(data=dict(lb=ahb[:-1], rb=ahb[1:], accuracy_counts=np.repeat(0, 19)))
        self.logprob_source = ColumnDataSource(data=dict(lb=ahb[:-1], rb=ahb[1:],
                                                         accuracy_counts=np.repeat(0, 19), logprob_counts=np.repeat(0, 19)))
        self.state_source = ColumnDataSource(
            data=dict(xs=[np.arange(0, 1, 0.01)] * self.num_states_slider.value,
                      ys=[np.zeros(100, dtype=float)] * self.num_states_slider.value,
                      color=self.curve_colors))
        self.hmm_loaded_source = ColumnDataSource(data=dict(params=[]))
        self.hmm_source = ColumnDataSource(data=dict(params=[]))
        self.html_source = ColumnDataSource(data=dict(html_text=[]))

    @property
    def nb_examples(self):
        return self.hmm_obj.data.shape[0]

    # @property
    # def nb_events_choices(self):
    #     return list(range(self.hmm_obj.nb_states))

    @property
    def curve_colors(self):
        # ceil(len(colors) // self.num_states_slider.value)
        return [colors[n] for n in (np.linspace(0, len(colors)-1, self.num_states_slider.value)).astype(int)]

    @property
    def col_mapper(self):
        # return LinearColorMapper(palette=self.curve_colors, low=0, high=self.num_states_slider.value - 1)
        return LinearColorMapper(palette=colors, low=0, high=0.99)

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

    def train_and_update(self):
        self.hmm_obj.train()
        tm_array = self.hmm_obj.trained_hmm.transmat_.reshape(-1, 1).squeeze()
        means = self.hmm_obj.trained_hmm.means_.reshape(-1, 1).squeeze()
        covars = self.hmm_obj.trained_hmm.covars_.reshape(-1, 1).squeeze()
        startprob = self.hmm_obj.trained_hmm.startprob_.reshape(-1, 1).squeeze()
        params = np.concatenate((tm_array, means, covars, startprob))
        self.hmm_source.data = dict(params=params.tolist())
        # self.hmm_source['params', 0] = params

    def load_hmm_params(self, attr, old, new):
        raw_contents = self.hmm_loaded_source.data['file_contents'][0]
        # remove the prefix that JS adds
        _, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents).decode('utf-8').split('\n')[1:-1]
        print(file_contents)
        params = [float(i) for i in file_contents]
        print(params)
        nb_states = param_state_dict[len(params)]
        print(f'nb_states {nb_states}')
        self.num_states_slider.value = nb_states
        self.hmm_obj.data.is_labeled = False
        self.hmm_obj.nb_states = nb_states
        new_hmm = hmm.GaussianHMM(n_components=nb_states, covariance_type='diag', init_params='')
        param_idx = np.cumsum([nb_states ** 2, nb_states, nb_states * 2])
        print(f'param_idx: {param_idx}')
        tm, start_prob, means, covars = np.split(params, param_idx)
        print(tm.reshape(nb_states, nb_states, 'F'))
        new_hmm.transmat_ = tm.reshape(nb_states, nb_states)
        print(means.reshape(nb_states, 2, 'F'))
        new_hmm.means_ = means.reshape(nb_states, 2)
        print(covars.reshape(nb_states, 2, 2))
        new_hmm.covars_ = covars.reshape(nb_states, 2, 2)
        print(start_prob)
        new_hmm.startprob_ = start_prob
        self.hmm_obj.trained_hmm = new_hmm
        self.hmm_obj.predict()
        self._redraw_all()

    def update_data(self, attr, old, new):
        raw_contents = self.new_source.data['file_contents'][0]
        # remove the prefix that JS adds
        _, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents).decode('utf-8')
        # file_contents = str(io.StringIO(bytes.decode(file_contents)))
        file_contents = np.column_stack([np.fromstring(n, sep=' ') for n in file_contents.split('\n') if len(n)]).T
        # file_io = io.StringIO(bytes.decode(file_contents))
        # print(file_contents)
        if len(file_contents):
            self.hmm_obj.add_data_tuple(self.new_source.data['file_name'], file_contents)
            self.example_select.options = self.hmm_obj.data.index.tolist()#  set(options=self.hmm_obj.data.index.tolist())
        if self.cur_example_idx is None:
            self.cur_example_idx = self.example_select.options[0]
            self.train_and_update()
            self._redraw_all()

    def update_example(self, attr, old, new):
        if old == new:
            return
        if old is not None:
            self.set_value('labels', self.source.data['labels'])
        self.train_and_update()
        self.cur_example_idx = new
        self._redraw_all()

    def update_example_logprob(self):
        if all(self.hmm_obj.data.is_labeled):
            self.notification.text = 'All examples have already\nbeen manually classified'
        else:
            self.train_and_update()
            # self.cur_example_idx = self.hmm_obj.data.loc[np.invert(self.hmm_obj.data.is_labeled), 'logprob'].idxmin()
            self.cur_example_idx = np.random.choice(self.hmm_obj.data.loc[np.invert(self.hmm_obj.data.is_labeled), 'logprob'].index)
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
                                labels=self.hmm_obj.data.loc[self.cur_example_idx, 'labels'],
                                labels_pct=self.hmm_obj.data.loc[self.cur_example_idx, 'labels'] * 1.0 / self.num_states_slider.value)

        self.update_accuracy_hist()
        self.update_logprob_hist()
        self.update_state_curves()
        self.update_stats_text()

    def generate_report(self):
        self.html_source.data['html_text'] = [FretReport(self).construct_html_report()]
        self.holder.text = 'trigger'

    def update_classification(self, attr, old, new):
        index = self.source.selected.indices
        if index:
            patch = {'labels': [(i, self.sel_state_slider.value - 1) for i in index],
                     'labels_pct': [(i, (self.sel_state_slider.value - 1) * 1.0 / self.num_states_slider.value) for i in index]}
            self.source.patch(patch)
            self.source.selected.indices = []
            self.update_accuracy_hist()
            self.update_stats_text()

    def update_accuracy_hist(self):
        acc_counts = np.histogram(self.hmm_obj.accuracy, bins=np.linspace(5, 100, num=20))[0]
        self.accuracy_source.data['accuracy_counts'] = acc_counts

    def update_logprob_hist(self):
        counts, edges = np.histogram(self.hmm_obj.data.logprob, bins='auto')
        self.logprob_source.data = {'logprob_counts': counts, 'lb': edges[:-1], 'rb': edges[1:]}

    def update_stats_text(self):
        pct_labeled = round(self.hmm_obj.data.is_labeled.sum() / self.hmm_obj.data.is_labeled.size * 100.0, 1)
        if pct_labeled == 0:
            acc = 'N/A'
        else:
            acc = round(self.hmm_obj.accuracy.mean(), 1)
        self.stats_text.text = ('Accuracy: {acc}\n'
                                'Manually classified: {mc}%\n').format(acc=acc,
                                                                       mc=pct_labeled)

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

    def update_num_states(self, attr, old, new):
        if new != self.hmm_obj.nb_states:
            print('changing hmm_obj states')
            self.hmm_obj.data.is_labeled = False
            self.hmm_obj.nb_states = new
            print('retraining hmm')
            self.hmm_obj.train()
            self.invalidate_cached_properties()
            print('setting new end for sel_state_slider')
            self.sel_state_slider.end = new
            self.update_state_curves()

            print('update example')
            self.update_example(None, '', self.cur_example_idx)

    def export_data(self):
        self.fretReport = FretReport(self)

    def make_document(self, doc):
        # --- Define widgets ---
        example_button = Button(label='Next example', button_type='success')
        load_button = Button(label='Upload', button_type='success')
        load_button.callback = CustomJS(args=dict(file_source=self.new_source), code=upload_impl)

        save_model_button = Button(label='Save model')
        save_model_button.callback = CustomJS(args=dict(file_source=self.hmm_source),
                                              code=download_csv_impl)
        load_model_button = Button(label='Load model')
        load_model_button.callback = CustomJS(args=dict(file_source=self.hmm_loaded_source),
                                              code=upload_model_impl)
        report_button = Button(label='report', button_type='success')
        report_button.on_click(self.generate_report)

        # --- Define plots ---
        # Main timeseries
        ts = figure(tools='xbox_select,wheel_zoom,save', plot_width=1000, plot_height=275, active_drag='xbox_select')
        ts.rect('time', 'rect_mid', height='rect_height', fill_color={'field': 'labels_pct', 'transform': self.col_mapper},
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
        self.new_source.on_change('data', self.update_data)
        self.hmm_loaded_source.on_change('data', self.load_hmm_params)
        self.example_select.on_change('value', self.update_example)
        example_button.on_click(self.update_example_logprob)
        self.num_states_slider.on_change('value', self.update_num_states)
        self.source.on_change('selected', self.update_classification)
        self.state_radio.on_change('active', lambda attr, old, new: self.update_state_curves())

        # hidden holder to generate report
        self.holder.js_on_change('text', CustomJS(args=dict(file_source=self.html_source),
                                                  code=download_report_impl))

        # --- Build layout ---
        state_block = row(state_curves, self.state_radio)
        widgets = column(load_button,
                         self.sel_state_slider,
                         self.num_states_slider,
                         example_button,
                         row(widgetbox(save_model_button, width=150), widgetbox(load_model_button, width=150), width=300),
                         report_button,
                         self.stats_text,
                         self.notification,
                         width=300)
        hists = row(acc_hist, logprob_hist, state_block)
        graphs = column(self.example_select, ts, hists, self.holder)
        layout = row(widgets, graphs)
        doc.add_root(layout)
        doc.title = "FRET with bokeh test"

    def start_gui(self):
        apps = {'/': Application(FunctionHandler(self.make_document))}
        server = Server(apps, port=0)
        server.show('/')
        loop = IOLoop.current()
        loop.start()

    def start_server(self):
        return self.make_document(curdoc())
