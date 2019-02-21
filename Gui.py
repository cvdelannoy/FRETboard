import numpy as np
import os
import tempfile
import shutil
from cached_property import cached_property
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import base64

from bokeh.layouts import row, column, widgetbox
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LinearColorMapper, CustomJS
from bokeh.models.widgets import Slider, Select, Button, PreText, RadioGroup, Div, CheckboxButtonGroup
from tornado.ioloop import IOLoop
from hmmlearn import hmm
from helper_functions import print_timestamp

from FretReport import FretReport
from ScrollStateTool import ScrollStateTool

line_opts = dict(line_width=1)
rect_opts = dict(width=1.01, alpha=1, line_alpha=0)
white_blue_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
pastel_colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
# col_mapper = LinearColorMapper(palette=white_blue_colors, low=0, high=1)
diverging_colors = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
colors = white_blue_colors

upload_impl = """
function read_file(filename, idx) {
    var reader = new FileReader();
    reader.onload = (function(i){
        return function(event){
            var b64string = event.target.result;
            if (i + 1 == input.files.length){
                console.log('final entry');
                var last_bool = true
            } else {
                var last_bool = false
            }
            last_bool
            file_source.data = {'file_contents' : [b64string], 'file_name': [input.files[i].name], 'last_bool': [last_bool]};
            file_source.change.emit();
        };
    })(idx);
    reader.onerror = error_handler;

    // readAsDataURL represents the file's data as a base64 encoded string
    var re = /(?:\.([^.]+))?$/g;
    var ext = (re.exec(input.files[idx].name))[1];
    if (ext == "dat" || ext == "traces"){
        reader.readAsDataURL(filename);
    } else{ alert(ext + " extension found, only .dat and .traces files accepted for now")};
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
    file_source.change.emit();
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

download_datzip_impl = """
function str2bytes (str) {
    var bytes = new Uint8Array(str.length);
    for (var i=0; i<str.length; i++) {
        bytes[i] = str.charCodeAt(i);
    }
    return bytes;
}

const filename = 'dat_files.zip'
var b64data = window.atob(file_source.data.datzip[0])
const blob = new Blob([str2bytes(b64data)], { type: 'application/octet-stream' })
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
const filename = 'FRETboard_report.html'
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

const filename = 'FRETboard_model.csv'
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


param_state_dict = {18: 2, 30: 3, 44: 4, 60: 5, 78: 6, 98: 7}


class Gui(object):
    def __init__(self, hmm_obj):
        self.version = '0.0.1'
        self.cur_example_idx = None

        # widgets
        self.example_select = Select(title='current_example', value='None', options=['None'])
        self.num_states_slider = Slider(title='Number of states', value=hmm_obj.nb_states, start=2, end=6, step=1)
        self.sel_state_slider = Slider(title='change selection to state', value=1, start=1,
                                       end=self.num_states_slider.value, step=1)
        self.influence_slider = Slider(title='Influence supervision', value=0.2, start=0.0, end=1.0, step=0.01)
        self.notification = PreText(text='', width=1000, height=15)
        self.acc_text = PreText(text='N/A')
        self.mc_text = PreText(text='0%')
        self.state_radio = RadioGroup(labels=hmm_obj.feature_list, active=0)
        self.report_holder = PreText(text='', css_classes=['hidden'])  # hidden holder to generate js callbacks
        self.datzip_holder = PreText(text='', css_classes=['hidden'])  # hidden holder to generate js callbacks

        # ColumnDataSources
        self.source = ColumnDataSource(data=dict(i_don=[], i_acc=[], time=[],
                                                 rect_height=[], rect_mid=[],
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
        self.hmm_loaded_source = ColumnDataSource(data=dict(params=[]))
        self.hmm_source = ColumnDataSource(data=dict(params=[]))
        self.html_source = ColumnDataSource(data=dict(html_text=[]))
        self.datzip_source = ColumnDataSource(data=dict(datzip=[]))
        self.scroll_state_source = ColumnDataSource(data=dict(new_state=[False]))
        self.model_loaded = False

        # HMM object
        self.hmm_obj = hmm_obj

    @property
    def hmm_obj(self):
        return self._hmm_obj

    @hmm_obj.setter
    def hmm_obj(self, hmm_obj):
        self._hmm_obj = hmm_obj
        if hmm_obj.data.shape[0] == 0:
            return
        self.example_select.options = self.hmm_obj.data.index.tolist()
        if self.cur_example_idx is None:
            self.cur_example_idx = self.example_select.options[0]
        self._redraw_all()


    @property
    def nb_examples(self):
        return self.hmm_obj.data.shape[0]

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
        self.hmm_obj.train(influence=self.influence_slider.value)
        tm_array = self.hmm_obj.trained_hmm.transmat_.reshape(-1, 1).squeeze()
        means = self.hmm_obj.trained_hmm.means_.reshape(-1, 1).squeeze()
        covar_list = [blk.reshape(1, -1).squeeze() for blk in
                      np.split(self.hmm_obj.trained_hmm.covars_, axis=0, indices_or_sections=3)]
        covars = np.concatenate(covar_list)
        # covars = self.hmm_obj.trained_hmm.covars_.reshape(-1, 1).squeeze()
        startprob = self.hmm_obj.trained_hmm.startprob_.reshape(-1, 1).squeeze()
        nb_states = np.expand_dims(self.num_states_slider.value, 0)
        params = np.concatenate((nb_states, tm_array, means, covars, startprob))
        self.hmm_source.data = dict(params=params.tolist())

    def load_hmm_params(self, attr, old, new):
        raw_contents = self.hmm_loaded_source.data['file_contents'][0]
        # remove the prefix that JS adds
        _, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents).decode('utf-8').split('\n')[1:-1]
        params = [float(i) for i in file_contents]
        nb_states = int(params[0])
        params = params[1:]
        self.num_states_slider.value = nb_states
        self.hmm_obj.nb_states = nb_states
        new_hmm = hmm.GaussianHMM(n_components=nb_states, covariance_type='full', init_params='')
        param_idx = np.cumsum([nb_states ** 2, nb_states * 2, nb_states * 2 * 2])
        print(f'param_idx: {param_idx}')
        tm, means, covars, start_prob = np.split(params, param_idx)
        new_hmm.transmat_ = tm.reshape(nb_states, nb_states)
        new_hmm.means_ = means.reshape(nb_states, 2)
        new_hmm.covars_ = covars.reshape(nb_states, 2, 2)
        new_hmm.startprob_ = start_prob
        self.hmm_obj.trained_hmm = new_hmm
        self.model_loaded = True
        self.notification.text += f'{print_timestamp()}Model loaded'
        if self.hmm_obj.data.shape[0] != 0:
            self.hmm_obj.data.is_labeled = False
            self.hmm_obj.predict()
            self._redraw_all()

    def update_data(self, attr, old, new):
        raw_contents = self.new_source.data['file_contents'][0]
        _, b64_contents = raw_contents.split(",", 1)  # remove the prefix that JS adds
        file_contents = base64.b64decode(b64_contents)
        fn = self.new_source.data['file_name'][0]
        last_bool = self.new_source.data['last_bool'][0]

        # Process .trace files
        if '.traces' in fn:
            nb_frames, _, nb_traces = np.frombuffer(file_contents, dtype=np.int16, count=3)
            traces_vec = np.frombuffer(file_contents, dtype=np.int16, count=nb_frames * nb_traces + 3)
            file_contents = traces_vec[3:].reshape((2, nb_traces // 2, nb_frames), order='F')
            fn_clean = os.path.splitext(fn)[0]
            for fi, f in enumerate(np.hsplit(file_contents, file_contents.shape[1])):
                last_trace_bool = fi+1 == file_contents.shape[1]
                self.hmm_obj.add_data_tuple(f'{fn_clean}_tr{fi+1}', f.squeeze(), last=last_bool * last_trace_bool)

        # Process .dat files
        elif '.dat' in fn:
            file_contents = base64.b64decode(b64_contents).decode('utf-8')
            file_contents = np.column_stack([np.fromstring(n, sep=' ') for n in file_contents.split('\n') if len(n)])[1:, :]
            if len(file_contents):
                self.hmm_obj.add_data_tuple(self.new_source.data['file_name'][0], file_contents, last=last_bool)

        if last_bool:
            if len(file_contents): self.example_select.options = self.hmm_obj.data.index.tolist()
            if self.cur_example_idx is None: self.cur_example_idx = self.example_select.options[0]
            if self.model_loaded:
                self.hmm_obj.predict()
            else:
                self.train_and_update()
                self.model_loaded = True
            self._redraw_all()

    def update_example(self, attr, old, new):
        if old == new:
            return
        if old is not None:
            self.set_value('labels', self.source.data['labels'])
        # self.train_and_update()
        self.cur_example_idx = new
        self._redraw_all()

    def update_example_retrain(self):
        """
        Assume current example is labeled correctly, retrain and display new random example
        Note: this should be the only way to set an example to 'labeled training example' mode (is_labeled == True)
        """
        if all(self.hmm_obj.data.is_labeled):
            self.notification.text += f'{print_timestamp()}All examples have already been manually classified'
        else:
            self.set_value('labels', self.hmm_obj.data.loc[self.cur_example_idx].prediction.copy())
            self.set_value('is_labeled', True)
            self.train_and_update()
            new_example_idx = np.random.choice(self.hmm_obj.data.loc[np.invert(self.hmm_obj.data.is_labeled), 'logprob'].index)
            self.example_select.value = self.cur_example_idx
            self.update_example(None, None, new_example_idx)

    def _redraw_all(self):
        self.invalidate_cached_properties()
        nb_samples = self.i_don.size
        if not len(self.hmm_obj.data.loc[self.cur_example_idx].labels):
            self.set_value('labels', self.hmm_obj.data.loc[self.cur_example_idx].prediction.copy())
        all_ts = np.concatenate((self.i_don, self.i_acc))
        rect_mid = (all_ts.min() + all_ts.max()) / 2
        rect_height = np.abs(all_ts.min()) + np.abs(all_ts.max())
        self.source.data = dict(i_don=self.i_don, i_acc=self.i_acc, time=np.arange(nb_samples),
                                rect_height=np.repeat(rect_height, nb_samples),
                                rect_mid=np.repeat(rect_mid, nb_samples),
                                labels=self.hmm_obj.data.loc[self.cur_example_idx, 'labels'],
                                labels_pct=self.hmm_obj.data.loc[self.cur_example_idx, 'labels'] * 1.0 / self.num_states_slider.value)
        self.update_accuracy_hist()
        self.update_logprob_hist()
        self.update_state_curves()
        self.update_stats_text()

    def generate_report(self):
        self.html_source.data['html_text'] = [FretReport(self).construct_html_report()]
        self.report_holder.text += ' '

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
        self.acc_text.text = f'{acc}'
        self.mc_text.text = f'{pct_labeled}'

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

            self.hmm_obj.data.is_labeled = False
            self.hmm_obj.nb_states = new

            # Update widget: show-me checkboxes
            showme_idx = [n - 1 for n in range(1,new)]
            showme_states = [str(n) for n in range(1,new)]
            self.showme_checkboxes.labels = showme_states
            self.showme_checkboxes.active = showme_idx

            # retraining hmm
            self.hmm_obj.train(influence=self.influence_slider.value)
            self.invalidate_cached_properties()

            # Update widget: selected state slider
            self.sel_state_slider.end = new
            self.update_state_curves()

            print('update example')
            self.update_example(None, '', self.cur_example_idx)

    def update_scroll_state(self, attr, old, new):
        if not self.scroll_state_source['new_state'][0]:
            return
        if self.sel_state_slider.value + 1 <= self.num_states_slider.value:
            self.sel_state_slide.value += 1
        else:
            self.sel_state_slider = 1
        self.scroll_state_source['new_state'][0] = False

    def export_data(self):
        self.fretReport = FretReport(self)

    def generate_dats(self):
        tfh = tempfile.TemporaryDirectory()
        for fn, tup in self.hmm_obj.data.iterrows():
            labels = tup.labels + 1 if len(tup.labels) != 0 else tup.prediction + 1
            sm_bool = [True for sm in self.saveme_checkboxes.active if sm + 1 in labels]
            if not all(sm_bool): continue
            time = np.arange(labels.size)
            arr_out = np.vstack((time, tup.i_don, tup.i_acc, labels)).T.astype(int)
            np.savetxt(f'{tfh.name}/{fn}', arr_out, fmt='%i')
        zip_dir = tempfile.TemporaryDirectory()
        zip_fn = shutil.make_archive(f'{zip_dir.name}/dat_files', 'zip', tfh.name)
        with open(zip_fn, 'rb') as f:
            self.datzip_source.data['datzip'] = [base64.b64encode(f.read()).decode('ascii')]
        tfh.cleanup()
        zip_dir.cleanup()
        self.datzip_holder.text += ' '

    def del_trace(self):
        self.hmm_obj.del_data_tuple(self.cur_example_idx)
        self.example_select.options.remove(self.cur_example_idx)
        self.update_example_retrain()

    def update_showme(self, attr, old, new):
        if old == new: return
        valid_bool = self.hmm_obj.data.apply(lambda x: np.invert(x.is_labeled)
                                                       and any(i in new for i in x.prediction), axis=1)
        if any(valid_bool):
            self.example_select.options = list(self.hmm_obj.data.index[valid_bool])
            if self.cur_example_idx not in self.example_select.options:
                new_example_idx = np.random.choice(self.example_select.options)
                self.update_example(None, '', new_example_idx)
        else:
            self.notification.text += f'''\n{print_timestamp()}No valid (unclassified) traces to display for classes {', '.join([str(ts+1) for ts in new]) }'''

    def make_document(self, doc):
        # --- Define widgets ---

        ff_title = Div(text=f"""<font size=15><b>FRETboard</b></font><br>v.{self.version}<hr>""",
                       width=280, height=70)

        # --- 1. Load ---
        load_button = Button(label='Data', button_type='success')
        load_button.callback = CustomJS(args=dict(file_source=self.new_source), code=upload_impl)

        load_model_button = Button(label='Model')
        load_model_button.callback = CustomJS(args=dict(file_source=self.hmm_loaded_source),
                                              code=upload_model_impl)


        # --- 2. Teach ---
        del_trace_button = Button(label='Delete', button_type='danger')
        del_trace_button.on_click(self.del_trace)

        showme_states = [str(n + 1) for n in range(self.num_states_slider.value)]
        showme_idx = list(range(self.num_states_slider.value))

        self.showme_checkboxes = CheckboxButtonGroup(labels=showme_states, active=showme_idx)
        self.showme_checkboxes.on_change('active', self.update_showme)
        showme_col = column(Div(text='Show traces with states:', width=300, height=16),
                            self.showme_checkboxes,
                            width=300)

        example_button = Button(label='Next example', button_type='success')

        # --- 3. Save ---
        save_model_button = Button(label='Model')
        save_model_button.callback = CustomJS(args=dict(file_source=self.hmm_source),
                                              code=download_csv_impl)
        save_data_button = Button(label='Data')
        save_data_button.on_click(self.generate_dats)

        report_button = Button(label='report', button_type='success')
        report_button.on_click(self.generate_report)

        self.saveme_checkboxes = CheckboxButtonGroup(labels=showme_states, active=showme_idx)
        saveme_col = column(Div(text='Save traces with states:', width=300, height=16),
                            self.saveme_checkboxes,
                            width=300)

        # --- Define plots ---
        # Main timeseries
        ts = figure(tools='xbox_select,save', plot_width=1000, plot_height=275, active_drag='xbox_select')
        ts.add_tools(ScrollStateTool(source=self.scroll_state_source))
        ts.rect('time', 'rect_mid', height='rect_height', fill_color={'field': 'labels_pct',
                                                                      'transform': self.col_mapper},
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

        # Stats in text
        stats_text = column( row(Div(text='Accuracy: ', width=120, height=18), self.acc_text, height=18),
                             row(Div(text='Manually classified: ', width=120, height=18), self.mc_text, height=18))

        # todo: Average accuracy and posterior

        # --- Define update behavior ---
        self.new_source.on_change('data', self.update_data)
        self.hmm_loaded_source.on_change('data', self.load_hmm_params)
        self.example_select.on_change('value', self.update_example)
        example_button.on_click(self.update_example_retrain)
        self.num_states_slider.on_change('value', self.update_num_states)
        self.source.on_change('selected', self.update_classification)
        self.state_radio.on_change('active', lambda attr, old, new: self.update_state_curves())
        self.scroll_state_source.on_change('data', self.update_scroll_state)

        # hidden holders to generate saves
        self.report_holder.js_on_change('text', CustomJS(args=dict(file_source=self.html_source),
                                                         code=download_report_impl))
        self.datzip_holder.js_on_change('text', CustomJS(args=dict(file_source=self.datzip_source),
                                                         code=download_datzip_impl))

        # --- Build layout ---
        state_block = row(state_curves, self.state_radio)
        widgets = column(ff_title,
                         Div(text="<font size=4>1. Load</font>", width=280, height=15),
                         row(widgetbox(load_button, width=150), widgetbox(load_model_button, width=150),
                             width=300),
                         Div(text="<font size=4>2. Teach</font>", width=280, height=15),
                         self.num_states_slider,
                         self.sel_state_slider,
                         self.influence_slider,
                         showme_col,
                         del_trace_button,
                         example_button,
                         Div(text="<font size=4>3. Save</font>", width=280, height=15),
                         saveme_col,
                         report_button,
                         row(widgetbox(save_data_button, width=150), widgetbox(save_model_button, width=150)),
                         width=300)
        hists = row(acc_hist, logprob_hist, state_block)
        graphs = column(self.example_select,
                        ts,
                        hists,
                        stats_text,
                        Div(text='Notifications: ', width=100, height=15),
                        self.notification,
                        self.report_holder, self.datzip_holder)
        layout = row(widgets, graphs)
        doc.add_root(layout)
        doc.title = f'FRETboard v. {self.version}'

    def start_gui(self):
        apps = {'/': Application(FunctionHandler(self.make_document))}
        server = Server(apps, port=0)
        server.show('/')
        loop = IOLoop.current()
        loop.start()

    def start_server(self):
        return self.make_document(curdoc())
