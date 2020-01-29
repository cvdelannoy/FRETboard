import os, fnmatch, warnings
import re

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper#, CustomJS
from bokeh.models.widgets import Select
from tornado.ioloop import IOLoop

import pandas as pd
import numpy as np
from os.path import basename
import argparse

def parse_input_path(location, pattern=None):
    """
    Take path, list of files or single file, Return list of files with path name concatenated.
    """
    if not isinstance(location, list):
        location = [location]
    all_files = []
    for loc in location:
        loc = os.path.abspath(loc)
        if os.path.isdir(loc):
            if loc[-1] != '/':
                loc += '/'
            for root, dirs, files in os.walk(loc):
                if pattern:
                    for f in fnmatch.filter(files, pattern):
                        all_files.append(os.path.join(root, f))
                else:
                    for f in files:
                        all_files.append(os.path.join(root, f))
        elif os.path.exists(loc):
            all_files.extend(loc)
        else:
            warnings.warn('Given file/dir %s does not exist, skipping' % loc, RuntimeWarning)
    if not len(all_files):
        ValueError('Input file location(s) did not exist or did not contain any files.')
    return all_files


# --- argument parsing ---
parser = argparse.ArgumentParser(description='View kinsoft traces files')
parser.add_argument('in_dir', type=str,
                    help='input directory containing kinsoft files')
parser.add_argument('--type', type=str, default='kinsoft',choices=['kinsoft','fretboard'],
                    help='type of input dat files')
parser.add_argument('--nb-states', type=int,
                    help='Number of states to expect in files.')
args = parser.parse_args()

# --- viz options ---
line_opts = dict(line_width=1)
rect_opts = dict(width=1, alpha=1, line_alpha=0)
white_blue_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
colors = white_blue_colors


class Viewer(object):
    def __init__(self, path, file_type, nb_states):
        self.file_type = file_type
        self.path = path
        self.nb_states = nb_states


        self.source = ColumnDataSource(data=dict(i_don=[], i_acc=[], time=[], labels=[],
                                         rect_height=[], rect_mid=[], labels_pct=[]))

        # widgets
        self.example_select = Select(title='Current example',
                                     value=list(self.trace_dict)[0],
                                     options=list(self.trace_dict))
        self.ts = figure(tools='save,xwheel_zoom,xwheel_pan', plot_width=1000, plot_height=275)
        self.ts.rect(x='time', y='rect_mid',
                     height='rect_height',
                     fill_color={'field': 'labels_pct', 'transform': LinearColorMapper(palette=colors,
                                                                                                     low=0, high=0.99)},
                source=self.source, **rect_opts)
        self.ts.line('time', 'i_don', color='#4daf4a', source=self.source, **line_opts)
        self.ts.line('time', 'i_acc', color='#e41a1c', source=self.source, **line_opts)

        # update behavior
        self.example_select.on_change('value', self.update_example)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if self.file_type == 'kinsoft':
            trace_list = parse_input_path(path, pattern='trace_*')
            state_list = parse_input_path(path, pattern='state_time_*')
            self.state_dict = {re.search('[0-9]+(?=.txt)', basename(fn)).group(0): fn for fn in state_list}
            self.trace_dict = {basename(trace): [trace, re.search('[0-9]+(?=.txt)', basename(trace)).group(0)] for trace in
                               trace_list}
        elif self.file_type == 'fretboard':
            trace_list = parse_input_path(path, pattern='*.dat')
            fn_list = [basename(tl) for tl in trace_list]
            trace_list = [tl for _,tl in sorted(zip(fn_list, trace_list))]
            self.trace_dict = {basename(trace): trace for trace in trace_list}
        self._path = path

    # --- update rules ---
    def update_example(self, attr, old, new):
        """
        Update the example currently on the screen.
        """
        if old == new:
            return
        new_df = self.get_df(new)

        nb_samples = len(new_df)
        if new_df.labels.max() >= self.nb_states:  # >= because 0-based labels
            raise ValueError(f'Number states given is {self.nb_states}, but trace {new} contains states of higher index!')

        all_ts = np.concatenate((new_df.i_don.to_numpy(), new_df.i_acc.to_numpy()))
        rect_mid = (all_ts.min() + all_ts.max()) / 2
        rect_height = np.abs(all_ts.min()) + np.abs(all_ts.max())
        labels_pct = new_df.labels * 1.0 / self.nb_states
        self.source.data = dict(i_don=new_df.i_don, i_acc=new_df.i_acc, time=np.arange(len(new_df)),
                                labels=new_df.labels, labels_pct=labels_pct,
                                rect_height=np.repeat(rect_height, nb_samples),
                                rect_mid=np.repeat(rect_mid, nb_samples))

    def get_df(self, new):
        if self.file_type == 'kinsoft':
            new_df = pd.read_csv(self.trace_dict[new][0], sep="\t", header=None, skiprows=1,
                                   names=['time', 'i_don', 'i_acc', 'i_aa', 'E_FRET'])
            ks_df = pd.read_csv(self.state_dict[self.trace_dict[new][1]], sep='\t')
            ks_df.loc[:, 't_end (s)'] = ks_df.loc[:, 't_start (s)'] + ks_df.loc[:, 't_dwell (s)']

            new_df.loc[:, 'labels'] = [ks_df.loc[ks_df[ks_df.loc[:, 't_start (s)'] <= t].index.max(), '%state'] for t in
                                   new_df.time]
            new_df.loc[:, 'labels'] -= new_df.loc[:, 'labels'].min()
            return new_df
        elif self.file_type == 'fretboard':
            new_df = pd.read_csv(self.trace_dict[new], sep='\t', header=0)
            if new_df.label.isna().any():
                new_df.labels = new_df.predicted
            else:
                new_df.labels = new_df.label
            new_df.drop(['predicted', 'label'], axis=1, inplace=True)
            new_df.labels -= 1  # labels must be 0-based
            return new_df

    def make_document(self, doc):
        layout = column(self.example_select, self.ts)
        doc.add_root(layout)
        doc.title = 'traceViewer'

    def start_gui(self, port=0):
        apps = {'/': Application(FunctionHandler(self.make_document))}
        server = Server(apps, port=port, websocket_max_message_size=100000000)
        server.show('/')
        loop = IOLoop.current()
        loop.start()


viewer = Viewer(args.in_dir, args.type, args.nb_states)
viewer.start_gui(port=5201)
