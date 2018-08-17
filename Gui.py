import numpy as np
from cached_property import cached_property
import holoviews as hv
hv.extension('bokeh')
from holoviews import streams

from bokeh.server.server import Server
# from bokeh.layouts import column
from bokeh.io import curdoc
from tornado.ioloop import IOLoop

rects_opts = dict(cmap='Blues', color_index='level', line_width=0.000001, colorbar=True)
curve_opts = dict(height=275, width=1000, xaxis=None, color='black', line_width=1)
curve_opts_don = dict(height=275, width=1000, xaxis=None, color='green', line_width=1)
curve_opts_acc = dict(height=275, width=1000, xaxis=None, tools=['xbox_select'], color='red', line_width=1)


class Gui(object):
    def __init__(self, hmm_obj):
        self.hmm_obj = hmm_obj
        self.cur_example_idx = self.hmm_obj.data.index[0]

    @property
    def nb_examples(self):
        return self.hmm_obj.data.shape[0]

    @property
    def nb_events_choices(self):
        return list(range(self.hmm_obj.nb_states))

    # @cached_property
    # def example(self):
    #     return self.hmm_obj.data.loc[self.cur_example_idx]

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
    def ts_don(self):
        return hv.Curve((np.arange(self.i_don.size), self.i_don)).options(**curve_opts_don).opts(norm=dict(framewise=True))

    @cached_property
    def ts_acc(self):
        return hv.Curve((np.arange(self.i_acc.size), self.i_acc)).options(**curve_opts_acc).opts(norm=dict(framewise=True))

    @cached_property
    def i_acc(self):
        return self.hmm_obj.data.loc[self.cur_example_idx].i_acc

    # @cached_property
    def accuracy_hist(self, boundsx):
        bins = np.arange(0.05,1.05,0.05)
        # labeled_data = self.hmm_obj.data.loc[self.hmm_obj.data.is_labeled, ('prediction', 'labels')]
        # nb_correct = labeled_data.apply(lambda x: np.sum(np.equal(x.prediction, x.labels)), axis=1)
        # nb_points = labeled_data.apply(lambda x: x.labels.size, axis=1)
        # accuracy = nb_correct / nb_points
        acc_hist, _ = np.histogram(self.hmm_obj.accuracy, bins=bins, range=(0, 1))
        return hv.Histogram((acc_hist, bins)).opts(norm=dict(framewise=True))

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

    def start_gui(self):
        stream = streams.BoundsX(source=self.ts_acc, boundsx=(0, 0))
        dm = (hv.DynamicMap(self.display_selection, kdims=['example_idx', 'sel_state'], streams=[stream]).
              redim.values(example_idx=self.hmm_obj.data.index, sel_state=self.nb_events_choices))
        renderer = hv.renderer('bokeh').instance(mode='server')
        hists = hv.DynamicMap(self.accuracy_hist, streams=[stream]).opts(plot=dict(shared_axes=False)).redim.range(x=(0, 1))

        # hists = (hv.DynamicMap(self.accuracy_hist, kdims=['example_idx', 'sel_state'], streams=[stream]).
        #     redim.values(example_idx=self.hmm_obj.data.index, sel_state=self.nb_events_choices).
        #     opts(plot=dict(shared_axes=False)).redim.range(x=(0, 1)))

        app = renderer.app(dm + hists)
        # widget1 = renderer.get_widget(dm, None, position='above').state
        # widget2 = renderer.get_widget(hists, None).state
        # doc = curdoc()
        # doc.add_root(column(widget1, widget2))
        # app = renderer.app(widget1)

        server = Server({'/': app}, port=0)
        server.show('/')
        loop = IOLoop.current()
        loop.start()

    def display_selection(self, example_idx, sel_state, boundsx):
        label = ''
        if example_idx != self.cur_example_idx:
            self.cur_example_idx = example_idx
            self.invalidate_cached_properties()
            self.hmm_obj.train()
            boundsx = (0, 0)
        boundsx = [round(bx) for bx in boundsx]
        if boundsx[1] > self.i_acc.size:
            boundsx[1] = self.i_acc.size
        if boundsx[0] < 0:
            boundsx[0] = 0
        if not self.hmm_obj.data.loc[self.cur_example_idx].is_labeled:
            self.set_value('labels', self.hmm_obj.data.loc[self.cur_example_idx].prediction.copy())
            # self.set_value('is_labeled', True)
        # example = self.hmm_obj.data.loc[self.cur_example_idx]
        if boundsx[1] - boundsx[0] > 0:
            if not self.hmm_obj.data.loc[self.cur_example_idx].is_labeled:
                self.set_value('is_labeled', True)
            index_list = list(range(*boundsx))
            self.set_value('labels', sel_state, idx=index_list)
        pred = self.hmm_obj.data.loc[self.cur_example_idx, 'labels'] #/ self.hmm_obj.nb_states
        time = np.arange(self.i_fret.size)
        pred = np.column_stack((time, pred))
        y_height = np.max(np.concatenate((self.i_acc, self.i_don)))
        pred_rects = hv.Polygons([{('x', 'y'): self.rectangle(x, height=y_height), 'level': z} for
                                  x, z in pred], vdims='level').options(**rects_opts).redim.range(level=(0,self.hmm_obj.nb_states))
        return pred_rects * self.ts_don * self.ts_acc  #* hv.Text(200, 0.6, label)

    @staticmethod
    def rectangle(x=0, y=0, width=1, height=1000):
        return np.array([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
