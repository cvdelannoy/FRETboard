import numpy as np
import holoviews as hv
hv.extension('bokeh')
from holoviews import streams

from bokeh.layouts import layout
from bokeh.io import curdoc
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.models import Button

rects_opts = dict(color=hv.Palette('Category10'), color_index='level', line_width=0.000001)
curve_opts = dict(height=250, width=1200, xaxis=None, tools=['box_select'], color='black', line_width=1)

class Gui(object):
    def __init__(self, **kwargs):
        self.renderer = hv.renderer('bokeh').instance(mode='server')
        self.hmm_obj = kwargs['hmm_obj']

        self.nb_events_choices = [2, 3, 4, 5]

    def plot_trace(self, nb_states):
        # if nb_states != self.hmm_obj.nb_states:
        #     self.hmm_obj.nb_states = nb_states
        self.hmm_obj.nb_states = nb_states
        self.hmm_obj.train()
        example = self.hmm_obj.data.loc[np.invert(self.hmm_obj.data.is_labeled)].sample(1)
        example.labels = example.prediction
        i_fret = example.i_fret.values[0]
        pred = example.prediction.values[0] / nb_states
        assert pred.size == i_fret.size
        time = np.arange(i_fret.size)
        pred = np.column_stack((time, pred))

        pred_rects = hv.Polygons([{('x', 'y'): self.rectangle(x), 'level': z} for
                                  x, z in pred], vdims='level').options(**rects_opts)
        ts = hv.Curve((np.arange(i_fret.size), i_fret)).options(**curve_opts)
        ts_selected = streams.Selection1D(source=ts)
        return pred_rects * ts

    def display_selection(sel_state, index):
        if index:
            example.prediction.values[0][index] = sel_state
        pred = example.prediction.values[0] / nb_states
        time = np.arange(i_fret.size)
        pred = np.column_stack((time, pred))
        pred_rects = hv.Polygons([{('x', 'y'): rectangle(x), 'level': z} for
                                  x, z in pred], vdims='level').options(**rects_opts)
        return pred_rects

    @staticmethod
    def rectangle(x=0, y=0, width=1, height=1):
        return np.array([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])

    def start_gui(self):
        # button = Button(label='Next', width=60)
        # hmap = hv.HoloMap(button)






        dmap = hv.DynamicMap(self.plot_trace, kdims=['nb_states']).\
            redim.values(nb_states=self.nb_events_choices)

        # button_layout = layout([button])
        # curdoc().add_root(button_layout)
        app = self.renderer.app(dmap)
        server = Server({'/': app}, port=0)
        server.show('/')
        loop = IOLoop.current()
        loop.start()
