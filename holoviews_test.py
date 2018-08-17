import numpy as np
import holoviews as hv
import holoviews.plotting.bokeh
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

renderer = hv.renderer('bokeh')

points = hv.Points(np.random.randn(1000,2 )).options(tools=['box_select', 'lasso_select'])
selection = hv.streams.Selection1D(source=points)

def selected_info(index):
    arr = points.array()[index]
    if index:
        label = 'Mean x, y: %.3f, %.3f' % tuple(arr.mean(axis=0))
    else:
        label = 'No selection'
    return points.clone(arr, label=label).options(color='red')

layout = points + hv.DynamicMap(selected_info, streams=[selection])

doc = renderer.server_doc(layout)
doc.title = 'HoloViews App'

def sine(frequency, phase, amplitude):
    xs = np.linspace(0, np.pi*4)
    return hv.Curve((xs, np.sin(frequency*xs+phase)*amplitude)).options(width=800)

ranges = dict(frequency=(1, 5), phase=(-np.pi, np.pi), amplitude=(-2, 2), y=(-2, 2))
dmap = hv.DynamicMap(sine, kdims=['frequency', 'phase', 'amplitude']).redim.range(**ranges)

app = renderer.app(dmap)

server = Server({'/': app}, port=0)
server.show('/')
loop = IOLoop.current()
loop.start()
