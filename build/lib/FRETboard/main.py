import sys, os
from Gui import Gui
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}')

# Required for bokeh serve
gui_obj = Gui()
gui_obj.create_gui()
gui_obj.start_threads()
gui_obj.doc
