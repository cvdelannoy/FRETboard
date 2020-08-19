import sys, os, yaml
from Gui import Gui
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}')

# Required for bokeh serve
with open(f'{__location__}/bokeh_serve_params.yml') as fh: param_dict = yaml.load(fh, Loader=yaml.SafeLoader)
gui_obj = Gui(nb_processes=param_dict['nb_cores'], allow_custom_scripts=param_dict['allow_custom_scripts'])
gui_obj.create_gui()
gui_obj.start_threads()
gui_obj.doc
