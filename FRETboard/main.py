import sys, yaml
from Gui import Gui
from pathlib import Path
__location__ = Path(__file__).parent.resolve()
sys.path.append(str(__location__))

# Required for bokeh serve
with open(__location__ / 'bokeh_serve_params.yml') as fh: param_dict = yaml.load(fh, Loader=yaml.SafeLoader)
gui_obj = Gui(nb_processes=param_dict['nb_cores'], allow_custom_scripts=param_dict['allow_custom_scripts'])
gui_obj.create_gui()
gui_obj.start_threads()
gui_obj.doc
