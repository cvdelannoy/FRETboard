import argparse
from HiddenMarkovModel import HiddenMarkovModel
from GuiBokeh import Gui
# from Gui import Gui
import helper_functions as hp

import holoviews as hv
hv.extension('bokeh')

parser = argparse.ArgumentParser(description='Detect FRET signal with less effort.')
parser.add_argument('data_path', type=str, nargs='+',
                    help='.dat trace files to classify or paths at which they can be found.')
parser.add_argument('--gui', action='store_true', default=False,
                    help='Start in GUI mode')
parser.add_argument('-n', '--nb-states', type=int, default=3,
                    help='Number of states to detect initially.')
args = parser.parse_args()

data_files = hp.parse_input_path(args.data_path)
hmm = HiddenMarkovModel(nb_states=args.nb_states, data=data_files)
hmm.train()

gui_obj = Gui(hmm)
gui_obj.start_gui()

