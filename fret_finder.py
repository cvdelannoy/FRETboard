import argparse
from HiddenMarkovModel import HiddenMarkovModel
from Gui import Gui
import helper_functions as hp

parser = argparse.ArgumentParser(description='Detect FRET signal with less effort.')
parser.add_argument('--data-path', type=str, nargs='+', required=False,
                    help='.dat trace files to classify or paths at which they can be found.')
parser.add_argument('-n', '--nb-states', type=int, default=3,
                    help='Number of states to detect initially.')
args = parser.parse_args()

if args.data_path:
    data_files = hp.parse_input_path(args.data_path)
    hmm = HiddenMarkovModel(nb_states=args.nb_states, data=data_files)
    hmm.train()
else:
    hmm = HiddenMarkovModel(nb_states=args.nb_states, data=[])

gui_obj = Gui(hmm)
gui_obj.start_gui()
