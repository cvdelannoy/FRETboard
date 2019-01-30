from Gui import Gui
from HiddenMarkovModel import HiddenMarkovModel

hmm = HiddenMarkovModel(nb_states=3, data=[])

gui_obj = Gui(hmm)
gui_obj.start_server()
