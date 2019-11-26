from FRETboard.Gui import Gui

# Required for bokeh serve
gui_obj = Gui()
gui_obj.create_gui()
gui_obj.start_threads()
gui_obj.doc
