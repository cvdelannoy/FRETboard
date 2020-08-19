import sys
import argparse
from FRETboard.Gui import Gui

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='FRETboard', description='Supervise FRET event detection algorithms')
    parser.add_argument('-n', '--nb-states', type=int, default=3,
                        help='Number of states to detect initially.')
    parser.add_argument('-c', '--nb-cores', type=int, default=4,
                        help='Number of cores (processes) to use.')
    parser.add_argument('-p', '--port', type=int, default=0,
                        help='Port where bokeh will listen.')
    parser.add_argument('--allow-custom-scripts', action='store_true',
                        help='Allow running of custom classification algorithms. WARNING: any custom code implies risk'
                             'of code injection. Use only on machines not exposed to external networks.')
    args = parser.parse_args(args)
    gui_obj = Gui(nb_processes=args.nb_cores, nb_states=args.nb_states, allow_custom_scripts=args.allow_custom_scripts)
    gui_obj.start_threads()
    gui_obj.start_ioloop(port=args.port)

if __name__ == '__main__':
    main()
