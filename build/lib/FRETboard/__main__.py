import sys
import argparse
from FRETboard.Gui import Gui

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='FRETboard', description='Supervise FRET event detection algorithms')
    parser.add_argument('-n', '--nb-states', type=int, default=3,
                        help='Number of states to detect initially.')
    args = parser.parse_args(args)
    gui_obj = Gui(args.nb_states, [])
    gui_obj.start_gui()

if __name__ == '__main__':
    main()
