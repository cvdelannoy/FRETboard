import argparse
import sys
import os
import re
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from not_for_upload.helper_functions import parse_output_dir, parse_input_path


parser = argparse.ArgumentParser(description='Convert sunghyuns format to FRETboard format')
parser.add_argument('--in-dats', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--label-file', action='store_true')

args = parser.parse_args()

dat_files = parse_input_path(args.in_dats, pattern='*.dat')
out_dir = parse_output_dir(args.out_dir, clean=True)
if args.label_file:
    names = ['time', 'label', 'E_FRET']
else:
    names = None

for dat in dat_files:
    fn_base = os.path.basename(dat)
    if args.label_file:
        fn_base = re.search('trace.+(?=_hidden)', fn_base).group(0) + '.dat'
    df = pd.read_csv(dat, header=None, delimiter=r"\s+", names=names)
    df.to_csv(f'{out_dir}{fn_base}', sep='\t', header=args.label_file, index=False)
