import argparse
from os.path import basename, splitext, abspath
from shutil import rmtree
import pathlib
import os, fnmatch, warnings

def parse_output_dir(out_dir, clean=False):
    out_dir = abspath(out_dir) + '/'
    if clean:
        rmtree(out_dir, ignore_errors=True)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir

def parse_input_path(location, pattern=None):
    """
    Take path, list of files or single file, Return list of files with path name concatenated.
    """
    if not isinstance(location, list):
        location = [location]
    all_files = []
    for loc in location:
        loc = os.path.abspath(loc)
        if os.path.isdir(loc):
            if loc[-1] != '/':
                loc += '/'
            for root, dirs, files in os.walk(loc):
                if pattern:
                    for f in fnmatch.filter(files, pattern):
                        all_files.append(os.path.join(root, f))
                else:
                    for f in files:
                        all_files.append(os.path.join(root, f))
        elif os.path.exists(loc):
            all_files.extend(loc)
        else:
            warnings.warn('Given file/dir %s does not exist, skipping' % loc, RuntimeWarning)
    if not len(all_files):
        ValueError('Input file location(s) did not exist or did not contain any files.')
    return all_files

parser = argparse.ArgumentParser(description='Convert KinSoft trace files to format readible by FRETboard.')
parser.add_argument('--indir', type=str, required=True, help='input directory')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
args = parser.parse_args()

in_files = parse_input_path(args.indir, pattern='trace_*.txt')
outdir = parse_output_dir(args.outdir, clean=True)

for fn in in_files:
    fn_new = f'{outdir}{splitext(basename(fn))[0]}.dat'
    with open(fn, 'r') as fh_old, open(fn_new, 'w') as  fh_new:
        lines = fh_old.readlines()
        fh_new.writelines(lines[1:])
