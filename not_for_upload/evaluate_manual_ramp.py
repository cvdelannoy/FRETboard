import argparse
import os, fnmatch, warnings
from matplotlib.colors import LinearSegmentedColormap
from copy import copy
from os.path import basename, splitext, abspath
from shutil import rmtree
import pathlib
import re
import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import pickle
from itertools import permutations
import warnings
from scipy.linalg import logm

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
from helper_functions import parse_input_path

mpl.rcParams['figure.dpi'] = 400

colmap = LinearSegmentedColormap.from_list('custom_blues', ['#FFFFFF', '#084594'])

def matlab_load_fun(fn):
    df = pd.read_csv(fn, sep='\t', names=['time', 'i_don', 'i_acc', 'label'])
    df.label += 1  # matlab classification is 0-based
    return df

def fretboard_load_fun(fn):
    df = pd.read_csv(fn, sep='\t', header=0)
    return df

def parse_output_dir(out_dir, clean=False):
    out_dir = abspath(out_dir) + '/'
    if clean:
        rmtree(out_dir, ignore_errors=True)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir

def get_category(fn):
    cat = [cat for cat in args.categories if cat in fn]
    if not len(cat):
        return
    elif len(cat) > 1:
        raise ValueError(f'trace {fb} falls in multiple categories, redefine categories')
    return cat[0]

parser = argparse.ArgumentParser(description='graph accuracy for dir of increasing number of manually curated traces.')
parser.add_argument('--fb', type=str, required=True, help='FRETboard manual ramp results dir')
parser.add_argument('--manual', type=str, required=True, help='manual dir')
parser.add_argument('--target-states', type=int, required=True, nargs='+',
                    help='States (as denoted in manually labeled files) to take into account.')
parser.add_argument('--manual-type', type=str, default='matlab', choices=['matlab', 'fretboard', 'kinsoft'],
                    help='Format of manual label files')
parser.add_argument('--svg', type=str, required=True, help='output plot')
parser.add_argument('--categories', type=str, nargs='+', default=[''],
                    help='In plots, split files and stats by strings that are to be recognized in path names')
parser.add_argument('--type-ramp', type=str, choices=['manual', 'lambda'])
args = parser.parse_args()


target_states_str = np.array(args.target_states, dtype=str)
fb_files = parse_input_path(args.fb, pattern='*.dat')
manual_files = parse_input_path(args.manual, pattern='*.dat')
manual_dict = {}
for fn in manual_files:
    cat = get_category(fn)
    if cat is None: continue
    manual_dict[(cat, basename(fn))] = fn

loader_fun = dict(matlab=matlab_load_fun, fretboard=fretboard_load_fun)[args.manual_type]
acc_df = pd.DataFrame(columns=['cat',  'nb_manual', 'correct', 'total'])

for fb in fb_files:
        cat = get_category(fb)
        if cat is None: continue
        fb_base = basename(fb)
        if (cat, fb_base) not in manual_dict: continue  # skip if no ground truth file available
        dat_df = pd.read_csv(fb, sep='\t')
        if not dat_df.label.isnull().all(): continue  # skip if read was used as labeled example

        if args.type_ramp == 'manual':
            man_nb_obj = re.search('(?<=/)[0-9]+(?=/)', fb) # find how many manual traces were ussed for classification todo
            if man_nb_obj is None:
                raise ValueError(f'Could not deduce number of manual traces for {fb}')
            man_nb = int(man_nb_obj.group(0))
        else:
            man_nb_obj = re.search('(?<=lambda)[0-9.]+',
                                   fb)  # find how many manual traces were ussed for classification todo
            if man_nb_obj is None:
                raise ValueError(f'Could not deduce number of manual traces for {fb}')
            man_nb = float(man_nb_obj.group(0))


        # load manual labels
        manual_df = loader_fun(manual_dict[(cat, fb_base)])

        dat_df.loc[:, 'manual'] = manual_df.label.astype(int)
        dat_df.loc[np.invert(np.in1d(dat_df.manual, args.target_states + [1])), 'manual'] = 1

        # Add EFRET
        i_sum = np.sum((dat_df.f_dex_dem, dat_df.f_dex_aem), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E_FRET = np.divide(dat_df.f_dex_aem, i_sum)
        E_FRET[i_sum == 0] = np.nan  # set to nan where i_don and i_acc after background correction cancel out
        dat_df.loc[:, "E_FRET"] = E_FRET

        # Record accuracy
        correct = (dat_df.predicted == dat_df.manual).sum()
        acc_df.loc[fb, :] = (cat, man_nb, correct, dat_df.shape[0])

plot_df = pd.DataFrame(columns=['nb_manual', 'category', 'accuracy']).set_index(['category', 'nb_manual'])
for cat, cdf in acc_df.groupby('cat'):
    for nbm, nbdf in cdf.groupby('nb_manual'):
        plot_df.loc[(cat, nbm), :] = float(nbdf.correct.sum() / nbdf.total.sum())
plot_df.reset_index(inplace=True)
plot_df.loc[:, 'accuracy'] = plot_df.loc[:, 'accuracy'].astype(float)

sns.lineplot(x='nb_manual', y='accuracy', hue='category', data=plot_df)
plt.xlabel('manually classified traces')
plt.savefig(args.svg, type='svg')
plot_df.to_csv(f'{args.svg}.tsv', sep='\t')
