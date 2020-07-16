import sys
import os
import matplotlib as mpl
mpl.use('Agg')
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
import re
import numpy as np
import pandas as pd
import importlib
from datetime import datetime
import argparse
from not_for_upload.helper_functions import parse_output_dir, parse_input_path
from os.path import basename
# from FRETboard.algorithms.VanillaHmm import Classifier
from not_for_upload.MainTable_simulate import MainTable
from not_for_upload.FretReport_sim import FretReport
import yaml

np.random.seed(1)

def store_results(cl, data, params_dict, out_dir):
    report_str = FretReport(cl, data, params_dict['algo'], params_dict['buffer_size'],
                            params_dict['supervision_influence'], params_dict['bootstrap_size']).construct_html_report()
    with open(out_dir + 'FRETboard_report.html', 'w') as fh:
        fh.write(report_str)
    tm_str = re.search('(?<=var text = ").+(?=";)', report_str).group(0).replace('''\\n''', '''\n''')

    with open(out_dir + 'FRETboard_data_transition_rates.csv', 'w') as fh:
        fh.write(tm_str)

    # dat files
    dat_out_dir = parse_output_dir(out_dir + '/dat_files', clean=True)
    for fn in data.trace_dict:
        data.trace_dict[fn].predicted += 1
        data.trace_dict[fn].loc[:, 'label'] = data.label_dict.get(fn, None)
        data.trace_dict[fn].to_csv(dat_out_dir + fn, sep='\t', header=True, na_rep='NA', index=False)


parser = argparse.ArgumentParser(description='Simulate classification procedure of FRETboard'
                                             ' given labeled dat files.')
parser.add_argument('--in-dats', type=str, required=True)
parser.add_argument('--nb-states', type=int, required=True)
parser.add_argument('--params-file', type=str, required=True,
                    help='parameter values for initialization of FRETboard classifier in yaml file')
parser.add_argument('--label-dats', type=str, required=True)
parser.add_argument('--nb-manual', type=int, required=True,
                    help='number of traces of which manual input is simulated')
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--store-intermediates', action='store_true',
                    help='Store FRETboard results after every round of training')
parser.add_argument('--supervision-influence', type=float,
                    help='override supervision influence (lambda) in params file')
parser.add_argument('--allow-restart', action='store_true',
                    help='Allow simulation to pick up after crash')
args = parser.parse_args()


dat_list = parse_input_path(args.in_dats, pattern='*.dat')
label_list = parse_input_path(args.label_dats, pattern='*.dat')
out_dir = parse_output_dir(args.out_dir, clean=not args.allow_restart)

base_label_list = [basename(lab) for lab in label_list]
label_dict = {bl: ll for bl, ll in zip(base_label_list, label_list)}

dat_list = [fn for fn in dat_list if basename(fn) in base_label_list]
with open(args.params_file, 'r') as fh:
    params_dict = yaml.load(fh, Loader=yaml.SafeLoader)
if params_dict['DBSCAN_eps'] == 'nan': params_dict['DBSCAN_eps'] = np.nan
if args.supervision_influence is not None:
    params_dict['supervision_influence'] = args.supervision_influence

# load data
data = MainTable(eps=params_dict['DBSCAN_eps'], l=0, d=0, gamma=1, alex=0, dat_list=dat_list)

# Initialize classifier, train unsupervised
Classifier = importlib.import_module('FRETboard.algorithms.' + params_dict['algo']).Classifier
cl = Classifier(data=data, nb_states=args.nb_states, **params_dict)

# Load previous model if possible
model_list = [fn for fn in os.listdir(out_dir) if 'model_round' in fn]
if args.allow_restart and len(model_list):
    model_dict = {int(re.search('(?<=model_round_)[0-9]+', fn).group(0)): fn for fn in model_list}
    max_it = max(model_dict)
    print(f'{datetime.now()}: Loading model {max_it}')
    with open(f'{out_dir}{model_dict[max_it]}', 'r') as fh: modstr = fh.read()
    cl.load_params(modstr)
else:
    print(f'{datetime.now()}: start initial training (unsupervised)')
    cl.train(data_dict=data.trace_dict, supervision_influence=params_dict['supervision_influence'])
    print(f'{datetime.now()}: initial training done')
    max_it = 0

# First round of predictions
for idx in data.index_table.index:
    data.trace_dict[idx].loc[:, 'predicted'], data.index_table.loc[idx, 'logprob'] = cl.predict(data.trace_dict[idx])

# semi-supervised rounds
for n in range(max_it, args.nb_manual):
    with open(f'{out_dir}/model_round_{n}.txt', 'w') as fh: fh.write(cl.get_params())  # save model
    if args.store_intermediates:
        intermediate_dir = parse_output_dir(f'{out_dir}{n}')
        store_results(cl, data, params_dict, intermediate_dir)
    if n == 0:
        min_lp_idx = data.index_table.sample(1).index[0]
        # min_lp_idx = data.index_table.index[0]
    else:
        idx_bool = [idx not in data.label_dict for idx in data.index_table.index]
        unlabeled_index_table = data.index_table.loc[idx_bool, :]
        min_lp_idx = unlabeled_index_table.logprob.idxmin()
    data.label_dict[min_lp_idx] = pd.read_csv(label_dict[min_lp_idx], sep='\t').label.to_numpy(dtype=int) - 1
    data.manual_table.loc[min_lp_idx] = {'is_labeled': True, 'is_junk': False}
    print(f'{datetime.now()}: start manual round {n}')
    try:
        cl.train(data.trace_dict, supervision_influence=params_dict['supervision_influence'])
    except:
        print(f'error occurred at manual labeling of {min_lp_idx}')
        print(f'label_file dims: {data.label_dict[min_lp_idx].shape}')
        print(f'trace_file dims: {data.trace_dict[min_lp_idx].shape}')
        raise
    print(f'{datetime.now()}: finished manual round {n}')

    # Repeat prediction
    for idx in data.index_table.index:
        data.trace_dict[idx].loc[:, 'predicted'], data.index_table.loc[idx, 'logprob'] = cl.predict(
            data.trace_dict[idx])

with open(f'{out_dir}/model_end.txt', 'w') as fh: fh.write(cl.get_params())  # save model

# Store results
if args.store_intermediates:
    od = parse_output_dir(f'{out_dir}{args.nb_manual}')
else:
    od = out_dir
store_results(cl, data, params_dict, od)
