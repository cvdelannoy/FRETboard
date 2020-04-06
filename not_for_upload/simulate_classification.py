import sys
import os
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
import re
import numpy as np
import pandas as pd
import importlib
import argparse
from not_for_upload.helper_functions import parse_output_dir, parse_input_path
from os.path import basename
# from FRETboard.algorithms.VanillaHmm import Classifier
from not_for_upload.MainTable_simulate import MainTable
from not_for_upload.FretReport_sim import FretReport
import yaml


parser = argparse.ArgumentParser(description='Simulate classification procedure of FRETboard'
                                             ' given labeled dat files.')
parser.add_argument('--in-dats', type=str, required=True)
parser.add_argument('--params-file', type=str, required=True,
                    help='parameter values for initialization of FRETboard classifier in yaml file')
parser.add_argument('--label-dats', type=str, required=True)
parser.add_argument('--nb-manual', type=int, required=True,
                    help='number of traces of which manual input is simulated')
parser.add_argument('--out-dir', type=str, required=True)
args = parser.parse_args()


dat_list = parse_input_path(args.in_dats, pattern='*.dat')
label_list = parse_input_path(args.label_dats, pattern='*.dat')
out_dir = parse_output_dir(args.out_dir, clean=True)
dat_out_dir = parse_output_dir(args.out_dir + '/dat_files', clean=True)

base_label_list = [basename(lab) for lab in label_list]
label_dict = {bl: ll for bl, ll in zip(base_label_list, label_list)}

dat_list = [fn for fn in dat_list if basename(fn) in base_label_list]
with open(args.params_file, 'r') as fh:
    params_dict = yaml.load(fh, Loader=yaml.SafeLoader)
if params_dict['DBSCAN_eps'] == 'nan': params_dict['DBSCAN_eps'] = np.nan

# load data
data = MainTable(eps=params_dict['DBSCAN_eps'], l=0, d=0, gamma=1, alex=0, dat_list=dat_list)

# Initialize classifier, train unsupervised
Classifier = importlib.import_module('FRETboard.algorithms.' + params_dict['algo']).Classifier
cl = Classifier(data=data, **params_dict)
cl.train(data_dict=data.trace_dict, supervision_influence=params_dict['supervision_influence'])

for idx in data.index_table.index:
    data.trace_dict[idx].loc[:, 'predicted'], data.index_table.loc[idx, 'logprob'] = cl.predict(data.trace_dict[idx])
for n in range(args.nb_manual):
    if n == 0:
        min_lp_idx = data.index_table.sample(1).index[0]
        # min_lp_idx = data.index_table.index[0]
    else:
        idx_bool = [idx not in data.label_dict for idx in data.index_table.index]
        unlabeled_index_table = data.index_table.loc[idx_bool, :]
        min_lp_idx = unlabeled_index_table.index[unlabeled_index_table.logprob.argmin()]
    # min_lp_idx = data.index_table.sample(1).index[0]
    data.label_dict[min_lp_idx] = pd.read_csv(label_dict[min_lp_idx], sep='\t').predicted.to_numpy() - 1
    data.manual_table.loc[min_lp_idx] = {'is_labeled': True, 'is_junk': False}
    cl.train(data.trace_dict, supervision_influence=params_dict['supervision_influence'])

    # Repeat prediction
    for idx in data.index_table.index:
        data.trace_dict[idx].loc[:, 'predicted'], data.index_table.loc[idx, 'logprob'] = cl.predict(
            data.trace_dict[idx])

# Store results

# Report
report_str = FretReport(cl, data, params_dict['algo'], params_dict['buffer_size'],
                        params_dict['supervision_influence'], params_dict['bootstrap_size']).construct_html_report()
with open(out_dir+'FRETboard_report.html', 'w') as fh:
    fh.write(report_str)
tm_str = re.search('(?<=var text = ").+(?=";)', report_str).group(0).replace('''\\n''', '''\n''')

with open(out_dir+'FRETboard_data_transition_rates.csv', 'w') as fh:
    fh.write(tm_str)

# dat files
for fn in data.trace_dict:
    data.trace_dict[fn].predicted += 1
    data.trace_dict[fn].loc[:, 'label'] = data.label_dict.get(fn, None)
    data.trace_dict[fn].to_csv(dat_out_dir+fn, sep='\t', header=True, na_rep='NA', index=False)
