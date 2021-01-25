import numpy as np
import pomegranate as pg
from pomegranate.kmeans import Kmeans
from datetime import datetime
from random import choices, sample
import itertools
from itertools import permutations
import yaml
from FRETboard.helper_functions import numeric_timestamp, get_edge_labels, discrete2continuous



def parallel_predict(tup_list, mod):
    vit_list = [mod.viterbi(tup) for tup in tup_list]
    logprob_list, trace_state_list = list(map(list, zip(*vit_list)))
    state_list = []
    for tsl in trace_state_list:
        state_list.append([ts[0] for ts in tsl])
    return logprob_list, state_list

class Classifier(object):
    """ HMM classifier that automatically adds 'edge states' to better recognize valid transitions between states.
    """
    def __init__(self, nb_states, data, **kwargs):
        """
        :param nb_states: number of states to detect
        :param data: object of class MainTable
        :param gui: Gui object [optional]
        :param buffer: int, size of buffer area around regular classes [required if gui not given]
        """
        self.trained = None
        self.framerate = None
        self.timestamp = numeric_timestamp()
        self.nb_threads = kwargs.get('nb_threads', 8)
        self.feature_list = kwargs['features']
        self.buffer = kwargs.get('buffer', 3)

        self.nb_states = nb_states
        self.data = data

        self.state_names = None  # names assigned to states in same order as pomegranate model
        self.pg_gui_state_dict = dict()
        self.gui_state_dict = dict()
        self.str2num_state_dict = dict()

    # --- training ---
    def train(self, data_dict, supervision_influence=1.0):
        """
        Generate trained hmm and predict examples
        """
        self.supervision_influence = supervision_influence
        self.trained = self.get_trained_hmm(data_dict)
        self.timestamp = numeric_timestamp()

    def get_trained_hmm(self, data_dict, bootstrap=False):
        nb_subsample = 100
        if self.framerate is None:
            self.framerate = 1 / np.concatenate([data_dict[tr].time.iloc[1:].to_numpy() -
                                                 data_dict[tr].time.iloc[:-1].to_numpy() for tr in data_dict]).mean()

        idx_list = list(data_dict)
        nb_labeled = self.data.manual_table.is_labeled.sum()
        nb_unlabeled = len(idx_list) - nb_labeled
        unlabeled_idx = [idx for idx in idx_list if idx not in self.data.manual_table.query('is_labeled').index]

        # Take bootstrapped /subsampled sample
        if bootstrap:
            labeled_seqs = [si for si in self.data.manual_table.query('is_labeled').index if si in data_dict]
            labeled_seqs = choices(labeled_seqs, k=nb_labeled)
            if nb_unlabeled <= nb_subsample:
                # bootstrap size m == n
                unlabeled_seqs = choices(unlabeled_idx, k=nb_unlabeled)
            else:
                # subsampling, m = 100
                unlabeled_seqs = sample(unlabeled_idx, k=nb_subsample)
            seq_idx = labeled_seqs + unlabeled_seqs
            data_dict = {si: data_dict[si] for si in seq_idx}
        elif nb_unlabeled > nb_subsample:
            labeled_seqs = self.data.manual_table.query('is_labeled').index.to_list()
            unlabeled_seqs = sample(unlabeled_idx, k=nb_subsample)
            seq_idx = labeled_seqs + unlabeled_seqs
            data_dict = {si: data_dict[si] for si in seq_idx}

        # Get initialized hmm (structure + initial parameters)
        hmm = self.get_untrained_hmm(data_dict)

        # Fit model on data
        # Case 1: supervised --> perform no training
        X = [data_dict[dd].loc[:, self.feature_list].to_numpy() for dd in data_dict]
        X = [x.reshape(-1).reshape(-1, len(self.feature_list)) for x in X]
        if self.supervision_influence < 1.0 and any(self.data.manual_table.is_labeled):
            # Case 2: semi-supervised --> perform training with lambda as weights
            labels = []
            for li in data_dict:
                if self.data.manual_table.loc[li, 'is_labeled'] and not self.data.manual_table.loc[li, 'is_junk']:
                    labels.append([f's{int(lab)}' for lab in self.data.label_dict[li]])
                else:
                    labels.append(None)
            nsi = 1.0 - self.supervision_influence
            weights = [nsi if lab is None else self.supervision_influence for lab in labels]
            labels = [self.add_boundary_labels(lab, hmm) if lab is not None else None for lab in labels]
            hmm.fit(X, weights=weights, labels=labels, use_pseudocount=True, algorithm='baum-welch', max_iterations=100,
                    # inertia=0.9,
                    # batches_per_epoch=20
                    )
        elif not any(self.data.manual_table.is_labeled):
            # Case 3: unsupervised --> just train
            hmm.fit(X, use_pseudocount=True, algorithm='baum-welch', max_iterations=100,
                    # batches_per_epoch=20
                    )
        return hmm

    def add_boundary_labels(self, labels, hmm):
        """
        Add transitional boundary layers when labels switch class
        """
        if not len(labels): return None
        states = hmm.states
        out_labels = [None] * len(labels)

        overhang_right = self.buffer // 2
        overhang_left = self.buffer - overhang_right  # -1 for current position

        oh_counter = 0
        prev_label = None
        cur_label = labels[0][1:]
        for li, l in enumerate(labels):
            if l[1:] == cur_label:
                if oh_counter != 0:
                    out_labels[li] = f'e{prev_label}_{cur_label}_{self.buffer - oh_counter}'
                    oh_counter -= 1
                else:
                    try:
                        out_labels[li] = f's{cur_label}'
                    except:
                        print(f'out_labels: {str(out_labels)}')
                        print(f'li: {li}')
                        print(f'nb states {len(states)}')
                        print(f'{str(self.str2num_state_dict)}')
                        print(f's{cur_label}')
                        raise
            else:
                if overhang_left > li:
                    # left margin does not allow for full set of border states; do not mark as overhang
                    out_labels[li] = f's{cur_label}'
                    continue
                prev_label = cur_label
                cur_label = l[1:]
                oh_counter = overhang_right
                out_labels[li-overhang_left + 1:li + 1] = [f'e{prev_label}_{cur_label}_{i}' for i in range(overhang_left)]
        return [hmm.start.name] + out_labels + [hmm.end]

    def get_untrained_hmm(self, data_dict):
        """
        return an untrained pomegranate hmm object with parameters filled in
        - If all data is unlabeled: finds emission parameters using k-means, transmission and start p are equal
        - If some data is labeled: initial estimate using given classifications
        """
        hmm = pg.HiddenMarkovModel()

        # Get emission distributions & transition probs
        states, edge_states, pg_gui_state_dict = self.get_states(data_dict)
        tm_dict, pstart_dict, pend_dict = self.get_transitions(data_dict)
        # for k in tm_dict: tm_dict[k] = max(tm_dict[k], 0.000001)  # reset 0-prob transitions to essentially 0, avoids nans on edges
        # for k in pstart_dict: pstart_dict[k] = max(pstart_dict[k], 0.000001)
        # for k in pend_dict: pend_dict[k] = max(pend_dict[k], 0.000001)

        # Add states, self-transitions, transitions to start/end state
        for s_name in states:
            s = states[s_name]
            hmm.add_state(s)
            hmm.add_transition(hmm.start, s, pstart_dict[s_name], pseudocount=0)
            hmm.add_transition(s, hmm.end, pend_dict[s_name], pseudocount=0)
            hmm.add_transition(s, s, tm_dict[(s_name, s_name)], pseudocount=0)

        # Make connections between states using edge states
        for es_name in edge_states:
            es_list = edge_states[es_name][0]
            s1, s2 = [states[s] for s in edge_states[es_name][1]]
            for es in es_list: hmm.add_state(es)
            hmm.add_transition(s1, es_list[0], tm_dict[edge_states[es_name][1]])
            for i in range(1, self.buffer):
                hmm.add_transition(es_list[i-1], es_list[i], 1.0, pseudocount=9999999)
            hmm.add_transition(es_list[-1], s2, 1.0, pseudocount=9999999)
        hmm.bake()

        state_names = np.array([state.name for state in hmm.states])
        self.pg_gui_state_dict = pg_gui_state_dict
        self.gui_state_dict = {si: pg_gui_state_dict.get(s, None) for si, s in enumerate(state_names)}
        self.str2num_state_dict = {str(si): ni for si, ni in zip(state_names, list(self.gui_state_dict))}
        return hmm

    def get_states(self, data_dict): # todo check
        """
        Return dicts of pomgranate states with initialized normal multivariate distributions
        """
        left_buffer = self.buffer // 2
        if not any(self.data.manual_table.is_labeled):
            # Estimate emission distributions (same as pomegranate does usually)
            data_vec = np.concatenate([dat.loc[:, self.feature_list].to_numpy() for dat in data_dict.values()], 0)
            if data_vec.shape[0] > 1000:  # avoid endless waiting for k-means guess in large dataset
                km_idx = np.random.choice(data_vec.shape[0], 1000, replace=False)
            else:
                km_idx = np.arange(data_vec.shape[0])
            km = Kmeans(k=self.nb_states, n_init=1).fit(X=data_vec[km_idx, :])
            y = km.predict(data_vec)
            def distfun(s1, s2):
                return self.get_dist(data_vec[np.logical_or(y == s1, y == s2), :].T)
        else:
            # Estimate emission distributions from given class labels
            labeled_indices = self.data.manual_table.query('is_labeled and not is_junk').index
            data_vec = np.concatenate([data_dict[idx].loc[:, self.feature_list].to_numpy()
                                       for idx in data_dict if idx in labeled_indices], 0)
            y = np.concatenate([self.data.label_dict[idx]
                                for idx in data_dict if idx in labeled_indices], 0)
            y_edge = np.concatenate([get_edge_labels(self.data.label_dict[idx].astype(int), self.buffer)
                                     for idx in data_dict if idx in labeled_indices], 0)
            def distfun(s1, s2):
                return self.get_dist(data_vec[y_edge == f'e{s1}_{s2}', :].T)

        # Create states
        pg_gui_state_dict = dict()
        states = dict()
        for i in range(self.nb_states):
            sn = f's{i}'
            X = data_vec[y == i, :].T.copy()
            if X.size == 0 and self.trained is not None:
                states[sn] = self.trained.states[self.str2num_state_dict[sn]]
            else:
                states[sn] = pg.State(self.get_main_dist(data_vec[y == i, :].T.copy()), name=f's{i}')
            pg_gui_state_dict[sn] = i
        present_states = list(states)

        # Create edge states
        edges = list(permutations(range(self.nb_states), 2))
        # edges = list(permutations(np.unique(y.astype(int)), 2))
        edge_states = dict()
        for edge in edges:
            # if not (f's{edge[0]}' in present_states and f's{edge[0]}' in present_states): continue
            sn = f'e{edge[0]}_{edge[1]}'
            estates_list = list()
            for i in range(self.buffer):
                estates_list.append(pg.State(distfun(*edge), name=f'e{edge[0]}_{edge[1]}_{i}'))
                pg_gui_state_dict[f'{sn}_{i}'] = int(edge[0]) if i < left_buffer else int(edge[1])
            edge_states[sn] = [estates_list, (f's{edge[0]}', f's{edge[1]}')]
        return states, edge_states, pg_gui_state_dict

    def get_main_dist(self, X):
        return self.get_dist(X)

    @staticmethod
    def get_dist(data_vec):
        dist_list = []
        for vec in data_vec:
            vec = vec[~np.isnan(vec)]
            if len(vec):
                dist_list.append(pg.NormalDistribution(np.nanmean(vec), max(np.std(vec), 1E-6)))
            else:
                dist_list.append(pg.NormalDistribution(0, 999999))
        return pg.IndependentComponentsDistribution(dist_list)

    def get_transitions(self, data_dict):
        labeled_indices = self.data.manual_table.query('is_labeled and not is_junk').index
        labels_array = [self.data.label_dict[dd] for dd in data_dict if dd in labeled_indices]
        nb_seqs = len(labels_array)
        if nb_seqs == 0:
            # Equal transition probs if no labeled sequences given
            ps_prior = 1.0 / self.nb_states
            t_prior = 1.0 / (self.nb_states + 1)
            pstart_dict = {f's{s}': ps_prior for s in range(self.nb_states)}
            pend_dict = {f's{s}': t_prior for s in range(self.nb_states)}
            out_dict = {(f's{perm[0]}', f's{perm[1]}'): t_prior
                        for perm in itertools.product(list(range(self.nb_states)), repeat=2)}
            return out_dict, pstart_dict, pend_dict

        # start prob matrix
        start_labels = np.array([tup[0] for tup in labels_array])
        pstart_dict = {f's{s}': np.sum(start_labels == s) / nb_seqs for s in range(self.nb_states)}

        # end prob matrix (filled with transition matrix)
        pend_dict = dict()

        # transition matrix
        transition_dict = {perm: 0 for perm in list(itertools.product(list(range(self.nb_states)), repeat=2)) +
                           [(st, -1) for st in range(self.nb_states)]}
        out_dict = dict()
        total_transitions = dict()

        # Count occurring transitions
        for labels in labels_array:
            labels = np.append(labels, -1)
            for n in range(1, len(labels)):
                dimer = (labels[n - 1], labels[n])
                transition_dict[dimer] += 1
        for tra in transition_dict:
            if total_transitions.get(tra[0]):
                total_transitions[tra[0]] += transition_dict[tra]
            else:
                total_transitions[tra[0]] = transition_dict[tra]
        for tra in transition_dict:
            tt = total_transitions[tra[0]]
            if tra[1] == -1:
                pend_dict[f's{tra[0]}'] = transition_dict[tra] / tt if tt != 0 else 0.0
            else:
                out_dict[(f's{tra[0]}', f's{tra[1]}')] = transition_dict[tra] / tt if tt != 0 else 0.0
        return out_dict, pstart_dict, pend_dict

    def predict(self, trace_df, hmm=None):
        """
        Predict labels for given indices.

        :param idx: index [str] in self.data.data_clean for which to predict labels
        :returns:
        pred_list: list of numpy arrays of length len(idx) containing predicted labels
        logprob_list: list of floats of length len(idx) containing posterior log-probabilities
        """
        if hmm is None: hmm = self.trained
        logprob, trace_state_list = hmm.viterbi(np.split(trace_df.loc[:, self.feature_list].to_numpy(),
                                                         len(trace_df), axis=0))
        if trace_state_list is None:  # sequence is impossible, logprob -inf
            return np.zeros(len(trace_df)), 1E-90
        state_list = np.vectorize(self.gui_state_dict.__getitem__)([ts[0] for ts in trace_state_list[1:-1]])
        return state_list, logprob

    def get_matrix(self, df):
        """
        convert pandas dataframe to numpy array of shape [nb_sequences, nb_samples_per_sequence, nb_features]
        """
        return np.stack([np.stack(list(tup), axis=-1) for tup in df.to_numpy()], 0)

    # --- parameters and performance measures ---
    def get_data_tm(self, trace_dict, out_labels, nb_bootstrap_iters):
        """
        Calculate bootstrapped confidence intervals on data-derived transition matrix values
        :return:
        """
        state_order_dict = {state.name: idx for idx, state in enumerate(self.trained.states)}

        # actual value
        # actual_tm = self.tm_from_hmm(self.trained, state_order_dict)
        trace_list = list(trace_dict.values())
        nb_traces = len(trace_list)
        if self.framerate is None:  # todo occurs when model is loaded, never retrained. Find better solution
            self.framerate = np.mean(trace_list[0].time[1:].to_numpy() - trace_list[0].time[:-1].to_numpy())
        actual_tm = self.tm_from_seq(trace_list)

        # CIs
        tm_array = []
        invalid_indices = [idx for idx, tup in self.data.manual_table.iterrows() if tup.is_junk]
        idx_list = [idx for idx in trace_dict if idx not in invalid_indices]
        trace_dict = {tr: trace_dict[tr] for tr in trace_dict if tr in idx_list}
        for n in range(nb_bootstrap_iters):
            # print(f'{datetime.now()}: bootstrap round {n}')
            # hmm = self.get_trained_hmm(trace_dict, bootstrap=True)
            # tm = self.tm_from_hmm(hmm, state_order_dict)
            bs_idx = np.random.choice(nb_traces, size=nb_traces)
            tm = self.tm_from_seq([trace_list[it] for it in bs_idx])
            if tm is None: continue
            tm_array.append(tm)
        tm_mat = np.stack(tm_array, axis=-1)
        sd_mat = np.std(tm_mat, axis=-1)
        mu_mat = np.mean(tm_mat, axis=-1)
        ci_mat = np.tile(np.expand_dims(mu_mat, -1), (1, 1, 2))
        ci_mat[:, :, 0] -= sd_mat * 2
        ci_mat[:, :, 1] += sd_mat * 2
        return actual_tm, ci_mat

    def tm_from_hmm(self, hmm, state_order_dict):
        try:
            full_tm = hmm.dense_transition_matrix()
            tm = np.zeros([self.nb_states, self.nb_states])
            state_list = list(range(self.nb_states))
            for s1, s2 in permutations(state_list, 2):
                tm[s1, s2] = full_tm[state_order_dict[f's{s1}'], state_order_dict[f'e{s1}_{s2}_0']]
            for s in state_list:
                tm[s,s] = full_tm[state_order_dict[f's{s}'], state_order_dict[f's{s}']]
        except:
            return None
        return discrete2continuous(tm, self.framerate)

    def tm_from_seq(self, seq_list):
        tm_out = np.zeros([self.nb_states, self.nb_states], dtype=int)
        for seq in seq_list:
            seq_pred = seq.predicted.astype(int)
            for tr in zip(seq_pred[:-1], seq_pred[1:]):
                tm_out[tr[0], tr[1]] += 1
        return discrete2continuous(tm_out / np.expand_dims(tm_out.sum(axis=1), -1), self.framerate)

    def get_mus(self, feature):
        state_names = [f's{i}' for i in range(self.nb_states)]
        fidx = np.argwhere(feature == np.array(self.feature_list))[0,0]
        mu_dict = {self.pg_gui_state_dict[state.name]: state.distribution.distributions[fidx].parameters[0]
                   for state in self.trained.states if state.name in state_names}
        mu_list = [mu_dict[mk] for mk in sorted(list(mu_dict))]
        return mu_list

    def get_sds(self, feature):
        state_names = [f's{i}' for i in range(self.nb_states)]
        fidx = np.argwhere(feature == np.array(self.feature_list))[0, 0]
        sd_dict = {self.pg_gui_state_dict[state.name]: state.distribution.distributions[fidx].parameters[1]
                   for state in self.trained.states if state.name in state_names}
        sd_list = [sd_dict[mk] for mk in sorted(list(sd_dict))]
        return sd_list

    # --- saving/loading models ---
    def get_params(self):
        mod_txt = self.trained.to_yaml()
        gui_state_dict_txt = yaml.dump(self.gui_state_dict)
        pg_gui_state_dict_txt = yaml.dump(self.pg_gui_state_dict)
        str2num_state_dict_txt = yaml.dump(self.str2num_state_dict)
        feature_txt = '\n'.join(self.feature_list)
        div = '\nSTART_NEW_SECTION\n'
        out_txt = ('EdgeAwareHmm'
                   + div + mod_txt
                   + div + feature_txt
                   + div + gui_state_dict_txt
                   + div + pg_gui_state_dict_txt
                   + div + str2num_state_dict_txt
                   + div + f'nb_states: {str(self.nb_states)}\nbuffer: {self.buffer}\ndbscan_epsilon: {self.data.eps}')
        return out_txt

    def load_params(self, file_contents):
        (mod_check,
         model_txt,
         feature_txt,
         gui_state_dict_txt,
         pg_gui_state_dict_txt,
         str2num_state_dict_txt,
         misc_txt) = file_contents.split('\nSTART_NEW_SECTION\n')
        if mod_check != 'EdgeAwareHmm':
            error_msg = '\nERROR: loaded model parameters are not for a substate HMM!'
            raise ValueError(error_msg)
        self.trained = pg.HiddenMarkovModel().from_yaml(model_txt)
        self.feature_list = feature_txt.split('\n')
        self.gui_state_dict = yaml.load(gui_state_dict_txt, Loader=yaml.FullLoader)
        self.pg_gui_state_dict = yaml.load(pg_gui_state_dict_txt, Loader=yaml.FullLoader)
        self.str2num_state_dict = yaml.load(str2num_state_dict_txt, Loader=yaml.FullLoader)
        misc_dict = yaml.load(misc_txt, Loader=yaml.SafeLoader)
        if misc_dict['dbscan_epsilon'] == 'nan': misc_dict['dbscan_epsilon'] = np.nan
        self.nb_states = misc_dict['nb_states']
        self.buffer = misc_dict['buffer']
        self.data.eps = misc_dict['dbscan_epsilon']
