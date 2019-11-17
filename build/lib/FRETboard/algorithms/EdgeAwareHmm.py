import pandas as pd
import numpy as np
import pomegranate as pg
from pomegranate.kmeans import Kmeans
from random import choices
import itertools
from itertools import permutations
import yaml
from joblib import Parallel, delayed


def parallel_predict(tup_list, mod):
    vit_list = [mod.viterbi(tup) for tup in tup_list]
    logprob_list, trace_state_list = list(map(list, zip(*vit_list)))
    state_list = []
    for tsl in trace_state_list:
        state_list.append([ts[0] for ts in tsl])
    return logprob_list, state_list


def get_dist(data_vec):
    dist_list = []
    for vec in data_vec:
        vec = vec[~np.isnan(vec)]
        if len(vec):
            dist_list.append(pg.NormalDistribution(np.nanmean(vec), max(np.std(vec), 1E-6)))
        else:
            dist_list.append(pg.NormalDistribution(0, 999999))
    return pg.IndependentComponentsDistribution(dist_list)


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
        self.nb_threads = kwargs.get('nb_threads', 8)
        self.feature_list = kwargs['features']

        self.nb_states = nb_states
        self.gui = kwargs.get('gui', None)
        self.data = data
        if self.gui:
            self.buffer = self.gui.buffer_slider.value
        else:
            self.buffer = kwargs['buffer']

        self.state_names = None  # names assigned to states in same order as pomegranate model
        self.pg_gui_state_dict = dict()
        self.gui_state_dict = dict()

    # --- training ---
    def train(self, supervision_influence=1.0):
        """
        Generate trained hmm and predict examples
        """
        self.trained = self.get_trained_hmm(supervision_influence=supervision_influence)

    def get_trained_hmm(self, supervision_influence=1.0, bootstrap=False):

        # Make selection of tuples, if bootstrapping (otherwise use all data)
        if bootstrap:
            nb_labeled = self.data.data_clean.is_labeled.sum()
            nb_unlabeled = len(self.data.data_clean) - nb_labeled
            labeled_seqs = choices(self.data.data_clean.index[self.data.data_clean.is_labeled], k=nb_labeled)
            self.data._data.is_labeled = self.data._data.is_labeled.astype(bool)
            unlabeled_seqs = choices(self.data.data_clean.index[np.invert(self.data.data_clean.is_labeled)], k=nb_unlabeled)
            seq_idx = labeled_seqs + unlabeled_seqs
        else:
            seq_idx = self.data.data_clean.index

        # Get initialized hmm (structure + initial parameters)
        hmm = self.get_untrained_hmm(seq_idx)

        # Fit model on data
        # Case 1: supervised --> perform no training
        if supervision_influence < 1.0:
            if any(self.data.data_clean.is_labeled):
                # Case 2: semi-supervised --> perform training with inertia on pre-determined labeled sequences
                labels = list(self.data.data_clean.labels.to_numpy(copy=True))
                labels = [list(lab) if len(lab) else [None] for lab in labels]
                hmm.fit(self.get_matrix(self.data.data_clean.loc[seq_idx, self.feature_list]),
                        inertia=supervision_influence, labels=labels, n_jobs=self.nb_threads,
                        use_pseudocounts=True)
            else:
                # Case 3: unsupervised --> just train
                hmm.fit(self.get_matrix(self.data.data_clean.loc[seq_idx, self.feature_list]),
                        inertia=0.0, n_jobs=self.nb_threads)
        return hmm

    def get_untrained_hmm(self, seq_idx):
        """
        return an untrained pomegranate hmm object with parameters filled in
        - If all data is unlabeled: finds emission parameters using k-means, transmission and start p are equal
        - If some data is labeled: initial estimate using given classifications
        """
        hmm = pg.HiddenMarkovModel()

        # Get emission distributions & transition probs
        states, edge_states, pg_gui_state_dict = self.get_states(seq_idx)
        tm_dict, pstart_dict, pend_dict = self.get_transitions(seq_idx)
        for k in tm_dict: tm_dict[k] = max(tm_dict[k], 0.000001)  # reset 0-prob transitions to essentially 0, avoids nans on edges

        # Add states, self-transitions, transitions to start/end state
        for sidx, s_name in enumerate(states):
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
        return hmm

    def get_states(self, seq_idx):
        """
        Return dicts of pomgranate states with initialized normal multivariate distributions
        """
        left_buffer = self.buffer // 2
        data = self.data.data_clean.loc[seq_idx, :]
        if not any(data.is_labeled):
            # Estimate emission distributions (same as pomegranate does usually)
            data_vec = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, self.feature_list].to_numpy()], 0)
            if data_vec.shape[0] > 20000:  # avoid endless waiting for k-means guess in large dataset
                km_idx = np.random.choice(data_vec.shape[0], 10000, replace=False)
            else:
                km_idx = np.arange(data_vec.shape[0])
            km = Kmeans(k=self.nb_states, n_init=1).fit(X=data_vec[km_idx, :], n_jobs=self.nb_threads)
            y = km.predict(data_vec)
            def distfun(s1, s2):
                return get_dist(data_vec[np.logical_or(y == s1, y == s2), :].T)
        else:
            # Estimate emission distributions from given class labels
            data = data.loc[data.is_labeled, :]
            y = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, 'labels'].to_numpy()], 0)
            y_edge = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, 'edge_labels'].to_numpy()], 0)
            data_vec = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, self.feature_list].to_numpy()], 0)
            def distfun(s1, s2):
                return get_dist(data_vec[y_edge == f'e{s1}{s2}', :].T)

        # Create states
        pg_gui_state_dict = dict()
        states = dict()
        for i in range(self.nb_states):
            sn = f's{i}'
            states[sn] = pg.State(get_dist(data_vec[y == i, :].T), name=f's{i}')
            pg_gui_state_dict[sn] = i

        # Create edge states
        edges = list(permutations(range(self.nb_states), 2))
        edge_states = dict()
        for edge in edges:
            sn = f'e{edge[0]}{edge[1]}'
            estates_list = list()
            for i in range(self.buffer):
                estates_list.append(pg.State(distfun(*edge), name=f'e{edge[0]}{edge[1]}_{i}'))
                pg_gui_state_dict[f'{sn}_{i}'] = edge[0] if i < left_buffer else edge[1]
            edge_states[sn] = [estates_list, (f's{edge[0]}', f's{edge[1]}')]
        return states, edge_states, pg_gui_state_dict

    def get_transitions(self, seq_idx):
        data = self.data.data_clean.loc[seq_idx, :]
        labels_array = data.loc[data.is_labeled, 'labels'].to_numpy()
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

    def predict(self, idx):
        """
        Predict labels for given indices.

        :param idx: list of indices in self.data.data_clean for which to predict labels
        :returns:
        pred_list: list of numpy arrays of length len(idx) containing predicted labels
        logprob_list: list of floats of length len(idx) containing posterior log-probabilities
        """
        tuple_list = np.array([np.stack(list(tup), axis=-1)
                               for tup in self.data.data_clean.loc[idx, self.feature_list].to_numpy()])

        # fwd/bwd also logprobs parallel
        nb_threads = min(len(idx), self.nb_threads)
        batch_idx = np.array_split(np.arange(len(tuple_list)), nb_threads)
        parallel_list = Parallel(n_jobs=nb_threads)(delayed(parallel_predict)(tuple_list[bi], self.trained)
                                                   for bi in batch_idx)
        logprob_list, pred_list = list(map(list, zip(*parallel_list)))
        logprob_list = list(itertools.chain.from_iterable(logprob_list))
        pred_list = [np.vectorize(self.gui_state_dict.__getitem__)(pred[1:-1])
                     for pred in itertools.chain.from_iterable(pred_list)]

        return pred_list, logprob_list

    def get_matrix(self, df):
        """
        convert pandas dataframe to numpy array of shape [nb_sequences, nb_samples_per_sequence, nb_features]
        """
        return np.stack([np.stack(list(tup), axis=-1) for tup in df.to_numpy()], 0)

    # --- parameters and performance measures ---
    @property
    def confidence_intervals(self):
        """
        Calculate bootstrapped confidence intervals on transition matrix values
        :return:
        """
        tm_array = []
        for _ in range(10):
            hmm = self.get_trained_hmm(supervision_influence=self.gui.supervision_slider.value, bootstrap=True)
            tm_array.append(self.get_tm(hmm).to_numpy())
        tm_mat = np.stack(tm_array, axis=-1)
        sd_mat = np.std(tm_mat, axis=-1)
        mu_mat = np.mean(tm_mat, axis=-1)
        ci_mat = np.tile(np.expand_dims(mu_mat, -1), (1, 1, 2))
        ci_mat[:, :, 0] -= sd_mat * 2
        ci_mat[:, :, 1] += sd_mat * 2
        return ci_mat

    def get_states_mu(self, feature):
        fidx = np.argwhere(feature == np.array(self.feature_list))[0,0]
        mu_dict = {self.pg_gui_state_dict[state.name]: state.distribution.distributions[fidx].parameters[0]
                   for state in self.trained.states if not state.is_silent()}
        mu_list = [mu_dict[mk] for mk in sorted(list(mu_dict))]
        return mu_list

    def get_states_sd(self, feature):
        fidx = np.argwhere(feature == np.array(self.feature_list))[0, 0]
        sd_dict = {self.pg_gui_state_dict[state.name]: state.distribution.distributions[fidx].parameters[1]
                   for state in self.trained.states if not state.is_silent()}
        sd_list = [sd_dict[mk] for mk in sorted(list(sd_dict))]
        return sd_list

    def get_tm(self, hmm):
        df = pd.DataFrame({st: [0] * self.nb_states for st in range(self.nb_states)})
        df.set_index(df.columns, inplace=True)
        state_idx_dict = {st.name: idx for idx, st in enumerate(hmm.states)}
        dense_tm = hmm.dense_transition_matrix()
        for s0, s1 in [(i1, i2) for i1 in range(self.nb_states) for i2 in range(self.nb_states)]:
            if s0 == s1:
                si0 = state_idx_dict[f's{s0}']; si1 = state_idx_dict[f's{s1}']
            else:
                si0 = state_idx_dict[f's{s0}']; si1 = state_idx_dict[f'e{s0}{s1}_0']
            df.loc[s0, s1] = dense_tm[si0, si1]
        # todo: switching to transition rates i.o. probs here, for kinSoft challenge
        mean_durations = self.data.data_clean.time.apply(lambda x: (x[-1] - x[0]) / len(x))
        duration_frame = np.mean(mean_durations)
        df[np.eye(self.nb_states, dtype=bool)] -= 1
        df /= duration_frame
        return df

    # --- saving/loading models ---
    def get_params(self):
        mod_txt = self.trained.to_yaml()
        gui_state_dict_txt = yaml.dump(self.gui_state_dict)
        pg_gui_state_dict_txt = yaml.dump(self.pg_gui_state_dict)
        feature_txt = '\n'.join(self.feature_list)
        div = '\nSTART_NEW_SECTION\n'
        out_txt = ('EdgeAwareHmm'
                   + div + mod_txt
                   + div + feature_txt
                   + div + gui_state_dict_txt
                   + div + pg_gui_state_dict_txt
                   + div + f'nb_states: {str(self.nb_states)}\nbuffer: {self.buffer}')
        return out_txt

    def load_params(self, file_contents):
        (mod_check,
         model_txt,
         feature_txt,
         gui_state_dict_txt,
         pg_gui_state_dict_txt,
         misc_txt) = file_contents.split('\nSTART_NEW_SECTION\n')
        if mod_check != 'EdgeAwareHmm':
            error_msg = '\nERROR: loaded model parameters are not for a boundary-aware HMM!'
            if self.gui:
                self.gui.text = error_msg
            else:
                raise ValueError(error_msg)
        self.trained = pg.HiddenMarkovModel().from_yaml(model_txt)
        self.feature_list = feature_txt.split('\n')
        self.gui_state_dict = yaml.load(gui_state_dict_txt, Loader=yaml.FullLoader)
        self.pg_gui_state_dict = yaml.load(pg_gui_state_dict_txt, Loader=yaml.FullLoader)
        misc_dict = yaml.load(misc_txt, Loader=yaml.SafeLoader)
        self.nb_states = misc_dict['nb_states']
        self.buffer = misc_dict['buffer']
