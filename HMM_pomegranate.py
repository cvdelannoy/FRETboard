import pandas as pd
import numpy as np
import pomegranate as pg
from pomegranate.kmeans import Kmeans
from random import choices
import itertools
from itertools import permutations
# to show hmm graph: plt.figure(dpi=600); hmm.plot(); plt.show()

class Classifier(object):
    """ HMM classifier that automatically adds 'edge states' to better recognize valid transitions between states.

    """

    def __init__(self, **kwargs):
        self.trained = None
        self.feature_list = ['E_FRET', 'i_sum', 'sd_roll']
        self.nb_states = kwargs['nb_states']
        self.gui = kwargs['gui']
        self.data = self.gui.data._data

        self.state_names = None  # names assigned to states in same order as pomegranate model
        self.pg_gui_state_dict = dict()
        self.gui_state_dict = dict()

    # --- training ---
    def train(self, supervision_influence=1.0):
        """
        Generate trained hmm and predict examples
        """
        # todo: ignore influence for now
        self.trained = self.get_trained_hmm(supervision_influence=supervision_influence)
        self.predict()

    def get_trained_hmm(self, supervision_influence=1.0, bootstrap=False):

        # Make selection of tuples, if bootstrapping (otherwise use all data)
        if bootstrap:
            nb_labeled = self.data.is_labeled.sum()
            nb_unlabeled = len(self.data) - nb_labeled
            labeled_seqs = choices(self.data.index[self.data.is_labeled], k=nb_labeled)
            unlabeled_seqs = choices(self.data.index[np.invert(self.data.is_labeled)], k=nb_unlabeled)
            seq_idx = labeled_seqs + unlabeled_seqs
        else:
            seq_idx = self.data.index

        # Get initialized hmm (structure + initial parameters)
        hmm = self.get_untrained_hmm(seq_idx)

        # Fit model on data
        # Case 1: supervised --> perform no training
        if supervision_influence < 1.0:
            if any(self.data.is_labeled):
                # Case 2: semi-supervised --> perform training with inertia on pre-determined labeled sequences
                labels = list(self.data.labels.to_numpy(copy=True))
                labels = [list(lab) if len(lab) else [None] for lab in labels]
                hmm.fit(self.get_matrix(self.data.loc[seq_idx, self.feature_list]),
                        inertia=supervision_influence, labels=labels)
            else:
                # Case 3: unsupervised --> just train
                hmm.fit(self.get_matrix(self.data.loc[seq_idx, self.feature_list]), inertia=0.0)
        return hmm

    def get_untrained_hmm(self, seq_idx):
        """
        return an untrained pomegranate hmm object with parameters filled in
        - If all data is unlabeled: finds emission parameters using k-means, transmission and start p are equal
        - If some data is labeled: initial estimate using given classifications
        :param data:
        :return:
        """
        hmm = pg.HiddenMarkovModel()
        buffer = self.gui.buffer_slider.value

        # Get emission distributions & transition probs
        states, edge_states, pg_gui_state_dict = self.get_states(seq_idx)
        tm_dict, pstart_dict, pend_dict = self.get_transitions(seq_idx)


        # Add states, self-transitions, transitions to start/end state
        for sidx, s_name in enumerate(states):
            s = states[s_name]
            hmm.add_state(s)
            hmm.add_transition(hmm.start, s, pstart_dict[s_name])
            hmm.add_transition(s, hmm.end, pend_dict[s_name])
            hmm.add_transition(s, s, tm_dict[(s_name, s_name)])

        # Make connections between states using edge states
        for es_name in edge_states:
            es_list = edge_states[es_name][0]
            s1, s2 = [states[s] for s in edge_states[es_name][1]]
            for es in es_list: hmm.add_state(es)
            hmm.add_transition(s1, es_list[0], tm_dict[edge_states[es_name][1]])
            for i in range(1, buffer):
                hmm.add_transition(es_list[i-1], es_list[i], 1)
            hmm.add_transition(es_list[-1], s2, 1.0)
        hmm.bake()

        state_names = np.array([state.name for state in hmm.states])
        self.pg_gui_state_dict = pg_gui_state_dict
        self.gui_state_dict = {si: pg_gui_state_dict.get(s, None) for si, s in enumerate(state_names)}
        return hmm

    def get_states(self, seq_idx):
        """
        Return dicts of pomgranate states with initialized normal multivariate distributions
        """
        buffer = self.gui.buffer_slider.value
        left_buffer = buffer // 2
        right_buffer = buffer - left_buffer
        data = self.data.loc[seq_idx, :]
        if not any(data.is_labeled):
            # Estimate emission distributions (same as pomegranate does usually)
            data_vec = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, self.feature_list].to_numpy()], 0)
            km = Kmeans(k=self.nb_states, n_init=1).fit(X=data_vec)
            y = km.predict(data_vec)
            def distfun(s1, s2):
                return pg.MultivariateGaussianDistribution.from_samples(data_vec[np.logical_or(y == s1, y == s2), :])
        else:
            # Estimate emission distributions from given class labels
            data = data.loc[data.is_labeled, :]
            y = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, 'labels'].to_numpy()], 0)
            y_edge = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, 'edge_labels'].to_numpy()], 0)
            data_vec = np.concatenate([np.stack(list(tup), axis=-1) for tup in data.loc[:, self.feature_list].to_numpy()], 0)
            def distfun(s1, s2):
                # return pg.MultivariateGaussianDistribution.from_samples(data_vec[np.logical_or(y == s1, y == s2), :])
                return pg.MultivariateGaussianDistribution.from_samples(data_vec[y_edge == f'e{s1}{s2}', :])

        # Create states
        pg_gui_state_dict = dict()
        states = dict()
        for i in range(self.nb_states):
            sn = f's{i}'
            states[sn] = pg.State(pg.MultivariateGaussianDistribution.from_samples(data_vec[y == i, :]), name=f's{i}')
            pg_gui_state_dict[sn] = i

        # Create edge states
        edges = list(permutations(range(self.nb_states), 2))
        edge_states = dict()
        for edge in edges:
            sn = f'e{edge[0]}{edge[1]}'
            estates_list = list()
            for i in range(buffer):
                estates_list.append(pg.State(distfun(*edge), name=f'e{edge[0]}{edge[1]}_{i}'))
                pg_gui_state_dict[f'{sn}_{i}'] = edge[0] if i < left_buffer else edge[1]
            edge_states[sn] = [estates_list, (f's{edge[0]}', f's{edge[1]}')]
        return states, edge_states, pg_gui_state_dict

    def get_transitions(self, seq_idx):
        data = self.data.loc[seq_idx, :]
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

    def predict(self):
        # dm = self.get_matrix(self.data.loc[:, self.feature_list])
        tuple_list = [np.stack(list(tup), axis=-1) for tup in self.data.loc[:, self.feature_list].to_numpy()]
        logprob_list = [self.trained.predict_log_proba(tup) for tup in tuple_list]
        logprob_path_list = [np.sum(np.max(logprob, axis=1)) for logprob in logprob_list]
        self.data.logprob = logprob_path_list
        pred_list = [np.argmax(logprob, axis=1) for logprob in logprob_list]
        pred_list = [np.vectorize(self.gui_state_dict.__getitem__)(pred) for pred in pred_list]
        self.data.prediction = pred_list

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

    def get_states_mu(self, fidx):
        mu_dict = {self.pg_gui_state_dict[state.name]: state.distribution.mu
                   for state in self.trained.states if not state.is_silent()}
        mu_list = [mu_dict[mk][fidx] for mk in sorted(list(mu_dict))]
        return mu_list

    def get_states_sd(self, fidx):
        dg = np.eye(len(self.feature_list), dtype=bool)
        sd_dict = {self.pg_gui_state_dict[state.name]: state.distribution.cov[dg]
                   for state in self.trained.states if not state.is_silent()}
        sd_list = [sd_dict[mk][fidx] for mk in sorted(list(sd_dict))]
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
        return df


    # --- saving/loading models ---
    def get_params(self):
        return [self.trained.to_yaml()]

    def load_params(self, file_contents):
        self.trained = pg.HiddenMarkovModel().from_yaml(file_contents[0])
