import pandas as pd
import numpy as np
import pomegranate as pg
from pomegranate.kmeans import Kmeans
from random import choices
import itertools
import yaml
from FRETboard.helper_functions import numeric_timestamp
from joblib import Parallel, delayed


def parallel_predict(tup_list, mod):
    vit_list = [mod.viterbi(tup) for tup in tup_list]
    logprob_list, trace_state_list = list(map(list, zip(*vit_list)))
    state_list = []
    for tsl in trace_state_list:
        state_list.append([ts[0] for ts in tsl])
    return logprob_list, state_list

class Classifier(object):
    """ HMM model class
    """

    # --- training ---
    def train(self, data_dict, supervision_influence=1.0):
        """
        Generate trained hmm and predict examples
        """
        self.supervision_influence = supervision_influence
        self.trained = self.get_trained_hmm(data_dict)
        self.timestamp = numeric_timestamp()

    def get_trained_hmm(self, data_dict, bootstrap=False):

        # Make selection of tuples, if bootstrapping (otherwise use all data)
        if bootstrap:
            nb_labeled = self.data.manual_table.is_labeled.sum()
            nb_unlabeled = len(self.data.index_table) - nb_labeled
            labeled_seqs = choices(self.data.index_table.index[self.data.manual_table.is_labeled], k=nb_labeled)
            unlabeled_seqs = choices(self.data.index_table.index[np.invert(self.data.manual_table.is_labeled)], k=nb_unlabeled)
            seq_idx = labeled_seqs + unlabeled_seqs
            data_dict = {si: data_dict[si] for si in seq_idx}
        # else:
        #     seq_idx = list(data_dict)

        # Get initialized hmm (structure + initial parameters)
        hmm = self.get_untrained_hmm(data_dict)

        # Fit model on data
        # Case 1: supervised --> perform no training
        if self.supervision_influence < 1.0:
            if any(self.data.manual_table.is_labeled):
                # Case 2: semi-supervised --> perform training with lambda as weights
                labels = [list(data_dict[dd].labels) if self.data.manual_table.loc[dd, 'is_labeled'] else None for dd in data_dict]  # todo check if training goes alright
                # labels = list(self.data.data_clean.labels.to_numpy(copy=True))
                nsi = 1.0 - self.supervision_influence
                weights = [nsi if lab is None else self.supervision_influence for lab in labels]
                hmm.fit([data_dict[dd].loc[:, self.feature_list] for dd in data_dict],  # todo check dimensions: [nb_sequences, nb_samples_per_sequence, nb_features]
                        weights=weights, labels=labels, n_jobs=self.nb_threads,
                        use_pseudocount=True)
            else:
                # Case 3: unsupervised --> just train
                hmm.fit([data_dict[dd].loc[:, self.feature_list] for dd in data_dict],
                        n_jobs=self.nb_threads, use_pseudocount=True)
        return hmm

    def get_untrained_hmm(self, data_dict):
        """
        return an untrained pomegranate hmm object with parameters filled in
        - If all data is unlabeled: finds emission parameters using k-means, transmission and start p are equal
        - If some data is labeled: initial estimate using given classifications
        """

        # Get emission distributions & transition probs
        dists, pg_gui_state_dict = self.get_states(data_dict)
        trans_df, pstart_dict, pend_dict = self.get_transitions(data_dict)
        trans_df.replace(0, 0.000001, inplace=True)
        # for k in tm_dict: tm_dict[k] = max(tm_dict[k], 0.000001)  # reset 0-prob transitions to essentially 0, avoids nans on edges
        for k in pstart_dict: pstart_dict[k] = max(pstart_dict[k], 0.000001)
        for k in pend_dict: pend_dict[k] = max(pend_dict[k], 0.000001)

        state_names = list(dists)
        tm_mat = trans_df.loc[state_names, state_names].to_numpy()
        hmm = pg.HiddenMarkovModel.from_matrix(transition_probabilities=tm_mat,
                                               distributions=dists.values(),
                                               starts=[pstart_dict[sn] for sn in state_names],
                                               ends=[pend_dict[sn] for sn in state_names],
                                               state_names=state_names)
        self.pg_gui_state_dict = pg_gui_state_dict
        self.gui_state_dict = {si: pg_gui_state_dict.get(s, None) for si, s in enumerate(state_names)}
        self.str2num_state_dict = {str(si): ni for si, ni in zip(state_names, list(self.gui_state_dict))}
        return hmm

    def get_states(self, data_dict):
        """
        Return dicts of pomgranate states with initialized normal multivariate distributions
        """
        if not any(self.data.manual_table.is_labeled):
            # Estimate emission distributions (same as pomegranate does usually)
            data_vec = np.concatenate([dat.loc[:, self.feature_list].to_numpy() for dat in data_dict.values()], 0)
            if data_vec.shape[0] > 1000:  # avoid endless waiting for k-means guess in large dataset
                km_idx = np.random.choice(data_vec.shape[0], 1000, replace=False)
            else:
                km_idx = np.arange(data_vec.shape[0])
            km = Kmeans(k=self.nb_states, n_init=1).fit(X=data_vec[km_idx, :])
            y = km.predict(data_vec)
        else:
            # Estimate emission distributions from given class labels
            labeled_indices = self.data.manual_table.query('is_labeled').index
            data_vec = np.concatenate([data_dict[idx].loc[:, self.feature_list].to_numpy()
                                       for idx in data_dict if idx in labeled_indices], 0)
            y = np.concatenate([self.data.label_dict[idx]
                                       for idx in data_dict if idx in labeled_indices], 0)

        # Create distributions
        pg_gui_state_dict = dict()
        dists = dict()
        for i in range(self.nb_states):
            sn = f's{i}'
            dists[sn] = self.get_dist(data_vec[y == i, :].T)
            pg_gui_state_dict[sn] = i
        return dists, pg_gui_state_dict

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
        labels_array = [self.data.label_dict[dd]
                        for dd in data_dict if self.data.manual_table.loc[dd, 'is_labeled']]
        nb_seqs = len(labels_array)
        state_names = [f's{s}' for s in range(self.nb_states)]
        trans_df = pd.DataFrame([], columns=state_names, index=state_names)
        if nb_seqs == 0:
            # Equal transition probs if no labeled sequences given
            ps_prior = 1.0 / self.nb_states
            t_prior = 1.0 / (self.nb_states + 1)
            pstart_dict = {f's{s}': ps_prior for s in range(self.nb_states)}
            pend_dict = {f's{s}': t_prior for s in range(self.nb_states)}
            for perm in itertools.product(list(range(self.nb_states)), repeat=2):
                trans_df.loc[f's{perm[0]}', f's{perm[1]}'] = t_prior
            return trans_df, pstart_dict, pend_dict

        # start prob matrix
        start_labels = np.array([tup[0] for tup in labels_array])
        pstart_dict = {f's{s}': np.sum(start_labels == s) / nb_seqs for s in range(self.nb_states)}

        # end prob matrix (filled with transition matrix)
        pend_dict = dict()

        # transition matrix
        transition_dict = {perm: 0 for perm in list(itertools.product(list(range(self.nb_states)), repeat=2)) +
                           [(st, -1) for st in range(self.nb_states)]}
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
                trans_df.loc[f's{tra[0]}', f's{tra[1]}'] = transition_dict[tra] / tt if tt != 0 else 0.0
        return trans_df, pstart_dict, pend_dict

    def predict(self, trace_df, hmm=None):
        """
        Predict labels for given indices.

        :param idx: list of indices in self.data.data_clean for which to predict labels
        :returns:
        pred_list: list of numpy arrays of length len(idx) containing predicted labels
        logprob_list: list of floats of length len(idx) containing posterior log-probabilities
        """
        if hmm is None: hmm = self.trained
        logprob, trace_state_list = hmm.viterbi(np.split(trace_df.loc[:, self.feature_list].to_numpy(),
                                                         len(trace_df), axis=0))
        state_list = np.vectorize(self.gui_state_dict.__getitem__)([ts[0] for ts in trace_state_list[1:-1]])
        return state_list, logprob

        # tuple_list = np.array([np.stack(list(tup), axis=-1)
        #                        for tup in self.data.data_clean.loc[idx, self.feature_list].to_numpy()])
        #
        # # fwd/bwd also logprobs parallel
        # nb_threads = min(len(idx), self.nb_threads)
        # batch_idx = np.array_split(np.arange(len(tuple_list)), nb_threads)
        # parallel_list = Parallel(n_jobs=nb_threads)(delayed(parallel_predict)(tuple_list[bi], self.trained)
        #                                            for bi in batch_idx)
        # logprob_list, pred_list = list(map(list, zip(*parallel_list)))
        # logprob_list = list(itertools.chain.from_iterable(logprob_list))
        # pred_list = [np.vectorize(self.gui_state_dict.__getitem__)(pred[1:-1])
        #              for pred in itertools.chain.from_iterable(pred_list)]
        #
        # return pred_list, logprob_list

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
        # actual value
        actual_tm = self.tm_from_seq(out_labels)

        # CIs
        tm_array = []
        invalid_indices = [idx for idx, tup in self.data.manual_table.iterrows() if tup.is_labeled or tup.is_junk]
        valid_indices = [idx for idx in self.data.index_table.index if idx not in invalid_indices]
        if len(valid_indices) > 100:
            idx_list = np.random.choice(valid_indices, 100)
        else:
            idx_list = valid_indices
        idx_list = np.concatenate((idx_list, self.data.manual_table.query('is_labeled').index), axis=0)
        trace_dict = {tr: trace_dict[tr] for tr in trace_dict if tr in idx_list}
        for _ in range(nb_bootstrap_iters):
            hmm = self.get_trained_hmm(trace_dict, bootstrap=True)
            seqs = [self.predict(trace_dict[idx], hmm)[0] for idx in idx_list]
            tm_array.append(self.tm_from_seq(seqs))
        tm_mat = np.stack(tm_array, axis=-1)
        sd_mat = np.std(tm_mat, axis=-1)
        mu_mat = np.mean(tm_mat, axis=-1)
        ci_mat = np.tile(np.expand_dims(mu_mat, -1), (1, 1, 2))
        ci_mat[:, :, 0] -= sd_mat * 2
        ci_mat[:, :, 1] += sd_mat * 2
        return actual_tm, ci_mat

    def tm_from_seq(self, seq_list):
        # concatenate seqs with stop symbol in between
        # seq = np.concatenate([seq + [99] for seq in seq_list])
        tm_out = np.zeros([self.nb_states, self.nb_states], dtype=int)
        # inventory of transitions
        for seq in seq_list:
            for tr in zip(seq[:-1], seq[1:]):
                tm_out[tr[0], tr[1]] += 1
        return tm_out / np.expand_dims(tm_out.sum(axis=1), -1)

    def get_confidence_intervals(self, data_dict):
        """
        Calculate bootstrapped confidence intervals on transition matrix values
        :return:
        """
        tm_array = []
        for _ in range(10):
            hmm = self.get_trained_hmm(data_dict=data_dict, bootstrap=True)
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
            si0 = state_idx_dict[f's{s0}']; si1 = state_idx_dict[f's{s1}']
            df.loc[s0, s1] = dense_tm[si0, si1]
        df = df / df.sum(1).T  # correct for start/end transitions
        # # todo: switching to transition rates i.o. probs here, for kinSoft challenge
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
        str2num_state_dict_txt = yaml.dump(self.str2num_state_dict)
        feature_txt = '\n'.join(self.feature_list)
        div = '\nSTART_NEW_SECTION\n'
        out_txt = ('VanillaHmm'
                   + div + mod_txt
                   + div + feature_txt
                   + div + gui_state_dict_txt
                   + div + pg_gui_state_dict_txt
                   + div + str2num_state_dict_txt
                   + div + f'nb_states: {str(self.nb_states)}\ndbscan_epsilon: {self.data.eps}')
        return out_txt

    def load_params(self, file_contents):
        (mod_check,
         model_txt,
         feature_txt,
         gui_state_dict_txt,
         pg_gui_state_dict_txt,
         str2num_state_dict_txt,
         misc_txt) = file_contents.split('\nSTART_NEW_SECTION\n')
        if mod_check != 'VanillaHmm':
            error_msg = '\nERROR: loaded model parameters are not for a Vanilla HMM!'
            # if self.gui:
            #     self.gui.notify(error_msg)
            #     return
            # else:
            raise ValueError(error_msg)
        self.trained = pg.HiddenMarkovModel().from_yaml(model_txt)
        self.feature_list = feature_txt.split('\n')
        self.gui_state_dict = yaml.load(gui_state_dict_txt, Loader=yaml.FullLoader)
        self.pg_gui_state_dict = yaml.load(pg_gui_state_dict_txt, Loader=yaml.FullLoader)
        self.str2num_state_dict = yaml.load(str2num_state_dict_txt, Loader=yaml.FullLoader)
        misc_dict = yaml.load(misc_txt, Loader=yaml.SafeLoader)
        if misc_dict['dbscan_epsilon'] == 'nan': misc_dict['dbscan_epsilon'] = np.nan
        self.nb_states = misc_dict['nb_states']
        self.data.eps = misc_dict['dbscan_epsilon']
