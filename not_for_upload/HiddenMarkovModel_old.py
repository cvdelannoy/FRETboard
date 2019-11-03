import pandas as pd
import numpy as np
from hmmlearn import hmm
from random import choices
import itertools
from sklearn.utils import check_array
from hmmlearn.utils import iter_from_X_lengths
import warnings
from FRETboard.helper_functions import rolling_var, rolling_cross_correlation, rolling_corr_coef

class HiddenMarkovModel(object):
    """ A class for the HMM used to detect FRET signals

    """

    def __init__(self, **kwargs):
        self.trained = None
        self.trained_edge = None
        self.feature_list = ['E_FRET', 'i_sum', 'sd_roll']
        # self.feature_list = ['E_FRET', 'sd_roll']
        self.nb_states = kwargs['nb_states']
        self.data = kwargs['data']

    @property
    def accuracy(self):
        if not any(self.data.is_labeled):
            return np.array([np.nan], dtype=float)
        labeled_data = self.data.loc[self.data.is_labeled, ('prediction', 'labels')]
        nb_correct = labeled_data.apply(lambda x: np.sum(np.equal(x.prediction, x.labels)), axis=1)
        nb_points = labeled_data.apply(lambda x: x.labels.size, axis=1)
        return nb_correct / nb_points * 100


    @property
    def confidence_intervals(self):
        tm_array = []
        for _ in range(10):
            hmm = self.get_trained_hmm(influence=1.0, bootstrap=True)
            tm_array.append(hmm.transmat_)
        tm_mat = np.stack(tm_array, axis=-1)
        sd_mat = np.std(tm_mat, axis=-1)
        mu_mat = np.mean(tm_mat, axis=-1)
        ci_mat = np.tile(np.expand_dims(mu_mat, -1), (1, 1, 2))
        ci_mat[:, :, 0] -= sd_mat * 2
        ci_mat[:, :, 1] += sd_mat * 2
        return ci_mat

    @property
    def conf95(self):
        """
        return bootstrapped 95 percent confidence intervals +/1
        :return:
        """
        tm_array = []
        for _ in range(10):
            hmm = self.get_trained_hmm(influence=1.0, bootstrap=True)
            tm_array.append(hmm.transmat_)
        tm_mat = np.stack(tm_array, axis=-1)
        sd_mat = np.std(tm_mat, axis=-1)
        return sd_mat * 2

    @property
    def params_list(self):
        tm_array = self.trained.transmat_.reshape(-1, 1).squeeze()
        means = self.trained.means_.reshape(-1, 1).squeeze()
        covar_list = [blk.reshape(1, -1).squeeze() for blk in
                      np.split(self.trained.covars_, axis=0,
                               indices_or_sections=self.nb_states)]
        covars = np.concatenate(covar_list)
        # covars = self.classifier.trained.covars_.reshape(-1, 1).squeeze()
        startprob = self.trained.startprob_.reshape(-1, 1).squeeze()
        nb_states = np.expand_dims(self.nb_states, 0)
        params = np.concatenate((nb_states, tm_array, means, covars, startprob))
        return params.tolist()

    @property
    def data(self):
        return self._data

    def decode(self, X, lengths=None, algorithm=None):
        """Basically a copy of build-in decode method of hmmlearn, but with posterior per sequence
        """
        algorithm = algorithm or self.trained.algorithm

        decoder = {
            "viterbi": self.trained._decode_viterbi,
            "map": self.trained._decode_map
        }[algorithm]

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = []
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                logprobij, state_sequenceij = decoder(X[i:j])
            logprob.append(logprobij)
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence

    def get_states_mu(self, fidx):
        return self.trained.means_[:,fidx]

    def get_states_sd(self, fidx):
        return self.trained.covars_[:, fidx, fidx]


    def get_untrained_hmm(self, influence, training_seqs=None):
        if any(self.data.is_labeled):
            print('generating supervised hmm')
            hmm_out = hmm.GaussianHMM(n_components=self.nb_states, covariance_type='diag', init_params='')
            (hmm_out.startprob_,
             hmm_out.transmat_,
             hmm_out.means_,
             hmm_out.covars_) = self.get_supervised_params(influence=influence, training_seqs=training_seqs)
            hmm_edge = hmm.GaussianHMM(n_components=2, covariance_type='diag', init_params='')
            (hmm_edge.startprob_,
             hmm_edge.transmat_,
             hmm_edge.means_,
             hmm_edge.covars_) = self.get_supervised_params(influence=influence, training_seqs=training_seqs,
                                                            labels='edge_labels')
        else:
            print('generating unsupervised hmm')
            hmm_out = hmm.GaussianHMM(n_components=self.nb_states, covariance_type='diag', init_params='stmc')
            hmm_edge = hmm.GaussianHMM(n_components=2, covariance_type='diag', init_params='stmc')
        return hmm_out, hmm_edge

    def get_trained_hmm(self, influence=1.0, bootstrap=False):
        nb_labeled = self.data.is_labeled.sum()
        nb_unlabeled = len(self.data) - nb_labeled
        labeled_seqs = choices(self.data.index[self.data.is_labeled], k=nb_labeled) if bootstrap else None
        hmm, hmm_edge = self.get_untrained_hmm(influence=influence, training_seqs=labeled_seqs)
        if not nb_unlabeled:
            return hmm, hmm_edge
        else:
            unlabeled_seqs = choices(self.data.index[np.invert(self.data.is_labeled)], k=nb_unlabeled) if bootstrap \
                else np.invert(self.data.is_labeled)

            # Train state hmm
            train_mat, train_seq_lengths = self.construct_matrix(unlabeled_seqs)
            with warnings.catch_warnings():  # catches deprecation warning sklearn: log_multivariate_normal_density
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                hmm.fit(train_mat, train_seq_lengths)

            # Train edge hmm
            train_mat, train_seq_lengths = self.construct_matrix(unlabeled_seqs)
            with warnings.catch_warnings():  # catches deprecation warning sklearn: log_multivariate_normal_density
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # hmm_edge.fit(train_mat, train_seq_lengths)

            # Train hierarchical hmm


            return hmm, hmm_edge

    def train(self, influence=1.0):
        self.trained, self.trained_edge = self.get_trained_hmm(influence=influence)
        # update predictions
        self.predict()

    def predict(self):
        self.data_mat, self.seq_lengths = self.construct_matrix()
        logprob, pred = self.decode(self.data_mat, self.seq_lengths)
        pred_list = np.split(pred, np.cumsum(self.seq_lengths)[:-1])
        self.data.prediction = pred_list
        self.data.logprob = logprob

    @data.setter
    def data(self, dat_files):
        nb_files = len(dat_files)
        df_out = pd.DataFrame({
            'time': [np.array([], dtype=np.int64)] * nb_files,
            'i_don': [np.array([], dtype=np.int64)] * nb_files,
            'i_acc': [np.array([], dtype=np.int64)] * nb_files,
            'i_sum': [np.array([], dtype=np.float64)] * nb_files,
            'E_FRET': [np.array([], dtype=np.float64)] * nb_files,
            'sd_roll': [np.array([], dtype=np.float64)] * nb_files,
            'labels': [np.array([], dtype=np.int64)] * nb_files,
            'edge_labels': [np.array([], dtype=np.int64)] * nb_files,
            'prediction': [np.array([], dtype=np.int64)] * nb_files,
            'logprob': [np.array([], dtype=np.float64)] * nb_files},
            index=dat_files)
        df_out['is_labeled'] = pd.Series([False] * nb_files, dtype=np.bool)
        for dat_file in dat_files:
            try:
                fc = np.loadtxt(dat_file)
                fc_out = self.read_line(fc.T)
                df_out.at[dat_file, 'time'] = fc_out[0]
                df_out.at[dat_file, 'i_don'] = fc_out[1]
                df_out.at[dat_file, 'i_acc'] = fc_out[2]
                df_out.at[dat_file, 'i_sum'] = fc_out[3]
                df_out.at[dat_file, 'E_FRET'] = fc_out[4]
                df_out.at[dat_file, 'sd_roll'] = fc_out[5]
            except:
                print('File {} could not be read, skipping'.format(dat_file))
                df_out.drop([dat_file], inplace=True)
        self._data = df_out
        self.data_mat, self.seq_lengths = self.construct_matrix()

    def read_line(self, fc, full_set=False):
        # window = 9
        window = 15
        ss = (window - 1) // 2  # sequence shortening
        fc[fc <= 0] = np.finfo(np.float64).eps  # hacky, required to get rid of overzealous background subtraction
        time = fc[0, :]
        i_don = fc[1, :]
        i_acc = fc[2, :]
        i_sum = np.sum((i_don, i_acc), axis=0)
        # i_sum = i_sum / i_sum.max()F
        E_FRET = np.divide(i_acc, np.sum((i_don, i_acc), axis=0))
        # sd_roll = rolling_var(E_FRET, window)
        sd_roll = rolling_corr_coef(i_don, i_acc, window)
        if full_set:
            return (time[ss:-ss], i_don[ss:-ss], i_acc[ss:-ss],
                    i_sum[ss:-ss], E_FRET[ss:-ss], sd_roll,
                    np.array([], dtype=np.int64), np.array([], dtype=np.int64),  # labels, edge labels
                    np.array([], dtype=np.int64),  # prediction
                    np.array([], dtype=np.float64), False)  # logprob, is_labeled
        return time[ss:-ss], i_don[ss:-ss], i_acc[ss:-ss], i_sum[ss:-ss], E_FRET[ss:-ss], sd_roll
        # return i_don, i_acc, i_sum, E_FRET, sd_roll

    # def del_data_tuple(self, idx):
    #     self._data.drop(idx)
    #     self.set_matrix()

    def get_supervised_params(self, influence, training_seqs=None, labels='labels'):
        """
        Extract stat probs, means, covars and transitions from lists of numpy vectors
        containing feature values and labels

        training_seqs: sequences to use in finding parameters. May be bool vector or index names
        """
        ws = influence
        wus = 1 - ws
        if training_seqs is None:
            training_seqs = self.data.is_labeled

        feature_list = list()
        for feature_name in self.feature_list:
            feature_vec = self.data.loc[training_seqs, feature_name]
            feature_vec = np.concatenate(list(feature_vec)).reshape(-1, 1)
            feature_list.append(feature_vec)
        nb_features = len(feature_list)
        # feature_list = self.data.loc[self.data.is_labeled, 'E_FRET']
        label_list = self.data.loc[training_seqs, labels]
        empty_seqs = label_list.index[label_list.apply(lambda x: len(x) == 0)]
        if len(empty_seqs):
            label_list.loc[empty_seqs] = self.data.loc[empty_seqs, 'prediction']  # TODO: for testing purposes!!
        label_vec = np.concatenate(list(label_list)).reshape(-1, 1)

        classes = np.arange(self.nb_states)

        # set start probs
        first_class_list = [ll[0] for ll in label_list]
        start_probs_s = np.array([sum([cl == cur_cl for cl in first_class_list])
                                / len(first_class_list)
                                for cur_cl in classes])
        if 1 in start_probs_s:
            sp_max_idx = start_probs_s == start_probs_s.max()
            start_probs_s[sp_max_idx] = 1 - 1e-10 * (self.nb_states - 1)
            start_probs_s[np.invert(sp_max_idx)] = 1e-10
        start_probs = ws * start_probs_s + wus * self.trained.startprob_

        # means & covars
        means_s = np.zeros([self.nb_states, nb_features])
        # covars_s = np.zeros([self.nb_states, nb_features, nb_features])
        covars_s = np.zeros([self.nb_states, nb_features])
        for cl in classes:
            feature_cl = np.row_stack([fl[label_vec == cl] for fl in feature_list])
            if feature_cl.shape[1] != 0:
                # covars_s[cl, :, :] = np.cov(feature_cl)
                covars_s[cl, :] = np.cov(feature_cl)[np.eye(nb_features,dtype=bool)]
                means_s[cl, :] = feature_cl.mean(axis=1)
        covars_prev = self.trained.covars_[np.tile(np.eye(nb_features, dtype=bool), (self.nb_states, 1, 1))].reshape(self.nb_states, nb_features)
        covars = covars_s * ws + covars_prev * wus
        covars[covars == 0] = np.finfo(float).eps
        # covars = covars_s * ws + self.trained.covars_ * wus
        means = means_s * ws + self.trained.means_ * wus

        # transition matrix
        transmat = np.zeros((self.nb_states, self.nb_states))
        transition_dict = {perm: 0 for perm in itertools.product(classes, repeat=2)}
        # Count occurring transitions
        for cur_labels in label_list:
            for n in range(1, len(cur_labels)):
                dimer = (cur_labels[n - 1], cur_labels[n])
                transition_dict[dimer] += 1
        total_transitions = dict()
        for tra in transition_dict:
            if total_transitions.get(tra[0]):
                total_transitions[tra[0]] += transition_dict[tra]
            else:
                total_transitions[tra[0]] = transition_dict[tra]
        for tra in transition_dict:
            tt = total_transitions[tra[0]]
            transmat[tra[0], tra[1]] = transition_dict[tra] / tt if tt != 0 else 0
        for ti in range(self.nb_states):
            transmat[ti, ti] = transmat[ti, ti] + (1.0 - transmat[ti, :].sum())  # ensure rows sum to 1
        transmat = transmat * ws + self.trained.transmat_ * wus
        return start_probs, transmat, means, covars

    def construct_matrix(self, bool_idx=None):
        if self.data.size == 0:
            return np.array([]), [0]
        if bool_idx is None:
            bool_idx = np.repeat(True, self.data.shape[0])
        data_subset = self.data.loc[bool_idx, :]
        vec_list = list()
        seq_lengths = data_subset.apply(lambda x: x.E_FRET.size, axis=1)
        for feature in self.feature_list:
            vec = np.concatenate(data_subset.loc[:, feature].values).reshape(-1,1)
            vec_list.append(vec)
        return np.concatenate(vec_list, axis=1), seq_lengths

    # def set_matrix(self):
    #     self.data_mat, self.seq_lengths = self.construct_matrix()

    def load_params(self, file_contents):
        params = [float(i) for i in file_contents]
        self.nb_states = int(params[0])
        params = params[1:]
        new_hmm = hmm.GaussianHMM(n_components=self.nb_states, covariance_type='full', init_params='')
        param_idx = np.cumsum([self.nb_states ** 2, self.nb_states * 2, self.nb_states * 2 * 2])
        tm, means, covars, start_prob = np.split(params, param_idx)
        new_hmm.transmat_ = tm.reshape(self.nb_states, self.nb_states)
        new_hmm.means_ = means.reshape(self.nb_states, 2)
        new_hmm.covars_ = covars.reshape(self.nb_states, 2, 2)
        new_hmm.startprob_ = start_prob
        self.trained = new_hmm
