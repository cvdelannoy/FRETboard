import pandas as pd
import numpy as np
from hmmlearn import hmm
import itertools
from sklearn.utils import check_array
from hmmlearn.utils import iter_from_X_lengths
import warnings

import holoviews as hv
hv.extension('bokeh')


class HiddenMarkovModel(object):
    """ A class for the HMM used to detect FRET signals

    """

    def __init__(self, **kwargs):
        self.trained_hmm = None
        self.feature_list = ['i_fret', 'i_sum']
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
        # overall_accuracy = np.sum(nb_correct) / np.sum(nb_points)
        # return overall_accuracy

    @property
    def data(self):
        return self._data

    @property
    def untrained_hmm(self):
        if any(self.data.is_labeled):
            hmm_out = hmm.GaussianHMM(n_components=self.nb_states, covariance_type='diag', init_params='')
            (hmm_out.startprob_,
             hmm_out.transmat_,
             hmm_out.means_,
             hmm_out.covars_) = self.get_supervised_params()
        else:
            hmm_out = hmm.GaussianHMM(n_components=self.nb_states, covariance_type='diag')
        return hmm_out

    def decode(self, X, lengths=None, algorithm=None):
        """Basically a copy of build-in decode method of hmmlearn, but with posterior per sequence
        """
        algorithm = algorithm or self.trained_hmm.algorithm

        decoder = {
            "viterbi": self.trained_hmm._decode_viterbi,
            "map": self.trained_hmm._decode_map
        }[algorithm]

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = []
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            logprobij, state_sequenceij = decoder(X[i:j])
            logprob.append(logprobij)
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence


    def train(self):
        train_mat, train_seq_lengths = self.construct_matrix(np.invert(self.data.is_labeled))
        hmm_obj = self.untrained_hmm
        with warnings.catch_warnings():  # catches deprecation warning sklearn: log_multivariate_normal_density
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            try:
                hmm_obj.fit(train_mat, train_seq_lengths)
            except:
                print(str(hmm_obj.n_components))
                print(str(hmm_obj.startprob_))
        self.trained_hmm = hmm_obj

        # update predictions todo: move into separate method some day(?)
        logprob, pred = self.decode(self.data_mat, self.seq_lengths)
        pred_list = np.split(pred, np.cumsum(self.seq_lengths)[:-1])
        self.data.prediction = pred_list
        self.data.logprob = logprob

    @data.setter
    def data(self, dat_files):
        nb_files = len(dat_files)
        df_out = pd.DataFrame({
            'i_don': [np.array([], dtype=np.int64)] * nb_files,
            'i_acc': [np.array([], dtype=np.int64)] * nb_files,
            'i_sum': [np.array([], dtype=np.float64)] * nb_files,
            'i_fret': [np.array([], dtype=np.float64)] * nb_files,
            'is_labeled': [False] * nb_files,
            'labels': [np.array([], dtype=np.int64)] * nb_files,
            'prediction': [np.array([], dtype=np.int64)] * nb_files,
            'logprob': [np.array([], dtype=np.float64)] * nb_files},
            index=dat_files)

        for dat_file in dat_files:
            try:
                fc = np.loadtxt(dat_file)
                i_don = fc[:, 1]
                i_acc = fc[:, 2]
                i_sum = np.sum((i_don, i_acc), axis=0)
                i_sum = i_sum / i_sum.max()
                i_fret = np.divide(i_acc, np.sum((i_don, i_acc), axis=0))
                df_out.at[dat_file, 'i_don'] = i_don
                df_out.at[dat_file, 'i_acc'] = i_acc
                df_out.at[dat_file, 'i_sum'] = i_sum
                df_out.at[dat_file, 'i_fret'] = i_fret
            except:
                print('File {} could not be read, skipping'.format(dat_file))
                df_out.drop([dat_file], inplace=True)
        self._data = df_out
        self.data_mat, self.seq_lengths = self.construct_matrix()

    def get_supervised_params(self):
        """
        Extract stat probs, means, covars and transitions from lists of numpy vectors
        containing feature values and labels
        """
        feature_list = list()
        for feature_name in self.feature_list:
            feature_vec = self.data.loc[self.data.is_labeled, feature_name]
            feature_vec = np.concatenate(feature_vec).reshape(-1,1)
            feature_list.append(feature_vec)
        # feature_list = self.data.loc[self.data.is_labeled, 'i_fret']
        label_list = self.data.loc[self.data.is_labeled, 'labels']
        label_vec = np.concatenate(label_list).reshape(-1, 1)

        # classes = np.sort(np.unique(label_vec))
        # nb_classes = classes.size
        classes = np.arange(self.nb_states)
        means = np.zeros([self.nb_states, len(feature_list)])
        covars = np.zeros([self.nb_states, len(feature_list)])

        # set start probs
        first_class_list = [ll[0] for ll in label_list]
        start_probs = np.array([sum([cl == cur_cl for cl in first_class_list])
                                / len(first_class_list)
                                for cur_cl in classes])
        # start_probs = np.expand_dims(start_probs, axis=1)
        if 1 in start_probs:
            sp_max_idx = start_probs == start_probs.max()
            start_probs[sp_max_idx] = 1 - 1e-10 * (self.nb_states - 1)
            start_probs[np.invert(sp_max_idx)] = 1e-10

        # means & covars
        for cl in classes:
            for fi, feature in enumerate(feature_list):
                feature_cl_vec = feature[label_vec == cl]
                if feature_cl_vec.size > 0:
                    means[cl, fi] = np.mean(feature_cl_vec)
                    covars[cl, fi] = np.var(feature_cl_vec)

        # transition matrix
        transmat = np.zeros([self.nb_states, self.nb_states])
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
            transmat[tra[0], tra[1]] = transition_dict[tra] / total_transitions[tra[0]]
        return start_probs, transmat, means, covars

    def construct_matrix(self, bool_idx=None):
        if bool_idx is None:
            bool_idx = np.repeat(True, self.data.shape[0])
        data_subset = self.data.loc[bool_idx, :]
        vec_list = list()
        seq_lengths = data_subset.apply(lambda x: x.i_fret.size, axis=1)
        for feature in self.feature_list:
            vec = np.concatenate(data_subset.loc[:,feature].values).reshape(-1,1)
            vec_list.append(vec)
        return np.concatenate(vec_list, axis=1), seq_lengths
