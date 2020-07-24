import pandas as pd
import numpy as np
from random import choices, sample
import yaml
from FRETboard.helper_functions import numeric_timestamp

# FRETBOARD MODEL TEMPLATE
# Using this template you can create your own FRETboard model. For a working model, you need to implement at most 4 steps:
# 1: a model generation/training procedure
# 2: a predict function
# 3: Optionally, parameter return functions
# 4: Optinally, model saving and loading functions
#
# You are completely free to implement any model, however as FRETboard is meant to rapidly estimate and re-estimate models
# to react to the user's input we advise to keep your model implementations light, e.g. fully training a neural network
# from the ground up might work but will probably slow down your analysis a lot.
#
# If you have written a model that you think might be useful for others, please consider submitting it as a pull request
# on the project github page:
#
# https://github.com/cvdelannoy/FRETboard.git)
#
# or get in touch with me:
#
# carlos.delannoy@wur.nl


class Classifier(object):
    """ A template for your own FRETboard-model!
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
        self.supervision_influence = kwargs['supervision_influence']

        self.nb_states = nb_states
        self.data = data

    # --- training ---
    def train(self, data_dict, supervision_influence=1.0):
        """
        Generate trained hmm and predict examples
        """
        self.supervision_influence = supervision_influence
        self.trained = self.get_trained_model(data_dict)
        self.timestamp = numeric_timestamp()

    def get_trained_model(self, data_dict, bootstrap=False):
        """
        STEP 1:
        Write a function that returns your trained model. The model should be trained on the traces provided
        in data_dict. If bootstrap is provided your model should be trained on a bootstrapped sample of data_dict.
        Don' worry about that though, a general implementation of bootstrapping that modifies the data_dict variable
        is already there waiting for you!


        :param data_dict: dict of traces. Keys are trace ids, values are pandas dataframes of shape
                          [nb_timepoints, nb_features + time]
        :param bootstrap: bool, if true, train on bootstrapped sample of data_dict instead
        :return: trained model
        """
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
            if nb_unlabeled <= 100:
                # bootstrap size m == n
                unlabeled_seqs = choices(unlabeled_idx, k=nb_unlabeled)
            else:
                # subsampling, m = 100
                unlabeled_seqs = sample(unlabeled_idx, k=100)
            seq_idx = labeled_seqs + unlabeled_seqs
            data_dict = {si: data_dict[si] for si in seq_idx}
        elif nb_unlabeled > 100:
            labeled_seqs = self.data.manual_table.query('is_labeled').index.to_list()
            unlabeled_seqs = sample(unlabeled_idx, k=100)
            seq_idx = labeled_seqs + unlabeled_seqs
            data_dict = {si: data_dict[si] for si in seq_idx}

        # Your implementation goes here


        return my_awesome_model



    def predict(self, trace_df, model=None):
        """
        STEP 2:
        implement your predict function. Given a pandas data frame containing the trace (trace_df), it should return
        a numpy array of integers indicating the class for each time point, and the log-probability of the classification
        being correct as a single float (log_prob). The latter will be reported in the FRETboard GUI in a histogram,
        however if you're just experimenting or your classifier can't generate such a measure for some reason, just
        provide a 0. Predictions should be made with the model in self.trained, but if a model is provided as a parameter
        (see below), please allow for this model to be used instead, to ensure proper functioning of bootstrapping
        confidence intervals.

        :param trace_df: a pandas dataframe containing a single trace of shape [nb_timepoints, nb_features + time]
        :param model: a model that is structured and works exactly as self.trained. Added for bootstrapping functionality.
        :returns: a 1D numpy array of states of length nb_timepoints, a single float
        """
        if model is None: model = self.trained

        # Your implementation goes here


        return state_list, logprob

    # --- parameters and performance measures ---
    def get_data_tm(self, trace_dict, nb_bootstrap_iters):
        """
        STEP 3.1 (OPTIONAL):
        To provide a confidence interval for transition rates FRETboard can train nb_bootstrap_iters models,
        classify data each time and derive transition rates from the new classification, as implemented here. For some
        types of models it makes more sense to derive transition matrices some other way; for example hidden markov
        models already estimates a transition matrix to work with. If this is also the case for you, you can
        change the implementation in this method to make the bootstrapping a lot faster or better.

        :param trace_dict:  dict of traces. Keys are trace ids, values are pandas dataframes of shape
                          [nb_timepoints, nb_features + time]
        :param nb_bootstrap_iters: integer number of bootstrap iterations to perform
        :returns tm: a 2D numpy array of size [nb_classes, nb_classes] containing transition rates estimated on final model
        :returns ci_mat: a 3D numpy array of size [nb_classes, nb_classes, 2]. [nb_classes, nb_classes 0] contains
                         lower bounds for CI for transition rates, [nb_classes, nb_classes, 1] contains upper bounds.
        """

        # Ensure junk traces didn't end up in trace_dict
        invalid_indices = [idx for idx, tup in self.data.manual_table.iterrows() if tup.is_labeled or tup.is_junk]
        idx_list = [idx for idx in self.data.index_table.index if idx not in invalid_indices]
        idx_list = np.concatenate((idx_list, self.data.manual_table.query('is_labeled').index), axis=0)
        trace_dict = {tr: trace_dict[tr] for tr in trace_dict if tr in idx_list}

        tm = self.tm_from_seq([self.predict(trace_dict[idx])[0] for idx in idx_list])

        # get bootstrapped CIs
        tm_array = []
        for _ in range(nb_bootstrap_iters):
            model = self.get_trained_model(trace_dict, bootstrap=True)
            seqs = [self.predict(trace_dict[idx], model)[0] for idx in idx_list]
            tm_array.append(self.tm_from_seq(seqs))
        tm_mat = np.stack(tm_array, axis=-1)
        sd_mat = np.std(tm_mat, axis=-1)
        mu_mat = np.mean(tm_mat, axis=-1)
        ci_mat = np.tile(np.expand_dims(mu_mat, -1), (1, 1, 2))
        ci_mat[:, :, 0] -= sd_mat * 2
        ci_mat[:, :, 1] += sd_mat * 2
        return tm, ci_mat

    def tm_from_seq(self, seq_list):
        """
        Used for the standard bootstrapping implementation, do not remove if that is still in use.
        """
        tm_out = np.zeros([self.nb_states, self.nb_states], dtype=int)
        # inventory of transitions
        for seq in seq_list:
            for tr in zip(seq[:-1], seq[1:]):
                tm_out[tr[0], tr[1]] += 1
        return tm_out / np.expand_dims(tm_out.sum(axis=1), -1)


    def get_mus(self, feature):
        """
        STEP 3.2 (OPTIONAL):
        Given a feature name, return a list of means of feature values, one for each state. This may be derived from your
        model or from your data. Values are reported in the GUI and the generated report. If not implemented,
        just use an array of zeros
        :param feature: string, name of feature for which mean values per class should be returned
        :return: list of floats of length nb_classes
        """
        mu_list = [0.0 for _ in self.nb_states]

        return mu_list

    def get_sds(self, feature):
        """
        STEP 3.3 (OPTIONAL):
        Given a feature name, return a list of standard deviation feature values, one for each state.
        This may be derived from your model or from your data. Values are reported in the GUI and the generated report.
        If not implemented, just use an array of zeros.
        :param feature: string, name of feature for which mean values per class should be returned
        :return: list of floats of length nb_classes
        """
        sd_list = [0.0 for _ in self.nb_states]
        return sd_list

    # --- saving/loading models ---
    def get_params(self):
        """
        STEP 4.1 (OPTIONAL):
        Return a string representation of your model that can be loaded into some other instance using the load_params
        function below. You are completely free on whether or how to implement this. If you do implement it, your
        implementation is vastly more useful as trained models can now be re-used.

        Tip: If you save your parameters in dicts you can easily store them in strings using yaml.dump, which can
             be easily decoded into dicts again using yaml.load

        :return: string
        """

        return 'MODEL SAVING NOT IMPLEMENTED'

    def load_params(self, file_contents):
        """
        STEP 4.2 (OPTIONAL):
        Load a previously trained model. Naturally, this only makes sense if the function save_params is also
        implemented. The result should be that the model in self.trained functions exactly the same way as it did
        when the model was first trained.
        """

        # Your implementation goes here

        self.timestamp = numeric_timestamp()  # time stamp for model updating methods, please leave as is
