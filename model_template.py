import numpy as np

import yaml
from FRETboard.helper_functions import numeric_timestamp, discrete2continuous

# FRETBOARD MODEL TEMPLATE
# Using this template you can create your own FRETboard model. Just follow the instructions listed in comments
# for each method.

# You are completely free to implement any model, however as FRETboard is meant to rapidly estimate and re-estimate models
# to react to the user's input we advise to keep your model implementations light, e.g. fully training a neural network
# from the ground up might work but will probably slow down your analysis a lot.
#
# If you have written a model that you think might be useful for others, please consider submitting it as a pull request
# on the project github page:
#
# https://github.com/cvdelannoy/FRETboard.git)

class Classifier(object):
    """ A template class for creating new algorithms for FRETboard
    """
    def __init__(self, nb_states, data, **kwargs):
        """
        :param nb_states: number of states to detect
        :param data: object of class MainTable
        :param gui: Gui object [optional]
        :param buffer: int, size of buffer area around regular classes [required if gui not given]
        """

        # Required parameters for training
        self.nb_states = nb_states
        self.feature_list = kwargs['features'] # list names of features that should be used in training/inference

        # These parameters are passed on from the GUI, usage optional
        self.buffer = kwargs['buffer']  # int number of points used in calculating the sliding-window features

        # Parameters necessary for proper functioning of algorithm. Do not change!
        self.timestamp = numeric_timestamp()
        self.data = data
        self.trained = None
        self.framerate = None

    # --- training ---
    def train(self, data_dict, supervision_influence=1.0):
        """
        Generate trained classifier
        """

        # Determine framerate for input data, leave as is
        if self.framerate is None:
            self.framerate = 1 / np.concatenate([data_dict[tr].time.iloc[1:].to_numpy() -
                                                 data_dict[tr].time.iloc[:-1].to_numpy() for tr in data_dict]).mean()

        # Write your code for generating a classifier object here. The only restrictions are:
        # - it should use the data in the data_dict argument
        # - it should store the object in self.trained
        # - the predict function of this class should be able to use it to run inference


        self.trained = None # <-- classifier object goes here!

        # Parameters necessary for proper functioning of algorithm. Do not change!
        self.timestamp = numeric_timestamp()

    def predict(self, trace_df):
        """
        Predict labels for a trace

        :trace_df: DataFrame containing all features of a single trace as its columns
        :returns:
        state_list:
        logprob_list: list of floats of length len(idx) containing posterior log-probabilities
        """

        # Write your code for predicting classification for the given traces here.

        state_array = None  # <- Redefine as numpy array of states, of which length equals len(trace_df)
        mean_logprob = None # <- Redefine as mean log-probability for the prediction

        return state_array, mean_logprob


    # --- parameters and performance measures ---
    def get_data_tm(self, trace_dict, out_labels, nb_bootstrap_iters):
        """
        Calculate bootstrapped confidence intervals on data-derived transition matrix values and convert to
        continous transition rates
        """

        # This method produces a transition rate matrix from classified data, with bootstrapped CIs. It should
        # work if predict is correctly implemented.

        state_order_dict = {state.name: idx for idx, state in enumerate(self.trained.states)}

        # actual estimate
        trace_list = list(trace_dict.values())
        nb_traces = len(trace_list)
        actual_tm = self.tm_from_seq(trace_list)

        # CIs
        tm_array = []
        for n in range(nb_bootstrap_iters):
            bs_idx = np.random.choice(nb_traces, size=nb_traces)
            tm = self.tm_from_seq([trace_list[it] for it in bs_idx])
            tm_array.append(tm)
        tm_mat = np.stack(tm_array, axis=-1)
        sd_mat = np.std(tm_mat, axis=-1)
        mu_mat = np.mean(tm_mat, axis=-1)
        ci_mat = np.tile(np.expand_dims(mu_mat, -1), (1, 1, 2))
        ci_mat[:, :, 0] -= sd_mat * 2
        ci_mat[:, :, 1] += sd_mat * 2
        return actual_tm, ci_mat

    def tm_from_seq(self, seq_list):
        # Necessary for get_data_tm(), leave as is
        tm_out = np.zeros([self.nb_states, self.nb_states], dtype=int)
        for seq in seq_list:
            seq_pred = seq.predicted.astype(int)
            for tr in zip(seq_pred[:-1], seq_pred[1:]):
                tm_out[tr[0], tr[1]] += 1
        return discrete2continuous(tm_out / np.expand_dims(tm_out.sum(axis=1), -1), self.framerate)

    def get_mus(self, feature):

        mu_list = None # <-- Return a list of mean peak values for all classes, for a feature of a given name

        return mu_list

    # --- saving/loading models ---
    def get_params(self):
        # Should return a representation of the trained model in text format, as a string. In principle,
        # The only requirement is that load_params() can read it and produce the exact same model again, however the
        # lines below give you a head start by defining some parameters. The load_params() method
        # similarly contains some code to read these out again.

        mod_txt = ''  # <- store model representation as string here
        some_other_stuff = 'foobar\n' # <- example variable to show how other information can be stored
        feature_txt = '\n'.join(self.feature_list)
        div = '\nSTART_NEW_SECTION\n'
        out_txt = ('CoolModelName'  # <- give your model a name (no spaces) so the load_params() function can immediately check if the model loaded is the correct one
                   + div + mod_txt
                   + div + some_other_stuff
                   + div + feature_txt
                   + div + f'nb_states: {str(self.nb_states)}\n'
                           f'dbscan_epsilon: {self.data.eps}\n'
                           f'supervision_influence: {self.supervision_influence}\n'
                           f'framerate: {self.framerate}')

        return out_txt

    def load_params(self, file_contents):
        # Should load the model from a string as it is stored by get_params(). The lines below should give you a head
        # start again.


        (mod_check,
         model_txt,
         some_other_stuff,  # <- example variable, see get_params()
         feature_txt,  # <- features
         misc_txt) = file_contents.split('\nSTART_NEW_SECTION\n')


        if mod_check != 'CoolModelName':
            error_msg = '\nERROR: loaded model parameters are not for this type of model!'
            raise ValueError(error_msg)
        self.feature_list = feature_txt.split('\n')
        misc_dict = yaml.load(misc_txt, Loader=yaml.SafeLoader)
        if misc_dict['dbscan_epsilon'] == 'nan': misc_dict['dbscan_epsilon'] = np.nan
        self.nb_states = misc_dict['nb_states']
        self.data.eps = misc_dict['dbscan_epsilon']
        self.supervision_influence = misc_dict['supervision_influence']
        self.framerate = misc_dict['framerate']
        self.timestamp = numeric_timestamp()
