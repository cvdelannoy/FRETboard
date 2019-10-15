


# Keep name of this class as Classifier!
class Classifier(object):

    def __init__(self, **kwargs):
        # Standard properties, do not change
        self.nb_states = kwargs['nb_states']
        self.gui = kwargs['gui']
        self.data = self.gui.data._data
        self.trained = None

        # Pick features to train your classifier on, may also include i_don (donor intensity) and i_acc (acceptor intensity)
        self.feature_list = ['E_FRET', 'i_sum', 'correlation_coefficient', 'E_FRET_sd']


    def train(self, supervision_influence=1.0):
        """
        Train the model and update predictions. Must store model object in self.trained!
        """
        self.trained =
        return


    def predict(self):
        """
        Classify examples in data using classifier in self.trained. Should generate two iterables:
        - pred_list: predicted labels, 1 vector per tuple to be stored in self.data.prediction
        - logprob_list: logprob of labels, 1 value per tuple to be stored in self.data.logprob
        """


        self.data.prediction = pred_list
        self.data.logprob = logprob_list

    # --- return model parameters
    def get_states_mu(self, fidx):
        """
        Return vector of mean values of states for feature number [fidx]
        :param fidx: int index of feature
        :return: 1d np array of length [nself.nb_states]
        """
        return

    def get_states_sd(self, fidx):
        """
        Return vector of mean values of states for feature number [fidx]
        :param fidx: int index of feature
        :return: 1d np array of length [nself.nb_states]
        """


        return

    def get_tm(self, classifier):
        """
        Return the transition matrix in an nb_states x nb_states 2d numpy array,
        given a classifier object (such as that stored in self.trained). May be the transmission matrix as calculated
        by an HMM, or may be deduced from predicted labels.

        In a valid transition matrix, the rows sum to 1.
        """

        return

    @property
    def confidence_intervals(self):
        """
        Return confidence intervals on transition matrix. Bootstrap CIs are used in included HMM implementations, but
        you are free to explore other options.
        :return: nb_states x nb_states x 2 numpy vector containing lower and upper bound in last dimension (in that order)
        """

        return

    # --- saving/loading models ---

    def get_params(self):
        """
        Return params as string, in same format as can be read by load_params. It is advisable to include a check
        for currently set algorithm (see example below and corresponding lines in load_params)
        """


        div = '\nSTART_NEW_SECTION\n'
        out_txt = ('Your algorithm name '
                   + div + some_parameters_as_string
                   + div + even_more_parameters)
        return out_txt

    def load_params(self, file_contents):
        """
        Load model parameters from same format as read from get_params. Example includes a check for currently set
        algorithm in GUI
        """
        (mod_check,
         some_parameters_as_string,
         even_more_parameters) = file_contents.split('\nSTART_NEW_SECTION\n')
        if mod_check != 'your algorithm name':
            self.gui.text += '\nERROR: loaded model parameters are not for this algorithm!'
