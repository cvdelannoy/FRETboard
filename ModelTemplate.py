

class ModelTemplate(object):

    def __init__(self, **kwargs):
        self.trained = None
        self.feature_list = ['E_FRET', 'i_sum', 'sd_roll']
        self.nb_states = kwargs['nb_states']
        self.gui = kwargs['gui']
        self.data = self.gui.data

    def train(self, influence=1.0):
        """
        Train the model and update predictions. Must store model object in self.trained!
        :param influence:
        :return:
        """

        self.trained =
        return


    def predict(self):
        """
        Classify examples in data using classifier in self.trained
        :return: None
        """
        return

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
        return
        """
        return

    # --- saving/loading models ---

    def get_params(self):
        """
        Return params as string, in same format as can be read by load_params
        """
        return

    def load_params(self, fc):
        """
        Load model parameters
        """
        return
