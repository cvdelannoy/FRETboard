import pandas as pd
import numpy as np
import pomegranate as pg
import itertools

class OneshotHmm(object):
    """ HMM classifier that automatically adds 'edge states' to better recognize valid transitions between states.
    """
    def __init__(self, nb_states, trace_df, labels):
        """
        :param nb_states: number of states to detect
        :param data: object of class MainTable
        :param gui: Gui object [optional]
        :param buffer: int, size of buffer area around regular classes [required if gui not given]
        """

        self.nb_states = nb_states
        self.state_names = None  # names assigned to states in same order as pomegranate model
        self.pg_gui_state_dict = dict()
        self.gui_state_dict = dict()
        self.str2num_state_dict = dict()
        self.fit(trace_df, labels)

    # --- training ---
    def fit(self, data_dict, labels):
        """
        return an untrained pomegranate hmm object with parameters filled in
        - If all data is unlabeled: finds emission parameters using k-means, transmission and start p are equal
        - If some data is labeled: initial estimate using given classifications
        """

        # Get emission distributions & transition probs
        dists, pg_gui_state_dict = self.get_states(data_dict, labels)
        trans_df = self.get_transitions(labels)
        trans_df.replace(0, 0.000001, inplace=True)

        state_names = list(dists)
        tm_mat = trans_df.loc[state_names, state_names].to_numpy()
        hmm = pg.HiddenMarkovModel.from_matrix(transition_probabilities=tm_mat,
                                               distributions=dists.values(),
                                               starts=[1.0 for _ in state_names],
                                               ends=[1.0 for _ in state_names],
                                               state_names=state_names)
        self.pg_gui_state_dict = pg_gui_state_dict
        self.gui_state_dict = {si: pg_gui_state_dict.get(s, None) for si, s in enumerate(state_names)}
        self.str2num_state_dict = {str(si): ni for si, ni in zip(state_names, list(self.gui_state_dict))}
        self.trained = hmm

    def get_states(self, data_df, y):
        """
        Return dicts of pomgranate states with initialized normal multivariate distributions
        """
        # Estimate emission distributions from given class labels
        data_vec = data_df.to_numpy()

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

    def get_transitions(self, labels):
        state_names = [f's{s}' for s in range(self.nb_states)]
        trans_df = pd.DataFrame([], columns=state_names, index=state_names)

        # transition matrix
        transition_dict = {perm: 0 for perm in list(itertools.product(list(range(self.nb_states)), repeat=2)) +
                           [(st, -1) for st in range(self.nb_states)]}
        total_transitions = dict()

        # Count occurring transitions
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
            if tra[1] != -1:
                trans_df.loc[f's{tra[0]}', f's{tra[1]}'] = transition_dict[tra] / tt if tt != 0 else 0.0
        return trans_df

    def predict(self, trace_df):
        """
        Predict labels for given indices.

        """
        logprob, trace_state_list = self.trained.viterbi(np.split(trace_df.to_numpy(),
                                                         len(trace_df), axis=0))
        state_list = np.vectorize(self.gui_state_dict.__getitem__)([ts[0] for ts in trace_state_list[1:-1]])
        return state_list
