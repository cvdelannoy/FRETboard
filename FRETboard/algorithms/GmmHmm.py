import os, sys, yaml
import numpy as np
import pomegranate as pg
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__)
import BoundaryAwareHmm


class Classifier(BoundaryAwareHmm.Classifier):

    def get_main_dist(self, X):
        if X.size == 0:
            # No prior model and no examples assigned --> initialize dummy with 0 and diagonal cov matrix
            return self.get_blank_distribution()

        # Determine number of components using BIC
        X = X.T.copy()
        bic_list, dist_list = self.select_distribution(X)
        if not len(bic_list):
            # Data was impossible under all distributions --> initialize dummy with 0 and diagonal cov matrix
            return self.get_blank_distribution()
        return dist_list[np.argmin(bic_list)]
        # return pg.GeneralMixtureModel.from_samples([pg.NormalDistribution for _ in range(len(self.feature_list))], n_components=nd, X=X.T.copy())

    def get_blank_distribution(self):
        nd = 3
        dist_list = []
        for _ in range(nd):
            dist_list.append(pg.IndependentComponentsDistribution(
                [pg.NormalDistribution(0, 1) for _ in range(len(self.feature_list))]))
        return pg.GeneralMixtureModel(dist_list, weights=[1 / nd] * nd)


    def select_distribution(self, X):
        bic_list, dist_list = [], []
        if X.shape[0] > 1000: X = X[np.random.choice(X.shape[0], size=1000), :]
        for nc in range(1, min(5, X.shape[0]) + 1):
            if nc == 1:
                dist = pg.IndependentComponentsDistribution.from_samples(X,
                    distributions=[pg.NormalDistribution for _ in range(len(self.feature_list))])
                p = dist.log_probability(X).sum()
                k = len(self.feature_list) * 2
            else:
                dist = pg.GeneralMixtureModel.from_samples(
                    [pg.NormalDistribution for _ in range(len(self.feature_list))],
                    n_components=nc, X=X, n_init=3)
                p = dist.log_probability(X).sum()
                k = (len(self.feature_list) * 2 + 1) * nc
            if not np.isnan(p):
                dist_list.append(dist)
                bic_list.append(k * np.log(X.shape[0]) - 2 * p)
        return bic_list, dist_list

    def get_mus(self, feature):
        state_names = [f's{i}' for i in range(self.nb_states)]
        fidx = np.argwhere(feature == np.array(self.feature_list))[0,0]
        mu_dict = {}
        for state in self.trained.states:
            if state.name not in state_names: continue
            dist = state.distribution
            if type(dist) == pg.GeneralMixtureModel:
                mu_dict[self.pg_gui_state_dict[state.name]] = dist.distributions[dist.weights.argmax()].parameters[0][fidx].parameters[0]
            elif type(dist) == pg.IndependentComponentsDistribution:
                mu_dict[self.pg_gui_state_dict[state.name]] = dist.distributions[fidx].parameters[0]
            else:
                raise ValueError(f'Expected IndependentComponentDistribution or GMM, got {type(dist)}')
        # dist_dict = {self.pg_gui_state_dict[state.name]: (state.distribution.distributions, state.distribution.weights)
        #            for state in self.trained.states if state.name in state_names}
        #
        # mu_dict = {dd: dist_dict[dd][0][dist_dict[dd][1].argmax()].parameters[0][fidx].parameters[0] for dd in dist_dict}
        mu_list = [mu_dict[mk] for mk in sorted(list(mu_dict))]
        return mu_list

    def get_sds(self, feature):
        state_names = [f's{i}' for i in range(self.nb_states)]
        fidx = np.argwhere(feature == np.array(self.feature_list))[0, 0]

        sd_dict = {}
        for state in self.trained.states:
            if state.name not in state_names: continue
            dist = state.distribution
            if type(dist) == pg.GeneralMixtureModel:
                sd_dict[self.pg_gui_state_dict[state.name]] = \
                dist.distributions[dist.weights.argmax()].parameters[0][fidx].parameters[1]
            elif type(dist) == pg.IndependentComponentsDistribution:
                sd_dict[self.pg_gui_state_dict[state.name]] = dist.distributions[fidx].parameters[1]
            else:
                raise ValueError(f'Expected IndependentComponentDistribution or GMM, got {type(dist)}')

        # dist_dict = {self.pg_gui_state_dict[state.name]: (state.distribution.distributions, state.distribution.weights)
        #              for state in self.trained.states if state.name in state_names}
        # sd_dict = {dd: dist_dict[dd][0][dist_dict[dd][1].argmax()].parameters[0][fidx].parameters[1] for dd in
        #            dist_dict}
        sd_list = [sd_dict[mk] for mk in sorted(list(sd_dict))]
        return sd_list

    # --- saving/loading models ---
    def get_params(self):
        mod_txt = self.trained.to_yaml()
        gui_state_dict_txt = yaml.dump(self.gui_state_dict)
        pg_gui_state_dict_txt = yaml.dump(self.pg_gui_state_dict)
        str2num_state_dict_txt = yaml.dump(self.str2num_state_dict)
        feature_txt = '\n'.join(self.feature_list)
        div = '\nSTART_NEW_SECTION\n'
        out_txt = ('GmmHmm'
                   + div + mod_txt
                   + div + feature_txt
                   + div + gui_state_dict_txt
                   + div + pg_gui_state_dict_txt
                   + div + str2num_state_dict_txt
                   + div + f'nb_states: {str(self.nb_states)}\nbuffer: {self.buffer}\ndbscan_epsilon: {self.data.eps}')
        return out_txt

    def load_params(self, file_contents):
        (mod_check,
         model_txt,
         feature_txt,
         gui_state_dict_txt,
         pg_gui_state_dict_txt,
         str2num_state_dict_txt,
         misc_txt) = file_contents.split('\nSTART_NEW_SECTION\n')
        if mod_check != 'GmmHmm':
            error_msg = '\nERROR: loaded model parameters are not for a GMM-HMM!'
            raise ValueError(error_msg)
        self.trained = pg.HiddenMarkovModel().from_yaml(model_txt)
        self.feature_list = feature_txt.split('\n')
        self.gui_state_dict = yaml.load(gui_state_dict_txt, Loader=yaml.FullLoader)
        self.pg_gui_state_dict = yaml.load(pg_gui_state_dict_txt, Loader=yaml.FullLoader)
        self.str2num_state_dict = yaml.load(str2num_state_dict_txt, Loader=yaml.FullLoader)
        misc_dict = yaml.load(misc_txt, Loader=yaml.SafeLoader)
        if misc_dict['dbscan_epsilon'] == 'nan': misc_dict['dbscan_epsilon'] = np.nan
        self.nb_states = misc_dict['nb_states']
        self.buffer = misc_dict['buffer']
        self.data.eps = misc_dict['dbscan_epsilon']
