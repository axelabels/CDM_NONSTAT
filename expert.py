from __future__ import print_function

from math import ceil
# from bandit import OffsetBandit
from scipy.stats import gmean

from sklearn.gaussian_process.kernels import RationalQuadratic, PairwiseKernel


from policy import *
from tools import *


EXPECTED_STD_REWARD = np.float64(.5)


MAX_SLICE_SIZE = 100
RANDOM_CONFIDENCE = .5


class Agent(object):

    def __init__(self, bandit, policy, name=None):
        self.bandit = bandit
        self.policy = policy

        self.value_mode = False

        self.prior_bandit = None

        self.t = 0
        self.reward_history = []
        self.context_history = []
        self.action_history = []
        self.cache_id = None

        self.oracle_confidence = False
        self.name = name
        self.confidence = np.zeros(bandit.k)
        self.mu = np.zeros(bandit.k)

    def value_estimates(self, contexts=None, cache_index=None, return_std=False, arm=None, batch=True):

        if cache_index is not None:
            assert self.is_prepared(
            ), "When an int is given as context the expert should be prepared in advance"
            self.mu, self.sigma = np.copy(self.cached_predictions[cache_index])
        else:
            self.mu, self.sigma = self.predict_normalized(
                contexts, arm=arm, batch=batch)
        if return_std:
            return self.mu, self.sigma
        return self.mu

    def is_prepared(self, cache_id=None):
        if cache_id is None:
            return self.cache_id == self.bandit.cache_id
        return self.cache_id == cache_id


    def reset(self):
        self.t = 0
        self.reward_history = []
        self.context_history = []
        self.action_history = []
        self.cache_id = None

    def cache_predictions(self, bandit, trials):
        if not self.is_prepared(bandit.cache_id) or len(self.cached_predictions) < trials:
            self.cached_predictions = np.array(
                self.predict_normalized(bandit.cached_contexts,batch=True))

            assert np.shape(self.cached_predictions) == (
                2, trials, bandit.k), np.shape(self.cached_predictions)
            self.cached_predictions = np.moveaxis(
                self.cached_predictions, 1, 0)
                
            self.cache_id = bandit.cache_id
            self.cached_probabilities = greedy_choice(
                self.cached_predictions[:, 0], axis=1)
            recomputed = True
       
        return recomputed


class OracleExpert(Agent):
    def __init__(self,bandit, policy):
        super().__init__(bandit,policy)


    def compute(self, contexts=None, mn=None, std=None):
        pass

    def predict_normalized(self, contexts, slice_size=None, arm=None, batch=False):
        assert np.shape(contexts)[1:] == (self.bandit.dims,)
        
        mu = self.bandit.cached_values 

    
        if arm is not None:
            mu = mu[..., arm]
        sigma = np.zeros_like(mu)
        return mu, sigma

    def prior_play(self, bandit=None):
        self.prior_bandit = bandit
