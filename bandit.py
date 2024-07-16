
import numpy as np

from tools import *


MAX_ATTEMPTS_BANDIT_DISTANCE = 100
BANDIT_DISTANCE_EPSILON = .05


INF = float("inf")



class MetaBandit():
    def __init__(self, dims, k):
        self.dims = dims
        self.k = k

UPPER = .8

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)
def shape_rewards(a,x):
    return a**x
class ArtificialBandit():
    def __init__(self, n_arms=1, problem='classification',  bernoulli=True):
        self.k = n_arms
        self.problem = problem 
        
        self.mixture_p=2**np.random.uniform(0,5)
        
        self.bernoulli=bernoulli
        self.cached_contexts = None
        self.dims=1
        self._expected_reward=None
        
    @property
    def expected_reward(self):
        if self.problem=='mixture' and self._expected_reward is None:
            self._expected_reward = np.mean((np.arange(self.k)/(self.k-1))**self.mixture_p) 
            
        return .5 if self.problem=='regression' else 1/self.k if self.problem=='classification' else self._expected_reward

    def reset(self):
        self.mixture_p=2**np.random.uniform(0,4)
        self._expected_reward=None
        
        self.cached_contexts = None
    
    
    def generate_random_values(self,shape,include_neg=False):

        if self.problem=='regression':
            values = np.repeat(np.arange(self.k)[None],shape[0],axis=0)
            values  = shuffle_along_axis(values ,axis=1)
            values = values/(self.k-1)
            if include_neg:
                return values,1-values
            return values
            
        elif self.problem=='classification':
            values = np.repeat(np.arange(self.k)[None],shape[0],axis=0)
            values  = shuffle_along_axis(values ,axis=1)
            values  =  softmax(np.log(10)*values ,axis=1)
            
            if include_neg:
                neg_values = np.repeat(np.arange(self.k)[None],shape[0],axis=0)
                neg_values  = shuffle_along_axis(neg_values ,axis=1)
                neg_class_values  =  softmax(np.log(10)*neg_values ,axis=1)
                
                return values,neg_class_values
            return values
        else:
            values = np.repeat(np.arange(self.k)[None],shape[0],axis=0)
            values  = shuffle_along_axis(values ,axis=1)
            neg_values = np.max(values)-values
            
            values = (values/(self.k-1))**self.mixture_p
            neg_values = (neg_values/(self.k-1))**self.mixture_p
            
            if include_neg:
                return values,neg_values

            else:
                return values
    
    def assign_cached_values(self,values):
        assert values.shape==self.cached_values.shape
        self.cached_values = values 
        self.cached_rewards = self.sample(self.cached_values)


    def cache_contexts(self, t, cache_id):
        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.random.uniform(
                0, 1, size=(t,  self.dims))

            
            self.cached_values,self.neg_cached_values = self.generate_random_values((t,self.k),include_neg=True) # 
            self.random_values = self.generate_random_values((t,self.k),include_neg=False) # 

            assert np.shape(self.cached_values) == (
                t, self.k), (np.shape(self.cached_values), (t, self.k))
            self.cached_rewards = self.sample(self.cached_values)

            assert np.shape(self.cached_rewards) == (t, self.k)
            self.cache_id = cache_id

        return self.cached_contexts

    def observe_contexts(self, center=.5, spread=1,  cache_index=None):
        if cache_index is not None:
            self.contexts = self.cached_contexts[cache_index]
            self.action_values = self.cached_values[cache_index]
        else:
           
            self.contexts = np.random.uniform(
                center - spread / 2, center + spread / 2, size=(self.dims))

            self.contexts = self.contexts % 1
            self.action_values = self.get(self.contexts[None, :])[0]
        self.optimal_value = np.max(self.action_values)

        return self.contexts
    def sample(self, values=None, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index]

        if values is None:
            values = self.action_values
        if self.bernoulli:
          
            return np.random.uniform(size=np.shape(values)) < values
        else:
            return values
