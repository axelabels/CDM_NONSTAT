
import numpy as np

from tools import greedy_choice, softmax



class Policy(object):
    def __init__(self, b=1):
        self.b = b
        self.key = 'value'

    def __str__(self):
        return 'generic policy'

    def probabilities(self, agent, contexts):
        a = agent.value_estimates(contexts)
        self.pi = softmax(a*self.b)
        return self.pi
        

    def choose(self, agent, contexts, greedy=False):
        
        self.pi = self.probabilities(agent, contexts)
        
        if greedy:
            self.pi = greedy_choice(self.pi)
            
        np.testing.assert_allclose(np.sum(self.pi),1,atol=1e-5,err_msg=str(agent)+" "+str(np.sum(self.pi))+" "+str(self.pi))
        action = np.searchsorted(np.cumsum(self.pi), np.random.rand(1))[0]

        return action
        
class AdaPolicy(Policy):
    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'

    def __str__(self):
        return 'Ada'

    def probabilities(self, agent, contexts):
        self.pi = agent.probabilities(contexts)
        self.pi = self.pi * (1 - self.eps) + self.eps / len(self.pi) 
        return self.pi


class RandomPolicy(Policy):

    def __init__(self):
        self.key = 'value'

    def __str__(self):
        return 'random'

    def probabilities(self, agent, contexts):
        self.pi = np.ones(agent.bandit.k)/agent.bandit.k
        return self.pi

class SoftmaxPolicy(Policy):

    def __init__(self, beta):
        self.beta = beta
        self.key = 'value'

    def __str__(self):
        return 'eps'.format(self.beta)

    def probabilities(self, agent, contexts):
        v = agent.value_estimates(contexts)
        self.pi = softmax(self.beta*v)
        return self.pi


class EpsilonGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.key = 'value'

    def __str__(self):
        return 'eps'.format(self.epsilon)

    def get_samples(self):
        return None

    @staticmethod
    def from_values(values,agent):
        assert len(values.shape)==1
        return  greedy_choice(values)
    def probabilities(self, agent, contexts):
        v = agent.value_estimates(contexts)
        self.pi = greedy_choice(v)       
        self.pi *= (1-self.epsilon)
        self.pi += self.epsilon/agent.bandit.k
        return self.pi


class GreedyPolicy(EpsilonGreedyPolicy):

    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return 'greedy'


class ProbabilityGreedyPolicy(Policy):
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.datas = []
        self.key = 'probability'

    def __str__(self):
        return 'PGP'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        
        self.pi = greedy_choice(agent.probabilities(contexts))
        
        if 'Cover' in str(agent) and agent.update_all:
            self.pi *= (1-self.epsilon)
            self.pi += self.epsilon/agent.bandit.k
        

        return self.pi


class UCBPolicy(Policy):

    def __init__(self):
        pass 
    def __str__(self):
        return 'GPUCB' 

    def probabilities(self, agent, contexts):
        self.pi = greedy_choice(agent.ucb_values(contexts))
        return self.pi

class Exp3Policy(Policy):
    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'

    def __str__(self):
        return 'E3P'

    def probabilities(self, agent, contexts):
        self.pi = agent.probabilities(contexts)
        
        self.pi = self.pi * (1 - self.eps) + self.eps / len(self.pi) 
        
        return self.pi

class SCBPolicy(Policy):

    def __init__(self, gamma=0):
        self.gamma = gamma
        self.key = 'probability'

    def __str__(self):
        return 'SCB'

    def probabilities(self, agent, contexts):

        values = agent.value_estimates(contexts)
        best_arm = np.argmax(values)
        self.pi = np.zeros_like(values)
        self.pi[:] = 1 / \
            (agent.bandit.k+self.gamma*(values[best_arm]-values))
        self.pi[best_arm] += (1-(np.sum(self.pi)))

        return self.pi


class BootstrapGreedyPolicy(Policy):

    def __init__(self, m=1,bootstraps=1000,epsilon=0):
        self.m = m
        self.key = 'value'
        self.bootstraps = bootstraps
        self.epsilon=epsilon

    def __str__(self):
        return 'eps'

    def get_samples(self):
        return self.bootstraps

    
    @staticmethod
    def from_values(values,agent):

        return greedy_choice(values,axis=1).mean(axis=0)
        

    def probabilities(self, agent, contexts):
        
        samples = agent.value_estimates(contexts,samples=self.bootstraps)
        self.pi =self.from_values(samples,agent)

        assert len(self.pi)==agent.bandit.k,(np.shape(self.pi))
        
        self.pi *= (1-self.epsilon)
        self.pi += self.epsilon/agent.bandit.k
        return self.pi

class LinEXP(Policy):

    def __init__(self, eta,gamma):
        self.eta = eta
        self.gamma = gamma
        self.key = 'probability'

    def __str__(self):
        return 'LinEXP'

    def get_samples(self):
        return None

    @staticmethod
    def from_values(values,agent):
        assert len(values.shape)==1
        p =  softmax(agent.t*agent.policy.eta*values)
        p = (1-agent.policy.gamma)*p+ (agent.policy.gamma)/agent.bandit.k
        return p
    def probabilities(self, agent, contexts):

        values = agent.value_estimates(contexts,mu_only=True)
        self.pi = self.from_values(values,agent)
        return self.pi