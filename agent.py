
from scipy.optimize import fsolve, root, bisect, brentq, newton
from numpy.random import Generator, PCG64
from sklearn.linear_model import Ridge, ElasticNet
from tools import generate_decays
import numpy as np
from numbers import Number

from expert import *

class Collective(Agent):
    def __init__(self, bandit, policy, n_experts,  gamma=None,  name=None,   alpha=1, beta=1,
                 ):

        super(Collective, self).__init__(bandit, policy)

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

        self.k = self.bandit.k
        self.n = n_experts
        self.name = name
        self.advice = np.zeros((self.n, self.bandit.k))
        self._value_estimates = np.zeros(self.k)
        self._probabilities = np.zeros(self.k)

        self.confidences = np.ones((self.n, self.bandit.k))

        self.initialize_w()

    def initialize_w(self):
        pass

    def short_name(self):
        return "Average"

    @property
    def info_str(self):
        info_str = ""
        return info_str

    def observe(self, reward, arm):

        self.t += 1

    def get_weights(self, contexts):

        self.confidences = np.ones((self.n, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']
        w = safe_logit(self.confidences)

        return w

    def __str__(self):
        if self.name is not None:
            return (self.name)
        return (self.short_name())

    def set_name(self, name):
        self.name = name

    @staticmethod
    def prior_play(experts,  base_bandit):

        for i, e in (list(enumerate(experts))):
            e.index = i
            e.bandit = base_bandit

    def choose(self, advice, greedy=False):
        return self.policy.choose(self, advice, greedy=greedy)

    def probabilities(self, contexts):
        self.advice = np.copy(contexts['advice'])
        if isinstance(self, Exp4) and not np.allclose(np.sum(self.advice, axis=1), 1):
            advice = np.zeros_like(self.advice)
            advice[self.advice == np.max(self.advice, axis=1)[:, None]] = 1
            self.advice = advice/np.sum(advice, axis=1)[:, None]

        w = self.get_weights(contexts)
        self._probabilities = np.sum(w * self.advice, axis=0)

        assert len(self._probabilities) == self.bandit.k
        return self._probabilities

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice - self.bandit.expected_reward), axis=0)

        return self._value_estimates

    def reset(self):
        super().reset()

        self.initialize_w()


class Exp4(Collective):
    def __init__(self, bandit, policy, n_experts, gamma, name=None, weights_decay=0,
                 p=1):
        super(Exp4, self).__init__(bandit, policy,
                                   n_experts, name=name, gamma=gamma)

        self.e_sum = 1
        self.p = p
        self.w = np.zeros(self.n)
        self.weights_decay = weights_decay

    def copy(self):
        return Exp4(self.bandit, self.policy, self.n, self.gamma,
                    crop=self.crop, prefix=self.prefix)

    def short_name(self):
        return f"EXP4.S[µ={(self.weights_decay )}]"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.ones(self.n)/self.n

    def reset(self):
        super(Exp4, self).reset()

    def get_weights(self, contexts):

        w = np.copy(self.w)
        w = np.repeat(w[:, np.newaxis], self.bandit.k, axis=1)

        return w

    @property
    def effective_weights_decay(self):
        return self.weights_decay

    def observe(self, reward, arm):

        assert np.allclose(np.sum(self.advice, axis=1),
                           1), "expected probability advice"
        x_t = self.advice[:, arm] * (self.bandit.expected_reward-reward)
        y_t = x_t / (self.policy.pi[arm]+self.gamma)

        self.e_sum = (1-self.effective_weights_decay)*self.e_sum + \
            np.sum(np.max(self.advice, axis=0))

        assert np.sum(np.max(self.advice, axis=0)) <= self.bandit.k
        numerator = self.e_sum
        lr = np.sqrt(np.log(self.n) /
                     numerator)
        if self.weights_decay != 0:
            lr = np.sqrt(np.log(self.n/self.weights_decay)
                         * self.weights_decay/self.bandit.k)
        p = -self.p*lr * y_t
        p -= np.max(p)
        self.w *= np.exp(p)
        self.w /= np.sum(self.w)

        self.w = (1-self.effective_weights_decay)*self.w + \
            self.effective_weights_decay/self.n

        np.testing.assert_allclose(
            self.w.sum(), 1, err_msg=f"{self.w.sum()} should be 1 {self.w.sum()-1} ")

        self.t += 1


class MAB(Collective):

    def __init__(self, bandit, policy, experts,
                 name=None,  gamma=None, weights_decay=None, base='d-TS', epsilon=0.5):
        assert base.upper() in ('D-TS', 'SW-TS', 'D-UCB',
                                'SW-UCB'), "base should be one of 'D-TS','SW-TS','D-UCB', or 'SW-UCB'"
        self.base = base.upper()
        super().__init__(bandit, policy,
                         experts, gamma=gamma,  name=name)

        self.weights_decay = weights_decay
        self.epsilon = epsilon

    def short_name(self):
        return f"{self.base}[µ={(self.weights_decay )}]"

    def initialize_w(self):
        self.reward_history = []
        self.context_history = []
        self.effective_n = self.n
        self.pull_history = []

        if 'TS' in self.base:
            self.betas = np.zeros(self.n)
            self.alphas = np.zeros(self.n)
        else:
            self.pulls = np.zeros(self.n)
            self.means = np.zeros(self.n)
        self.chosen_expert = np.random.randint(self.n)
        self.w = np.ones(self.n)/self.n
        self.unselected = np.ones(self.n, dtype=bool)

    def get_weights(self, contexts):

        if self.t < self.n and 'UCB' in self.base:
            # randomly choose an expert with 0 pulls
            self.chosen_expert = randargmax(self.unselected)
        else:

            if 'TS' in self.base:
                expert_values = np.random.beta(self.alphas+1,  self.betas+1)
            else:
                c = (2 if 'D-' in self.base else 1)*np.sqrt(self.epsilon *
                                                            np.log(np.sum(self.pulls))/(self.pulls+1e-10))

                m = self.means+0
                m[self.pulls > 0] /= self.pulls[self.pulls > 0]
                expert_values = m + c
            self.chosen_expert = randargmax(expert_values)

        w = np.zeros((self.n, self.bandit.k))
        w[self.chosen_expert, :] = 1
        # print(w)
        return w

    def observe(self, reward, arm):
        self.unselected[self.chosen_expert] = False
        self.pull_history.append(self.chosen_expert)
        self.reward_history.append(reward)

        window_size = 0 if self.weights_decay == 0 else int(
            1/self.weights_decay)

        if self.base == 'D-TS':
            self.alphas *= (1-self.weights_decay)
            self.betas *= (1-self.weights_decay)

            self.alphas[self.chosen_expert] += reward
            self.betas[self.chosen_expert] += 1 - reward
        elif self.base == 'D-UCB':

            self.means *= (1-self.weights_decay)
            self.pulls *= (1-self.weights_decay)

            self.means[self.chosen_expert] += reward
            self.pulls[self.chosen_expert] += 1

        elif self.base.startswith('SW-'):
            if 'UCB' in self.base:
                self.means.fill(0)
                self.pulls.fill(0)
            else:
                self.alphas.fill(0)
                self.betas.fill(0)
            for chosen_expert, r in zip(self.pull_history[-window_size:], self.reward_history[-window_size:]):
                if 'UCB' in self.base:
                    self.means[chosen_expert] += r
                    self.pulls[chosen_expert] += 1
                else:
                    self.alphas[chosen_expert] += r
                    self.betas[chosen_expert] += 1 - r

        self.t += 1




class MultiOnlineRidge():
    def __init__(self, alpha, adaptive=False, weights_decay=0,  beta=None):
        self._model = None
        self.beta = beta
        self.alpha = alpha

        self.weights_decay = weights_decay
        self.adaptive = adaptive
        

    @property
    def model(self):
        if self._model is None:

            self._model = self._init_model({})

        return self._model

    @property
    def gamma(self):
        return (1-self.weights_decay)

    def _init_model(self, model):
        model['A'] = np.identity(self.context_dimension) * self.alpha
        model['A_inv'] = np.identity(self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))

        self.A_history = [model['A']]
        self.b_history = []
        self.action_context_history = []
        self.reward_history = []
        self.t = 0
        return model

    def partial_fit(self, X, Y, arm=None, w=1):

        self.context_dimension = np.shape(X)[1]

        for x, y in zip(X, Y):
            x = x[..., None]/((1*len(x))**0.5)
            action_context = x*(w**.5)
            reward = y*(w**.5)

            b = ((reward) * action_context)

            if self.adaptive:

                self.model['A'] = self.model['A']*self.gamma + action_context.dot(
                    action_context.T) + (1-self.gamma)*self.alpha*np.identity(self.context_dimension)

                self.model['b'] = self.model['b'] * \
                    self.gamma + b

                self.model['A_inv'] = np.linalg.inv(
                    self.model['A'])
            else:

                self.model['A'] += x.dot(x.T)
                self.model['A_inv'] = SMInv(
                    self.model['A_inv'], x, x, 1)

                self.model['b'] += b

        self.model['theta'] = (
            self.model['A_inv'].dot(self.model['b']))
        self.t += 1

    def get_beta(self):
        if self.beta is None:
            gamma = min(1-1e-10, self.gamma)
            
            S = 1
            L = 1
            delta = .1
            s = 1
            
            adapt_beta = np.sqrt(self.alpha)*S + s * np.sqrt(2*np.log(1/delta) + self.context_dimension*np.log(
                1+(L**2*(1-gamma**(2*(self.t)))/(self.alpha*self.context_dimension*(1-gamma**2)))))
                
            return adapt_beta
        else:
            return self.beta

    def uncertainties(self, X):

        Xp = X/(X.shape[1]**.5)
        values = np.sqrt(((Xp @ self.model['A_inv']) * Xp).sum(axis=1))

        return np.asarray(values)

    def predict(self, X):
        self.context_dimension = X.shape[1]

        theta = self.model['theta'][None, ]

        X = X/((1*X.shape[1])**0.5)
        return (X*theta[:, :, 0]).sum(-1)



class LinUCB(Collective):
    def __init__(self, bandit, policy, experts, beta=None, name=None, trials=1,
                 alpha=1,  fixed=False, inverse_propensity=True,
                 adaptive=False, weights_decay=0,  n_sub=3,enable_corval=False,corval_exploration_type="max",
                 mode='UCB'):

        self.trials = trials
        self._model = None
        self.fixed = fixed
        self.mode = mode
        self.n_sub = n_sub
        self.adaptive = adaptive
        
        self.inverse_propensity = inverse_propensity
        self.enable_corval = enable_corval
        self.weights_decay =weights_decay
    
        self.gamma = (1-self.weights_decay)
        
        self.corval_exploration_type =corval_exploration_type
        self.context_dimension = experts
        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, beta=beta)

    def copy(self):
        return LinUCB(self.bandit, self.policy, self.n, beta=self.beta,
                      alpha=self.alpha)

    def short_name(self):
        if self.enable_corval:
            s = f"CORVAL[{(self.corval_exploration_type)}]"
       

        else:
            s = f"MCB[µ={(self.weights_decay )}]"
       
        return s

    @property
    def model(self):
        if self._model is None:            
            self._model = MultiOnlineRidge(
                    self.alpha, adaptive=self.adaptive, weights_decay=self.weights_decay, beta=1 if self.enable_corval and self.corval_exploration_type in ("max","weighted") else self.beta)

        return self._model

    def get_values(self, contexts, return_std=True,):

        self.estimated_rewards = self.model.predict(contexts)
        self.estimated_rewards = self.map_predictions(self.estimated_rewards)
        if return_std:
            
            if self.enable_corval and self.inverse_propensity:

                all_uncertainties = [sub.get_beta()*sub.uncertainties(self.shifted_advice.T,
                                                                        ) for sub in self.sub_models]
                indices = self.estimated_rewards+all_uncertainties
                self.sub_pis = [self.policy.from_values(idx, self) for idx in indices]
            if self.enable_corval and self.corval_exploration_type in ("max","weighted"):
                all_uncertainties = [sub.get_beta()*sub.uncertainties(self.shifted_advice.T,
                                                                    ) for sub in self.sub_models]

                if self.corval_exploration_type == "max":
                    uncertainties = (all_uncertainties[np.argmax(self.model.model["theta"].flatten())])
                else:
                    uncertainties = (all_uncertainties*np.abs(self.model.model["theta"].flatten()[:, None])).sum(axis=0)/(np.abs(self.model.model["theta"]).sum()+1e-10)

                    assert uncertainties.shape == (all_uncertainties[np.argmax(self.model.model["theta"].flatten())]).shape, (np.shape(
                        all_uncertainties), uncertainties.shape, (all_uncertainties[np.argmax(self.model.model["theta"].flatten())]).shape)

            else:
                uncertainties = self.model.uncertainties(contexts )
            return self.estimated_rewards, uncertainties
        else:
            return self.estimated_rewards

    def initialize_w(self):
        self._model = None
        self.sub_models = []
        if self.enable_corval:# str(self.weights_decay).startswith("prop") or str(self.weights_decay).startswith("auto") or str(self.weights_decay).startswith("base"):
            for weight_decay in generate_decays(self.n_sub):
                self.sub_models.append(MultiOnlineRidge(
                    self.alpha, adaptive=True,
                    weights_decay=weight_decay, beta=1))  
    def map_advice(self, advice):
        self.shifted_advice = (np.array(advice) - self.bandit.advice_offset)

        self.mean_predictions = np.zeros(self.bandit.k)
        
        assert np.shape(self.shifted_advice)[1] == self.bandit.k
        
        if self.enable_corval:# str(self.weights_decay).startswith("prop") or str(self.weights_decay).startswith("auto") or str(self.weights_decay).startswith("base"):
            model_predictions = np.array(
                [sub.predict(self.shifted_advice.T) for sub in self.sub_models]).T
         
            return model_predictions-self.mean_predictions[:, None]
        else:
            return self.shifted_advice.T

    def map_predictions(self, predictions):
        assert np.shape(predictions) == np.shape(self.mean_predictions), (np.shape(
            predictions), np.shape(self.mean_predictions))
        return predictions + self.mean_predictions

    def value_estimates(self, contexts, mu_only=False, samples=None):
        self.advice = np.copy(contexts['advice'])
        self.meta_contexts = self.map_advice(self.advice)
        
        mu, sigma = self.get_values(self.meta_contexts)
        
        if mu_only:
            return mu

        return mu + sigma*self.model.get_beta()

    def reset(self):
        super().reset()

    def observe(self, reward, arm):

        selection = np.zeros(self.bandit.k)
        selection[arm] = 1
        
        action_context = self.meta_contexts[arm]
        
        if self.enable_corval:
            [sub_model.partial_fit([self.shifted_advice.T[arm]], [(reward-self.bandit.expected_reward)], arm,
                                   w=self.sub_pis[mi][arm]/self.policy.pi[arm] if self.inverse_propensity else 1) for mi, sub_model in enumerate(self.sub_models)]
                            
        self.model.partial_fit(
            [action_context], [(reward-self.bandit.expected_reward)], arm)
        
        self.t += 1


def LOMD(p_t, loss_t, eta_t, prev_val):
    # Helper function to find lambda that satisfies the LOMD condition
    def equation(lamb):
        return np.sum(1 / (1 / p_t + eta_t * (loss_t - lamb))) - 1
   
    lambda_val = newton(equation, prev_val, tol=1e-5)
    p_next = 1 / (1 / p_t + eta_t * (loss_t - lambda_val))
    p_next = p_next/np.sum(p_next)
    return p_next, lambda_val


class CORRAL(Collective):
    def __init__(self, bandit, policy, experts, beta=None, name=None, trials=1,
                 alpha=1, inverse_propensity=True, n_sub=3,
                  strict=True,
                 mode='UCB'):

        self.strict = strict
        self.n_sub = n_sub
      
        self.trials = trials
        self.gamma = 1 / trials
        self.sub_beta = 1
        self.beta = np.exp(1 / np.log(trials))
        self.mode = mode
        self.inverse_propensity = inverse_propensity
        self.context_dimension = experts
        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, beta=np.exp(1 / np.log(trials)), gamma=self.gamma)
        self.beta = np.exp(1 / np.log(trials))

    def short_name(self):
        s = f"CORRAL"

        return s

    def reset(self):
        self.lambda_val = None
        super().reset()

    def choose(self, contexts,):

        self.advice = np.copy(contexts['advice'])

        assert np.abs(1-self.p_bar_t.sum()
                      ) < 1e-5, (self.p_bar_t, self.p_bar_t.sum())
        self.i_t = np.random.choice(
            len(self.sub_models), p=self.p_bar_t/self.p_bar_t.sum())

        votes = self.get_votes(self.advice, samples=self.policy.get_samples())
        self.policy.pi = (votes*self.p_bar_t[:, None]).sum(axis=0)

        assert self.policy.pi.shape == (
            self.bandit.k,), (self.policy.pi.shape, self.bandit.k)
        choice = np.random.choice(self.bandit.k, p=votes[self.i_t])

        return choice

    def initialize_w(self):

        self.sub_models = []
        
        for weight_decay in generate_decays(self.n_sub):
            self.sub_models.append(MultiOnlineRidge(
                self.alpha, adaptive=True,
                weights_decay=weight_decay, beta=self.sub_beta))
        M = len(self.sub_models)

        def R(T):
            return (self.n**2 * 0.0001*T**2)**(1/3)
            
        self.eta = 1/np.sqrt(self.trials)
        
        self.eta_t = np.array([self.eta] * M)
        self.scale_t = np.array([2 * M] * M)
        self.p_t = np.array([1 / M] * M)
        self.p_bar_t = np.copy(self.p_t)

    def get_votes(self, advice, samples=None):
        self.shifted_advice = (np.array(advice) - self.bandit.advice_offset)
        assert np.shape(self.shifted_advice)[1] == self.bandit.k
        predictions = np.array([sub.predict(self.shifted_advice.T)
                               for sub in self.sub_models])

        sample_to_scale = None
        uncertainties = [sub.get_beta()*sub.uncertainties(self.shifted_advice.T,
                                                        ) for sub in self.sub_models]
        combined = predictions + \
            uncertainties if samples is None else predictions[:,
                                                              None]+uncertainties
                                                              
        self.sub_pis = np.array(
            [self.policy.from_values(c, self) for c in combined])

        assert np.shape(self.sub_pis) == (len(self.sub_models), self.bandit.k), (np.shape(
            self.sub_pis), (len(self.sub_models), self.bandit.k))
            
        return self.sub_pis

    def observe(self, reward, arm):
        loss_t = 1-(reward)
        
        M = len(self.sub_models)
        f_t_i = np.zeros(M)
        f_t_i[self.i_t] = loss_t / self.p_bar_t[self.i_t]

        # Update p using LOMD
        if self.lambda_val is None:
            self.lambda_val = np.mean(f_t_i)
        self.p_t, self.lambda_val = LOMD(
            self.p_t, f_t_i, self.eta_t, self.lambda_val)
            
        self.p_bar_t = (1 - self.gamma) * self.p_t + \
            self.gamma * np.ones(M) / M
            
        # Update scales and learning rates
        for i in range(M):
            if 1 / self.p_bar_t[i] > self.scale_t[i]:
                
                self.scale_t[i] = 2 / self.p_bar_t[i]
                self.eta_t[i] = self.beta * self.eta_t[i]

        [sub_model.partial_fit([self.shifted_advice.T[arm]], [reward-self.bandit.expected_reward], arm,
                                   w=1 / self.p_bar_t[self.i_t] if mi == self.i_t else 0) for mi, sub_model in enumerate(self.sub_models)]
       
        self.t += 1
