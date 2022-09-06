
from numbers import Number

from expert import *

EXPECTED_AVG_REWARD = .5


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
                 p=16):
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
        return f"EXP4.S(decay={self.weights_decay})"

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
        return f"{self.base}(decay={self.weights_decay})"

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
            # randomly chose an expert with 0 pulls
            self.chosen_expert = randargmax(self.unselected)
            # print(self.t,self.base,self.chosen_expert)
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
    def __init__(self, alpha, adaptive=False, weights_decay=0, adaptive_uncertainty=False):
        self._model = None
        self.alpha = alpha

        self.weights_decay = weights_decay
        self.adaptive = adaptive
        self.adaptive_uncertainty = adaptive_uncertainty

    @property
    def model(self):
        if self._model is None:

            self._model = self._init_model({})

        return self._model

    @property
    def gamma(self):
        return 1-self.weights_decay

    def _init_model(self, model):
        model['A'] = np.identity(self.context_dimension) * self.alpha
        model['A_ap'] = np.identity(self.context_dimension) * self.alpha
        model['A_inv'] = np.identity(self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))

        self.A_history = [model['A']]
        self.b_history = []
        self.action_context_history = []
        self.reward_history = []

        return model

    def partial_fit(self, X, Y, arm=None):

        self.context_dimension = np.shape(X)[1]

        for x, y in zip(X, Y):
            x = x[..., None]/((1*len(x))**0.5)
            action_context = x
            reward = y

            self.reward_history.append(reward)
            self.action_context_history.append(action_context)
            self.b_history.append((reward) * action_context)

            if self.adaptive:
                self.model['A'] = self.model['A']*self.gamma + action_context.dot(
                    action_context.T) + (1-self.gamma)*self.alpha*np.identity(self.context_dimension)
                self.model['A_ap'] = self.model['A_ap']*self.gamma**2 + action_context.dot(
                    action_context.T) + (1-self.gamma**2)*self.alpha*np.identity(self.context_dimension)

                self.model['b'] = self.model['b'] * \
                    self.gamma + self.b_history[-1]

                self.model['A_inv'] = np.linalg.inv(
                    self.model['A'])
            else:

                self.model['A'] += x.dot(x.T)
                self.model['A_inv'] = SMInv(
                    self.model['A_inv'], x, x, 1)

                self.model['b'] += self.b_history[-1]

        self.model['theta'] = (
            self.model['A_inv'].dot(self.model['b']))

    def uncertainties(self, X, sample=None):

        Xp = X/(X.shape[1]**.5)

        if self.adaptive and self.adaptive_uncertainty:
            matrix = (np.array(self.model['A_inv'][None, ]) *
                      (self.model['A_ap'][None, ])*(self.model['A_inv'][None, ]))[0]
        else:
            matrix = self.model['A_inv']

        values = np.sqrt(
            ((Xp[:, :, np.newaxis]*matrix[None, ]).sum(axis=1)*Xp).sum(-1))

        np.testing.assert_allclose(values,  np.sqrt((Xp@matrix*Xp).sum(-1)))

        if sample is not None:
            vvalues = sample * \
                np.random.normal(np.zeros_like(values),
                                 values, size=values.shape)

            assert not np.isnan(vvalues).any(), (values, vvalues, matrix, ((
                Xp[:, :, np.newaxis]*matrix[None, ]).sum(axis=1)*Xp).sum(-1))
            values = vvalues
        else:
            np.random.normal(np.zeros_like(values), values, size=values.shape)

        assert not np.isnan(values).any(), values

        return np.asarray(values)

    def predict(self, X):
        self.context_dimension = X.shape[1]

        theta = self.model['theta'][None, ]

        X = X/((1*X.shape[1])**0.5)
        return (X*theta[:, :, 0]).sum(-1)


class LinUCB(Collective):
    def __init__(self, bandit, policy, experts, beta=None, name=None, trials=1,
                 alpha=1,  fixed=False,
                 adaptive=False, weights_decay=0, adaptive_uncertainty=False,
                 mode='UCB'):

        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, beta=beta)
        self.trials = trials
        self._model = None
        self.fixed = fixed
        self.mode = mode
        self.adaptive = adaptive
        self.adaptive_uncertainty = adaptive_uncertainty

        self.weights_decay = weights_decay
        self.gamma = (1-weights_decay)
        self.context_dimension = experts

    def copy(self):
        return LinUCB(self.bandit, self.policy, self.n, beta=self.beta,
                      alpha=self.alpha)

    def short_name(self):
        return f"D-Meta-CMAB(decay={(self.weights_decay)})"

    @property
    def model(self):
        if self._model is None:
            self.counts = {}
            self._model = MultiOnlineRidge(
                self.alpha, adaptive=self.adaptive, weights_decay=self.weights_decay, adaptive_uncertainty=self.adaptive_uncertainty)

        return self._model

    def get_values(self, contexts, return_std=True):

        estimated_rewards = self.model.predict(
            contexts)
        if return_std:
            uncertainties = self.model.uncertainties(contexts, sample=None)

            return estimated_rewards, uncertainties
        else:
            return estimated_rewards

    def initialize_w(self):
        self._model = None

        self.context_history = []
        self.full_context_history = []
        self.reward_history = []
        self.action_history = []

        self.selection_history = []

    def value_estimates(self, contexts, mu_only=False):
        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice) - self.bandit.expected_reward

        self.meta_contexts = centered_advice.T
        mu, sigma = self.get_values(self.meta_contexts)
        if mu_only:
            return mu
        assert self.context_dimension > 0

        if self.beta is None:

            weights = np.array([0.04592796, 0.12515091, 0.05823158])

            beta1 = np.sqrt(.5*np.log(2*self.trials*self.bandit.k*self.trials))
            beta3 = 0 if self.t == 0 else np.sqrt(
                max(0, 2*np.log(self.trials)+np.log(1/np.linalg.det(self.model.model["A_inv"]))))
            beta4 = 1

            adapt_beta = weights.dot([beta1, beta3, beta4])

            return mu + sigma*adapt_beta

        else:

            return mu + sigma*self.beta

    def reset(self):
        super().reset()

    def observe(self, reward, arm):
        self.full_context_history.append(self.meta_contexts)
        self.context_history.append(self.meta_contexts[arm])
        selection = np.zeros(self.bandit.k)
        selection[arm] = 1
        self.selection_history.append(selection)
        self.action_history.append(arm)

        action_context = self.meta_contexts[arm]
        self.reward_history.append(reward)
        if not self.fixed:

            self.model.partial_fit(
                [action_context], [(reward-self.bandit.expected_reward)], arm)

        self.t += 1
