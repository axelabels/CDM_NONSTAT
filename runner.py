
from sklearn.metrics import mean_absolute_error
from functools import lru_cache
from scipy.signal import square
import numpy as np

import pandas as pd
from tqdm import tqdm, trange
from itertools import product
import sys
import h5py
import numpy as np
from agent import *
from expert import OracleExpert
from bandit import *
from policy import *
from tqdm import tqdm
from tools import running_max, generate_decays
import os
import shutil
seed = int(sys.argv[1])

arm_counts = [8]
expert_count_bins = [[2], [4], [8], [16], [32], [64], [128], [256], [512],[1024]]
compute_misspecification_performance = "--extra-figure" in sys.argv 

bin_idx = seed % len(expert_count_bins)
expert_counts = expert_count_bins[bin_idx]
seed = seed//len(expert_count_bins)
np.random.seed(seed)
np.random.shuffle(expert_counts)

periods = (100, 500, 2500, 'static')[:]

modes = [2]

n_trials = 5000

verbose = os.getenv("VSC_SCRATCH", ".") == "."

work_path = output_folder = os.getenv(
    "VSC_SCRATCH", ".")+"/nonstationary_results/"

os.makedirs(work_path, exist_ok=True)
problems = ['mixture']


def collect_results(experiment, problem, n_arms, n_experts,  period, mode):

    filename = work_path+"period"+"_" + \
        "_".join(map(str, (experiment, n_arms, n_experts,
                           0.5, problem, period)))+".fth"
    if verbose:
        print(filename)
    if os.path.exists(filename):
        return

    def initialize_experiment(bandit, learners, experts, cache_id,   reset=True):
        if reset:
            bandit.reset()
            [e.reset() for e in experts]
        for learner in learners:
            learner.reset()
        if reset:
            learner.prior_play(experts, bandit)
            bandit.cache_contexts(n_trials, cache_id)
            for i, e in (enumerate(experts)):
                e.cache_predictions(bandit, n_trials)
                e.cached_predictions[:, 0] = bandit.cached_values

    data = []

    bandit = ArtificialBandit(n_arms=n_arms, problem=problem, bernoulli=True)
    bandit.advice_offset = bandit.expected_reward

    experts = [OracleExpert(bandit, GreedyPolicy()) for i in range(n_experts)]

    # set up learners
    learners = []
    learners += [Collective(bandit, GreedyPolicy(), n_experts)]

    weight_decays = (0,)+tuple(generate_decays(3))
    for weights_decay in weight_decays:
        learners += [MAB(bandit, GreedyPolicy(), n_experts,
                         weights_decay=weights_decay)]
        exp4_gamma = .5*(2*np.log(n_experts) /
                         (n_arms * n_trials)) ** (1 / 2)
        learners += [Exp4(bandit, Exp3Policy(), n_experts,
                          gamma=exp4_gamma, weights_decay=weights_decay)]
        learners += [LinUCB(bandit, GreedyPolicy(), n_experts, 
                            beta=1, trials=n_trials, adaptive=weights_decay > 0,
                            weights_decay=weights_decay,)]
    M = 3
    learners += [LinUCB(bandit, GreedyPolicy(), n_experts,
                            trials=n_trials,
                        adaptive=True, enable_corval=True, n_sub=M)]
    learners += [CORRAL(bandit, GreedyPolicy(), n_experts,
                        trials=n_trials,
                        n_sub=M)]

    # set up experiment (initializes bandits and experts)
    np.random.seed(experiment)

    initialize_experiment(bandit, learners, experts,
                          experiment,   reset=True)
    np.random.seed(experiment)

    cached_advice = np.array([e.cached_predictions for e in experts])[:, :, 0]
    phases = np.random.uniform(0, 1, size=n_experts)

    # generate non-stationary advice
    optimal_advice = bandit.cached_values

    per_round_expertises = []
    np.random.seed(experiment)

    def get_expertise(phase, period, t, duty=1/2):
        periods = np.random.normal(period, (period/2), size=100)
     
        a = np.mean([np.sin((np.random.uniform()+np.arange(n_trials*3)/p)*2*np.pi)
                    for i, p in enumerate(periods)], axis=0)
        a -= np.mean(a)
        a = a/running_max(np.abs(a), window_size=period)

        return (a/2+.5)[t+n_trials]

    biased_advices = []
    for i in trange(n_experts,disable=not verbose):
        expertise = (get_expertise(
            phases[i], 100 if period == 'static' else period, np.arange(n_trials), ))
        if period == 'static':
            expertise[:] = expertise[n_trials//2]

        dishonest_advice = bandit.neg_cached_values
        bandit.generate_random_values(optimal_advice.shape)
        (np.random.uniform(size=len(cached_advice[i])) < expertise)[:, None]
        (np.random.uniform(size=len(cached_advice[i])) >= 0)[:, None]

        def expertise_to_weights(expertise):

            dist = np.array(1-expertise)[:, None]
            dist = dist*np.ones_like(cached_advice[i])
            w_0 = np.maximum(0, (1-dist*2))**1
            w_1 = np.maximum(0, (dist*2-1))**1
            w_r = 1-w_0-w_1

            w_p = np.array((w_0, w_1, w_r))[..., 0].T

            cumulative = np.cumsum(w_p, axis=1)
            random_values = np.random.rand(len(w_p), 1)
            x = (random_values < cumulative)
            random_flip = np.argmax(x, axis=1)

            w = np.zeros_like(w_p)
            w[np.arange(n_trials), random_flip] = 1
            w = np.repeat(w.T[..., None], bandit.k, axis=-1)

            return w

        random_values = bandit.generate_random_values(optimal_advice.shape)
        if compute_misspecification_performance:
            per_round_biased_advice = [(np.array((cached_advice[i][..., 0], bandit.neg_cached_values[..., 0], random_values[..., 0]))
                                        * expertise_to_weights(np.repeat([e], n_trials))[..., 0]).sum(axis=0) for e in expertise]
            biased_advices.append(per_round_biased_advice)

        cached_advice[i] = (np.array((cached_advice[i], bandit.neg_cached_values,
                            bandit.random_values))*expertise_to_weights(expertise)).sum(axis=0)

        round_choices = greedy_choice(cached_advice[i], axis=1)

        expert_period = n_trials if period == 'static' else period//2
        expertsrounds = (round_choices*bandit.cached_rewards).sum(
            axis=1).reshape((-1, expert_period)).mean(axis=-1)
        per_round_expertises.append(expertsrounds)
        

    from sklearn.linear_model import Ridge

    best_expert_performances = np.repeat(
        np.max(per_round_expertises, axis=0), expert_period)
    if compute_misspecification_performance:
        biased_advices = np.array(biased_advices)

        model_errors = []
        for tt in range(np.shape(biased_advices)[1]):
            X = np.array(biased_advices[:, tt]).T
            y = bandit.cached_values[..., 0]
            model = Ridge().fit(X, y)
            model_errors.append(mean_absolute_error(y, model.predict(X)))

        model_error = np.mean(
            model_errors)/np.mean(np.abs(bandit.cached_values-np.mean(bandit.cached_values)))
        if verbose:
            print(n_arms, n_experts, period, problem, mode, "expperf",
                  best_expert_performances.mean(), "model er", model_error)
    else:
        model_error = 0

    results = np.zeros((n_experts+len(learners)+2, n_trials))
    step_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=n_trials)

    for t in trange(n_trials, desc=str((n_arms, n_experts, problem, period, mode)), smoothing=0, disable=not verbose):
        # Get current context and expert advice
        np.random.seed(step_seeds[t])
        context = bandit.observe_contexts(cache_index=t)
        sampled_rewards = bandit.sample(cache_index=t)

        advice = cached_advice[:, t]

        meta_context = {'advice': advice}
        for n, learner in enumerate(learners):
            np.random.seed(step_seeds[t])

            action = learner.choose(meta_context)

            reward = float(sampled_rewards[action])
            results[n_experts+n, t] = reward

            learner.observe(reward, action)

        # Log expert performance
        for e, expert in enumerate(experts):
            choice = np.argmax(advice[e])
            results[e, t] = sampled_rewards[choice]

        results[-1, t] = np.max(bandit.action_values)  # Best expected reward
        results[-2, t] = np.mean(bandit.action_values)  # Random policy

    for learner in learners:
        if verbose and type(learner) == CORRAL and learner.strict:
            print(t, learner.p_t)
    WINDOW = 10
    for s in np.arange(0, n_trials, WINDOW):

        for n, learner in enumerate(learners):
            if verbose and s == 0:
                print(np.mean(results[n_experts+n]), learner.short_name(),)
            learner_score = np.mean(results[n_experts+n, s:s+WINDOW])
            decay = getattr(learner, 'weights_decay', 0) 
            # decay = -1 if decay == "auto" else -2 if decay == "base" else - \
            #     3 if type(learner) == CORRAL else decay
            data.append([s, learner.short_name().replace("(decay=0)", ""), experiment, learner_score, "value", n_arms,
                         n_experts,   period, decay, problem, model_error, bandit.mixture_p])

        for sort_n, n in enumerate(sorted(range(n_experts), key=lambda n: -np.mean(results[n]))):
            if sort_n == 0:
                data.append([s, f"expert {sort_n}", experiment, np.mean(
                    best_expert_performances[s:s+WINDOW]), "value", n_arms, n_experts,  period, 0, problem, model_error, bandit.mixture_p])
            if verbose and s == 0:
                print(np.mean(results[n]), f"expert {n}",)

        data.append([s, f"random", experiment, np.mean(results[-2, s:s+WINDOW]), "value",
                     n_arms, n_experts,  period, 0, problem, model_error, bandit.mixture_p])
        data.append([s, f"optimal", experiment, np.mean(results[-1, s:s+WINDOW]), "value",
                     n_arms, n_experts,  period, 0, problem, model_error, bandit.mixture_p])
    header = ['t', 'algorithm', 'experiment', 'performance', 'type', 'n_arms', 'n_experts',
              "period", "decay", 'problem', 'model_error', 'shape']

    df = pd.DataFrame(data, columns=header)
    df.to_feather(filename,compression_level=4)


if __name__ == "__main__":
    np.random.seed(seed)
    configurations = np.array(list(product((seed,), problems, arm_counts,
                                           expert_counts,  periods, modes)), dtype=object)

    # configurations = sorted([conf for conf in configurations if (
    #     conf[2] == 8 or conf[3] == 8)], key=lambda conf: (conf[0], max(conf[2], conf[3]), conf[2]+conf[3]))[::1]
    np.random.seed(seed)
    for conf in tqdm(configurations, disable=not verbose):
        collect_results(*conf)
