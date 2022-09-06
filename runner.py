
from functools import lru_cache
from scipy.signal import square
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from itertools import product
import sys

import numpy as np
from agent import *
from expert import OracleExpert
from bandit import *
from policy import *
from tqdm import tqdm
import os
seed = int(sys.argv[1])
arm_counts = (2, 8,32,128)
expert_counts = (2, 8,32,128)

periods = ( 60, 240, 960,'static')
modes = (0, 1)

verbose = True

work_path = "results_period/"


os.makedirs(work_path, exist_ok=True)
problem = ('regression','classification')


def collect_results(experiment, problem, n_arms, n_experts,  period, mode):
    
    filename = work_path+"period"+"_" + \
        "_".join(map(str, (experiment, n_arms, n_experts,
                           0.5, problem, period,f"{mode:.03f}")))+".fth"
    if os.path.exists(filename): return

    n_trials = 960

    def generate_expert(bandit, i):
        experts = [OracleExpert(bandit, GreedyPolicy())]
        return experts[i % len(experts)]

    def generate_experts(n, bandit):
        return [generate_expert(bandit, i) for i in range(n)]

    def initialize_experiment(bandit, learners, experts, cache_id,   reset=True):
        if reset:
            bandit.reset()
            [e.reset() for e in experts]
        for learner in learners:
            learner.reset()
        if reset:
            learner.prior_play(experts, 
                               bandit)

            bandit.cache_contexts(n_trials, cache_id)

            truth = bandit.cached_values.flatten()
       
            desired_var=0.25
           
            advice = np.zeros((n_experts,)+truth.shape) 
            advice_reshaped = advice.reshape((n_experts,)+bandit.cached_values.shape)
            
            for n in range(n_experts):
                advice_reshaped[n] = bandit.cached_values+0
                
                e_idx = np.random.choice(len(bandit.cached_values),size=int(len(bandit.cached_values)*desired_var),replace=False)
                
                expert_erroneous_advice = get_err_advice(truth,desired_var).reshape(bandit.cached_values.shape) #

                if bandit.problem=='classification' :
                    advice_reshaped[n,e_idx] = bandit.generate_random_values(bandit.cached_values.shape)[e_idx]  
                else:
                    advice_reshaped[n] = expert_erroneous_advice
                    
                
    
            for i, e in (enumerate(experts)):
                e.cache_predictions(bandit, n_trials) 
                e.cached_predictions[:,0] = advice[i].reshape(bandit.cached_values.shape)
            

            
    data = []


    bandit = ArtificialBandit(n_arms=n_arms,problem=problem,bernoulli=True)
        

    experts = generate_experts(n_experts, bandit)

    @lru_cache(maxsize=None)
    def get_smoothed_data(phase, period, smoothing,  duty):
        if not smoothing:
            return np.sin((phase+np.arange(n_trials)/period)*2*np.pi)
        else:
            return square((phase+np.arange(n_trials)/period)*2*np.pi, duty=duty)

    def get_expertise(phase, period, t, smoothing, duty=1/2):
        return get_smoothed_data(phase, period, smoothing,  duty)[t]/2+.5
    ALPHA=1
    # set up learners
    learners = []
    learners += [Collective(bandit, GreedyPolicy(), n_experts)]


    for weights_decay in (0,)+tuple(np.logspace(-8, -.1, 10, base=2))[:]+(2**-.01, 1,):
        # experts doubled for inversion trick
        learners += [MAB(bandit, GreedyPolicy(), n_experts *
                         2, weights_decay=weights_decay)]
        exp4_gamma = .5*(2*np.log(n_experts*2) /
                         (n_arms * n_trials)) ** (1 / 2)
        learners += [Exp4(bandit, Exp3Policy(), n_experts*2,
                            gamma=exp4_gamma, weights_decay=weights_decay)]
        
        learners += [LinUCB(bandit, GreedyPolicy(), n_experts, alpha=ALPHA, trials=n_trials, adaptive=weights_decay > 0,
                                    weights_decay=weights_decay,)]
    
    
        
    # set up experiment (initializes bandits and experts)
    np.random.seed(experiment)
    initialize_experiment(bandit, learners, experts,
                          experiment,   reset=True)
    np.random.seed(experiment)

    cached_advice = np.array([e.cached_predictions for e in experts])[:, :, 0]
    
    phases = np.random.uniform(0, 1, size=n_experts)

    # generate non-stationary advice
    optimal_advice = bandit.cached_values
    common_dishonest_advice = bandit.generate_random_values(optimal_advice.shape)
    
    for i in range(n_experts):
        if period=='static':
            expertise = get_expertise(phases[i], 60, 0, mode) 
        else:
            expertise = get_expertise(phases[i], period, np.arange(n_trials), mode)
            
        dishonest_advice = 1-cached_advice[i]
        if bandit.problem=='classification':
            dishonest_advice = common_dishonest_advice
            individual_adv_advice = bandit.generate_random_values(optimal_advice.shape)
        else: 
            individual_adv_advice = dishonest_advice
            
        choices = (np.random.uniform(size=len(cached_advice[i]))<expertise)[:,None]
        

        coor_flags = (np.random.uniform(size=len(cached_advice[i]))<.75)[:,None]
        cached_advice[i] = cached_advice[i]*choices + ((1-choices)*coor_flags)*dishonest_advice + ((1-choices)*(1-coor_flags))*individual_adv_advice


    # run experiment
    results = np.zeros((n_experts+len(learners)+2, n_trials))
    step_seeds = np.random.randint(0, np.iinfo(np.int32).max,size=n_trials)
 
 
    for t in trange(n_trials, desc=str((n_arms, n_experts, problem, period,mode)),disable=not verbose):
        # Get current context and expert advice
        np.random.seed(step_seeds[t])
        context = bandit.observe_contexts(cache_index=t)
        sampled_rewards = bandit.sample(cache_index=t)
    
        advice = cached_advice[:,t]
     
        meta_context = {'advice': advice,}
        expanded_meta_context = {'advice': np.vstack([advice, 1-advice])}
        for n, learner in enumerate(learners):
            np.random.seed(step_seeds[t])
            if type(learner) in (MAB, Exp4):  # invert method
                action = learner.choose(expanded_meta_context)
            else:
                action = learner.choose(meta_context)

            reward = float(sampled_rewards[action])
            results[n_experts+n, t] = reward
          
            learner.observe(reward, action)

        # Log expert performance
        for e, expert in enumerate(experts):
            probabilities = greedy_choice(advice[e])
            results[e, t] = np.dot(sampled_rewards,probabilities)

        results[-1, t] = np.max(bandit.action_values) # Best expected reward
        results[-2, t] = np.mean(bandit.action_values)  # Random policy

    WINDOW = 5
    for s in np.arange(0, n_trials, WINDOW):

        for n, learner in enumerate(learners):
            learner_score = np.mean(results[n_experts+n, s:s+WINDOW])
            data.append([s, learner.short_name().replace("(decay=0)", ""), experiment, learner_score, "value", n_arms,
                         n_experts,   period, getattr(learner, 'weights_decay', 0), problem])

        for sort_n, n in enumerate(sorted(range(n_experts), key=lambda n: -np.mean(results[n]))):
            data.append([s, f"expert {sort_n}", experiment, np.mean(
                results[n, s:s+WINDOW]), "value", n_arms, n_experts,  period, 0, problem])
            break
        data.append([s, f"random", experiment, np.mean(results[-2, s:s+WINDOW]), "value",
                     n_arms, n_experts,  period, 0, problem])
        data.append([s, f"optimal", experiment, np.mean(results[-1, s:s+WINDOW]), "value",
                     n_arms, n_experts,  period, 0, problem])
    header = ['t', 'algorithm', 'experiment', 'performance', 'type', 'n_arms', 'n_experts',
               "period", "decay", 'problem']

    df = pd.DataFrame(data,columns=header)
    df.to_feather(filename)


if __name__ == "__main__":
    configurations = np.array(list(product((seed,), problem, arm_counts,
                                           expert_counts,  periods, modes)), dtype=object)

    configurations = [conf for conf in configurations if (conf[2]==8 or conf[3] ==8)]
    for conf in tqdm(configurations,disable=not verbose):
        collect_results(*conf)
