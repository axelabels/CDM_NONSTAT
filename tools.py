from scipy import stats
import math
import re
import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae


def greedy_choice(a, axis=None):
    max_values = np.amax(a, axis=axis, keepdims=True)
    choices = (a == max_values).astype(np.float)
    return choices / np.sum(choices, axis=axis, keepdims=True)


def prob_round(x):
    x = np.asarray(x)
    probs = x - np.floor(x)
    added = probs > np.random.uniform(size=np.shape(probs))
    return (np.floor(x)+added).astype(int)


def permutate(v, n):
    assert n != 1
    v = np.copy(v)
    if n == 0:
        return v
    orig = np.random.choice(len(v), size=n, replace=False)
    dest = np.copy(orig)
    dest[:-1] = orig[1:]
    dest[-1] = orig[0]
    straight = np.arange(len(v))
    straight[orig] = straight[dest]

    return v[straight]

import sys

def get_truth_advice( cov_matr,  n_trials,alpha_beta=.1):
    
    x = np.random.multivariate_normal(mean=np.zeros(len(cov_matr)), cov=cov_matr, size=n_trials)
    norm = stats.norm()
    x_unif = norm.cdf(x)

    # compute parameters to satisfy given means and variance

    alpha = [alpha_beta]*len(cov_matr) 
    beta = [alpha_beta]*len(cov_matr) 
    advice = [stats.beta(a=alpha[i], b=beta[i]).ppf(x_unif[:, i]) for i in range(len(cov_matr))]



    return advice[0],np.array(advice[1:])

def get_err_advice(truth,dist):
    b = np.random.uniform(0,1,size=truth.shape)



    w_0 =max(0,(1-2*dist))**1
    w_1 = max(0,(2*dist-1))**1
    w_r = 1-w_0-w_1
    return truth *w_0 + b*w_r+ w_1*(1-truth)
    

def get_r_advice(truth,dist):
    
    weights = np.random.choice([0,1],p=[1-dist,dist],size=truth.shape)
    return truth*(1-weights)+ (1-truth)*weights
   
def get_advice( cov_matr, desired_var, truth,n_trials=None):
    
    if np.shape(desired_var)==():
        desired_var = np.zeros(len(cov_matr))+desired_var

    if n_trials is None:
            n_trials=len(truth)
    x = np.random.multivariate_normal(mean=np.zeros(len(cov_matr)), cov=cov_matr, size=n_trials)
    norm = stats.norm()
    x_unif = norm.cdf(x)

    # compute parameters to satisfy given means and variance

    desired_var = np.clip(desired_var,1e-5,1-1e-5)[:,None]+truth*0
    max_error = np.maximum(0,((1-truth)-truth)**2-1e-10)**.5 
    
    desired_var *= max_error
    
    error = desired_var
    max_variance = truth - truth**2 
    
    mu_biased = truth + (truth<0.5)*error - (truth>0.5)*error

    var = mu_biased-mu_biased**2-1e-10
   
    var = np.clip(var,np.minimum(1e-5,(truth-truth**2-1e-10)/10), .01)
  
    alpha = ((1-mu_biased)/var -1/mu_biased)*mu_biased**2
    beta = alpha*(1/mu_biased-1)
  

    assert not np.isnan(alpha).any(),(var[np.isnan(alpha)],mu_biased[np.isnan(alpha)],( mu_biased-mu_biased**2)[np.isnan(alpha)],np.min(alpha))
    assert np.min(alpha)>0,(var[alpha<=0],mu_biased[alpha<=0],( mu_biased-mu_biased**2)[alpha<=0],np.min(alpha))
    assert np.min(beta)>0,(var[beta<=0],mu_biased[beta<=0],( mu_biased-mu_biased**2)[beta<=0])

    if np.min(alpha)<0 or np.min(beta)<0:
        print("clipping parameters",file=sys.stderr)
        print(np.min(truth),np.max(truth))
        alpha = np.maximum(0,alpha)
        beta = np.maximum(0,beta)


    advice = [stats.beta(a=alpha[i], b=beta[i]).ppf(x_unif[:, i]) for i in range(len(cov_matr))]
    return np.array(advice)


def arr2str(array):
    """Converts an array into a one line string

    Arguments:
        array {array} -- Array to convert

    Returns:
        str -- The string representation
    """
    return re.sub(r'\s+', ' ',
                  str(array).replace('\r', '').replace('\n', '').replace(
                      "array", "").replace("\t", " "))


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))


def SMInv(Ainv, u, v, alpha=1):
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))
    
    return Ainv - np.dot(Ainv, np.dot(np.dot(u, v.T), Ainv)) / (1 + np.dot(v.T, np.dot(Ainv, u)))


def randargmax(b, **kw):
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def scale(a):
    return normalize(a, offset=np.min(a), scale=np.ptp(a))


def max_scale(a):
    return normalize(a, offset=np.min(a), scale=np.ptp(a) if np.ptp(a) != 0 else 1e-9)


def normalize(a, offset=None, scale=None, axis=None):
    a = np.asarray(a)
    if offset is None:
        offset = np.mean(a, axis=axis)
    if scale is None:
        scale = np.std(a, axis=axis)
    if type(scale) in (float, np.float64, np.uint8, np.int64):
        if scale == 0:
            print(f"forcing scale to 1, was {scale}")
            scale = 1
    else:
        try:
            scale[scale == 0] = 1
        except:
            print("a:", np.shape(a), "scale:", np.shape(scale), scale)
            raise
    return (a - offset) / scale


def rescale(a, mu, std):
    a = np.asarray(a)
    return a * std + mu


def get_coords(dims=2, complexity=3, precision=100, flatten=True):

    lin = np.linspace(0, complexity, int(
        (precision)**(2/dims)) + 1, endpoint=True)
    coords = np.array(np.meshgrid(*(lin for _ in range(dims)))).T / complexity
    if flatten:
        coords = coords.reshape((-1, dims))
    return coords
