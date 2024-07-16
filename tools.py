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


def bootstrap_ci(data, n_bootstrap=1000):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = data.sample(n=len(data), replace=True)
        bootstrap_means.append(sample.mean())
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    return np.mean(bootstrap_means), lower_bound, upper_bound
def bootstrap_confidence_interval(data, num_bootstrap_samples=500, alpha=0.05, axis=0,window_size=None):
    """
    Compute the bootstrapped 95% confidence intervals along a given axis.

    Parameters:
    data (ndarray): The input data array.
    num_bootstrap_samples (int): The number of bootstrap samples to generate. Default is 1000.
    alpha (float): The significance level. Default is 0.05 for a 95% confidence interval.
    axis (int): The axis along which to compute the confidence interval. Default is 0.

    Returns:
    tuple: Lower and upper confidence intervals.
    """
    if window_size is not None:
        data=np.rollaxis(data.reshape((data.shape[0],-1,window_size)),2,1).reshape((data.shape[0]*window_size,-1))
    # Ensure the data is a NumPy array
    data = np.asarray(data)

    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(data.shape[axis], (num_bootstrap_samples, data.shape[axis]), replace=True)
    
    # Compute the statistic (mean) for each bootstrap sample
    bootstrap_statistics = np.mean(np.take(data, bootstrap_samples, axis=axis),axis=axis+1)
    # print(np.take(data, bootstrap_samples, axis=axis).shape,bootstrap_statistics.shape)
    # Compute the confidence intervals
    lower_bound = np.percentile(bootstrap_statistics, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2), axis=0)
    med = np.median(bootstrap_statistics,axis=0)
    return lower_bound, upper_bound,med 
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

def generate_decays(n_decays):
    return (np.logspace(-3,0,n_decays+1)/2)[1:][::-1]
from numpy.lib.stride_tricks import sliding_window_view
def running_max(data, window_size):
    # Pad the data on both sides to handle the window at the edges
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')
    
    sliding_windows = sliding_window_view(padded_data, window_size)
    
    max_values = np.max(sliding_windows, axis=1)
    
    return max_values[:len(data)]

def get_r_advice(truth,dist):
    
    weights = np.random.choice([0,1],p=[1-dist,dist],size=truth.shape)
    return truth*(1-weights)+ (1-truth)*weights
   
def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into 
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    try:
                        maxlen = maxlens.max().astype(int) 
                    except:
                        maxlen = int(maxlens.max())
                    col_type = 'S%s' % maxlen# ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values            
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype



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
    # print(np.dot(np.dot(u, v.T), Ainv))
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
