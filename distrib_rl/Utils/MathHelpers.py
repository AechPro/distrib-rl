import scipy.signal
import numpy as np
import torch
import functools


def compute_torch_normal_entropy(sigma):
    return (0.5 + 0.5*np.log(2*np.pi) + torch.log(sigma)).sum(dim=-1).item()


def minmax_norm(x, min_val, max_val):
    if min_val == max_val:
        return x
    return (x - min_val) / (max_val - min_val)


def compute_array_stats(arr):
    if len(arr) == 0 or type(arr) not in (list, tuple, np.ndarray):
        return 0,1,0,0
    return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)


def compute_discounted_future_sum(arr, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], arr[::-1], axis=0)[::-1]


@functools.lru_cache()
def apply_affine_map(value, from_min, from_max, to_min, to_max):
    if from_max == from_min or to_max == to_min:
        return to_min

    mapped = (value - from_min) * (to_max - to_min) / (from_max - from_min)
    mapped += to_min

    return mapped


@functools.lru_cache()
def map_policy_to_continuous_action(policy_output):
    n = policy_output.shape[-1]//2
    if len(policy_output.shape) == 1:
        mean = policy_output[:n]
        std = policy_output[n:]

    else:
        mean = policy_output[:, :n]
        std = policy_output[:, n:]

    std = apply_affine_map(std, -1, 1, 1e-1, 1)
    return mean, std