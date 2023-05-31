import numpy as np
from scipy.stats import bootstrap

def batch_expected_diff_norm(X, Y, batch_size=1000):
    """
    Compute E[||X-Y||] for energy distance.

    Inputs:
        param X: (N, D)
        param Y: (M, D)
        batch_size: number of samples

    Output:
        E[||X-Y||]
    """

    total_size = Y.shape[0]
    cur = 0
    total = 0
    while cur < total_size:
        cur_size = min(total_size - cur, batch_size)
        tmp = X.unsqueeze(1) - Y[cur:cur+cur_size].unsqueeze(0)
        tmp = tmp.square().sum(-1).sqrt().sum()
        total += tmp.item() / X.shape[0]
        cur += cur_size
    return total / total_size


def compute_energy_dist(source_samples, target_samples):
    """
    Compute energy distance

    Inputs:
        source_samples: samples from model distribution
        target_samples: samples from target distribution

    Outputs:
        energy distance
    """
    SS = batch_expected_diff_norm(source_samples, source_samples)
    ST = batch_expected_diff_norm(source_samples, target_samples)
    TT = batch_expected_diff_norm(target_samples, target_samples)
    return 2 * ST - SS - TT


def return_confidence_interval(results_array):
    data = (results_array,)
    res = bootstrap(data, np.mean, confidence_level=0.9)
    upper = res.confidence_interval[1]
    lower = res.confidence_interval[0]
    return lower, upper