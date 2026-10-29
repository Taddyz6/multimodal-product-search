"""Frozen scaler and 28→64→32→1 ListNet inference in NumPy.

Mean/std may be float64. Casting them early changes the deployed computation;
standardize first, then cast the standardized values to float32.
"""

import numpy as np


class ListNetRanker:
    def __init__(self, weights):
        self.weights = weights

    def scale(self, x):
        w = self.weights
        return np.asarray((np.asarray(x, np.float32) - w["mean"]) / w["std"], np.float32)

    def forward(self, z):
        w = self.weights
        for j in [1, 2]:
            z = np.maximum(z @ w[f"w{j}"].T + w[f"b{j}"], 0)
        return (z @ w["w3"].T + w["b3"]).reshape(-1)


def ordered_indices(scores, keys):
    s = np.asarray(scores)
    if s.shape != (len(keys),) or not np.isfinite(s).all():
        raise ValueError("Finite per-candidate scores required")
    return np.lexsort(([k[1] for k in keys], [k[0] for k in keys], -s))
