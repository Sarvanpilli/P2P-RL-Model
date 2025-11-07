# utils/metrics.py
import numpy as np

def gini_coefficient(x):
    """
    Compute Gini coefficient for array x (non-negative). Returns 0..1, 0 is perfect equality.
    """
    x = np.array(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    # ensure non-negative
    x = np.abs(x)
    x_sorted = np.sort(x)
    n = x.size
    cum = np.cumsum(x_sorted)
    sum_x = cum[-1]
    if sum_x == 0:
        return 0.0
    # Gini formula
    idx = np.arange(1, n+1)
    gini = (2.0 * np.sum(idx * x_sorted) - (n+1) * sum_x) / (n * sum_x)
    return float(gini)
