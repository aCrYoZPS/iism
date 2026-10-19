import numpy as np

from params import Params


def generate_pairs(n, rng):
    """x = F_X^-1(u1), y = F^-1(u2 | x) for independent u1, u2 ~ R(0, 1)."""
    u1 = rng.random(n)
    u2 = rng.random(n)
    x = Params.cdf_x_inv(u1)
    y = Params.cdf_y_given_x_inv(u2, x)
    return x, y


def generate_pairs_discrete(n, matrix, values_x, values_y, rng):
    """Split (0, 1) into segments: by the row sums for x, then by p_ij / p_i. for y."""
    row_sums = matrix.sum(axis=1)
    x_bounds = np.cumsum(row_sums)
    x_index = np.minimum(np.searchsorted(x_bounds, rng.random(n), side="right"), len(row_sums) - 1)

    y_bounds = np.cumsum(matrix / row_sums[:, None], axis=1)
    u2 = rng.random(n)
    y_index = np.minimum((u2[:, None] >= y_bounds[x_index]).sum(axis=1), matrix.shape[1] - 1)

    return values_x[x_index], values_y[y_index]
