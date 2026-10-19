"""Distribution matrices of the two-dimensional discrete RV.

MATRICES[name][i, j] = P(X = VALUES_X[i], Y = VALUES_Y[j]). To change the
distribution, rewrite this file: the rest of the code takes the matrices and
the theoretical characteristics only from here.
"""

import numpy as np

VALUES_X = np.arange(1, 6, dtype=float)
VALUES_Y = np.arange(1, 6, dtype=float)

# Most of the probability sits on the diagonal: Y tends to repeat X.
DEPENDENT = np.array(
    [
        [0.090, 0.050, 0.015, 0.010, 0.010],
        [0.050, 0.090, 0.050, 0.015, 0.010],
        [0.015, 0.050, 0.090, 0.050, 0.015],
        [0.010, 0.015, 0.050, 0.090, 0.050],
        [0.010, 0.010, 0.015, 0.050, 0.090],
    ]
)

# Product of the marginals of DEPENDENT: the same M and D, but no dependence.
INDEPENDENT = np.outer(DEPENDENT.sum(axis=1), DEPENDENT.sum(axis=0))

MATRICES = {"Dependent": DEPENDENT, "Independent": INDEPENDENT}


def marginals(matrix):
    return matrix.sum(axis=1), matrix.sum(axis=0)


def characteristics(matrix):
    """Theoretical M[X], M[Y], D[X], D[Y], rho of the matrix."""
    p_x, p_y = marginals(matrix)
    mean_x = float(VALUES_X @ p_x)
    mean_y = float(VALUES_Y @ p_y)
    var_x = float(((VALUES_X - mean_x) ** 2) @ p_x)
    var_y = float(((VALUES_Y - mean_y) ** 2) @ p_y)
    cov = float((VALUES_X - mean_x) @ matrix @ (VALUES_Y - mean_y))
    return mean_x, mean_y, var_x, var_y, cov / np.sqrt(var_x * var_y)
