import numpy as np


class Params:
    MEAN_X = 1.0
    MEAN_Y = 1.0
    VAR_X = 1.0
    VAR_Y = 1.0
    RHO = 0.0

    @staticmethod
    def f(x, y):
        inside = (x >= 0) & (y >= 0)
        return np.where(inside, np.exp(-np.where(inside, x + y, 0.0)), 0.0)

    @staticmethod
    def f_x(x):
        return np.where(x >= 0, np.exp(-np.maximum(x, 0.0)), 0.0)

    @staticmethod
    def f_y(y):
        return np.where(y >= 0, np.exp(-np.maximum(y, 0.0)), 0.0)

    @staticmethod
    def f_y_given_x(y, x):
        return np.where(y >= 0, np.exp(-np.maximum(y, 0.0)), 0.0) * np.ones_like(x, dtype=float)

    @staticmethod
    def cdf_x(x):
        return np.where(x >= 0, 1.0 - np.exp(-np.maximum(x, 0.0)), 0.0)

    @staticmethod
    def cdf_y(y):
        return np.where(y >= 0, 1.0 - np.exp(-np.maximum(y, 0.0)), 0.0)

    @staticmethod
    def cdf_x_inv(u):
        return -np.log(1.0 - u)

    @staticmethod
    def cdf_y_inv(u):
        return -np.log(1.0 - u)

    @staticmethod
    def cdf_y_given_x_inv(u, x):
        return -np.log(1.0 - u) * np.ones_like(x, dtype=float)
