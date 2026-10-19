from dataclasses import dataclass

import numpy as np
from scipy import stats

CONFIDENCE = 0.95


@dataclass(frozen=True)
class Estimates:
    mean: float
    variance: float
    mean_interval: tuple
    variance_interval: tuple


@dataclass(frozen=True)
class HypothesisTest:
    statistic: float
    p_value: float

    @property
    def rejected(self):
        return self.p_value < 1.0 - CONFIDENCE


@dataclass(frozen=True)
class Correlation:
    r: float
    interval: tuple
    t_statistic: float
    p_value: float


@dataclass(frozen=True)
class IndependenceTest:
    statistic: float
    dof: int
    critical: float
    p_value: float
    min_expected: float

    @property
    def rejected(self):
        return self.statistic > self.critical


def estimates(sample):
    n = len(sample)
    if n < 2:
        return None

    mean = float(np.mean(sample))
    variance = float(np.var(sample, ddof=1))
    dof = n - 1
    alpha = 1.0 - CONFIDENCE

    half_width = stats.t.ppf(1.0 - alpha / 2.0, dof) * np.sqrt(variance / n)

    # The chi-square interval assumes a normal sample; this one does not.
    variance_half_width = stats.norm.ppf(1.0 - alpha / 2.0) * _variance_std_error(sample, variance)
    variance_interval = (variance - variance_half_width, variance + variance_half_width)

    return Estimates(
        mean=mean,
        variance=variance,
        mean_interval=(mean - half_width, mean + half_width),
        variance_interval=variance_interval,
    )


def _variance_std_error(sample, variance):
    """sd of s^2 is about s^2 * sqrt((kurtosis - 1) / n); kurtosis is 3 for a normal sample."""
    centered = sample - np.mean(sample)
    kurtosis = np.mean(centered**4) / np.mean(centered**2) ** 2
    return variance * np.sqrt((kurtosis - 1.0) / len(sample))


def mean_test(sample, mean0):
    """H0: M = mean0, t ~ Student(n - 1)."""
    n = len(sample)
    if n < 2:
        return None

    t_statistic = (np.mean(sample) - mean0) / np.sqrt(np.var(sample, ddof=1) / n)
    p_value = 2.0 * stats.t.sf(abs(t_statistic), n - 1)
    return HypothesisTest(float(t_statistic), float(p_value))


def variance_test(sample, variance0):
    """H0: D = variance0, z ~ N(0, 1) for large n; no normality of the sample is assumed."""
    if len(sample) < 2:
        return None

    variance = np.var(sample, ddof=1)
    z_statistic = (variance - variance0) / _variance_std_error(sample, variance)
    p_value = 2.0 * stats.norm.sf(abs(z_statistic))
    return HypothesisTest(float(z_statistic), float(p_value))


def correlation(x, y):
    n = len(x)
    if n < 4:
        return None

    r = float(np.corrcoef(x, y)[0, 1])

    z = np.arctanh(r)
    half_width = stats.norm.ppf(1.0 - (1.0 - CONFIDENCE) / 2.0) / np.sqrt(n - 3)
    interval = (float(np.tanh(z - half_width)), float(np.tanh(z + half_width)))

    t_statistic = r * np.sqrt((n - 2) / (1.0 - r * r))
    p_value = 2.0 * stats.t.sf(abs(t_statistic), n - 2)

    return Correlation(r, interval, float(t_statistic), float(p_value))


def density_histogram(sample, low, high, bins):
    counts, edges = np.histogram(sample, bins=bins, range=(low, high))
    width = (high - low) / bins
    return counts / (len(sample) * width), edges


def density_histogram_2d(x, y, x_range, y_range, bins):
    """Cell heights n_ij / (n * dx * dy): the volume of all cells is the share of points inside."""
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=(x_range, y_range))
    cell_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    return counts / (len(x) * cell_area), x_edges, y_edges


def contingency_table(x, y, values_x, values_y):
    """Counts n_ij of the pairs (values_x[i], values_y[j]) in a discrete sample."""
    cells = np.searchsorted(values_x, x) * len(values_y) + np.searchsorted(values_y, y)
    return np.bincount(cells, minlength=len(values_x) * len(values_y)).reshape(len(values_x), len(values_y))


def independence_test_table(observed):
    """Pearson chi-square for independence: E_ij = (row sum * column sum) / n.

    The row and column probabilities are estimated from the sample, so
    dof = (r - 1)(c - 1). Empty rows and columns are dropped.
    """
    observed = observed[observed.sum(axis=1) > 0][:, observed.sum(axis=0) > 0]
    r, c = observed.shape
    if r < 2 or c < 2:
        return None
    n = observed.sum()

    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    dof = (r - 1) * (c - 1)

    return IndependenceTest(
        statistic=statistic,
        dof=dof,
        critical=float(stats.chi2.ppf(CONFIDENCE, dof)),
        p_value=float(stats.chi2.sf(statistic, dof)),
        min_expected=float(expected.min()),
    )


def class_indices(sample, edges):
    return np.searchsorted(edges, sample, side="right")


def independence_test(x, y, edges_x, edges_y):
    r = len(edges_x) + 1
    c = len(edges_y) + 1
    n = len(x)

    cells = class_indices(x, edges_x) * c + class_indices(y, edges_y)
    observed = np.bincount(cells, minlength=r * c)
    expected = n / (r * c)

    statistic = float(np.sum((observed - expected) ** 2 / expected))
    dof = r * c - 1

    return IndependenceTest(
        statistic=statistic,
        dof=dof,
        critical=float(stats.chi2.ppf(CONFIDENCE, dof)),
        p_value=float(stats.chi2.sf(statistic, dof)),
        min_expected=expected,
    )
