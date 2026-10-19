"""Fit three curve shapes to each game's weekly review counts and compare them by AIC/BIC.

Reads processed/weekly_series.csv (from build_weekly_series.py) and writes
processed/fits_simple.csv with one row per game.

Models for the expected weekly count mu(t), t = weeks since first review:
  exp        A * exp(-lam * t)
  power      A * (1 + t) ** (-alpha)
  lognormal  A * exp(-(ln(1 + t) - mu)^2 / (2 * sigma^2))   (curve shape, not a probability density)
Counts are modelled as negative binomial with mean mu(t) and one dispersion parameter r per game
(variance = mu + mu^2 / r), fitted by maximum likelihood.

Usage:
    python fit_decay.py [--series processed/weekly_series.csv] [--out processed/fits_simple.csv]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MIN_WEEKS = 26  # games observed for fewer weeks are listed but not fitted
SMOOTH = 4  # window (weeks) for the peak location and for starting values
LOG_R_BOUNDS = (-5.0, 15.0)


def nb_loglik(y: np.ndarray, mu: np.ndarray, r: float) -> float:
    mu = np.maximum(mu, 1e-9)
    return float(np.sum(
        gammaln(y + r) - gammaln(r) - gammaln(y + 1)
        + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def _exp(x):
    return np.exp(np.clip(x, -50.0, 50.0))


# Each model: parameter names, mean function of (params, t), bounds on the transformed
# parameters the optimiser sees, and a function producing starting points.
def mean_exp(p, t):
    return _exp(p[0] - _exp(p[1]) * t)


def mean_power(p, t):
    return _exp(p[0] - _exp(p[1]) * np.log1p(t))


def mean_lognormal(p, t):
    return _exp(p[0] - (np.log1p(t) - p[1]) ** 2 / (2.0 * _exp(2.0 * p[2])))


def starts_exp(a0, peak_a0, peak_week):
    return [[np.log(a0), np.log(lam)] for lam in (0.02, 0.1, 0.5)]


def starts_power(a0, peak_a0, peak_week):
    return [[np.log(a0), np.log(alpha)] for alpha in (0.5, 1.0, 2.0)]


def starts_lognormal(a0, peak_a0, peak_week):
    mus = (np.log1p(peak_week), 0.0, np.log(27.0))
    return [[np.log(peak_a0), mu, np.log(sigma)] for mu in mus for sigma in (0.7, 2.0)]


MODELS = {
    "exp": dict(params=["A", "lam"], mean=mean_exp, starts=starts_exp,
                to_natural=lambda p: [np.exp(p[0]), np.exp(p[1])],
                bounds=[(-5, 20), (-8, 3)]),
    "power": dict(params=["A", "alpha"], mean=mean_power, starts=starts_power,
                  to_natural=lambda p: [np.exp(p[0]), np.exp(p[1])],
                  bounds=[(-5, 20), (-8, 3)]),
    "lognormal": dict(params=["A", "mu", "sigma"], mean=mean_lognormal, starts=starts_lognormal,
                      to_natural=lambda p: [np.exp(p[0]), p[1], np.exp(p[2])],
                      bounds=[(-5, 20), (-3, 8), (-3, 3)]),
}


def fit_model(spec: dict, y: np.ndarray, t: np.ndarray, a0: float, peak_a0: float,
              peak_week: int) -> dict:
    """Best of several starting points; returns natural-scale parameters and fit statistics."""
    def objective(theta):
        value = -nb_loglik(y, spec["mean"](theta[:-1], t), np.exp(theta[-1]))
        return value if np.isfinite(value) else 1e12

    bounds = spec["bounds"] + [LOG_R_BOUNDS]
    best = None
    for start in spec["starts"](a0, peak_a0, peak_week):
        res = minimize(objective, np.array(start + [np.log(2.0)]), method="L-BFGS-B",
                       bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    k = len(spec["params"]) + 1  # curve parameters plus dispersion
    ll = -best.fun
    out = dict(zip(spec["params"], spec["to_natural"](best.x[:-1])))
    out.update(r=float(np.exp(best.x[-1])), loglik=ll, aic=2 * k - 2 * ll,
               bic=k * np.log(len(y)) - 2 * ll, converged=bool(best.success and ll > -1e11))
    return out


def peak_week_of(y: np.ndarray) -> int:
    """Start of the 4-week window with the most reviews."""
    sums = np.convolve(y, np.ones(SMOOTH), mode="valid")
    return int(np.argmax(sums))


def fit_game(item: tuple) -> dict:
    app_id, group, y = item
    row = {"app_id": app_id, "group": group, "n_weeks": len(y), "n_reviews": int(y.sum())}
    if len(y) < MIN_WEEKS:
        row["status"] = "too_short"
        return row
    t = np.arange(len(y), dtype=float)
    smooth = np.convolve(y, np.ones(SMOOTH) / SMOOTH, mode="valid")
    peak_week = peak_week_of(y)
    a0 = max(float(y[:SMOOTH].mean()), 1.0)
    peak_a0 = max(float(smooth.max()), 1.0)
    row["peak_week"] = peak_week
    row["status"] = "fitted"
    for name, spec in MODELS.items():
        fit = fit_model(spec, y.astype(float), t, a0, peak_a0, peak_week)
        row.update({f"{name}_{key}": value for key, value in fit.items()})
    for criterion in ("aic", "bic"):
        scores = {n: row[f"{n}_{criterion}"] for n in MODELS if row[f"{n}_converged"]}
        row[f"best_{criterion}"] = min(scores, key=scores.get) if scores else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=Path("processed/weekly_series.csv"))
    parser.add_argument("--out", type=Path, default=Path("processed/fits_simple.csv"))
    args = parser.parse_args()

    series = pd.read_csv(args.series).sort_values(["app_id", "week"])
    items = [(app_id, frame["group"].iloc[0], frame["n_reviews"].to_numpy())
             for app_id, frame in series.groupby("app_id")]
    print(f"fitting {len(items):,} games with {len(MODELS)} models")

    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(fit_game, items, chunksize=50))

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    fitted = result[result["status"] == "fitted"]
    print(f"fitted: {len(fitted):,} | too short (<{MIN_WEEKS} weeks): {(result.status == 'too_short').sum():,}")
    for name in MODELS:
        print(f"{name}: not converged in {(~fitted[f'{name}_converged'].astype(bool)).sum():,} games")
    for criterion in ("aic", "bic"):
        print(f"best by {criterion}:", fitted[f"best_{criterion}"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
