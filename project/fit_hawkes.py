"""Fit discrete-time Hawkes models with a decaying background to each game's weekly review counts.

Reads processed/weekly_series.csv and processed/fits_simple.csv, writes processed/fits_hawkes.csv.

Expected count in week t, given the earlier weeks:
    mu_t = B * (1 + t) ** (-alpha)  +  eta * s_t
    s_t  = sum_{k>=1} rho**(k-1) * (1 - rho) * y_{t-k}        (s_0 = 0)
The first term is a background that decays as a power law; the second is the echo of earlier weeks,
fading with factor rho per week. eta is the branching ratio (how many new reviews one review brings
on average). It is capped by an upper bound, and the variants differ only in that bound:
    hawkes    eta <= 0.99
    hawkes80  eta <= 0.80
    hawkes50  eta <= 0.50
With eta = 0 each model is exactly the power-law model from fit_decay.py, so its likelihood can
never be lower than that fit. Counts are negative binomial with mean mu_t and one dispersion
parameter r per game. Columns in the output are prefixed with the variant name.

Usage:
    python fit_hawkes.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

import fit_decay

SERIES = Path("processed/weekly_series.csv")
SIMPLE_FITS = Path("processed/fits_simple.csv")
OUT = Path("processed/fits_hawkes.csv")
VARIANTS = {"hawkes": 0.99, "hawkes80": 0.80, "hawkes50": 0.50}  # name -> upper bound of eta
RHO_BOUNDS = (0.01, 0.99)
N_PARAMS = 5  # B, alpha, eta, rho, r


def hawkes_mean(B: float, alpha: float, eta: float, rho: float, y: np.ndarray) -> np.ndarray:
    """One-step-ahead expected counts given the observed earlier weeks."""
    t = np.arange(len(y), dtype=float)
    echo = lfilter([0.0, 1.0 - rho], [1.0, -rho], y.astype(float))
    return B * (1.0 + t) ** (-alpha) + eta * echo


def fit_variant(name: str, eta_max: float, y: np.ndarray, power_A: float,
                power_alpha: float) -> dict:
    def objective(theta):
        B, alpha, eta, rho = np.exp(theta[0]), np.exp(theta[1]), theta[2], theta[3]
        value = -fit_decay.nb_loglik(y, hawkes_mean(B, alpha, eta, rho, y), np.exp(theta[4]))
        return value if np.isfinite(value) else 1e12

    bounds = [(-5, 20), (-8, 3), (0.0, eta_max), RHO_BOUNDS, fit_decay.LOG_R_BOUNDS]
    log_A, log_alpha = np.log(power_A), np.log(power_alpha)
    # eta = 0 reproduces the power-law fit, so one start sits (almost) there.
    starts = [[log_A, log_alpha, 0.01, 0.5], [log_A, log_alpha, 0.3, 0.5],
              [log_A, log_alpha, 0.6, 0.8], [log_A - 0.7, log_alpha, 0.5, 0.3]]
    best = None
    for start in starts:
        start[2] = min(start[2], 0.9 * eta_max)
        res = minimize(objective, np.array(start + [np.log(2.0)]), method="L-BFGS-B",
                       bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    ll = -best.fun
    return {f"{name}_B": float(np.exp(best.x[0])), f"{name}_alpha": float(np.exp(best.x[1])),
            f"{name}_eta": float(best.x[2]), f"{name}_rho": float(best.x[3]),
            f"{name}_r": float(np.exp(best.x[4])), f"{name}_loglik": ll,
            f"{name}_aic": 2 * N_PARAMS - 2 * ll,
            f"{name}_bic": N_PARAMS * np.log(len(y)) - 2 * ll,
            f"{name}_converged": bool(best.success and ll > -1e11)}


def fit_one(item: tuple) -> dict:
    app_id, group, y, power_A, power_alpha = item
    row = {"app_id": app_id, "group": group}
    for name, eta_max in VARIANTS.items():
        row.update(fit_variant(name, eta_max, y.astype(float), power_A, power_alpha))
    return row


def main() -> None:
    series = pd.read_csv(SERIES).sort_values(["app_id", "week"])
    simple = pd.read_csv(SIMPLE_FITS)
    simple = simple[(simple["status"] == "fitted") & simple["power_converged"].astype(bool)]
    simple = simple.set_index("app_id")
    items = [(app_id, frame["group"].iloc[0], frame["n_reviews"].to_numpy(),
              simple.at[app_id, "power_A"], simple.at[app_id, "power_alpha"])
             for app_id, frame in series.groupby("app_id") if app_id in simple.index]
    print(f"fitting {len(VARIANTS)} Hawkes variants to {len(items):,} games")

    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(fit_one, items, chunksize=50))
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)

    merged = result.merge(simple.reset_index()[
        ["app_id", "exp_aic", "power_aic", "lognormal_aic", "power_loglik", "peak_week"]],
        on="app_id")
    aic_cols = ["exp_aic", "power_aic", "lognormal_aic"] + [f"{n}_aic" for n in VARIANTS]
    best = merged[aic_cols].idxmin(axis=1).str.replace("_aic", "")
    print("best by AIC among all models:", best.value_counts().to_dict())
    for name, eta_max in VARIANTS.items():
        worse = (merged[f"{name}_loglik"] < merged["power_loglik"] - 1e-3).sum()
        print(f"{name}: not converged {(~merged[f'{name}_converged']).sum()}, "
              f"below nested power-law fit {worse}, "
              f"eta at upper bound {(merged[f'{name}_eta'] > eta_max - 1e-3).sum()}, "
              f"eta median {merged[f'{name}_eta'].median():.3f}, "
              f"rho median {merged[f'{name}_rho'].median():.3f}")


if __name__ == "__main__":
    main()
