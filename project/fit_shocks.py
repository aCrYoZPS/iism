"""Hawkes model with random shocks, fitted two ways, for each game's weekly review counts.

Reads processed/weekly_series.csv, processed/fits_simple.csv and processed/fits_hawkes.csv, writes
processed/fits_shocks.csv.

Base: the Hawkes model with eta <= 0.5 ("hawkes50", see fit_hawkes.py):
    mu_t = B * (1 + t) ** (-alpha) + eta * s_t          (s_t = echo of the earlier weeks)
On top of it, each week a shock arrives with probability p; its size S is lognormal(m, sigma) and is
added to the expected count of that week. What happens after a shock is carried by the echo.
    week t is  NB(mu_t, r)  with probability 1 - p,   NB(mu_t + S, r)  with probability p.

Two ways to estimate the shock part:
  shock_rule   two stages. Weeks whose count exceeds the 99.5% quantile of NB(mu_t, r) by at least
               MIN_EXCESS reviews are flagged as shocks; p is the flagged share of weeks and (m, sigma)
               come from the log of the excess y_t - mu_t. Hawkes parameters are kept from hawkes50.
  shock_joint  all parameters, including p, m and sigma, are fitted together by maximum likelihood.
               Which weeks hold a shock is not observed, so it is summed out: each week's likelihood is
               the mixture above, with the lognormal size integrated by Gauss-Hermite quadrature.
With p -> 0 both reduce to hawkes50, so shock_joint can never have a lower likelihood than hawkes50.
Columns in the output are prefixed with the model name.

Usage:
    python fit_shocks.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom

import fit_decay
import fit_hawkes

SERIES = Path("processed/weekly_series.csv")
SIMPLE_FITS = Path("processed/fits_simple.csv")
HAWKES_FITS = Path("processed/fits_hawkes.csv")
OUT = Path("processed/fits_shocks.csv")
BASE = "hawkes50"
QUANTILE = 0.995
MIN_EXCESS = 5.0
P_BOUNDS = (1e-4, 0.3)
N_NODES = 20
NODES, WEIGHTS = np.polynomial.hermite.hermgauss(N_NODES)
LOG_WEIGHTS = np.log(WEIGHTS / np.sqrt(np.pi))
N_PARAMS = 8  # B, alpha, eta, rho, r, p, m, sigma


def nb_logpmf(y, mu, r):
    mu = np.maximum(mu, 1e-9)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def mixture_loglik(y: np.ndarray, mu: np.ndarray, r: float, p: float, m: float,
                   sigma: float) -> float:
    """Log-likelihood of the weekly counts with a per-week shock that is summed out."""
    sizes = np.exp(m + np.sqrt(2.0) * sigma * NODES)
    no_shock = nb_logpmf(y, mu, r)
    with_shock = logsumexp(nb_logpmf(y[:, None], mu[:, None] + sizes[None, :], r)
                           + LOG_WEIGHTS[None, :], axis=1)
    return float(np.sum(np.logaddexp(np.log1p(-p) + no_shock, np.log(p) + with_shock)))


def stats_row(name: str, params: dict, ll: float, n: int, converged: bool, n_shocks) -> dict:
    row = {f"{name}_{key}": value for key, value in params.items()}
    row.update({f"{name}_loglik": ll, f"{name}_aic": 2 * N_PARAMS - 2 * ll,
                f"{name}_bic": N_PARAMS * np.log(n) - 2 * ll, f"{name}_converged": converged,
                f"{name}_n_shocks": n_shocks})
    return row


def fit_rule(y: np.ndarray, base: dict) -> dict:
    mu = fit_hawkes.hawkes_mean(base["B"], base["alpha"], base["eta"], base["rho"], y)
    r = base["r"]
    threshold = nbinom.ppf(QUANTILE, r, r / (r + np.maximum(mu, 1e-9)))
    flagged = (y > threshold) & (y - mu >= MIN_EXCESS)
    n_shocks = int(flagged.sum())
    if n_shocks == 0:
        p, m, sigma = P_BOUNDS[0], 0.0, 1.0
    else:
        p = float(np.clip(n_shocks / len(y), *P_BOUNDS))
        log_excess = np.log(y[flagged] - mu[flagged])
        m = float(log_excess.mean())
        sigma = float(max(log_excess.std(ddof=1), 0.1)) if n_shocks > 1 else 1.0
    ll = mixture_loglik(y, mu, r, p, m, sigma)
    params = dict(B=base["B"], alpha=base["alpha"], eta=base["eta"], rho=base["rho"], r=r,
                  p=p, m=m, sigma=sigma)
    return stats_row("shock_rule", params, ll, len(y), base["converged"], n_shocks)


def fit_joint(y: np.ndarray, base: dict, rule: dict) -> dict:
    def objective(theta):
        B, alpha, eta, rho = np.exp(theta[0]), np.exp(theta[1]), theta[2], theta[3]
        mu = fit_hawkes.hawkes_mean(B, alpha, eta, rho, y)
        value = -mixture_loglik(y, mu, np.exp(theta[4]), theta[5], theta[6], np.exp(theta[7]))
        return value if np.isfinite(value) else 1e12

    bounds = [(-5, 20), (-8, 3), (0.0, fit_hawkes.VARIANTS[BASE]), fit_hawkes.RHO_BOUNDS,
              fit_decay.LOG_R_BOUNDS, P_BOUNDS, (-1, 12), (-2.5, 2.0)]
    core = [np.log(base["B"]), np.log(base["alpha"]), min(base["eta"], 0.45), base["rho"],
            np.log(base["r"])]
    typical = np.log(max(5.0, 3.0 * float(y.mean())))
    starts = [core + [0.001, typical, 0.0],            # almost no shocks: close to hawkes50
              core + [0.03, typical, 0.0],
              core + [0.1, typical - 1.0, 0.0]]
    if rule["shock_rule_n_shocks"] > 0:
        starts.append(core + [rule["shock_rule_p"], rule["shock_rule_m"],
                              np.log(rule["shock_rule_sigma"])])
    best = None
    for start in starts:
        res = minimize(objective, np.array(start), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    ll = -best.fun
    x = best.x
    params = dict(B=float(np.exp(x[0])), alpha=float(np.exp(x[1])), eta=float(x[2]),
                  rho=float(x[3]), r=float(np.exp(x[4])), p=float(x[5]), m=float(x[6]),
                  sigma=float(np.exp(x[7])))
    return stats_row("shock_joint", params, ll, len(y), bool(best.success and ll > -1e11),
                     float(x[5] * len(y)))  # expected number of shock weeks


def fit_one(item: tuple) -> dict:
    app_id, group, y, base = item
    y = y.astype(float)
    row = {"app_id": app_id, "group": group}
    rule = fit_rule(y, base)
    row.update(rule)
    row.update(fit_joint(y, base, rule))
    return row


def main() -> None:
    series = pd.read_csv(SERIES).sort_values(["app_id", "week"])
    fits = pd.read_csv(HAWKES_FITS).set_index("app_id")
    fits = fits[fits[f"{BASE}_converged"].astype(bool)]
    suffixes = ["B", "alpha", "eta", "rho", "r", "converged"]
    items = [(app_id, frame["group"].iloc[0], frame["n_reviews"].to_numpy(),
              {s: fits.at[app_id, f"{BASE}_{s}"] for s in suffixes})
             for app_id, frame in series.groupby("app_id") if app_id in fits.index]
    print(f"fitting shock models to {len(items):,} games")

    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(fit_one, items, chunksize=25))
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)

    merged = result.merge(fits.reset_index()[["app_id", f"{BASE}_loglik", f"{BASE}_aic"]],
                          on="app_id")
    simple = pd.read_csv(SIMPLE_FITS)[["app_id", "exp_aic", "power_aic", "lognormal_aic"]]
    merged = merged.merge(simple, on="app_id").merge(
        fits.reset_index()[["app_id", "hawkes_aic", "hawkes80_aic"]], on="app_id")
    aic_cols = ["exp_aic", "power_aic", "lognormal_aic", "hawkes_aic", "hawkes80_aic",
                f"{BASE}_aic", "shock_rule_aic", "shock_joint_aic"]
    best = merged[aic_cols].idxmin(axis=1).str.replace("_aic", "")
    print("best by AIC among all eight models:", best.value_counts().to_dict())
    # p cannot go below P_BOUNDS[0], which costs about n * p_min = 0.01 in log-likelihood, so a
    # strict comparison with the nested fit would flag harmless differences.
    below = (merged["shock_joint_loglik"] < merged[f"{BASE}_loglik"] - 0.05).sum()
    print(f"shock_joint not converged: {(~merged['shock_joint_converged']).sum()}, "
          f"below the nested {BASE} fit: {below}")
    for name in ("shock_rule", "shock_joint"):
        shocks = merged[f"{name}_n_shocks"]
        print(f"{name}: games with at least one shock {(shocks >= 0.5).sum():,}, "
              f"median shocks per game {shocks.median():.2f}, "
              f"p median {merged[f'{name}_p'].median():.4f}, "
              f"size m median {merged[f'{name}_m'].median():.2f}, "
              f"sigma median {merged[f'{name}_sigma'].median():.2f}")


if __name__ == "__main__":
    main()
