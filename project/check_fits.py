"""Check how well the fitted curves describe real games, mainly by simulation.

Reads processed/weekly_series.csv and processed/fits_simple.csv. Writes:
  processed/sim_check.csv   per game x model x statistic: observed value and its percentile
                            among N_SIMS simulated series (plus lag-1 residual autocorrelation)
  figures/*.png             example games: data, three fitted curves, simulation band of the best model

Simulation: for each game and model, draw N_SIMS series from the negative binomial with the
fitted mean curve and dispersion, treating the fitted parameters as exact (so the check is a bit
optimistic). A model describes a statistic well if the observed value is typical among the draws.

Usage:
    python check_fits.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_decay
import fit_hawkes

SERIES = Path("processed/weekly_series.csv")
FITS = Path("processed/fits_simple.csv")
HAWKES_FITS = Path("processed/fits_hawkes.csv")
SHOCK_FITS = Path("processed/fits_shocks.csv")
OUT = Path("processed/sim_check.csv")
FIG_DIR = Path("figures")
N_SIMS = 200
EARLY_WEEKS = 8
HAWKES_NAMES = list(fit_hawkes.VARIANTS)
SHOCK_NAMES = ["shock_rule", "shock_joint"]
RECURSIVE_NAMES = HAWKES_NAMES + SHOCK_NAMES  # models where a week depends on the earlier weeks
MODEL_NAMES = list(fit_decay.MODELS) + RECURSIVE_NAMES
PLOT_NAMES = list(fit_decay.MODELS) + ["hawkes"]
STAT_NAMES = ["total", "peak_week", "early_share", "t50"]
MEANS = {"exp": fit_decay.mean_exp, "power": fit_decay.mean_power,
         "lognormal": fit_decay.mean_lognormal}
MAX_MEAN = 1e8  # guards the sampler against runaway values in explosive Hawkes draws


def mean_curve(name: str, row, t: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """Expected weekly count from the natural-scale parameters stored in the fits file.

    For the Hawkes and shock models this is the one-step-ahead mean given the observed earlier
    weeks (y), without the shock term.
    """
    if name in RECURSIVE_NAMES:
        return fit_hawkes.hawkes_mean(row[f"{name}_B"], row[f"{name}_alpha"], row[f"{name}_eta"],
                                      row[f"{name}_rho"], y)
    if name == "exp":
        theta = [np.log(row["exp_A"]), np.log(row["exp_lam"])]
    elif name == "power":
        theta = [np.log(row["power_A"]), np.log(row["power_alpha"])]
    else:
        theta = [np.log(row["lognormal_A"]), row["lognormal_mu"], np.log(row["lognormal_sigma"])]
    return MEANS[name](theta, t)


def simulate(mu: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    mu = np.clip(mu, 1e-9, MAX_MEAN)
    return rng.negative_binomial(r, r / (r + mu), size=(n, len(mu)))


def simulate_hawkes(row, name: str, n_weeks: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Free-running draws: each week is generated from the weeks simulated before it.

    Models with shock parameters (name_p, name_m, name_sigma) also add, with probability p each
    week, a lognormal shock to the expected count.
    """
    rho, eta, r = row[f"{name}_rho"], row[f"{name}_eta"], row[f"{name}_r"]
    t = np.arange(n_weeks, dtype=float)
    background = row[f"{name}_B"] * (1.0 + t) ** (-row[f"{name}_alpha"])
    has_shocks = f"{name}_p" in row.index
    y = np.zeros((n, n_weeks), dtype=np.int64)
    echo = np.zeros(n)
    for week in range(n_weeks):
        if week > 0:
            echo = rho * echo + (1.0 - rho) * y[:, week - 1]
        mu = background[week] + eta * echo
        if has_shocks:
            arrives = rng.random(n) < row[f"{name}_p"]
            mu = mu + arrives * rng.lognormal(row[f"{name}_m"], row[f"{name}_sigma"], size=n)
        mu = np.clip(mu, 1e-9, MAX_MEAN)
        y[:, week] = rng.negative_binomial(r, r / (r + mu))
    return y


def statistics(y: np.ndarray) -> dict:
    """Summary statistics for each row of y (series x weeks)."""
    y = np.atleast_2d(y).astype(float)
    total = y.sum(axis=1)
    windows = np.apply_along_axis(
        lambda v: np.convolve(v, np.ones(fit_decay.SMOOTH), mode="valid"), 1, y)
    safe = np.where(total > 0, total, np.nan)
    cum = np.cumsum(y, axis=1)
    return {
        "total": total,
        "peak_week": windows.argmax(axis=1).astype(float),
        "early_share": y[:, :EARLY_WEEKS].sum(axis=1) / safe,
        "t50": (cum >= 0.5 * total[:, None]).argmax(axis=1).astype(float),
    }


def percentile_of(observed: float, draws: np.ndarray) -> float:
    """Mid-rank percentile of the observed value among simulated values (ties count half)."""
    draws = draws[~np.isnan(draws)]
    if np.isnan(observed) or len(draws) == 0:
        return np.nan
    return float((np.sum(draws < observed) + 0.5 * np.sum(draws == observed)) / len(draws))


def lag1_autocorr(resid: np.ndarray) -> float:
    if resid.std() == 0:
        return np.nan
    return float(np.corrcoef(resid[:-1], resid[1:])[0, 1])


def check_game(item: tuple) -> list:
    app_id, group, y, fit_row = item
    t = np.arange(len(y), dtype=float)
    rng = np.random.default_rng(int(app_id))
    observed = {k: v[0] for k, v in statistics(y).items()}
    rows = []
    for name in MODEL_NAMES:
        if not fit_row[f"{name}_converged"]:
            continue
        mu = mean_curve(name, fit_row, t, y)
        r = fit_row[f"{name}_r"]
        draws = (simulate_hawkes(fit_row, name, len(y), N_SIMS, rng) if name in RECURSIVE_NAMES
                 else simulate(mu, r, N_SIMS, rng))
        sims = statistics(draws)
        resid = (y - mu) / np.sqrt(np.maximum(mu, 1e-9) + np.maximum(mu, 1e-9) ** 2 / r)
        autocorr = lag1_autocorr(resid)
        for stat in STAT_NAMES:
            rows.append(dict(app_id=app_id, group=group, model=name, stat=stat,
                             observed=observed[stat],
                             percentile=percentile_of(observed[stat], sims[stat]),
                             width=interval_width(stat, sims[stat]),
                             crps=crps_for(stat, observed[stat], sims[stat]),
                             resid_lag1=autocorr))
    return rows


def crps_ensemble(draws: np.ndarray, observed: float) -> float:
    """CRPS of an ensemble forecast, using the fair estimator for a finite ensemble.

    mean|X - y| - sum_{i,j}|X_i - X_j| / (2 m (m - 1)); smaller is better.
    """
    draws = np.sort(draws[~np.isnan(draws)])
    m = len(draws)
    if m < 2 or np.isnan(observed):
        return np.nan
    # sum over pairs i<j of |x_i - x_j| for sorted x, then double it for the full double sum
    pair_sum = np.sum((2.0 * np.arange(m) - m + 1.0) * draws)
    return float(np.mean(np.abs(draws - observed)) - 2.0 * pair_sum / (2.0 * m * (m - 1)))


def crps_for(stat: str, observed: float, draws: np.ndarray) -> float:
    """CRPS on the scale used for comparison: log(1 + x) for the review total, raw units otherwise."""
    if stat == "total":
        return crps_ensemble(np.log1p(draws), np.log1p(observed))
    return crps_ensemble(draws, observed)


def interval_width(stat: str, draws: np.ndarray) -> float:
    """Width of the central 95% of simulated values.

    Week-valued statistics (peak_week, t50) are in weeks; total and early_share are divided by the
    simulated median so that games of very different size are comparable.
    """
    draws = draws[~np.isnan(draws)]
    if len(draws) == 0:
        return np.nan
    low, mid, high = np.percentile(draws, [2.5, 50, 97.5])
    if stat in ("peak_week", "t50"):
        return float(high - low)
    return float((high - low) / mid) if mid > 0 else np.nan


def summarise(result: pd.DataFrame, fits: pd.DataFrame) -> None:
    result = result.merge(fits[["app_id", "peak_week"]].rename(columns={"peak_week": "obs_peak"}),
                          on="app_id")
    result["outside95"] = (result["percentile"] < 0.025) | (result["percentile"] > 0.975)
    result["peak_type"] = np.where(result["obs_peak"] > 26, "late peak", "early peak")
    print("\nshare of games where the observed value is outside the central 95% of simulations")
    print("(about 5% would be expected if the model were exactly right)")
    print(result.pivot_table(index="model", columns="stat", values="outside95",
                             aggfunc="mean").round(3).to_string())
    for peak_type, part in result.groupby("peak_type"):
        print(f"\n{peak_type} games:")
        print(part.pivot_table(index="model", columns="stat", values="outside95",
                               aggfunc="mean").round(3).to_string())
    print("\nmedian width of the simulated central 95% interval (narrower = sharper)")
    print("(total, early_share: relative to the simulated median; peak_week, t50: weeks)")
    print(result.pivot_table(index="model", columns="stat", values="width",
                             aggfunc="median").round(2).to_string())
    print("\nmean CRPS per model (smaller is better; total is on the log(1+x) scale,")
    print("early_share is a share, peak_week and t50 are in weeks)")
    print(result.pivot_table(index="model", columns="stat", values="crps",
                             aggfunc="mean").round(3).to_string())
    crps_differences(result)
    print("\nmedian lag-1 autocorrelation of residuals per model (0 = no leftover pattern):")
    print(result.drop_duplicates(["app_id", "model"]).groupby("model")["resid_lag1"]
          .median().round(3).to_string())


def crps_differences(result: pd.DataFrame, n_boot: int = 2000) -> None:
    """Mean per-game CRPS difference (model - baseline) with a bootstrap CI over games."""
    wide = result.pivot_table(index=["app_id", "peak_type", "stat"], columns="model",
                              values="crps")
    rng = np.random.default_rng(0)
    pairs = [(v, "power") for v in RECURSIVE_NAMES] + [
        ("shock_rule", "hawkes50"), ("shock_joint", "hawkes50"), ("shock_joint", "shock_rule")]
    print("\nmean CRPS difference (model - baseline) per game (negative = model better),")
    print("95% bootstrap CI over games")
    rows = []
    for variant, baseline in pairs:
        pair = wide[[variant, baseline]].dropna()
        diff = pair[variant] - pair[baseline]
        for (peak_type, stat), part in diff.groupby(level=["peak_type", "stat"]):
            d = part.to_numpy()
            boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(model=variant, baseline=baseline, peak_type=peak_type, stat=stat,
                             games=len(d), mean_diff=d.mean(), ci_low=low, ci_high=high))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def plot_examples(series: pd.DataFrame, fits: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    pool = fits[(fits["status"] == "fitted") & (fits["n_weeks"] == 104) & (fits["n_reviews"] >= 500)]
    chosen = pd.concat([
        pool[pool["peak_week"] <= EARLY_WEEKS].sample(6, random_state=0),
        pool[pool["peak_week"] > 26].sample(6, random_state=0)])
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
        y = series.loc[series["app_id"] == row["app_id"], "n_reviews"].to_numpy()
        t = np.arange(len(y), dtype=float)
        ax.plot(t, y, ".", color="0.4", markersize=4, label="reviews per week")
        for name in PLOT_NAMES:
            label = "hawkes (1-step mean)" if name == "hawkes" else name
            ax.plot(t, mean_curve(name, row, t, y), label=label, linestyle=":" if name == "hawkes" else "-")
        best = row["best_aic"]
        draws = simulate(mean_curve(best, row, t, y), row[f"{best}_r"], N_SIMS, rng)
        low, high = np.percentile(draws, [5, 95], axis=0)
        ax.fill_between(t, low, high, alpha=0.15, label=f"90% band ({best})")
        ax.set_yscale("symlog")
        ax.set_title(f"app {int(row['app_id'])}, peak week {int(row['peak_week'])}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("weeks since first review")
    fig.supylabel("reviews per week (symlog scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_fits.png", dpi=120)
    plt.close(fig)


def main() -> None:
    series = pd.read_csv(SERIES).sort_values(["app_id", "week"])
    fits = (pd.read_csv(FITS)
            .merge(pd.read_csv(HAWKES_FITS).drop(columns="group"), on="app_id", how="left")
            .merge(pd.read_csv(SHOCK_FITS).drop(columns="group"), on="app_id", how="left"))
    fitted = fits[fits["status"] == "fitted"].set_index("app_id", drop=False)
    for name in RECURSIVE_NAMES:
        fitted[f"{name}_converged"] = fitted[f"{name}_converged"].eq(True)
    items = [(app_id, frame["group"].iloc[0], frame["n_reviews"].to_numpy(), fitted.loc[app_id])
             for app_id, frame in series.groupby("app_id") if app_id in fitted.index]
    print(f"simulating {N_SIMS} series x {len(MODEL_NAMES)} models for {len(items):,} games")

    with ProcessPoolExecutor() as pool:
        rows = [row for game_rows in pool.map(check_game, items, chunksize=25) for row in game_rows]
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    summarise(result, fitted.reset_index(drop=True))
    plot_examples(series, fitted.reset_index(drop=True))
    print(f"\nwrote {OUT} and {FIG_DIR / 'example_fits.png'}")


if __name__ == "__main__":
    main()
