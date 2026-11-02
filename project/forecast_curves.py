"""Forecast the whole weekly review curve from the first weeks and compare models on held-out weeks.

For each game with the full 104-week window, models are fitted only to the first L weeks
(L = 12 and 26) and asked for the distribution of every later week (200 simulated continuations).
Models:
  flat       constant level fitted on the last 8 training weeks (naive baseline)
  exp, power, lognormal   the curve shapes from fit_decay.py, fitted to the training weeks
  hawkes50   Hawkes model with eta <= 0.5, fitted to the training weeks; the forecast starts from the
             real training history and runs freely afterwards
  template   mean share of reviews per week over other games (5-fold cross-validation by game),
             scaled by the game's training total
  power_c_*  power law plus a long-run level C:  A (1 + t)^-alpha + C
      power_c_free    C fitted per game together with A, alpha, r
      power_c_pop     C = kappa * S, S = training total of the game, kappa = median over OTHER games
                      (same folds as the template) of plateau / S, plateau = mean weekly reviews in
                      weeks 78..103 (one review added to the window so that zero plateaus stay finite)
      power_c_shrink  C fitted per game with the penalty (log C - log(kappa S))^2 / (2 sigma^2),
                      sigma = standard deviation of log(plateau / S) over other games
  hawkes50_c_pop      Hawkes with eta <= 0.5 whose background is B (1 + t)^-alpha + kappa S
Counts are negative binomial with the model mean; dispersion is fitted per game (template: pooled).
Scores on the log(1 + x) scale: CRPS per week, absolute error of the median, coverage of the 90%
interval, and the CRPS of the total over the forecast window. Averages are taken within horizon
buckets (weeks 12-25, 26-51, 52-103). A diagnostic reports how well the fitted C tracks the real
later level.

Reads processed/weekly_series.csv, processed/fits_simple.csv. Writes processed/forecast_scores.csv,
processed/forecast_floor_fits.csv, results/forecast_summary.txt, figures/forecast_crps.png and
figures/forecast_examples.png.

Usage:
    python forecast_curves.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter
from scipy.stats import spearmanr

import fit_decay
import fit_hawkes

PROCESSED = Path("processed")
FIG_DIR = Path("figures")
RESULTS = Path("results")
WINDOW = 104
LS = [12, 26]
N_SIMS = 200
N_FOLDS = 5
FLAT_WEEKS = 8
PLATEAU_WEEKS = (78, 104)
BUCKETS = {"12-25": (12, 26), "26-51": (26, 52), "52-103": (52, 104)}
MODELS = ["flat", "exp", "power", "lognormal", "hawkes50", "template",
          "power_c_free", "power_c_pop", "power_c_shrink", "hawkes50_c_pop"]
PLOT_MODELS = ["flat", "template", "power", "hawkes50",
               "power_c_free", "power_c_pop", "power_c_shrink", "hawkes50_c_pop"]
BASELINES = ["template", "flat", "power"]
ETA_CAP = 0.5
LOG_C_BOUNDS = (-7.0, 20.0)
MAX_MEAN = 1e8
SEED = 0


def nb_draws(mu: np.ndarray, r: float, rng: np.random.Generator) -> np.ndarray:
    mu = np.clip(mu, 1e-9, MAX_MEAN)
    return rng.negative_binomial(r, r / (r + mu), size=(N_SIMS, len(mu)))


def crps_weekly(draws_log: np.ndarray, obs_log: np.ndarray) -> np.ndarray:
    """Fair ensemble CRPS for each column (week); draws_log is (simulations, weeks)."""
    n = draws_log.shape[0]
    ordered = np.sort(draws_log, axis=0)
    pair_sum = ((2.0 * np.arange(n) - n + 1.0)[:, None] * ordered).sum(axis=0)
    return np.abs(draws_log - obs_log).mean(axis=0) - pair_sum / (n * (n - 1))


def best_of(objective, starts: list, bounds: list):
    best = None
    for start in starts:
        res = minimize(objective, np.array(start, dtype=float), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best


def fit_flat(y_last: np.ndarray) -> tuple:
    def objective(theta):
        value = -fit_decay.nb_loglik(y_last, np.full(len(y_last), np.exp(theta[0])), np.exp(theta[1]))
        return value if np.isfinite(value) else 1e12
    start = [np.log(max(float(y_last.mean()), 0.5)), np.log(2.0)]
    res = minimize(objective, start, method="L-BFGS-B",
                   bounds=[(-5, 20), fit_decay.LOG_R_BOUNDS])
    return float(np.exp(res.x[0])), float(np.exp(res.x[1])), bool(res.success)


def curve_theta(name: str, p: dict) -> list:
    """Optimiser-scale parameters from the natural-scale ones returned by fit_decay.fit_model."""
    if name == "exp":
        return [np.log(p["A"]), np.log(p["lam"])]
    if name == "power":
        return [np.log(p["A"]), np.log(p["alpha"])]
    return [np.log(p["A"]), p["mu"], np.log(p["sigma"])]


def power_floor_mean(log_A: float, log_alpha: float, C: float, t: np.ndarray) -> np.ndarray:
    return fit_decay._exp(log_A - fit_decay._exp(log_alpha) * np.log1p(t)) + C


def fit_power_floor(y: np.ndarray, t: np.ndarray, A0: float, alpha0: float, mode: str,
                    c_pop: float, sigma: float) -> dict:
    """Power law plus long-run level C; mode is 'free', 'pop' (C fixed) or 'shrink' (penalised)."""
    log_c_pop = np.log(max(c_pop, 1e-3))
    log_A0, log_alpha0 = np.log(max(A0, 0.5)), np.log(max(alpha0, 1e-3))
    if mode == "pop":
        def objective(theta):
            mu = power_floor_mean(theta[0], theta[1], c_pop, t)
            value = -fit_decay.nb_loglik(y, mu, np.exp(theta[2]))
            return value if np.isfinite(value) else 1e12
        bounds = [(-5, 20), (-8, 3), fit_decay.LOG_R_BOUNDS]
        starts = [[log_A0, log_alpha0, np.log(2.0)], [log_A0, log_alpha0 + 0.7, np.log(2.0)]]
        best = best_of(objective, starts, bounds)
        return dict(log_A=best.x[0], log_alpha=best.x[1], C=c_pop, r=float(np.exp(best.x[2])),
                    ok=bool(best.success))

    def objective(theta):
        mu = power_floor_mean(theta[0], theta[1], np.exp(theta[2]), t)
        value = -fit_decay.nb_loglik(y, mu, np.exp(theta[3]))
        if mode == "shrink":
            value += (theta[2] - log_c_pop) ** 2 / (2.0 * sigma ** 2)
        return value if np.isfinite(value) else 1e12
    bounds = [(-5, 20), (-8, 3), LOG_C_BOUNDS, fit_decay.LOG_R_BOUNDS]
    starts = [[log_A0, log_alpha0, log_c_pop, np.log(2.0)],
              [log_A0, log_alpha0, log_c_pop - 2.0, np.log(2.0)],
              [log_A0, log_alpha0 + 0.7, log_c_pop + 1.0, np.log(2.0)]]
    best = best_of(objective, starts, bounds)
    return dict(log_A=best.x[0], log_alpha=best.x[1], C=float(np.exp(best.x[2])),
                r=float(np.exp(best.x[3])), ok=bool(best.success))


def fit_hawkes_floor(y: np.ndarray, B0: float, alpha0: float, floor: float) -> dict:
    """hawkes50 whose background is B (1 + t)^-alpha + floor; mirrors fit_hawkes.fit_variant."""
    def objective(theta):
        B, alpha, eta, rho = np.exp(theta[0]), np.exp(theta[1]), theta[2], theta[3]
        mu = fit_hawkes.hawkes_mean(B, alpha, eta, rho, y) + floor
        value = -fit_decay.nb_loglik(y, mu, np.exp(theta[4]))
        return value if np.isfinite(value) else 1e12
    bounds = [(-5, 20), (-8, 3), (0.0, ETA_CAP), fit_hawkes.RHO_BOUNDS, fit_decay.LOG_R_BOUNDS]
    log_B, log_alpha = np.log(max(B0, 0.5)), np.log(max(alpha0, 1e-3))
    starts = [[log_B, log_alpha, 0.01, 0.5, np.log(2.0)], [log_B, log_alpha, 0.3, 0.5, np.log(2.0)],
              [log_B, log_alpha, 0.45, 0.8, np.log(2.0)], [log_B - 0.7, log_alpha, 0.3, 0.3, np.log(2.0)]]
    best = best_of(objective, starts, bounds)
    x = best.x
    return dict(hawkes50_B=float(np.exp(x[0])), hawkes50_alpha=float(np.exp(x[1])),
                hawkes50_eta=float(x[2]), hawkes50_rho=float(x[3]), hawkes50_r=float(np.exp(x[4])),
                floor=floor, ok=bool(best.success))


def hawkes_forecast(p: dict, y_train: np.ndarray, horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Free-running continuation from the real training history."""
    rho, eta, r = p["hawkes50_rho"], p["hawkes50_eta"], p["hawkes50_r"]
    L = len(y_train)
    echo_series = lfilter([0.0, 1.0 - rho], [1.0, -rho], y_train.astype(float))
    echo = np.full(N_SIMS, rho * echo_series[-1] + (1.0 - rho) * y_train[-1])
    t = np.arange(L, L + horizon, dtype=float)
    background = p["hawkes50_B"] * (1.0 + t) ** (-p["hawkes50_alpha"]) + p.get("floor", 0.0)
    out = np.zeros((N_SIMS, horizon), dtype=np.int64)
    for i in range(horizon):
        mu = np.clip(background[i] + eta * echo, 1e-9, MAX_MEAN)
        out[:, i] = rng.negative_binomial(r, r / (r + mu))
        echo = rho * echo + (1.0 - rho) * out[:, i]
    return out


def forecasts_for_game(y: np.ndarray, L: int, tau: np.ndarray, r_template: float, kappa: float,
                       sigma: float, rng: np.random.Generator) -> tuple:
    """Simulated continuations (simulations x future weeks) per model, and the fitted levels C."""
    y_train = y[:L].astype(float)
    t_train = np.arange(L, dtype=float)
    t_future = np.arange(L, WINDOW, dtype=float)
    horizon = WINDOW - L
    S = max(float(y_train.sum()), 1.0)
    c_pop = kappa * S
    out, levels = {}, {"c_pop": c_pop}

    A, r, ok = fit_flat(y_train[-FLAT_WEEKS:])
    if ok:
        out["flat"] = nb_draws(np.full(horizon, A), r, rng)

    smooth = np.convolve(y_train, np.ones(fit_decay.SMOOTH) / fit_decay.SMOOTH, mode="valid")
    a0, peak_a0 = max(float(y_train[:fit_decay.SMOOTH].mean()), 1.0), max(float(smooth.max()), 1.0)
    peak_week = fit_decay.peak_week_of(y_train)
    A0, alpha0 = a0, 1.0
    for name in ("exp", "power", "lognormal"):
        spec = fit_decay.MODELS[name]
        fit = fit_decay.fit_model(spec, y_train, t_train, a0, peak_a0, peak_week)
        if not fit["converged"]:
            continue
        if name == "power":
            A0, alpha0 = fit["A"], fit["alpha"]
        mu = spec["mean"](curve_theta(name, fit), t_future)
        out[name] = nb_draws(mu, fit["r"], rng)

    h = fit_hawkes.fit_variant("hawkes50", ETA_CAP, y_train, A0, alpha0)
    if h["hawkes50_converged"]:
        out["hawkes50"] = hawkes_forecast(h, y_train, horizon, rng)

    out["template"] = nb_draws(tau * S, r_template, rng)

    for mode in ("free", "pop", "shrink"):
        fit = fit_power_floor(y_train, t_train, A0, alpha0, mode, c_pop, sigma)
        levels[f"c_{mode}"] = fit["C"]
        if fit["ok"]:
            mu = power_floor_mean(fit["log_A"], fit["log_alpha"], fit["C"], t_future)
            out[f"power_c_{mode}"] = nb_draws(mu, fit["r"], rng)

    hf = fit_hawkes_floor(y_train, A0, alpha0, c_pop)
    if hf["ok"]:
        out["hawkes50_c_pop"] = hawkes_forecast(hf, y_train, horizon, rng)
    return out, levels


def score_game(y: np.ndarray, L: int, sims: dict) -> list:
    """Scores per model and bucket (plus the total over the forecast window)."""
    obs = y[L:].astype(float)
    obs_log = np.log1p(obs)
    rows = []
    for model, draws in sims.items():
        draws_log = np.log1p(draws)
        crps = crps_weekly(draws_log, obs_log)
        mae = np.abs(np.median(draws_log, axis=0) - obs_log)
        low, high = np.percentile(draws, [5, 95], axis=0)
        covered = ((obs >= low) & (obs <= high)).astype(float)
        for bucket, (start, stop) in BUCKETS.items():
            lo, hi = max(start, L) - L, stop - L
            if hi <= lo:
                continue
            rows.append(dict(model=model, bucket=bucket, crps=crps[lo:hi].mean(),
                             mae=mae[lo:hi].mean(), covered90=covered[lo:hi].mean()))
        total_log = np.log1p(draws.sum(axis=1))[:, None]
        total_crps = crps_weekly(total_log, np.array([np.log1p(obs.sum())]))[0]
        rows.append(dict(model=model, bucket="total", crps=total_crps,
                         mae=abs(np.median(total_log) - np.log1p(obs.sum())), covered90=np.nan))
    return rows


def actual_plateau(y: np.ndarray) -> float:
    lo, hi = PLATEAU_WEEKS
    return (float(y[lo:hi].sum()) + 1.0) / (hi - lo)


def run_game(item: tuple) -> tuple:
    app_id, group, y, L, tau, r_template, kappa, sigma = item
    rng = np.random.default_rng(int(app_id) * 10 + L)
    sims, levels = forecasts_for_game(y, L, tau, r_template, kappa, sigma, rng)
    rows = [dict(app_id=app_id, group=group, L=L, **row) for row in score_game(y, L, sims)]
    levels.update(app_id=app_id, L=L, plateau_actual=actual_plateau(y), S=float(y[:L].sum()))
    return rows, levels


def build_templates(Y: np.ndarray, folds: np.ndarray, L: int) -> tuple:
    """Per fold, estimated on the other folds only: template shares (ratio of sums), pooled template
    dispersion, kappa (median plateau / training total) and sigma (sd of its log)."""
    taus, rs, kappas, sigmas = {}, {}, {}, {}
    lo, hi = PLATEAU_WEEKS
    for f in range(N_FOLDS):
        other = folds != f
        totals = np.clip(Y[other, :L].sum(axis=1), 1, None)
        tau = Y[other, L:].sum(axis=0) / totals.sum()
        mu = tau[None, :] * totals[:, None]
        sample = np.random.default_rng(SEED).choice(other.sum(), size=min(1500, other.sum()), replace=False)
        y_s, mu_s = Y[other][sample, L:], mu[sample]
        res = minimize_scalar(lambda lr: -fit_decay.nb_loglik(y_s, mu_s, np.exp(lr)),
                              bounds=fit_decay.LOG_R_BOUNDS, method="bounded")
        ratio = ((Y[other, lo:hi].sum(axis=1) + 1.0) / (hi - lo)) / totals
        taus[f], rs[f] = tau, float(np.exp(res.x))
        kappas[f], sigmas[f] = float(np.median(ratio)), float(np.std(np.log(ratio)))
    return taus, rs, kappas, sigmas


def bootstrap_diff(diff: np.ndarray, n_boot: int = 2000) -> tuple:
    rng = np.random.default_rng(SEED)
    boot = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return diff.mean(), *np.percentile(boot, [2.5, 97.5])


def summarise(scores: pd.DataFrame, peak: pd.Series, levels: pd.DataFrame) -> str:
    lines = []
    for L in LS:
        part = scores[scores["L"] == L]
        lines.append(f"\n=== trained on weeks 0..{L - 1} ===")
        lines.append("mean CRPS (log scale, smaller is better) by horizon bucket")
        lines.append(part.pivot_table(index="model", columns="bucket", values="crps",
                                      aggfunc="mean").reindex(MODELS).round(3).to_string())
        lines.append("\nmean absolute error of the median forecast (log scale)")
        lines.append(part.pivot_table(index="model", columns="bucket", values="mae",
                                      aggfunc="mean").reindex(MODELS).round(3).to_string())
        lines.append("\ncoverage of the 90% interval (0.90 would be ideal)")
        lines.append(part[part["bucket"] != "total"].pivot_table(
            index="model", columns="bucket", values="covered90", aggfunc="mean")
            .reindex(MODELS).round(3).to_string())
        lines.append("\nCRPS difference (model - baseline), mean per game with 95% bootstrap CI")
        wide = part.pivot_table(index=["app_id", "bucket"], columns="model", values="crps")
        rows = []
        for baseline in BASELINES:
            for model in MODELS:
                if model == baseline:
                    continue
                pair = wide[[model, baseline]].dropna()
                for bucket, grp in pair.groupby(level="bucket"):
                    mean, lo, hi = bootstrap_diff((grp[model] - grp[baseline]).to_numpy())
                    rows.append(dict(baseline=baseline, model=model, bucket=bucket, games=len(grp),
                                     mean_diff=mean, ci_low=lo, ci_high=hi))
        lines.append(pd.DataFrame(rows).round(3).to_string(index=False))
        late = part["app_id"].map(peak) > 26
        for name, mask in (("early-peak games", ~late), ("late-peak games", late)):
            lines.append(f"\nmean CRPS, {name}")
            lines.append(part[mask].pivot_table(index="model", columns="bucket", values="crps",
                                                aggfunc="mean").reindex(MODELS).round(3).to_string())
        lv = levels[levels["L"] == L]
        lines.append("\nfitted long-run level C vs the real level in weeks 78..103 (Spearman over games)")
        target = np.log(lv["plateau_actual"])
        for col in ("c_free", "c_shrink", "c_pop"):
            rho, _ = spearmanr(np.log(np.clip(lv[col], 1e-3, None)), target)
            lines.append(f"  {col:9s} rho = {rho:.3f}   median C = {lv[col].median():.2f}   "
                         f"median real level = {lv['plateau_actual'].median():.2f}")
        at_bound = ((lv["c_free"] <= np.exp(LOG_C_BOUNDS[0]) * 1.01)).mean()
        lines.append(f"  c_free at the lower bound (C ~ 0): {at_bound:.1%} of games")
    return "\n".join(lines)


def plot_crps(scores: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(LS), figsize=(15, 5), sharey=True)
    for ax, L in zip(axes, LS):
        table = scores[(scores["L"] == L) & (scores["bucket"] != "total")].pivot_table(
            index="bucket", columns="model", values="crps", aggfunc="mean")[MODELS]
        table.plot.bar(ax=ax, rot=0, width=0.85)
        ax.set_title(f"trained on weeks 0-{L - 1}")
        ax.set_xlabel("forecast weeks")
        ax.set_ylabel("mean CRPS, log scale (smaller is better)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_crps.png", dpi=120)
    plt.close(fig)


def plot_examples(Y: np.ndarray, ids: np.ndarray, templates: tuple, folds: np.ndarray,
                  peak: pd.Series) -> None:
    L = LS[0]
    taus, rs, kappas, sigmas = templates
    rng = np.random.default_rng(SEED)
    early = [i for i, a in enumerate(ids) if peak[a] <= 8 and Y[i].sum() >= 500]
    late = [i for i, a in enumerate(ids) if peak[a] > 26 and Y[i].sum() >= 500]
    chosen = list(rng.choice(early, 6, replace=False)) + list(rng.choice(late, 6, replace=False))
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    t_future = np.arange(L, WINDOW)
    for ax, i in zip(axes.flat, chosen):
        f = folds[i]
        sims, _ = forecasts_for_game(Y[i], L, taus[f], rs[f], kappas[f], sigmas[f],
                                     np.random.default_rng(int(ids[i])))
        ax.plot(np.arange(WINDOW), Y[i], ".", color="0.4", markersize=4, label="reviews per week")
        for name in PLOT_MODELS:
            if name in sims:
                ax.plot(t_future, np.median(sims[name], axis=0), label=name, linewidth=1.2,
                        linestyle="--" if "_c_" in name else "-")
        band = "power_c_pop" if "power_c_pop" in sims else "template"
        low, high = np.percentile(sims[band], [5, 95], axis=0)
        ax.fill_between(t_future, low, high, alpha=0.12, label=f"90% band ({band})")
        ax.axvline(L - 0.5, color="k", linestyle="--", linewidth=0.8)
        ax.set_yscale("symlog")
        ax.set_title(f"app {int(ids[i])}, peak week {int(peak[ids[i]])}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("weeks since first review (dashed line: end of training window; dashed curves: models with a floor)")
    fig.supylabel("reviews per week (symlog scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_examples.png", dpi=120)
    plt.close(fig)


def main() -> None:
    series = pd.read_csv(PROCESSED / "weekly_series.csv")
    simple = pd.read_csv(PROCESSED / "fits_simple.csv")
    keep = simple[(simple["group"] == "main") & (simple["n_weeks"] == WINDOW)]
    peak = keep.set_index("app_id")["peak_week"]
    ids = np.sort(keep["app_id"].to_numpy())
    wide = series[series["app_id"].isin(ids)].pivot(index="app_id", columns="week",
                                                    values="n_reviews").loc[ids]
    Y = wide.to_numpy(dtype=float)
    folds = np.random.default_rng(SEED).permutation(len(ids)) % N_FOLDS
    groups = dict(zip(ids, keep.set_index("app_id").loc[ids, "group"]))
    print(f"{len(ids):,} games, training windows {LS}, {N_SIMS} simulations each")

    all_rows, all_levels, templates = [], [], {}
    for L in LS:
        templates[L] = build_templates(Y, folds, L)
        taus, rs, kappas, sigmas = templates[L]
        print(f"L = {L}: kappa per fold {np.round(list(kappas.values()), 4).tolist()}, "
              f"sigma per fold {np.round(list(sigmas.values()), 2).tolist()}")
        items = [(a, groups[a], Y[i], L, taus[folds[i]], rs[folds[i]], kappas[folds[i]], sigmas[folds[i]])
                 for i, a in enumerate(ids)]
        with ProcessPoolExecutor() as pool:
            for rows, levels in pool.map(run_game, items, chunksize=25):
                all_rows.extend(rows)
                all_levels.append(levels)
        print(f"L = {L}: done")
    scores = pd.DataFrame(all_rows)
    scores.to_csv(PROCESSED / "forecast_scores.csv", index=False)
    levels = pd.DataFrame(all_levels)
    levels.to_csv(PROCESSED / "forecast_floor_fits.csv", index=False)

    fits_per_model = scores[scores["bucket"] == "total"].groupby(["L", "model"]).size().unstack()[MODELS]
    print("\ngames with a successful forecast per model (out of", len(ids), ")")
    print(fits_per_model.to_string())

    text = summarise(scores, peak, levels)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "forecast_summary.txt").write_text(text, encoding="utf-8")
    plot_crps(scores)
    plot_examples(Y, ids, templates[LS[0]], folds, peak)


if __name__ == "__main__":
    main()
