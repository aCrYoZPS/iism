"""Six cheap improvements to the curve forecast, each as its own model so the gain is visible.

Reference models: power_c_shrink (power law + long-run level C shrunk to the population) and the
population template, both as in forecast_curves.py. New models:
  shrink_rpop        same mean, but one dispersion r for all games, estimated on the OTHER folds from
                     the forecast errors on their future weeks (so it includes the error of the curve)
  shrink_conf        same draws, interval stretched around the median by a factor s chosen on the other
                     folds so that the 90% interval covers 90% of their game-weeks (split conformal)
  shrink_laplace     parameters drawn from the normal approximation around the optimum (inverse
                     numerical Hessian of the penalised objective); each simulation uses its own draw
  power_c_reg        C = kappa_i * S with kappa_i from a regression log(plateau / S) ~ 1 + log S +
                     first-tag dummies, fitted on the other folds; C fixed
  power_c_regshrink  the same kappa_i as the centre of the shrinkage penalty
  template_knn       template built from the 50 other-fold games whose first L weeks are closest
                     (euclidean on log(1 + y)), ratio of sums as for the template
  mix_shrink_template, mix_shrink_knn   100 simulations of each member
Scores as in forecast_curves.py (log(1 + x) scale, CRPS by horizon bucket, coverage of the 90%
interval), with paired differences against both reference models.

Reads processed/weekly_series.csv, processed/fits_simple.csv and the Anton games files. Writes
processed/forecast_improve_scores.csv, results/forecast_improve_summary.txt,
figures/forecast_improve_crps.png.

Usage:
    python forecast_improve.py --data-dir datasets/kaggle/antonkozyriev-game-recommendations-on-steam

Sensitivity runs (sensitivity.py): the window is taken from the environment variable FI_WINDOW
(read at import so that worker processes see it too); --series, --groups, --min-reviews, --seed
and --tag select the games, the fold split and the output names.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

import fit_decay
import forecast_curves as fc
from cluster_types import load_games

PROCESSED, FIG_DIR, RESULTS = fc.PROCESSED, fc.FIG_DIR, fc.RESULTS
WINDOW = int(os.environ.get("FI_WINDOW", fc.WINDOW))
if WINDOW != fc.WINDOW:   # shorter window: plateau = last 26 weeks, last bucket cut at the window
    fc.WINDOW = WINDOW
    fc.PLATEAU_WEEKS = (WINDOW - 26, WINDOW)
    fc.BUCKETS = {f"{a}-{min(b, WINDOW) - 1}": (a, min(b, WINDOW)) for a, b in fc.BUCKETS.values() if a < WINDOW}
LS, N_SIMS, N_FOLDS, SEED = fc.LS, fc.N_SIMS, fc.N_FOLDS, fc.SEED
K_NEIGHBOURS = 50
HESS_STEP = 1e-3
CONF_MIN_HALF_WIDTH = 0.1  # log-scale floor so that a degenerate interval cannot dominate the factor
CONF_MAX_SCORE = 50.0
MODELS = ["power_c_shrink", "template", "shrink_rpop", "shrink_conf", "shrink_laplace",
          "power_c_reg", "reg_conf", "power_c_regshrink", "template_knn", "mix_shrink_template",
          "mix_shrink_knn", "mix_regconf_knn"]
BASELINES = ["power_c_shrink", "template"]
THETA_LO = np.array([-5.0, -8.0, fc.LOG_C_BOUNDS[0], fit_decay.LOG_R_BOUNDS[0]])
THETA_HI = np.array([20.0, 3.0, fc.LOG_C_BOUNDS[1], fit_decay.LOG_R_BOUNDS[1]])


# ---------- population-side estimates (always on the other folds) ----------

def plateau_regression(Y: np.ndarray, folds: np.ndarray, L: int, tags: np.ndarray) -> tuple:
    """Per game: predicted kappa and residual sd from log(plateau / S) ~ 1 + log S + tag dummies."""
    lo, hi = fc.PLATEAU_WEEKS
    S = np.clip(Y[:, :L].sum(axis=1), 1, None)
    target = np.log(((Y[:, lo:hi].sum(axis=1) + 1.0) / (hi - lo)) / S)
    levels = sorted(set(tags))
    X = np.column_stack([np.ones(len(Y)), np.log(S)] + [(tags == t).astype(float) for t in levels[1:]])
    kappa, sigma, slope = np.zeros(len(Y)), np.zeros(len(Y)), {}
    for f in range(N_FOLDS):
        other = folds != f
        beta, *_ = np.linalg.lstsq(X[other], target[other], rcond=None)
        resid = target[other] - X[other] @ beta
        kappa[~other] = np.exp(X[~other] @ beta)
        sigma[~other] = resid.std()
        slope[f] = float(beta[1])
    return kappa, sigma, slope


def knn_templates(Y: np.ndarray, folds: np.ndarray, L: int) -> np.ndarray:
    """Per game: future shares from its K nearest other-fold games (distance on log(1 + y) of the
    training weeks); ratio of sums as for the plain template."""
    feats = np.log1p(Y[:, :L])
    tau = np.zeros((len(Y), WINDOW - L))
    for f in range(N_FOLDS):
        other, mine = np.where(folds != f)[0], np.where(folds == f)[0]
        _, idx = NearestNeighbors(n_neighbors=K_NEIGHBOURS).fit(feats[other]).kneighbors(feats[mine])
        for row, neighbours in zip(mine, other[idx]):
            tau[row] = Y[neighbours, L:].sum(axis=0) / max(Y[neighbours, :L].sum(), 1.0)
    return tau


def pooled_r(items: list) -> float:
    """One dispersion for the forecast errors of other-fold games: ML over (future y, forecast mean)."""
    y = np.concatenate([it["y_future"] for it in items])
    mu = np.concatenate([it["mu_shrink"] for it in items])
    res = minimize_scalar(lambda lr: -fit_decay.nb_loglik(y, mu, np.exp(lr)),
                          bounds=fit_decay.LOG_R_BOUNDS, method="bounded")
    return float(np.exp(res.x))


def conformal_factor(items: list, key: str = "q") -> float:
    """Factor s such that stretching the 90% interval of the reference draws (quantiles stored under
    `key`) by s around the median covers 90% of the other-fold game-weeks."""
    scores = []
    for it in items:
        obs = np.log1p(it["y_future"])
        q05, q50, q95 = it[key]
        up = np.maximum(q95 - q50, CONF_MIN_HALF_WIDTH)
        down = np.maximum(q50 - q05, CONF_MIN_HALF_WIDTH)
        s = np.where(obs >= q50, (obs - q50) / up, (q50 - obs) / down)
        scores.append(np.minimum(s, CONF_MAX_SCORE))
    return float(np.quantile(np.concatenate(scores), 0.9))


# ---------- per-game fitting (pass 1) ----------

def shrink_objective(y: np.ndarray, t: np.ndarray, log_c_pop: float, sigma: float):
    def objective(theta):
        mu = fc.power_floor_mean(theta[0], theta[1], np.exp(theta[2]), t)
        value = (-fit_decay.nb_loglik(y, mu, np.exp(theta[3]))
                 + (theta[2] - log_c_pop) ** 2 / (2.0 * sigma ** 2))
        return value if np.isfinite(value) else 1e12
    return objective


def laplace_chol(objective, theta: np.ndarray):
    """Cholesky factor of the inverse numerical Hessian at theta, or None if not positive definite."""
    k, h = len(theta), HESS_STEP
    H = np.zeros((k, k))
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h
        for j in range(i, k):
            ej = np.zeros(k)
            ej[j] = h
            H[i, j] = H[j, i] = (objective(theta + ei + ej) - objective(theta + ei - ej)
                                 - objective(theta - ei + ej) + objective(theta - ei - ej)) / (4 * h * h)
    if not np.all(np.isfinite(H)):
        return None
    try:
        return np.linalg.cholesky(np.linalg.inv(H))
    except np.linalg.LinAlgError:
        return None


def pass1(item: tuple) -> dict:
    app_id, y, L, kappa_pop, sigma_pop, kappa_reg, sigma_reg = item
    y_train = y[:L].astype(float)
    t_train, t_future = np.arange(L, dtype=float), np.arange(L, WINDOW, dtype=float)
    S = max(float(y_train.sum()), 1.0)
    smooth = np.convolve(y_train, np.ones(fit_decay.SMOOTH) / fit_decay.SMOOTH, mode="valid")
    a0, peak_a0 = max(float(y_train[:fit_decay.SMOOTH].mean()), 1.0), max(float(smooth.max()), 1.0)
    fit = fit_decay.fit_model(fit_decay.MODELS["power"], y_train, t_train, a0, peak_a0,
                              fit_decay.peak_week_of(y_train))
    A0, alpha0 = (fit["A"], fit["alpha"]) if fit["converged"] else (a0, 1.0)

    out = {"app_id": app_id, "L": L, "S": S, "y_future": y[L:].astype(float)}
    shrink = fc.fit_power_floor(y_train, t_train, A0, alpha0, "shrink", kappa_pop * S, sigma_pop)
    out["shrink"] = shrink
    out["mu_shrink"] = fc.power_floor_mean(shrink["log_A"], shrink["log_alpha"], shrink["C"], t_future)
    theta_hat = np.array([shrink["log_A"], shrink["log_alpha"], np.log(max(shrink["C"], 1e-6)),
                          np.log(shrink["r"])])
    out["theta_hat"] = theta_hat
    out["chol"] = laplace_chol(shrink_objective(y_train, t_train, np.log(max(kappa_pop * S, 1e-3)),
                                                sigma_pop), theta_hat)
    for mode, name in (("pop", "reg"), ("shrink", "regshrink")):
        f = fc.fit_power_floor(y_train, t_train, A0, alpha0, mode, kappa_reg * S, sigma_reg)
        out[name] = f
        out[f"mu_{name}"] = fc.power_floor_mean(f["log_A"], f["log_alpha"], f["C"], t_future)
    rng = np.random.default_rng(int(app_id) * 10 + L)
    out["q"] = np.percentile(np.log1p(fc.nb_draws(out["mu_shrink"], shrink["r"], rng)), [5, 50, 95], axis=0)
    out["q_reg"] = np.percentile(np.log1p(fc.nb_draws(out["mu_reg"], out["reg"]["r"], rng)), [5, 50, 95], axis=0)
    return out


# ---------- simulation and scoring (pass 2) ----------

def laplace_draws(p1: dict, t_future: np.ndarray, rng: np.random.Generator) -> tuple:
    if p1["chol"] is None:
        return fc.nb_draws(p1["mu_shrink"], p1["shrink"]["r"], rng), False
    thetas = p1["theta_hat"] + rng.standard_normal((N_SIMS, 4)) @ p1["chol"].T
    thetas = np.clip(thetas, THETA_LO, THETA_HI)
    mu = (np.exp(np.clip(thetas[:, [0]] - np.exp(thetas[:, [1]]) * np.log1p(t_future)[None, :], -50, 50))
          + np.exp(thetas[:, [2]]))
    mu = np.clip(mu, 1e-9, fc.MAX_MEAN)
    r = np.exp(thetas[:, [3]])
    return rng.negative_binomial(r, r / (r + mu)), True


def conformal(draws: np.ndarray, s: float) -> np.ndarray:
    lg = np.log1p(draws)
    med = np.median(lg, axis=0, keepdims=True)
    return np.expm1(np.clip(med + s * (lg - med), 0.0, None))


def pass2(item: tuple) -> tuple:
    app_id, group, y, L, p1, r_pop, s_conf, s_conf_reg, tau, r_template, tau_knn = item
    rng = np.random.default_rng(int(app_id) * 100 + L)
    t_future = np.arange(L, WINDOW, dtype=float)
    S = p1["S"]
    sims = {}
    sims["power_c_shrink"] = fc.nb_draws(p1["mu_shrink"], p1["shrink"]["r"], rng)
    sims["template"] = fc.nb_draws(tau * S, r_template, rng)
    sims["shrink_rpop"] = fc.nb_draws(p1["mu_shrink"], r_pop, rng)
    sims["shrink_conf"] = conformal(sims["power_c_shrink"], s_conf)
    sims["shrink_laplace"], laplace_ok = laplace_draws(p1, t_future, rng)
    sims["power_c_reg"] = fc.nb_draws(p1["mu_reg"], p1["reg"]["r"], rng)
    sims["reg_conf"] = conformal(sims["power_c_reg"], s_conf_reg)
    sims["power_c_regshrink"] = fc.nb_draws(p1["mu_regshrink"], p1["regshrink"]["r"], rng)
    sims["template_knn"] = fc.nb_draws(tau_knn * S, r_template, rng)
    half = N_SIMS // 2
    sims["mix_shrink_template"] = np.vstack([sims["power_c_shrink"][:half], sims["template"][:half]])
    sims["mix_shrink_knn"] = np.vstack([sims["power_c_shrink"][:half], sims["template_knn"][:half]])
    sims["mix_regconf_knn"] = np.vstack([sims["reg_conf"][:half], sims["template_knn"][:half]])
    rows = [dict(app_id=app_id, group=group, L=L, **row) for row in fc.score_game(y, L, sims)]
    extra = dict(app_id=app_id, L=L, laplace_ok=laplace_ok, c_reg=p1["reg"]["C"],
                 c_regshrink=p1["regshrink"]["C"], c_shrink=p1["shrink"]["C"],
                 plateau_actual=fc.actual_plateau(y))
    return rows, extra


# ---------- reporting ----------

def summarise(scores: pd.DataFrame, peak: pd.Series, extras: pd.DataFrame, diag: dict) -> str:
    lines = []
    for L in LS:
        part = scores[scores["L"] == L]
        lines.append(f"\n=== trained on weeks 0..{L - 1} ===")
        d = diag[L]
        lines.append(f"pooled forecast dispersion r per fold: {np.round(d['r_pop'], 3).tolist()}")
        lines.append(f"conformal stretch factor per fold: {np.round(d['s_conf'], 3).tolist()}")
        lines.append(f"conformal stretch factor for power_c_reg per fold: {np.round(d['s_conf_reg'], 3).tolist()}")
        lines.append(f"regression slope of log(plateau/S) on log S per fold: {np.round(d['slope'], 3).tolist()}")
        ex = extras[extras["L"] == L]
        lines.append(f"Laplace approximation available for {ex['laplace_ok'].mean():.1%} of games")
        target = np.log(ex["plateau_actual"])
        for col in ("c_shrink", "c_reg", "c_regshrink"):
            rho, _ = spearmanr(np.log(np.clip(ex[col], 1e-3, None)), target)
            lines.append(f"  Spearman of {col} with the real level in weeks 78..103: {rho:.3f}")
        lines.append("\nmean CRPS (log scale, smaller is better) by horizon bucket")
        lines.append(part.pivot_table(index="model", columns="bucket", values="crps",
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
                    mean, lo, hi = fc.bootstrap_diff((grp[model] - grp[baseline]).to_numpy())
                    rows.append(dict(baseline=baseline, model=model, bucket=bucket, games=len(grp),
                                     mean_diff=mean, ci_low=lo, ci_high=hi))
        lines.append(pd.DataFrame(rows).round(3).to_string(index=False))
        late = part["app_id"].map(peak) > 26
        for name, mask in (("early-peak games", ~late), ("late-peak games", late)):
            lines.append(f"\nmean CRPS, {name}")
            lines.append(part[mask].pivot_table(index="model", columns="bucket", values="crps",
                                                aggfunc="mean").reindex(MODELS).round(3).to_string())
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
    fig.savefig(FIG_DIR / "forecast_improve_crps.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--series", type=Path, default=PROCESSED / "weekly_series.csv")
    parser.add_argument("--groups", type=Path, default=PROCESSED / "games_groups.csv")
    parser.add_argument("--min-reviews", type=int, default=0, help="extra filter on reviews in the window")
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the fold split")
    parser.add_argument("--tag", default="", help="suffix for output files (sensitivity runs)")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    # games: main group with a full window and at least --min-reviews reviews inside it
    series = pd.read_csv(args.series)
    main_ids = set(pd.read_csv(args.groups).query("group == 'main'")["app_id"])
    series = series[series["app_id"].isin(main_ids) & (series["week"] < WINDOW)]
    per_game = series.groupby("app_id").agg(n_weeks=("week", "size"), n_reviews=("n_reviews", "sum"))
    ids = np.sort(per_game[(per_game["n_weeks"] == WINDOW) & (per_game["n_reviews"] >= args.min_reviews)].index.to_numpy())
    Y = series[series["app_id"].isin(ids)].pivot(index="app_id", columns="week",
                                                  values="n_reviews").loc[ids].to_numpy(dtype=float)
    peak = pd.Series([fit_decay.peak_week_of(y) for y in Y], index=ids)
    folds = np.random.default_rng(args.seed).permutation(len(ids)) % N_FOLDS
    groups = dict(zip(ids, ["main"] * len(ids)))
    tags = load_games(args.data_dir).reindex(ids)["first_tag"].fillna("none").to_numpy()
    print(f"{len(ids):,} games, training windows {LS}, {N_SIMS} simulations each")

    all_rows, all_extras, diag = [], [], {}
    for L in LS:
        taus, rs, kappas, sigmas = fc.build_templates(Y, folds, L)
        kappa_reg, sigma_reg, slope = plateau_regression(Y, folds, L, tags)
        tau_knn = knn_templates(Y, folds, L)
        items = [(a, Y[i], L, kappas[folds[i]], sigmas[folds[i]], kappa_reg[i], sigma_reg[i])
                 for i, a in enumerate(ids)]
        with ProcessPoolExecutor() as pool:
            p1 = list(pool.map(pass1, items, chunksize=25))
        r_pop = {f: pooled_r([p for p, fo in zip(p1, folds) if fo != f]) for f in range(N_FOLDS)}
        s_conf = {f: conformal_factor([p for p, fo in zip(p1, folds) if fo != f]) for f in range(N_FOLDS)}
        s_conf_reg = {f: conformal_factor([p for p, fo in zip(p1, folds) if fo != f], "q_reg")
                      for f in range(N_FOLDS)}
        diag[L] = dict(r_pop=list(r_pop.values()), s_conf=list(s_conf.values()),
                       s_conf_reg=list(s_conf_reg.values()), slope=list(slope.values()))
        print(f"L = {L}: pass 1 done; r_pop {np.round(list(r_pop.values()), 3).tolist()}, "
              f"s_conf {np.round(list(s_conf.values()), 3).tolist()}, "
              f"s_conf_reg {np.round(list(s_conf_reg.values()), 3).tolist()}")
        items = [(a, groups[a], Y[i], L, p1[i], r_pop[folds[i]], s_conf[folds[i]], s_conf_reg[folds[i]],
                  taus[folds[i]], rs[folds[i]], tau_knn[i]) for i, a in enumerate(ids)]
        with ProcessPoolExecutor() as pool:
            for rows, extra in pool.map(pass2, items, chunksize=25):
                all_rows.extend(rows)
                all_extras.append(extra)
        print(f"L = {L}: pass 2 done")

    scores = pd.DataFrame(all_rows)
    scores.to_csv(PROCESSED / f"forecast_improve_scores{suffix}.csv", index=False)
    extras = pd.DataFrame(all_extras)
    text = summarise(scores, peak, extras, diag)
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"forecast_improve_summary{suffix}.txt").write_text(text, encoding="utf-8")
    if not args.tag:
        plot_crps(scores)


if __name__ == "__main__":
    main()
