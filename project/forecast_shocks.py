"""Random shocks inside the held-out forecast, against the blind conformal stretch.

The final model (power_c_reg) is calibrated by stretching its weekly intervals by one factor. Shocks are
the mechanistic alternative: sales and updates add a heavy right tail to some weeks. With 12 or 26
training weeks a game's own shock rate cannot be estimated, so the shock rule is a population
quantity, taken from the other folds:
  1. every game is fitted over the full window with the power law plus a floor fixed at its real
     plateau (mean of the plateau weeks); week t is flagged as a shock when the count exceeds the
     99.5% quantile of NB(mu_t, r) by at least MIN_EXCESS (the rule of fit_shocks.py);
  2. the shock probability per horizon bucket p_b = share of flagged game-weeks, the relative size
     z = (y - mu) / mu of the flagged weeks is lognormal(m, s).
Forecast draws: NB(mu_t * (1 + B_t * Z_t), r) with B_t ~ Bernoulli(p_b), Z_t ~ lognormal(m, s).
Models scored: power_c_reg, reg_conf, reg_shock, reg_shock_conf (shocks and then the conformal factor
re-estimated for them), template_knn, mix_shrink_knn (the best mixture so far), mix_shock_knn.
Diagnostic: on the held-out weeks, how often a corridor miss falls on a flagged week (the flags use the
game's own full series, so this is a description, not a forecast).

Reads processed/weekly_series.csv, games_groups.csv. Writes processed/full_power_fits.csv (per game:
full-window fit, used by game_findings.py), shock_flags.npz, forecast_shocks_scores.csv,
forecast_shocks_diag.csv, results/forecast_shocks_summary.txt.

Usage:
    python forecast_shocks.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

import fit_decay
import forecast_curves as fc
import forecast_improve as fi

DATA_DIR = Path("datasets/kaggle/antonkozyriev-game-recommendations-on-steam")
PROCESSED, RESULTS = fc.PROCESSED, fc.RESULTS
WINDOW, LS, N_SIMS, N_FOLDS, SEED = fc.WINDOW, fc.LS, fc.N_SIMS, fc.N_FOLDS, fc.SEED
QUANTILE, MIN_EXCESS = 0.995, 5.0
MODELS = ["power_c_reg", "reg_conf", "reg_shock", "reg_shock_conf", "template_knn", "mix_shrink_knn", "mix_shock_knn"]
DIAG_MODELS = ["power_c_reg", "reg_conf", "reg_shock"]


# ---------- full-window fit and shock flags ----------

def full_fit(item: tuple) -> dict:
    app_id, y = item
    y = y.astype(float)
    t = np.arange(WINDOW, dtype=float)
    smooth = np.convolve(y, np.ones(fit_decay.SMOOTH) / fit_decay.SMOOTH, mode="valid")
    a0, peak_a0 = max(float(y[:fit_decay.SMOOTH].mean()), 1.0), max(float(smooth.max()), 1.0)
    fit = fit_decay.fit_model(fit_decay.MODELS["power"], y, t, a0, peak_a0, fit_decay.peak_week_of(y))
    A0, alpha0 = (fit["A"], fit["alpha"]) if fit["converged"] else (a0, 1.0)
    C = fc.actual_plateau(y)
    f = fc.fit_power_floor(y, t, A0, alpha0, "pop", C, 1.0)
    mu = fc.power_floor_mean(f["log_A"], f["log_alpha"], C, t)
    r = f["r"]
    threshold = nbinom.ppf(QUANTILE, r, r / (r + mu))
    flag = (y > threshold) & (y - mu >= MIN_EXCESS)
    z = (y - mu) / np.maximum(mu, 1e-9)
    return dict(app_id=app_id, log_A=f["log_A"], alpha=float(np.exp(f["log_alpha"])), C=C, r=r, ok=f["ok"],
                S=float(y.sum()), S12=float(y[:12].sum()), n_shocks=int(flag.sum()), flag=flag, z=z)


def shock_rule(flags: np.ndarray, zs: np.ndarray, other: np.ndarray) -> dict:
    """Shock probability per horizon bucket and lognormal size, from the other-fold games."""
    p = np.zeros(WINDOW)
    for a, b in fc.BUCKETS.values():
        p[a:b] = flags[other, a:b].mean()
    p[:min(a for a, _ in fc.BUCKETS.values())] = flags[other, :12].mean()
    lz = np.log(zs[other][flags[other]])
    return dict(p=p, m=float(lz.mean()), s=float(lz.std()))


# ---------- per-game passes ----------

def shock_draws(mu: np.ndarray, r: float, rule: dict, L: int, rng: np.random.Generator) -> np.ndarray:
    H = len(mu)
    B = rng.random((N_SIMS, H)) < rule["p"][L:L + H][None, :]
    Z = np.exp(rule["m"] + rule["s"] * rng.standard_normal((N_SIMS, H)))
    mu_s = np.clip(mu[None, :] * (1.0 + B * Z), 1e-9, fc.MAX_MEAN)
    return rng.negative_binomial(r, r / (r + mu_s))


def pass1(item: tuple) -> dict:
    *base, rule = item
    out = fi.pass1(tuple(base))
    L = base[2]
    rng = np.random.default_rng(int(base[0]) * 10 + L + 7)
    draws = shock_draws(out["mu_reg"], out["reg"]["r"], rule, L, rng)
    out["q_shock"] = np.percentile(np.log1p(draws), [5, 50, 95], axis=0)
    return out


def pass2(item: tuple) -> tuple:
    app_id, y, L, p1, s_conf_reg, s_conf_shock, rule, tau_knn, r_template, flag = item
    rng = np.random.default_rng(int(app_id) * 100 + L + 7)
    S = p1["S"]
    sims = {}
    sims["power_c_reg"] = fc.nb_draws(p1["mu_reg"], p1["reg"]["r"], rng)
    sims["reg_conf"] = fi.conformal(sims["power_c_reg"], s_conf_reg)
    sims["reg_shock"] = shock_draws(p1["mu_reg"], p1["reg"]["r"], rule, L, rng)
    sims["reg_shock_conf"] = fi.conformal(sims["reg_shock"], s_conf_shock)
    sims["template_knn"] = fc.nb_draws(tau_knn * S, r_template, rng)
    shrink = fc.nb_draws(p1["mu_shrink"], p1["shrink"]["r"], rng)
    half = N_SIMS // 2
    sims["mix_shrink_knn"] = np.vstack([shrink[:half], sims["template_knn"][:half]])
    sims["mix_shock_knn"] = np.vstack([sims["reg_shock"][:half], sims["template_knn"][:half]])
    rows = [dict(app_id=app_id, L=L, **row) for row in fc.score_game(y, L, sims)]
    # diagnostic on the horizon weeks
    obs = y[L:].astype(float)
    obs_log = np.log1p(obs)
    fl = flag[L:]
    diag = dict(app_id=app_id, L=L, n_weeks=len(obs), n_flag=int(fl.sum()))
    for m in DIAG_MODELS:
        d = sims[m]
        lo, hi = np.percentile(d, [5, 95], axis=0)
        above, below = obs > hi, obs < lo
        crps = fc.crps_weekly(np.log1p(d), obs_log)
        diag.update({f"{m}_above": int(above.sum()), f"{m}_above_flag": int((above & fl).sum()),
                     f"{m}_below": int(below.sum()), f"{m}_below_flag": int((below & fl).sum()),
                     f"{m}_crps_flag": float(crps[fl].sum()), f"{m}_crps_noflag": float(crps[~fl].sum())})
    return rows, diag


# ---------- reporting ----------

def summarise(scores: pd.DataFrame, diag: pd.DataFrame, rules: dict, fits: pd.DataFrame, s_conf: dict) -> str:
    out = ["shocks inside the held-out forecast vs the conformal stretch", "",
           f"full-window fits: {len(fits)} games, flagged shock weeks {fits['n_shocks'].sum():,} "
           f"({fits['n_shocks'].sum() / (len(fits) * WINDOW):.3%} of game-weeks); games with no shock: "
           f"{(fits['n_shocks'] == 0).mean():.1%}; median shocks per game {fits['n_shocks'].median():.0f}", ""]
    for L in LS:
        rl = rules[L]
        out.append(f"L = {L}: shock rule (fold 0): p by bucket "
                   + ", ".join(f"{k} {rl[0]['p'][a]:.3f}" for k, (a, _) in fc.BUCKETS.items())
                   + f"; log relative size m {rl[0]['m']:.2f}, s {rl[0]['s']:.2f} "
                   f"(median multiplier x{1 + np.exp(rl[0]['m']):.1f}); "
                   f"conformal factors reg {np.round(s_conf[L]['reg'], 3).tolist()}, shock {np.round(s_conf[L]['shock'], 3).tolist()}")
        sc = scores[scores["L"] == L]
        crps = sc.pivot_table(index="model", columns="bucket", values="crps", aggfunc="mean").reindex(MODELS)
        cov = sc[sc["bucket"] != "total"].pivot_table(index="model", columns="bucket", values="covered90", aggfunc="mean").reindex(MODELS)
        out += ["  CRPS (log scale) by bucket and for the total:", crps.round(3).to_string(),
                "  coverage of the 90% interval:", cov.round(3).to_string()]
        wide = sc[sc["bucket"] == "total"].pivot_table(index="app_id", columns="model", values="crps")
        wk = sc[sc["bucket"] != "total"].groupby(["app_id", "model"])["crps"].mean().unstack("model")
        for a, b in (("reg_shock", "power_c_reg"), ("reg_shock", "reg_conf"), ("reg_shock_conf", "reg_conf"),
                     ("mix_shock_knn", "mix_shrink_knn")):
            mt, lt, ht = fc.bootstrap_diff((wide[a] - wide[b]).dropna().to_numpy())
            mw, lw, hw = fc.bootstrap_diff((wk[a] - wk[b]).dropna().to_numpy())
            out.append(f"  {a} - {b}: total {mt:+.3f} [{lt:+.3f}; {ht:+.3f}], weekly {mw:+.4f} [{lw:+.4f}; {hw:+.4f}]")
        dg = diag[diag["L"] == L]
        share_flag = dg["n_flag"].sum() / dg["n_weeks"].sum()
        out.append(f"  diagnostic: flagged share of horizon weeks {share_flag:.3f}")
        for m in DIAG_MODELS:
            above, af = dg[f"{m}_above"].sum(), dg[f"{m}_above_flag"].sum()
            below, bf = dg[f"{m}_below"].sum(), dg[f"{m}_below_flag"].sum()
            cf = dg[f"{m}_crps_flag"].sum() / max(dg["n_flag"].sum(), 1)
            cn = dg[f"{m}_crps_noflag"].sum() / max((dg["n_weeks"] - dg["n_flag"]).sum(), 1)
            out.append(f"    {m}: misses above {above / dg['n_weeks'].sum():.3f} of weeks, of them on flagged weeks "
                       f"{af / max(above, 1):.3f}; misses below {below / dg['n_weeks'].sum():.3f} (flagged {bf / max(below, 1):.3f}); "
                       f"CRPS on flagged weeks {cf:.3f} vs other weeks {cn:.3f}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    series = pd.read_csv(PROCESSED / "weekly_series.csv")
    main_ids = set(pd.read_csv(PROCESSED / "games_groups.csv").query("group == 'main'")["app_id"])
    series = series[series["app_id"].isin(main_ids) & (series["week"] < WINDOW)]
    n_weeks = series.groupby("app_id")["week"].size()
    ids = np.sort(n_weeks[n_weeks == WINDOW].index.to_numpy())
    Y = series[series["app_id"].isin(ids)].pivot(index="app_id", columns="week", values="n_reviews").loc[ids].to_numpy(float)
    folds = np.random.default_rng(SEED).permutation(len(ids)) % N_FOLDS
    tags = fi.load_games(DATA_DIR).reindex(ids)["first_tag"].fillna("none").to_numpy()
    print(f"{len(ids):,} games", flush=True)

    with ProcessPoolExecutor() as pool:
        full = list(pool.map(full_fit, [(a, Y[i]) for i, a in enumerate(ids)], chunksize=50))
    flags = np.array([f["flag"] for f in full])
    zs = np.array([f["z"] for f in full])
    fits = pd.DataFrame([{k: v for k, v in f.items() if k not in ("flag", "z")} for f in full])
    fits.to_csv(PROCESSED / "full_power_fits.csv", index=False)
    np.savez_compressed(PROCESSED / "shock_flags.npz", ids=ids, flags=flags)
    print("full-window fits done", flush=True)

    all_rows, all_diag, rules, s_conf = [], [], {}, {}
    for L in LS:
        taus, rs, kappas, sigmas = fc.build_templates(Y, folds, L)
        kappa_reg, sigma_reg, _ = fi.plateau_regression(Y, folds, L, tags)
        tau_knn = fi.knn_templates(Y, folds, L)
        rules[L] = {f: shock_rule(flags, zs, folds != f) for f in range(N_FOLDS)}
        items = [(a, Y[i], L, kappas[folds[i]], sigmas[folds[i]], kappa_reg[i], sigma_reg[i], rules[L][folds[i]])
                 for i, a in enumerate(ids)]
        with ProcessPoolExecutor() as pool:
            p1 = list(pool.map(pass1, items, chunksize=25))
        s_reg = {f: fi.conformal_factor([p for p, fo in zip(p1, folds) if fo != f], "q_reg") for f in range(N_FOLDS)}
        s_shock = {f: fi.conformal_factor([p for p, fo in zip(p1, folds) if fo != f], "q_shock") for f in range(N_FOLDS)}
        s_conf[L] = dict(reg=list(s_reg.values()), shock=list(s_shock.values()))
        items = [(a, Y[i], L, p1[i], s_reg[folds[i]], s_shock[folds[i]], rules[L][folds[i]], tau_knn[i],
                  rs[folds[i]], flags[i]) for i, a in enumerate(ids)]
        with ProcessPoolExecutor() as pool:
            for rows, dg in pool.map(pass2, items, chunksize=25):
                all_rows.extend(rows)
                all_diag.append(dg)
        print(f"L = {L} done", flush=True)

    scores, diag = pd.DataFrame(all_rows), pd.DataFrame(all_diag)
    scores.to_csv(PROCESSED / "forecast_shocks_scores.csv", index=False)
    diag.to_csv(PROCESSED / "forecast_shocks_diag.csv", index=False)
    text = summarise(scores, diag, rules, fits, s_conf)
    print(text)
    (RESULTS / "forecast_shocks_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
