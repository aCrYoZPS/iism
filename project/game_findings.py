"""The parameters of the final model as findings about games (descriptive, no causal claims).

Per game (main group, full 104-week window; full-window fits from forecast_shocks.py):
  alpha          speed of the power-law decay (power law + floor fixed at the real plateau)
  plateau_share  real plateau (mean of weeks 78..103) divided by the launch total S12 (weeks 0..11)
  early_share    share of all reviews in weeks 0..3
  pos_train      share of positive reviews in weeks 0..11 (observable at forecast time)
  pos_trend      positive share of weeks 4..11 minus weeks 0..3 (NaN unless both parts have >= 20 reviews)
Attributes from the catalogue: tags (all, top TOP_TAGS by frequency), price (final), release year.
The catalogue positive_ratio is as of 2022 and includes the horizon, so it is shown but never used as
a predictor.
Three outputs:
  1. medians of alpha, plateau_share (raw and adjusted for S12 by the fold-free regression on log S12)
     and early_share by tag (tag vs the rest), price band and release year, with bootstrap CIs of the
     median (resampling games);
  2. positive share: Spearman correlations of pos_train and pos_trend with alpha and the adjusted
     plateau, with bootstrap CIs, and medians by quartile of pos_train;
  3. the plateau regression extended with price, free-to-play flag, release year, pos_train, pos_trend
     and tag dummies, fitted on the other folds: out-of-fold residual sd against the base regression
     (log S + first tag), and the held-out forecast of power_c_reg with the extended kappa against the
     base one (paired CRPS, L = 12 and 26).
Writes processed/game_findings.csv, game_findings_groups.csv, game_findings_reg_scores.csv,
results/game_findings_summary.txt.

Usage:
    python game_findings.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import forecast_curves as fc
import forecast_improve as fi

DATA_DIR = Path("datasets/kaggle/antonkozyriev-game-recommendations-on-steam")
PROCESSED, RESULTS = fc.PROCESSED, fc.RESULTS
WINDOW, LS, N_SIMS, N_FOLDS, SEED = fc.WINDOW, fc.LS, fc.N_SIMS, fc.N_FOLDS, fc.SEED
TOP_TAGS, MIN_POS_REVIEWS, N_BOOT = 15, 20, 1000
PRICE_BANDS = [(-0.01, 0.0, "free"), (0.0, 4.99, "<= 4.99"), (4.99, 14.99, "5-14.99"), (14.99, 1e9, ">= 15")]
YEAR_BANDS = [(0, 2015, "<= 2015"), (2016, 2017, "2016-17"), (2018, 2019, "2018-19"), (2020, 2100, "2020+")]
MEASURES = ["alpha", "plateau_share", "plateau_adj", "early_share"]


def boot_median(x: np.ndarray, rng: np.random.Generator) -> tuple:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), *np.percentile(meds, [2.5, 97.5])


def boot_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    idx = rng.integers(0, len(x), size=(200, len(x)))
    boots = [spearmanr(x[i], y[i]).correlation for i in idx]
    return float(spearmanr(x, y).correlation), *np.percentile(boots, [2.5, 97.5]), len(x)


def load_catalogue(ids: np.ndarray) -> tuple:
    games = pd.read_csv(DATA_DIR / "games.csv", parse_dates=["date_release"]).set_index("app_id").reindex(ids)
    tags = {}
    with open(DATA_DIR / "games_metadata.json", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            tags[rec["app_id"]] = rec.get("tags") or []
    tag_lists = [tags.get(a, []) for a in ids]
    counts = pd.Series([t for lst in tag_lists for t in lst]).value_counts()
    top = list(counts.head(TOP_TAGS).index)
    tag_matrix = pd.DataFrame({t: [t in lst for lst in tag_lists] for t in top}, index=ids)
    return games, tag_matrix


def band(values: pd.Series, bands: list) -> pd.Series:
    out = pd.Series("none", index=values.index, dtype=object)
    for lo, hi, name in bands:
        out[(values > lo) & (values <= hi)] = name
    return out


def per_game_table(ids: np.ndarray, Y: np.ndarray, P: np.ndarray, fits: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    S, S12 = Y.sum(axis=1), np.clip(Y[:, :12].sum(axis=1), 1, None)
    n_a, n_b = Y[:, :4].sum(axis=1), Y[:, 4:12].sum(axis=1)
    pos_a, pos_b = P[:, :4].sum(axis=1) / np.clip(n_a, 1, None), P[:, 4:12].sum(axis=1) / np.clip(n_b, 1, None)
    tab = pd.DataFrame(index=ids)
    tab["S"], tab["S12"] = S, S12
    tab["alpha"] = fits.set_index("app_id").reindex(ids)["alpha"].to_numpy()
    tab["plateau_share"] = fc.actual_plateau_vector(Y) / S12 if hasattr(fc, "actual_plateau_vector") else \
        ((Y[:, fc.PLATEAU_WEEKS[0]:fc.PLATEAU_WEEKS[1]].sum(axis=1) + 1.0) / (fc.PLATEAU_WEEKS[1] - fc.PLATEAU_WEEKS[0])) / S12
    tab["early_share"] = Y[:, :4].sum(axis=1) / np.clip(S, 1, None)
    tab["pos_train"] = P[:, :12].sum(axis=1) / S12
    tab["pos_trend"] = np.where((n_a >= MIN_POS_REVIEWS) & (n_b >= MIN_POS_REVIEWS), pos_b - pos_a, np.nan)
    tab["price"] = games["price_final"].to_numpy()
    tab["price_band"] = band(tab["price"], PRICE_BANDS).to_numpy()
    tab["year"] = games["date_release"].dt.year.to_numpy()
    tab["year_band"] = band(tab["year"], YEAR_BANDS).to_numpy()
    tab["positive_ratio_2022"] = games["positive_ratio"].to_numpy()
    # plateau adjusted for size: residual of log(plateau_share) on log S12, as a ratio to the typical game
    x, y = np.log(S12), np.log(tab["plateau_share"].to_numpy())
    beta = np.polyfit(x, y, 1)
    tab["plateau_adj"] = np.exp(y - np.polyval(beta, x))
    tab.attrs["slope"] = float(beta[0])
    return tab


def group_table(tab: pd.DataFrame, tag_matrix: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []

    def add(kind, level, mask):
        row = dict(kind=kind, level=level, n=int(mask.sum()), median_S=float(np.median(tab.loc[mask, "S"])))
        for m in MEASURES:
            med, lo, hi = boot_median(tab.loc[mask, m].to_numpy(float), rng)
            row.update({m: med, f"{m}_lo": lo, f"{m}_hi": hi})
            if kind == "tag":
                rest = tab.loc[~mask, m].to_numpy(float)
                row[f"{m}_rest"] = float(np.nanmedian(rest))
        rows.append(row)

    add("all", "all games", np.ones(len(tab), bool))
    for t in tag_matrix.columns:
        add("tag", t, tag_matrix[t].to_numpy())
    for _, _, name in PRICE_BANDS:
        add("price", name, (tab["price_band"] == name).to_numpy())
    for _, _, name in YEAR_BANDS:
        add("year", name, (tab["year_band"] == name).to_numpy())
    q = pd.qcut(tab["pos_train"], 4, labels=["q1 (lowest)", "q2", "q3", "q4 (highest)"])
    for name in q.cat.categories:
        add("pos_train quartile", str(name), (q == name).to_numpy())
    return pd.DataFrame(rows)


# ---------- extended plateau regression and its forecast ----------

def design(tab: pd.DataFrame, tag_matrix: pd.DataFrame, first_tag: np.ndarray, extended: bool) -> np.ndarray:
    cols = [np.ones(len(tab)), np.log(tab["S12"].to_numpy(float))]
    levels = sorted(set(first_tag))
    cols += [(first_tag == t).astype(float) for t in levels[1:]]
    if extended:
        trend = tab["pos_trend"].to_numpy(float)
        cols += [np.log1p(tab["price"].to_numpy(float)), (tab["price"].to_numpy(float) == 0).astype(float),
                 tab["pos_train"].to_numpy(float), np.nan_to_num(trend), np.isnan(trend).astype(float),
                 (tab["year"].to_numpy(float) - 2018.0)]
        cols += [tag_matrix[t].to_numpy(float) for t in tag_matrix.columns]
    return np.column_stack(cols)


def oof_regression(X: np.ndarray, target: np.ndarray, folds: np.ndarray) -> tuple:
    kappa, sigma, resid_oof = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
    for f in range(N_FOLDS):
        other = folds != f
        beta, *_ = np.linalg.lstsq(X[other], target[other], rcond=None)
        kappa[~other] = np.exp(X[~other] @ beta)
        sigma[~other] = (target[other] - X[other] @ beta).std()
        resid_oof[~other] = target[~other] - X[~other] @ beta
    return kappa, sigma, resid_oof


def score_reg(item: tuple) -> list:
    app_id, y, L, p1 = item
    rng = np.random.default_rng(int(app_id) * 100 + L)
    draws = fc.nb_draws(p1["mu_reg"], p1["reg"]["r"], rng)
    return [dict(app_id=app_id, L=L, **row) for row in fc.score_game(y, L, {"power_c_reg": draws})]


def forecast_with(ids, Y, folds, L, kappas, sigmas, kappa_reg, sigma_reg, label) -> pd.DataFrame:
    items = [(a, Y[i], L, kappas[folds[i]], sigmas[folds[i]], kappa_reg[i], sigma_reg[i]) for i, a in enumerate(ids)]
    with ProcessPoolExecutor() as pool:
        p1 = list(pool.map(fi.pass1, items, chunksize=25))
        rows = [r for rr in pool.map(score_reg, [(a, Y[i], L, p1[i]) for i, a in enumerate(ids)], chunksize=25) for r in rr]
    c_reg = dict(zip(ids, [p["reg"]["C"] for p in p1]))
    df = pd.DataFrame(rows)
    return df.assign(variant=label, c_reg=df["app_id"].map(c_reg))


def summarise(tab, groups, corr, reg_info, reg_scores) -> str:
    out = ["the parameters of the final model as findings about games", "",
           f"{len(tab)} games; slope of log(plateau_share) on log S12: {tab.attrs['slope']:+.3f} "
           f"(plateau_adj = plateau_share relative to the typical game of the same launch size)", ""]
    g = groups.set_index(["kind", "level"])
    fmt = lambda r, m: f"{r[m]:.3f} [{r[f'{m}_lo']:.3f}; {r[f'{m}_hi']:.3f}]"
    for kind in ("all", "price", "year", "pos_train quartile", "tag"):
        out.append(f"by {kind}: n, median S | alpha | plateau_share | plateau_adj | early_share (medians, 95% bootstrap CI)")
        for level, r in g.loc[kind].iterrows():
            line = f"  {level:<22} {int(r['n']):>5} {r['median_S']:>8.0f} | " + " | ".join(fmt(r, m) for m in MEASURES)
            if kind == "tag":
                line += f"  (rest: {r['alpha_rest']:.3f} / {r['plateau_share_rest']:.3f} / {r['plateau_adj_rest']:.3f} / {r['early_share_rest']:.3f})"
            out.append(line)
        out.append("")
    out.append("positive share (Spearman, 95% bootstrap CI, n):")
    for name, (rho, lo, hi, n) in corr.items():
        out.append(f"  {name:<40} {rho:+.3f} [{lo:+.3f}; {hi:+.3f}]  n = {n}")
    out.append("")
    out.append("extended plateau regression (other folds), out-of-fold residual sd of log(plateau/S12):")
    for L, info in reg_info.items():
        out.append(f"  L = {L}: base {info['sd_base']:.3f} -> extended {info['sd_ext']:.3f}; "
                   f"largest standardised coefficients: {info['top_coefs']}")
    out.append("")
    out.append("held-out forecast of power_c_reg with the extended kappa against the base kappa:")
    for L in LS:
        sc = reg_scores[reg_scores["L"] == L]
        piv = sc.pivot_table(index="variant", columns="bucket", values="crps", aggfunc="mean")
        cov = sc[sc["bucket"] != "total"].pivot_table(index="variant", columns="bucket", values="covered90", aggfunc="mean")
        out += [f"  L = {L}: CRPS", piv.round(4).to_string(), "  coverage", cov.round(3).to_string()]
        wide = sc[sc["bucket"] == "total"].pivot_table(index="app_id", columns="variant", values="crps")
        wk = sc[sc["bucket"] != "total"].groupby(["app_id", "variant"])["crps"].mean().unstack("variant")
        mt, lt, ht = fc.bootstrap_diff((wide["extended"] - wide["base"]).dropna().to_numpy())
        mw, lw, hw = fc.bootstrap_diff((wk["extended"] - wk["base"]).dropna().to_numpy())
        out.append(f"  extended - base: total {mt:+.4f} [{lt:+.4f}; {ht:+.4f}], weekly {mw:+.4f} [{lw:+.4f}; {hw:+.4f}]")
    return "\n".join(out)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true", help="reuse the saved forecast scores")
    args = ap.parse_args()
    series = pd.read_csv(PROCESSED / "weekly_series.csv")
    main_ids = set(pd.read_csv(PROCESSED / "games_groups.csv").query("group == 'main'")["app_id"])
    series = series[series["app_id"].isin(main_ids) & (series["week"] < WINDOW)]
    n_weeks = series.groupby("app_id")["week"].size()
    ids = np.sort(n_weeks[n_weeks == WINDOW].index.to_numpy())
    sub = series[series["app_id"].isin(ids)]
    Y = sub.pivot(index="app_id", columns="week", values="n_reviews").loc[ids].to_numpy(float)
    P = sub.pivot(index="app_id", columns="week", values="n_positive").loc[ids].to_numpy(float)
    fits = pd.read_csv(PROCESSED / "full_power_fits.csv")
    games, tag_matrix = load_catalogue(ids)
    tab = per_game_table(ids, Y, P, fits, games)
    tab.join(tag_matrix.add_prefix("tag_")).rename_axis("app_id").to_csv(PROCESSED / "game_findings.csv")
    groups = group_table(tab, tag_matrix)
    groups.to_csv(PROCESSED / "game_findings_groups.csv", index=False)

    rng = np.random.default_rng(SEED)
    corr = {}
    for x in ("pos_train", "pos_trend"):
        for y in ("alpha", "plateau_adj", "early_share"):
            corr[f"{x} vs {y}"] = boot_spearman(tab[x].to_numpy(float), tab[y].to_numpy(float), rng)
    corr["pos_train vs positive_ratio_2022"] = boot_spearman(tab["pos_train"].to_numpy(float), tab["positive_ratio_2022"].to_numpy(float), rng)
    print("per-game table and groups done", flush=True)

    folds = np.random.default_rng(SEED).permutation(len(ids)) % N_FOLDS
    first_tag = fi.load_games(DATA_DIR).reindex(ids)["first_tag"].fillna("none").to_numpy()
    reg_info, all_scores = {}, []
    for L in LS:
        taus, rs, kappas, sigmas = fc.build_templates(Y, folds, L)
        lo, hi = fc.PLATEAU_WEEKS
        S_L = np.clip(Y[:, :L].sum(axis=1), 1, None)
        target = np.log(((Y[:, lo:hi].sum(axis=1) + 1.0) / (hi - lo)) / S_L)
        # training-window versions of the observable covariates
        tabL = tab.copy()
        tabL["S12"] = S_L
        tabL["pos_train"] = P[:, :L].sum(axis=1) / S_L
        na, nb = Y[:, :4].sum(axis=1), Y[:, 4:L].sum(axis=1)
        tabL["pos_trend"] = np.where((na >= MIN_POS_REVIEWS) & (nb >= MIN_POS_REVIEWS),
                                     P[:, 4:L].sum(axis=1) / np.clip(nb, 1, None) - P[:, :4].sum(axis=1) / np.clip(na, 1, None), np.nan)
        X_base, X_ext = design(tabL, tag_matrix, first_tag, False), design(tabL, tag_matrix, first_tag, True)
        k_base, s_base, r_base = oof_regression(X_base, target, folds)
        k_ext, s_ext, r_ext = oof_regression(X_ext, target, folds)
        beta, *_ = np.linalg.lstsq(X_ext, target, rcond=None)
        names = ["const", "log S"] + [f"first:{t}" for t in sorted(set(first_tag))[1:]] + \
                ["log price", "free", "pos_train", "pos_trend", "trend missing", "year"] + [f"tag:{t}" for t in tag_matrix.columns]
        std = beta * X_ext.std(axis=0)
        order = np.argsort(-np.abs(std))[:8]
        reg_info[L] = dict(sd_base=float(r_base.std()), sd_ext=float(r_ext.std()),
                           top_coefs=", ".join(f"{names[i]} {std[i]:+.3f}" for i in order if names[i] != "const"))
        print(f"L = {L}: residual sd base {r_base.std():.3f}, extended {r_ext.std():.3f}", flush=True)
        if args.summary_only:
            continue
        for label, kr, sr in (("base", k_base, s_base), ("extended", k_ext, s_ext)):
            all_scores.append(forecast_with(ids, Y, folds, L, kappas, sigmas, kr, sr, label))
            print(f"L = {L}: forecast {label} done", flush=True)
    if args.summary_only:
        reg_scores = pd.read_csv(PROCESSED / "game_findings_reg_scores.csv")
    else:
        reg_scores = pd.concat(all_scores)
        reg_scores.to_csv(PROCESSED / "game_findings_reg_scores.csv", index=False)
    text = summarise(tab, groups, corr, reg_info, reg_scores)
    print(text)
    (RESULTS / "game_findings_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
