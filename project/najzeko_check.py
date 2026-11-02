"""External check of the forecast on the najzeko Steam-reviews dataset (~300 popular games, to Jan 2021).

Stage 1 (--build, slow: the file is 8 GB): count reviews per game and day, write
processed/najzeko_daily.csv (app_id, date, n, n_pos, app_name).

Stage 2 (default): weekly series with the same conventions as build_weekly_series.py (week 0 = week
of the first review, main group = first review 0..30 days after the catalogue release date, full
104-week window), then
  a) agreement of the weekly series of the games present in both datasets (Spearman of the weekly
     counts, Pearson of the log changes, ratio of the totals);
  b) the held-out forecast (L = 12 and 26) on the najzeko series, with every population quantity
     (template shares, kNN templates, plateau regression, pooled dispersion, conformal factor)
     estimated on the Anton games only, excluding the games that najzeko also holds. Nothing is
     refitted on najzeko. Scores are compared with the Anton games that have >= 500 reviews in the
     window, because najzeko holds only popular games.
Writes processed/najzeko_weekly.csv, najzeko_games.csv, najzeko_agreement.csv, najzeko_scores.csv,
results/najzeko_summary.txt.

Usage:
    python najzeko_check.py --build
    python najzeko_check.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import forecast_curves as fc
import forecast_improve as fi

RAW = Path("datasets/kaggle/najzeko-steam-reviews-2021/steam_reviews.csv")
DATA_DIR = Path("datasets/kaggle/antonkozyriev-game-recommendations-on-steam")
PROCESSED, RESULTS = fc.PROCESSED, fc.RESULTS
WINDOW, LS, N_SIMS, N_FOLDS, SEED = fc.WINDOW, fc.LS, fc.N_SIMS, fc.N_FOLDS, fc.SEED
LATE_DAYS, MIN_REVIEWS, CHUNK = 30, 200, 2_000_000
MODELS = ["template", "template_knn", "power_c_shrink", "power_c_reg", "reg_conf", "mix_shrink_knn"]


# ---------- stage 1 ----------

def build_daily() -> pd.DataFrame:
    daily, names = None, {}
    for i, chunk in enumerate(pd.read_csv(RAW, usecols=["app_id", "app_name", "timestamp_created", "recommended"],
                                          chunksize=CHUNK)):
        chunk = chunk.dropna(subset=["app_id", "timestamp_created"])
        day = pd.to_datetime(chunk["timestamp_created"].astype("int64"), unit="s").dt.normalize()
        pos = chunk["recommended"].astype(str).str.lower().eq("true").astype(int)
        g = (pd.DataFrame({"app_id": chunk["app_id"].astype("int64"), "date": day, "pos": pos})
             .groupby(["app_id", "date"]).agg(n=("pos", "size"), n_pos=("pos", "sum")))
        daily = g if daily is None else daily.add(g, fill_value=0)
        for a, nm in chunk.groupby("app_id")["app_name"].first().items():
            names.setdefault(int(a), nm)
        print(f"chunk {i + 1}: {len(chunk):,} rows, {daily.index.get_level_values(0).nunique()} games", flush=True)
    daily = daily.reset_index()
    daily["app_name"] = daily["app_id"].map(names)
    daily.to_csv(PROCESSED / "najzeko_daily.csv", index=False)
    return daily


# ---------- stage 2 ----------

def weekly_from_daily(daily: pd.DataFrame) -> tuple:
    games = pd.read_csv(DATA_DIR / "games.csv", parse_dates=["date_release"]).set_index("app_id")
    daily["date"] = pd.to_datetime(daily["date"])
    record_end = daily["date"].max()
    first = daily.groupby("app_id")["date"].min()
    tab = pd.DataFrame({"first_review": first, "n_total": daily.groupby("app_id")["n"].sum(),
                        "title": daily.groupby("app_id")["app_name"].first()})
    tab["date_release"] = games["date_release"].reindex(tab.index)
    tab["gap_days"] = (tab["first_review"] - tab["date_release"]).dt.days
    tab["observed_weeks"] = ((record_end - tab["first_review"]).dt.days // 7 + 1).clip(upper=WINDOW)
    reason = np.where(tab["date_release"].isna(), "no catalogue entry",
             np.where(tab["gap_days"] < 0, "first review before release",
             np.where(tab["gap_days"] > LATE_DAYS, f"first review > {LATE_DAYS} days after release",
             np.where(tab["n_total"] < MIN_REVIEWS, f"< {MIN_REVIEWS} reviews",
             np.where(tab["observed_weeks"] < WINDOW, "window not complete", "main")))))
    tab["group"] = np.where(reason == "main", "main", "excluded")
    tab["reason"] = reason
    anton_groups = pd.read_csv(PROCESSED / "games_groups.csv").set_index("app_id")["group"]
    tab["anton_group"] = anton_groups.reindex(tab.index).fillna("absent")
    keep = tab[tab["group"] == "main"]
    d = daily[daily["app_id"].isin(keep.index)].copy()
    d["week"] = (d["date"] - d["app_id"].map(first)).dt.days // 7
    d = d[d["week"] < WINDOW]
    weekly = (d.groupby(["app_id", "week"]).agg(n_reviews=("n", "sum"), n_positive=("n_pos", "sum"))
              .reindex(pd.MultiIndex.from_product([keep.index, range(WINDOW)], names=["app_id", "week"]), fill_value=0)
              .reset_index())
    return tab.reset_index(), weekly


def agreement(weekly: pd.DataFrame) -> pd.DataFrame:
    anton = pd.read_csv(PROCESSED / "weekly_series.csv")
    anton = anton[anton["week"] < WINDOW]
    rows = []
    for a, g in weekly.groupby("app_id"):
        o = anton[anton["app_id"] == a].set_index("week")["n_reviews"].reindex(range(WINDOW))
        if o.isna().any():
            continue
        x, y = g.set_index("week")["n_reviews"].to_numpy(float), o.to_numpy(float)
        rows.append(dict(app_id=a, total_najzeko=x.sum(), total_anton=y.sum(), ratio=y.sum() / max(x.sum(), 1),
                         rho_level=spearmanr(x, y).correlation,
                         r_diff=pearsonr(np.diff(np.log1p(x)), np.diff(np.log1p(y)))[0],
                         first_week_shift=int(np.argmax(y > 0)) - int(np.argmax(x > 0))))
    return pd.DataFrame(rows)


def external_forecast(weekly: pd.DataFrame, tab: pd.DataFrame) -> tuple:
    """najzeko games are fold 0; Anton games (minus the ones najzeko holds) fill folds 1..4, so every
    population quantity used for fold 0 comes from Anton only."""
    ids_n = np.sort(weekly["app_id"].unique())
    Y_n = weekly.pivot(index="app_id", columns="week", values="n_reviews").loc[ids_n].to_numpy(float)
    series = pd.read_csv(PROCESSED / "weekly_series.csv")
    main_ids = set(pd.read_csv(PROCESSED / "games_groups.csv").query("group == 'main'")["app_id"])
    series = series[series["app_id"].isin(main_ids) & (series["week"] < WINDOW)]
    n_weeks = series.groupby("app_id")["week"].size()
    ids_a = np.sort(n_weeks[(n_weeks == WINDOW)].index.to_numpy())
    ids_a = ids_a[~np.isin(ids_a, set(tab["app_id"]))]          # drop every game najzeko holds
    Y_a = series[series["app_id"].isin(ids_a)].pivot(index="app_id", columns="week", values="n_reviews").loc[ids_a].to_numpy(float)
    Y = np.vstack([Y_n, Y_a])
    ids = np.concatenate([ids_n, ids_a])
    folds = np.concatenate([np.zeros(len(ids_n), int), 1 + np.random.default_rng(SEED).permutation(len(ids_a)) % (N_FOLDS - 1)])
    tags = fi.load_games(DATA_DIR).reindex(ids)["first_tag"].fillna("none").to_numpy()
    print(f"external forecast: {len(ids_n)} najzeko games, population {len(ids_a)} Anton games", flush=True)
    rows, extras, diag = [], [], {}
    mine = folds == 0
    for L in LS:
        taus, rs, kappas, sigmas = fc.build_templates(Y, folds, L)
        kappa_reg, sigma_reg, slope = fi.plateau_regression(Y, folds, L, tags)
        tau_knn = fi.knn_templates(Y, folds, L)
        items = [(a, Y[i], L, kappas[folds[i]], sigmas[folds[i]], kappa_reg[i], sigma_reg[i]) for i, a in enumerate(ids)]
        with ProcessPoolExecutor() as pool:
            p1 = list(pool.map(fi.pass1, items, chunksize=25))
        others = [p for p, m in zip(p1, mine) if not m]
        r_pop, s_conf, s_conf_reg = fi.pooled_r(others), fi.conformal_factor(others), fi.conformal_factor(others, "q_reg")
        diag[L] = dict(r_pop=r_pop, s_conf=s_conf, s_conf_reg=s_conf_reg, slope=slope[0])
        items = [(a, "najzeko", Y[i], L, p1[i], r_pop, s_conf, s_conf_reg, taus[0], rs[0], tau_knn[i])
                 for i, a in enumerate(ids) if mine[i]]
        with ProcessPoolExecutor() as pool:
            for rr, extra in pool.map(fi.pass2, items, chunksize=10):
                rows.extend(rr)
                extras.append(extra)
        print(f"L = {L} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(extras), diag


def summarise(tab: pd.DataFrame, agree: pd.DataFrame, scores: pd.DataFrame, extras: pd.DataFrame, diag: dict) -> str:
    out = ["najzeko external check", "",
           "games by reason:", tab["reason"].value_counts().to_string(), "",
           f"main games: {int((tab['group'] == 'main').sum())}; of them in Anton main group: "
           f"{int(((tab['group'] == 'main') & (tab['anton_group'] == 'main')).sum())}", ""]
    if len(agree):
        out += [f"agreement of weekly series on {len(agree)} common games (Anton / najzeko):",
                agree[["ratio", "rho_level", "r_diff", "first_week_shift"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string(), ""]
    ref = pd.read_csv(PROCESSED / "forecast_improve_scores_min500.csv")
    for L in LS:
        sc = scores[scores["L"] == L]
        rf = ref[ref["L"] == L]
        out += [f"L = {L}: r_pop {diag[L]['r_pop']:.3f}, s_conf {diag[L]['s_conf']:.3f}, s_conf_reg {diag[L]['s_conf_reg']:.3f}, "
                f"plateau slope {diag[L]['slope']:.3f}; {sc['app_id'].nunique()} games"]
        for name, part in (("najzeko", sc), ("Anton >= 500 (reference)", rf)):
            piv = part.pivot_table(index="model", columns="bucket", values="crps", aggfunc="mean").reindex(MODELS)
            cov = part[part["bucket"] != "total"].pivot_table(index="model", columns="bucket", values="covered90", aggfunc="mean").reindex(MODELS)
            out += [f"  CRPS, {name}:", piv.round(3).to_string(), f"  coverage 90%, {name}:", cov.round(2).to_string()]
        wide = sc[sc["bucket"] == "total"].pivot_table(index="app_id", columns="model", values="crps")
        for a, b in (("mix_shrink_knn", "template"), ("power_c_reg", "power_c_shrink"), ("power_c_reg", "template")):
            mean, lo, hi = fc.bootstrap_diff((wide[a] - wide[b]).dropna().to_numpy())
            out.append(f"  total CRPS {a} - {b}: {mean:+.3f} [{lo:+.3f}; {hi:+.3f}]")
        ex = extras[extras["L"] == L]
        S = sc[sc["bucket"] == "total"].groupby("app_id").size().index
        out.append(f"  median log(C_reg / plateau_actual): {np.median(np.log(ex['c_reg'] / ex['plateau_actual'])):+.3f}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--thin", type=float, default=None,
                    help="keep each review with this probability (binomial thinning) before forecasting; "
                         "0.5 mimics the Anton sampling rate; outputs get the suffix _thin")
    args = ap.parse_args()
    if args.build:
        build_daily()
        return
    daily = pd.read_csv(PROCESSED / "najzeko_daily.csv")
    tab, weekly = weekly_from_daily(daily)
    suffix = ""
    if args.thin is not None:
        rng = np.random.default_rng(SEED)
        pos = rng.binomial(weekly["n_positive"].to_numpy(int), args.thin)
        neg = rng.binomial((weekly["n_reviews"] - weekly["n_positive"]).to_numpy(int), args.thin)
        weekly = weekly.assign(n_reviews=pos + neg, n_positive=pos)
        suffix = "_thin"
    tab.to_csv(PROCESSED / f"najzeko_games{suffix}.csv", index=False)
    weekly.to_csv(PROCESSED / f"najzeko_weekly{suffix}.csv", index=False)
    agree = agreement(weekly)
    agree.to_csv(PROCESSED / f"najzeko_agreement{suffix}.csv", index=False)
    scores, extras, diag = external_forecast(weekly, tab)
    scores.to_csv(PROCESSED / f"najzeko_scores{suffix}.csv", index=False)
    text = (f"najzeko series thinned: each review kept with probability {args.thin}\n\n" if args.thin else "") \
        + summarise(tab, agree, scores, extras, diag)
    print(text)
    (RESULTS / f"najzeko_summary{suffix}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
