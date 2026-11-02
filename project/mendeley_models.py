"""M2: the forecasting models of forecast_curves.py applied to weekly player counts (Mendeley).

Two settings, because the player record (14 Dec 2017 - 12 Aug 2020) does not contain the start of
life of any game:
  part "release": 69 games whose Steam release date falls inside the record. Inspection shows these
     are early-access games (68 of 69 have players before the release date), so week 0 is the exit
     from early access: a spike on top of an existing base. The pre-release level (mean of the 8
     weeks before release) is a KNOWN floor, which gives a direct test of the "+C" idea:
     power_c_known (C fixed to the pre-release level) and flat_pre (forecast = pre-release level).
     Window 72 weeks after release (the shortest record), training weeks 0..11.
  part "record": games released before the record; week 0 = start of the record, 104 weeks
     (ends 12 Dec 2019, before the lockdown spike), training weeks 0..25. Checks whether the
     plateau/decay machinery holds on mature games where the curve is mostly plateau.
Weekly value = mean of the 5-minute (or hourly) player counts, rounded to an integer; the negative
binomial is used as a working variance model (variance grows with the level), not as a count model.
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import forecast_curves as fc
from mendeley_proxy import daily_means

RECORD_START = pd.Timestamp("2017-12-14")
RECORD_END = pd.Timestamp("2020-08-12")
PARTS = {
    "release": dict(window=72, L=12, plateau=(52, 72), pre_weeks=8,
                    buckets={"12-25": (12, 26), "26-51": (26, 52), "52-71": (52, 72)}),
    "record": dict(window=104, L=26, plateau=(78, 104), pre_weeks=0,
                   buckets={"26-51": (26, 52), "52-103": (52, 104)}),
}
MIN_MEDIAN_PLAYERS = 5
MIN_DAYS_PER_WEEK = 4
N_FOLDS = 5
SEED = 7
MODELS = ["flat", "flat_pre", "exp", "power", "lognormal", "hawkes50", "template",
          "power_c_free", "power_c_pop", "power_c_shrink", "power_c_known", "hawkes50_c_pop",
          "mix_shrink_template"]
BASELINES = ["template", "flat"]


def weekly_from(daily: pd.Series, start: pd.Timestamp, n_weeks: int) -> np.ndarray:
    days = pd.date_range(start, periods=n_weeks * 7, freq="D")
    vals = daily.reindex(days).to_numpy(dtype=float).reshape(n_weeks, 7)
    n_ok = np.isfinite(vals).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(vals, axis=1)
    return np.where(n_ok >= MIN_DAYS_PER_WEEK, mean, np.nan)


def build_series(mendeley_dir: Path) -> pd.DataFrame:
    info = pd.read_csv(mendeley_dir / "applicationInformation.csv", encoding="cp1252")
    info["release"] = pd.to_datetime(info["releasedate"], format="%d-%b-%y", errors="coerce")
    files = {int(p.stem): p for part in ("PlayerCountHistoryPart1", "PlayerCountHistoryPart2")
             for p in (mendeley_dir / part).glob("*.csv")}
    info = info[info["appid"].isin(files) & (info["type"] == "game")].drop_duplicates("appid")
    rows = []
    for _, g in info.iterrows():
        a = int(g["appid"])
        daily = daily_means(files[a])
        if len(daily) == 0:
            continue
        if pd.notna(g["release"]) and g["release"] >= RECORD_START:
            cfg = PARTS["release"]
            start = g["release"] - pd.Timedelta(days=7 * cfg["pre_weeks"])
            y = weekly_from(daily, start, cfg["window"] + cfg["pre_weeks"])
            weeks = np.arange(-cfg["pre_weeks"], cfg["window"])
            part = "release"
        elif pd.notna(g["release"]) and g["release"] < RECORD_START:
            cfg = PARTS["record"]
            y = weekly_from(daily, RECORD_START, cfg["window"])
            weeks = np.arange(cfg["window"])
            part = "record"
        else:
            continue
        post = y[weeks >= 0]
        if not np.all(np.isfinite(post)) or np.nanmedian(post) < MIN_MEDIAN_PLAYERS:
            continue
        if part == "release" and np.isfinite(y[weeks < 0]).sum() < 4:
            continue
        for w, v in zip(weeks, y):
            rows.append(dict(app_id=a, part=part, name=g["name"], week=int(w),
                             players=float(v) if np.isfinite(v) else np.nan))
    return pd.DataFrame(rows)


def configure(part: str) -> None:
    cfg = PARTS[part]
    fc.WINDOW, fc.PLATEAU_WEEKS, fc.BUCKETS, fc.N_FOLDS = cfg["window"], cfg["plateau"], cfg["buckets"], N_FOLDS


def run_game(item: tuple) -> tuple:
    part, app_id, y, pre, tau, r_template, kappa, sigma = item
    configure(part)
    L = PARTS[part]["L"]
    rng = np.random.default_rng(int(app_id) * 10 + L)
    sims, levels = fc.forecasts_for_game(y, L, tau, r_template, kappa, sigma, rng)
    horizon = PARTS[part]["window"] - L
    if pre is not None:
        t_train, t_future = np.arange(L, dtype=float), np.arange(L, PARTS[part]["window"], dtype=float)
        A_pre, r_pre, _ = fc.fit_flat(pre)   # baseline: keep even if L-BFGS-B reports no convergence
        sims["flat_pre"] = fc.nb_draws(np.full(horizon, A_pre), r_pre, rng)
        y_train = y[:L].astype(float)
        smooth = np.convolve(y_train, np.ones(4) / 4, mode="valid")
        fit = fc.fit_power_floor(y_train, t_train, max(float(smooth.max()), 1.0), 1.0, "pop",
                                 float(pre.mean()), sigma)
        levels["c_known"] = float(pre.mean())
        if fit["ok"]:
            mu = fc.power_floor_mean(fit["log_A"], fit["log_alpha"], fit["C"], t_future)
            sims["power_c_known"] = fc.nb_draws(mu, fit["r"], rng)
    if "power_c_shrink" in sims and "template" in sims:
        half = fc.N_SIMS // 2
        sims["mix_shrink_template"] = np.vstack([sims["power_c_shrink"][:half], sims["template"][:half]])
    rows = [dict(app_id=app_id, part=part, **row) for row in fc.score_game(y, L, sims)]
    levels.update(app_id=app_id, part=part, plateau_actual=fc.actual_plateau(y), S=float(y[:L].sum()),
                  jump=float(y[:4].mean() / max(pre.mean(), 1e-3)) if pre is not None else np.nan)
    return rows, levels


def summarise(scores: pd.DataFrame, levels: pd.DataFrame, part: str) -> str:
    cfg = PARTS[part]
    s = scores[scores["part"] == part]
    models = [m for m in MODELS if m in set(s["model"])]
    lines = [f"\n=== part '{part}': {s['app_id'].nunique()} games, window {cfg['window']} weeks, "
             f"trained on weeks 0..{cfg['L'] - 1} ==="]
    lines.append("mean CRPS (log scale, smaller is better) by horizon bucket")
    lines.append(s.pivot_table(index="model", columns="bucket", values="crps", aggfunc="mean")
                 .reindex(models).round(3).to_string())
    lines.append("\ncoverage of the 90% interval")
    lines.append(s[s["bucket"] != "total"].pivot_table(index="model", columns="bucket",
                                                        values="covered90", aggfunc="mean")
                 .reindex(models).round(3).to_string())
    lines.append("\nCRPS difference (model - baseline), mean per game with 95% bootstrap CI")
    wide = s.pivot_table(index=["app_id", "bucket"], columns="model", values="crps")
    rows = []
    for baseline in BASELINES:
        for model in models:
            if model == baseline or model not in wide:
                continue
            pair = wide[[model, baseline]].dropna()
            for bucket, grp in pair.groupby(level="bucket"):
                mean, lo, hi = fc.bootstrap_diff((grp[model] - grp[baseline]).to_numpy())
                rows.append(dict(baseline=baseline, model=model, bucket=bucket, games=len(grp),
                                 mean_diff=mean, ci_low=lo, ci_high=hi))
    lines.append(pd.DataFrame(rows).round(3).to_string(index=False))
    lv = levels[levels["part"] == part]
    lines.append(f"\nfitted level C vs the real level in weeks {cfg['plateau'][0]}..{cfg['plateau'][1] - 1} "
                 f"(Spearman over games)")
    target = np.log(lv["plateau_actual"])
    cols = ["c_free", "c_shrink", "c_pop"] + (["c_known"] if part == "release" else [])
    for col in cols:
        rho, _ = spearmanr(np.log(np.clip(lv[col], 1e-3, None)), target)
        lines.append(f"  {col:9s} rho = {rho:.3f}   median C = {lv[col].median():.1f}   "
                     f"median real level = {lv['plateau_actual'].median():.1f}")
    if part == "release":
        ratio = lv["plateau_actual"] / lv["c_known"]
        lines.append(f"  release jump (weeks 0-3 / pre-release level): median {lv['jump'].median():.2f}, "
                     f"q25 {lv['jump'].quantile(0.25):.2f}, q75 {lv['jump'].quantile(0.75):.2f}")
        lines.append(f"  plateau / pre-release level: median {ratio.median():.2f}, "
                     f"q25 {ratio.quantile(0.25):.2f}, q75 {ratio.quantile(0.75):.2f}; "
                     f"share of games ending above the old level: {(ratio > 1).mean():.2f}")
    return "\n".join(lines)


def plot(scores: pd.DataFrame, series: pd.DataFrame, levels: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, part in zip(axes[0, :2], PARTS):
        s = scores[(scores["part"] == part) & (scores["bucket"] == "total")]
        m = s.groupby("model")["crps"].mean().reindex([x for x in MODELS if x in set(s["model"])])
        ax.barh(m.index, m.values)
        ax.invert_yaxis()
        ax.set_title(f"'{part}': CRPS of the total over the forecast window")
    ax = axes[0, 2]
    s = scores[(scores["part"] == "release") & (scores["bucket"] != "total")]
    piv = s.pivot_table(index="bucket", columns="model", values="crps", aggfunc="mean")
    for model in ["flat", "flat_pre", "template", "power", "power_c_pop", "power_c_known", "mix_shrink_template"]:
        if model in piv:
            ax.plot(piv.index, piv[model], marker="o", label=model)
    ax.set_title("'release': weekly CRPS by horizon")
    ax.legend(fontsize=8)
    rel = series[series["part"] == "release"]
    lv = levels[levels["part"] == "release"].set_index("app_id")
    picks = [lv["jump"].idxmax(), lv["jump"].sub(lv["jump"].median()).abs().idxmin(), lv["jump"].idxmin()]
    for ax, a, lab in zip(axes[1], picks, ["largest release jump", "median jump", "smallest jump"]):
        g = rel[rel["app_id"] == a]
        ax.plot(g["week"], g["players"], color="tab:red")
        ax.axvline(0, color="grey", ls="--")
        ax.axhline(lv.loc[a, "c_known"], color="tab:blue", ls=":", label="pre-release level")
        ax.axvspan(0, PARTS["release"]["L"], color="grey", alpha=0.15, label="training weeks")
        ax.set_yscale("log")
        ax.set_title(f"{str(g['name'].iloc[0])[:30]} ({lab}, x{lv.loc[a, 'jump']:.1f})", fontsize=9)
        ax.set_xlabel("week since release (early-access exit)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mendeley-dir", type=Path, default=Path("datasets/mendeley/player_counts"))
    ap.add_argument("--processed", type=Path, default=Path("processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--limit", type=int, default=0, help="games per part (smoke test)")
    args = ap.parse_args()

    cache = args.processed / "mendeley_weekly.csv"
    if cache.exists():
        series = pd.read_csv(cache)
    else:
        series = build_series(args.mendeley_dir)
        series.to_csv(cache, index=False)
    print(series.groupby("part")["app_id"].nunique().to_dict(), "games")

    all_rows, all_levels = [], []
    for part, cfg in PARTS.items():
        configure(part)
        sub = series[series["part"] == part]
        ids = np.sort(sub["app_id"].unique())
        if args.limit:
            ids = ids[:args.limit]
        wide = sub.pivot(index="app_id", columns="week", values="players").loc[ids]
        Y = np.rint(wide.loc[:, 0:].to_numpy(dtype=float))
        pre = {a: np.rint(wide.loc[a, :-1].dropna().to_numpy(dtype=float)) for a in ids} \
            if cfg["pre_weeks"] else {a: None for a in ids}
        folds = np.random.default_rng(SEED).permutation(len(ids)) % N_FOLDS
        taus, rs, kappas, sigmas = fc.build_templates(Y, folds, cfg["L"])
        print(f"part '{part}': {len(ids)} games, kappa per fold {np.round(list(kappas.values()), 3).tolist()}, "
              f"template r {np.round(list(rs.values()), 2).tolist()}")
        items = [(part, int(a), Y[i], pre[a], taus[folds[i]], rs[folds[i]], kappas[folds[i]], sigmas[folds[i]])
                 for i, a in enumerate(ids)]
        with ProcessPoolExecutor() as pool:
            for rows, levels in pool.map(run_game, items, chunksize=10):
                all_rows.extend(rows)
                all_levels.append(levels)
    scores, levels = pd.DataFrame(all_rows), pd.DataFrame(all_levels)
    scores.to_csv(args.processed / "mendeley_forecast_scores.csv", index=False)
    levels.to_csv(args.processed / "mendeley_floor_fits.csv", index=False)
    text = "\n".join(summarise(scores, levels, part) for part in PARTS)
    print(text)
    (args.results / "mendeley_forecast_summary.txt").write_text(text, encoding="utf-8")
    plot(scores, series, levels, args.figures / "mendeley_forecast.png")
    print("figure saved")


if __name__ == "__main__":
    main()
