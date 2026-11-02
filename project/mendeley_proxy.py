"""M1: are weekly review counts a usable proxy for the weekly number of players?

For every game that has both a review series (processed/weekly_series.csv) and a Mendeley
player-count file, the 5-minute player counts are averaged over the game's own review weeks
(week k = [first_review + 7k, first_review + 7k + 7)). Per game:
  rho_level  Spearman correlation of log(1 + reviews) and log(1 + players) over common weeks
  r_diff     Pearson correlation of week-to-week changes of the two log series (shape without trend)
  best_lag   lag in weeks maximising the change correlation; > 0 means reviews react after players
  ratio      median reviews per 1000 player-weeks; sd_log_ratio = its within-game spread (scale)
Control: the same statistics with the player series of N_NULL random other games on the same
calendar windows (what common events such as sales or lockdown give "for free").
Decision rule fixed in advance: proxy is "good" if the median r_diff of own pairs exceeds the
median of the control AND at least half of the games lie above their own control 95th percentile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

WINDOW = 104
MIN_WEEKS = 26
LONG_WEEKS = 52
MIN_DAYS_PER_WEEK = 4          # a week of player data counts if >= 4 daily means exist
MIN_SAMPLES_PER_DAY = 12       # a day counts if >= 12 samples (files are 5-minute or hourly)
LAGS = range(-4, 5)
N_NULL = 40
SEED = 20261031


def daily_means(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    d["Time"] = pd.to_datetime(d["Time"])
    d = d.dropna(subset=["Playercount"])
    g = d.groupby(d["Time"].dt.normalize())["Playercount"]
    out = g.mean()[g.size() >= MIN_SAMPLES_PER_DAY]
    return out


def weekly_players(daily: pd.Series, first_review: pd.Timestamp) -> np.ndarray:
    """Mean players per review week (NaN if fewer than MIN_DAYS_PER_WEEK days observed)."""
    days = pd.date_range(first_review, periods=WINDOW * 7, freq="D")
    vals = daily.reindex(days).to_numpy(dtype=float).reshape(WINDOW, 7)
    n_ok = np.isfinite(vals).sum(axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(vals, axis=1)
    return np.where(n_ok >= MIN_DAYS_PER_WEEK, mean, np.nan)


def shape_stats(reviews: np.ndarray, players: np.ndarray) -> dict | None:
    ok = np.isfinite(players) & np.isfinite(reviews)
    if ok.sum() < MIN_WEEKS:
        return None
    lr, lp = np.log1p(reviews[ok]), np.log1p(players[ok])
    weeks = np.where(ok)[0]
    rho = spearmanr(lr, lp).correlation
    # differences only between consecutive observed weeks
    consec = np.diff(weeks) == 1
    dr, dp = np.diff(lr)[consec], np.diff(lp)[consec]
    if len(dr) < MIN_WEEKS - 1 or dr.std() == 0 or dp.std() == 0:
        return None
    r_diff = pearsonr(dr, dp)[0]
    lag_corr = {}
    for lag in LAGS:   # corr(dr[t], dp[t - lag]): positive lag = reviews follow players
        if lag >= 0:
            a, b = dr[lag:], dp[:len(dp) - lag]
        else:
            a, b = dr[:len(dr) + lag], dp[-lag:]
        lag_corr[lag] = pearsonr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan
    best_lag = max(lag_corr, key=lambda k: np.nan_to_num(lag_corr[k], nan=-9))
    pos = players[ok] > 0
    log_ratio = np.log(reviews[ok][pos] + 0.5) - np.log(players[ok][pos])
    return dict(n_weeks=int(ok.sum()), rho_level=rho, r_diff=r_diff, best_lag=best_lag,
                r_lag_best=lag_corr[best_lag], r_lag0=lag_corr[0],
                ratio_per_1000=float(np.exp(np.median(log_ratio)) * 1000),
                sd_log_ratio=float(log_ratio.std()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mendeley-dir", type=Path, default=Path("datasets/mendeley/player_counts"))
    ap.add_argument("--processed", type=Path, default=Path("processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    groups = pd.read_csv(args.processed / "games_groups.csv", parse_dates=["first_review"])
    groups = groups[groups["group"].isin(["main", "censored"])].set_index("app_id")
    series = pd.read_csv(args.processed / "weekly_series.csv")
    files = {int(p.stem): p for part in ("PlayerCountHistoryPart1", "PlayerCountHistoryPart2")
             for p in (args.mendeley_dir / part).glob("*.csv")}
    info = pd.read_csv(args.mendeley_dir / "applicationInformation.csv", encoding="cp1252",
                       usecols=["appid", "name"]).drop_duplicates("appid").set_index("appid")["name"]

    # daily player means for every candidate game whose review window touches the record
    record_start = pd.Timestamp("2017-12-14")
    cand = [a for a in groups.index if a in files
            and groups.loc[a, "first_review"] + pd.Timedelta(days=7 * WINDOW) > record_start]
    print(f"games with both sources: {len(cand)} (of {len(files)} player files)")
    daily = {a: daily_means(files[a]) for a in cand}
    daily = {a: d for a, d in daily.items() if len(d) > 0}

    rev = {a: g.set_index("week")["n_reviews"].reindex(range(WINDOW)).to_numpy(dtype=float)
           for a, g in series[series["app_id"].isin(cand)].groupby("app_id")}
    weekly = {a: weekly_players(daily[a], groups.loc[a, "first_review"]) for a in daily}

    rows, null_rows, examples = [], [], {}
    for a in daily:
        own = shape_stats(rev[a], weekly[a])
        if own is None:
            continue
        rows.append(dict(app_id=a, group=groups.loc[a, "group"], name=info.get(a, ""), **own))
        examples[a] = (rev[a], weekly[a])
        others = [b for b in daily if b != a]
        for b in rng.choice(others, size=N_NULL, replace=False):
            w = weekly_players(daily[b], groups.loc[a, "first_review"])
            st = shape_stats(rev[a], w)
            if st is not None:
                null_rows.append(dict(app_id=a, other=b, **st))
    own_df = pd.DataFrame(rows).set_index("app_id")
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(args.processed / "proxy_null.csv", index=False)

    # game-specific control thresholds
    q95 = null_df.groupby("app_id")[["rho_level", "r_diff"]].quantile(0.95)
    own_df["null_q95_level"] = q95["rho_level"].reindex(own_df.index)
    own_df["null_q95_diff"] = q95["r_diff"].reindex(own_df.index)
    own_df["above_control_level"] = own_df["rho_level"] > own_df["null_q95_level"]
    own_df["above_control_diff"] = own_df["r_diff"] > own_df["null_q95_diff"]
    own_df.to_csv(args.processed / "proxy_scores.csv")

    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    say(f"games with >= {MIN_WEEKS} common weeks: {len(own_df)}; >= {LONG_WEEKS}: {(own_df.n_weeks >= LONG_WEEKS).sum()}")
    say(f"groups: {own_df.group.value_counts().to_dict()}")
    for label, sub in [("all", own_df), (f">={LONG_WEEKS} weeks", own_df[own_df.n_weeks >= LONG_WEEKS])]:
        nl = null_df[null_df.app_id.isin(sub.index)]
        say(f"\n[{label}] n={len(sub)}")
        for col, key, name in [("rho_level", "level", "Spearman of log levels"),
                               ("r_diff", "diff", "Pearson of log changes")]:
            q = sub[col].quantile([0.25, 0.5, 0.75]).round(3).tolist()
            qn = nl[col].quantile([0.25, 0.5, 0.75]).round(3).tolist()
            share = (sub[col] > sub[f"null_q95_{key}"]).mean()
            say(f"  {name:24s} own q25/50/75 {q}  control {qn}  share above own control 95th pct {share:.2f}")
        say(f"  best lag (reviews after players): {sub.best_lag.value_counts().sort_index().to_dict()}")
        say(f"  gain of best lag over lag 0 (median): {(sub.r_lag_best - sub.r_lag0).median():.3f}")
        say(f"  reviews per 1000 player-weeks: median {sub.ratio_per_1000.median():.2f}, "
            f"q10 {sub.ratio_per_1000.quantile(0.1):.2f}, q90 {sub.ratio_per_1000.quantile(0.9):.2f}")
        say(f"  within-game sd of log(reviews/players): median {sub.sd_log_ratio.median():.2f}")
    ok_median = own_df.r_diff.median() > null_df.r_diff.median()
    ok_share = own_df.above_control_diff.mean() >= 0.5
    say(f"\nPre-registered rule: median r_diff own > control: {ok_median}; "
        f"share above control >= 0.5: {ok_share} -> proxy {'GOOD' if ok_median and ok_share else 'WEAK'}")
    (args.results / "proxy_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # figure: histograms, lag bar, three examples
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col, title in [(axes[0, 0], "rho_level", "Spearman, log levels"),
                           (axes[0, 1], "r_diff", "Pearson, weekly log changes")]:
        bins = np.linspace(-1, 1, 41)
        ax.hist(null_df[col].dropna(), bins=bins, density=True, alpha=0.5, label="control: other game players")
        ax.hist(own_df[col].dropna(), bins=bins, density=True, alpha=0.6, label="own players")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xlabel("correlation")
    ax = axes[0, 2]
    lag_tab = own_df.best_lag.value_counts().sort_index()
    ax.bar(lag_tab.index, lag_tab.values)
    ax.set_title("best lag (weeks, + = reviews after players)")
    picks = [own_df.r_diff.idxmax(), own_df.r_diff.sub(own_df.r_diff.median()).abs().idxmin(),
             own_df.r_diff.idxmin()]
    labels = ["best r_diff", "median r_diff", "worst r_diff"]
    for ax, a, lab in zip(axes[1], picks, labels):
        r, p = examples[a]
        ok = np.isfinite(p)
        t = np.arange(WINDOW)
        ax.plot(t, r, color="tab:blue", label="reviews / week")
        ax2 = ax.twinx()
        ax2.plot(t[ok], p[ok], color="tab:red", label="players (weekly mean)")
        ax.set_yscale("log")
        ax2.set_yscale("log")
        ax.set_title(f"{str(own_df.loc[a, 'name'])[:28]}  r_diff={own_df.loc[a, 'r_diff']:.2f} ({lab})", fontsize=9)
        ax.set_xlabel("week since first review")
    fig.tight_layout()
    fig.savefig(args.figures / "proxy_check.png", dpi=120)
    print("figure saved")


if __name__ == "__main__":
    main()
