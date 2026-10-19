"""Build summary tables of all model comparisons from the saved fit and check files.

Reads processed/sim_check.csv, processed/fits_simple.csv, processed/fits_hawkes.csv and
processed/fits_shocks.csv. Writes results/model_summary.md (readable) and
results/model_summary_crps.csv (mean CRPS per model, statistic and peak type).

Usage:
    python summarize_results.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("processed")
RESULTS = Path("results")
LATE_PEAK_WEEK = 26
STATS = ["early_share", "peak_week", "t50", "total"]
MODEL_ORDER = ["exp", "power", "lognormal", "hawkes", "hawkes80", "hawkes50",
               "shock_rule", "shock_joint"]
STAT_LABELS = {
    "early_share": "share of reviews in first 8 weeks",
    "peak_week": "peak week (weeks)",
    "t50": "week with 50% of reviews (weeks)",
    "total": "total reviews, log(1+x)",
}


def table(frame: pd.DataFrame, digits: int = 3) -> str:
    frame = frame.reindex([m for m in MODEL_ORDER if m in frame.index])
    return frame.round(digits).to_markdown()


def main() -> None:
    sim = pd.read_csv(PROCESSED / "sim_check.csv")
    simple = pd.read_csv(PROCESSED / "fits_simple.csv")
    peak = simple.set_index("app_id")["peak_week"]
    sim["peak_type"] = np.where(sim["app_id"].map(peak) > LATE_PEAK_WEEK, "late peak", "early peak")
    sim["outside95"] = (sim["percentile"] < 0.025) | (sim["percentile"] > 0.975)

    n_games = sim["app_id"].nunique()
    n_by_type = sim.drop_duplicates("app_id").groupby("peak_type").size()
    lines = ["# Model comparison summary", "",
             f"Generated from processed/sim_check.csv: {n_games:,} games "
             f"({n_by_type.get('early peak', 0):,} early-peak, {n_by_type.get('late peak', 0):,} "
             f"late-peak; late peak = first 4-week peak window starts after week {LATE_PEAK_WEEK}).",
             "Statistics are computed on free-running simulations (200 per game and model).", ""]

    def add(title: str, note: str, frame: pd.DataFrame, digits: int = 3) -> None:
        lines.extend([f"## {title}", "", note, "", table(frame, digits), ""])

    crps = sim.pivot_table(index="model", columns="stat", values="crps", aggfunc="mean")[STATS]
    add("Mean CRPS (smaller is better)",
        "Accounts for both accuracy and width of the simulated spread. "
        + "; ".join(f"{k}: {v}" for k, v in STAT_LABELS.items()) + ".", crps)
    for peak_type, part in sim.groupby("peak_type"):
        add(f"Mean CRPS, {peak_type} games", "Same measure restricted to the group.",
            part.pivot_table(index="model", columns="stat", values="crps", aggfunc="mean")[STATS])

    best = pd.concat({
        "all": crps.idxmin(),
        **{pt: part.pivot_table(index="model", columns="stat", values="crps", aggfunc="mean")[STATS]
           .idxmin() for pt, part in sim.groupby("peak_type")}}, axis=1)
    lines.extend(["## Lowest mean CRPS per statistic", "", best.to_markdown(), ""])

    outside = sim.pivot_table(index="model", columns="stat", values="outside95",
                              aggfunc="mean")[STATS]
    add("Share of games with the observed value outside the central 95% of simulations",
        "About 0.05 would be expected for a model that is exactly right; lower values can also "
        "come from intervals that are too wide, so read together with the width table.", outside)
    width = sim.pivot_table(index="model", columns="stat", values="width", aggfunc="median")[STATS]
    add("Median width of the central 95% simulation interval",
        "early_share and total: relative to the simulated median; peak_week and t50: weeks.",
        width, 2)
    resid = sim.drop_duplicates(["app_id", "model"]).groupby("model")["resid_lag1"].median()
    add("Median lag-1 autocorrelation of one-step residuals",
        "Near 0 means no leftover pattern. Shock models use the mean without the shock term, so "
        "their values are not comparable.", resid.to_frame("resid_lag1"))

    # In-sample AIC wins across all eight models (only games fitted by every model)
    hawkes = pd.read_csv(PROCESSED / "fits_hawkes.csv")
    shocks = pd.read_csv(PROCESSED / "fits_shocks.csv")
    merged = (simple[simple["status"] == "fitted"].merge(hawkes.drop(columns="group"), on="app_id")
              .merge(shocks.drop(columns="group"), on="app_id"))
    aic = merged[[f"{m}_aic" for m in MODEL_ORDER]]
    wins = aic.idxmin(axis=1).str.replace("_aic", "").value_counts()
    wins = (wins / len(merged)).reindex(MODEL_ORDER).fillna(0)
    add("Share of games where each model has the lowest AIC (in-sample, one-step likelihood)",
        f"{len(merged):,} games fitted by all models. AIC uses the real earlier weeks, so it "
        "favours the recursive (Hawkes, shock) models; it does not measure free-run quality.",
        wins.to_frame("share"))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    sim.groupby(["peak_type", "model", "stat"])["crps"].mean().reset_index().to_csv(
        RESULTS / "model_summary_crps.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
