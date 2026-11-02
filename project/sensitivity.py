"""Sensitivity of the forecast conclusions to the data choices.

Reruns forecast_improve.py under alternative settings and compares the numbers that the report
relies on. Variants (tag: setting):
  base     the settings of the main analysis (>= 200 total reviews, 104-week window, fold seed 0)
  min100   threshold lowered to 100 total reviews (series rebuilt in processed/sens_min100)
  min500   at least 500 reviews inside the 104-week window
  win78    window of 78 weeks (plateau = weeks 52..77)
  seed1    another random split into folds
Compared per variant at L = 12: number of games, CRPS of the total for the template, the shrink
reference, the final model and the best mixture, weekly CRPS and coverage of the calibrated final
model, the plateau-regression slope, and the paired differences (mixture - template, final - shrink)
with bootstrap CIs. Usage: python sensitivity.py --run <tags...> (runs), then python sensitivity.py
(summary only).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_curves as fc

DATA_DIR = "datasets/kaggle/antonkozyriev-game-recommendations-on-steam"
VARIANTS = {
    "base": dict(args=[], env={}),
    "min100": dict(args=["--series", "processed/sens_min100/weekly_series.csv",
                         "--groups", "processed/sens_min100/games_groups.csv"], env={}),
    "min500": dict(args=["--min-reviews", "500"], env={}),
    "win78": dict(args=[], env={"FI_WINDOW": "78"}),
    "seed1": dict(args=["--seed", "1"], env={}),
}
L = 12
KEY_MODELS = ["template", "power_c_shrink", "power_c_reg", "reg_conf", "mix_shrink_knn"]


def run(tag: str) -> None:
    v = VARIANTS[tag]
    env = dict(os.environ, **v["env"])
    cmd = [sys.executable, "forecast_improve.py", "--data-dir", DATA_DIR, "--tag", tag] + v["args"]
    print("running", tag, " ".join(cmd[2:]), v["env"], flush=True)
    with open(f"processed/forecast_improve_output_{tag}.txt", "w", encoding="utf-8") as out:
        subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def summarise(tags: list) -> str:
    rows, diffs = [], []
    for tag in tags:
        path = Path(f"processed/forecast_improve_scores_{tag}.csv")
        if not path.exists():
            continue
        sc = pd.read_csv(path)
        sc = sc[sc["L"] == L]
        text = Path(f"results/forecast_improve_summary_{tag}.txt").read_text(encoding="utf-8")
        m = re.search(r"regression slope of log\(plateau/S\) on log S per fold: \[([^\]]*)\]", text)
        slope = np.mean([float(x) for x in m.group(1).split(",")]) if m else np.nan
        tot = sc[sc["bucket"] == "total"].groupby("model")["crps"].mean()
        weekly = sc[sc["bucket"] != "total"]
        first_bucket = sorted(weekly["bucket"].unique(), key=lambda b: int(b.split("-")[0]))
        wk = weekly.pivot_table(index="model", columns="bucket", values="crps", aggfunc="mean")[first_bucket]
        cov = weekly.pivot_table(index="model", columns="bucket", values="covered90", aggfunc="mean")[first_bucket]
        row = dict(variant=tag, games=sc["app_id"].nunique(), slope=round(slope, 3))
        for mdl in KEY_MODELS:
            row[f"total_{mdl}"] = round(tot.get(mdl, np.nan), 3)
        row["weekly_reg_conf"] = " / ".join(f"{x:.3f}" for x in wk.loc["reg_conf"])
        row["cov_reg_conf"] = " / ".join(f"{x:.2f}" for x in cov.loc["reg_conf"])
        row["cov_power_c_reg"] = " / ".join(f"{x:.2f}" for x in cov.loc["power_c_reg"])
        rows.append(row)
        wide = sc[sc["bucket"] == "total"].pivot_table(index="app_id", columns="model", values="crps")
        for a, b in (("mix_shrink_knn", "template"), ("power_c_reg", "power_c_shrink"),
                     ("power_c_reg", "template"), ("mix_shrink_knn", "power_c_reg")):
            pair = wide[[a, b]].dropna()
            mean, lo, hi = fc.bootstrap_diff((pair[a] - pair[b]).to_numpy())
            diffs.append(dict(variant=tag, diff=f"{a} - {b}", games=len(pair), mean=round(mean, 3),
                              ci_low=round(lo, 3), ci_high=round(hi, 3)))
    lines = [f"sensitivity of the forecast conclusions, L = {L}, CRPS of the total unless stated",
             pd.DataFrame(rows).set_index("variant").T.to_string(),
             "\npaired differences of the total CRPS (mean per game, 95% bootstrap CI)",
             pd.DataFrame(diffs).to_string(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[], help="variants to (re)run before summarising")
    args = ap.parse_args()
    for tag in args.run:
        run(tag)
    text = summarise(list(VARIANTS))
    print(text)
    Path("results/sensitivity_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
