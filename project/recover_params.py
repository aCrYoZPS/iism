"""Parameter recovery for the final curve y_t ~ NB(mu_t, r), mu_t = A (1+t)^-alpha + C.

The model is used as a GENERATOR: "true" parameters are the free-C fits over the full 104 weeks of
N_TRUTH random main games (games whose fitted C sits at the lower bound are dropped: no plateau to
recover). From every truth, N_REP synthetic games of 104 weeks are drawn; the model is then fitted
to the first L weeks (L = 12, 26) as in the real forecast: free C, population C (kappa * S) and
shrinkage towards it. Reported per parameter and method: log(estimate / truth) quantiles, share of
estimates at a bound, and the error of the forecast sum over weeks L..103.

Expectation written down before running: at L = 12 free C is not recoverable (collapses to the
bound or spreads over orders of magnitude), at L = 26 it is recovered with visible spread; the
shrinkage fit has the smallest forecast error at both L.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast_curves as fc

WINDOW = 104
LS = [12, 26]
N_TRUTH = 400
N_REP = 20
SEED = 11
MODES = ["free", "pop", "shrink"]


def population_rule(Y: np.ndarray, L: int) -> tuple:
    """kappa and sigma of the plateau rule from the real population (same definition as forecast_curves)."""
    lo, hi = fc.PLATEAU_WEEKS
    S = np.clip(Y[:, :L].sum(axis=1), 1, None)
    ratio = ((Y[:, lo:hi].sum(axis=1) + 1.0) / (hi - lo)) / S
    return float(np.median(ratio)), float(np.std(np.log(ratio)))


def fit_truth(item: tuple) -> dict:
    app_id, y, A0, alpha0 = item
    t = np.arange(WINDOW, dtype=float)
    fit = fc.fit_power_floor(y, t, A0, alpha0, "free", 1.0, 1.0)
    return dict(app_id=app_id, A=float(np.exp(fit["log_A"])), alpha=float(np.exp(fit["log_alpha"])),
                C=fit["C"], r=fit["r"], ok=fit["ok"],
                at_bound=bool(fit["C"] <= np.exp(fc.LOG_C_BOUNDS[0]) * 1.01))


def generate(truth: dict) -> tuple:
    """N_REP synthetic games from one truth, and the true mean curve."""
    rng = np.random.default_rng(SEED * 1000 + int(truth["app_id"]))
    t = np.arange(WINDOW, dtype=float)
    mu = fc.power_floor_mean(np.log(truth["A"]), np.log(truth["alpha"]), truth["C"], t)
    ys = rng.negative_binomial(truth["r"], truth["r"] / (truth["r"] + mu), size=(N_REP, WINDOW)).astype(float)
    return ys, mu


def recover(item: tuple) -> list:
    truth, ys, mu, rules = item
    t = np.arange(WINDOW, dtype=float)
    rows = []
    for rep in range(N_REP):
        y = ys[rep]
        for L in LS:
            kappa, sigma = rules[L]
            y_train, t_train = y[:L], t[:L]
            S = max(float(y_train.sum()), 1.0)
            smooth = np.convolve(y_train, np.ones(4) / 4, mode="valid")
            A0 = max(float(smooth.max()), 1.0)
            for mode in MODES:
                fit = fc.fit_power_floor(y_train, t_train, A0, 1.0, mode, kappa * S, sigma)
                mu_hat = fc.power_floor_mean(fit["log_A"], fit["log_alpha"], fit["C"], t[L:])
                rows.append(dict(
                    app_id=truth["app_id"], rep=rep, L=L, mode=mode, ok=fit["ok"],
                    true_alpha=truth["alpha"], true_C=truth["C"], true_A=truth["A"],
                    est_alpha=float(np.exp(fit["log_alpha"])), est_C=fit["C"], est_A=float(np.exp(fit["log_A"])),
                    c_at_bound=bool(fit["C"] <= np.exp(fc.LOG_C_BOUNDS[0]) * 1.01) if mode != "pop" else False,
                    alpha_at_bound=bool(fit["log_alpha"] <= -7.99 or fit["log_alpha"] >= 2.99),
                    forecast_log_err=float(np.log(mu_hat.sum() + 1.0) - np.log(mu[L:].sum() + 1.0)),
                    realised_log_err=float(np.log(mu_hat.sum() + 1.0) - np.log(y[L:].sum() + 1.0)),
                ))
    return rows


def summarise(res: pd.DataFrame, truths: pd.DataFrame) -> str:
    lines = [f"truths: {len(truths)} games (of which {int(truths['at_bound'].sum())} with C at the bound "
             f"were dropped before generation); {N_REP} synthetic games each; L = {LS}"]
    lines.append("true parameter ranges (q10 / median / q90): " + ", ".join(
        f"{p} {truths[p].quantile(0.1):.3g} / {truths[p].median():.3g} / {truths[p].quantile(0.9):.3g}"
        for p in ("A", "alpha", "C", "r")))
    for L in LS:
        part = res[res["L"] == L]
        lines.append(f"\n=== fitted on weeks 0..{L - 1} ===")
        lines.append("log(estimate / truth): q10 / median / q90, and share at a bound")
        for mode in MODES:
            m = part[part["mode"] == mode]
            la = np.log(m["est_alpha"] / m["true_alpha"])
            lc = np.log(np.clip(m["est_C"], 1e-3, None) / m["true_C"])
            lines.append(f"  {mode:7s} alpha {la.quantile(0.1):6.2f} / {la.median():6.2f} / {la.quantile(0.9):6.2f}"
                         f"  (at bound {m['alpha_at_bound'].mean():.2f})   "
                         f"C {lc.quantile(0.1):6.2f} / {lc.median():6.2f} / {lc.quantile(0.9):6.2f}"
                         f"  (at bound {m['c_at_bound'].mean():.2f})")
        lines.append("forecast sum over weeks L..103, log(predicted / true mean): q10 / median / q90, "
                     "mean |error|")
        for mode in MODES:
            m = part[part["mode"] == mode]
            e = m["forecast_log_err"]
            lines.append(f"  {mode:7s} {e.quantile(0.1):6.2f} / {e.median():6.2f} / {e.quantile(0.9):6.2f}   "
                         f"mean|err| {e.abs().mean():.3f}   share within factor 2: {(e.abs() < np.log(2)).mean():.2f}")
        # does recovery of C depend on how large the plateau is relative to the start?
        m = part[part["mode"] == "free"].copy()
        m["rel"] = pd.qcut(np.log(m["true_C"] / m["true_A"]), 4, labels=["C/A low", "q2", "q3", "C/A high"])
        g = m.groupby("rel", observed=True).apply(
            lambda d: pd.Series(dict(at_bound=d["c_at_bound"].mean(),
                                     med_logC=np.log(np.clip(d["est_C"], 1e-3, None) / d["true_C"]).median(),
                                     mean_abs_fc=d["forecast_log_err"].abs().mean())), include_groups=False)
        lines.append("free C by quartile of true C/A (plateau relative to start): share at bound, "
                     "median log(C_hat/C), mean |forecast error|")
        lines.append(g.round(3).to_string())
    return "\n".join(lines)


def plot(res: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, L in enumerate(LS):
        part = res[res["L"] == L]
        ax = axes[row, 0]
        for mode in ("free", "shrink"):
            m = part[part["mode"] == mode]
            ax.hist(np.log(np.clip(m["est_C"], 1e-3, None) / m["true_C"]), bins=np.linspace(-8, 4, 49),
                    alpha=0.6, label=mode, density=True)
        ax.axvline(0, color="k", ls="--")
        ax.set_title(f"L={L}: log(C_hat / C_true)")
        ax.legend()
        ax = axes[row, 1]
        for mode in ("free", "shrink"):
            m = part[part["mode"] == mode]
            ax.hist(np.log(m["est_alpha"] / m["true_alpha"]), bins=np.linspace(-2, 2, 41),
                    alpha=0.6, label=mode, density=True)
        ax.axvline(0, color="k", ls="--")
        ax.set_title(f"L={L}: log(alpha_hat / alpha_true)")
        ax.legend()
        ax = axes[row, 2]
        data = [part[part["mode"] == mode]["forecast_log_err"].clip(-4, 4) for mode in MODES]
        ax.boxplot(data, tick_labels=MODES, showfliers=False)
        ax.axhline(0, color="k", ls="--")
        ax.set_title(f"L={L}: log(forecast sum / true sum), weeks {L}..103")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("processed"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--n-truth", type=int, default=N_TRUTH)
    args = ap.parse_args()

    series = pd.read_csv(args.processed / "weekly_series.csv")
    simple = pd.read_csv(args.processed / "fits_simple.csv")
    keep = simple[(simple["group"] == "main") & (simple["n_weeks"] == WINDOW) & simple["power_converged"].eq(True)]
    ids_all = np.sort(keep["app_id"].to_numpy())
    wide = series[series["app_id"].isin(ids_all)].pivot(index="app_id", columns="week", values="n_reviews").loc[ids_all]
    Y = wide.to_numpy(dtype=float)
    print("real population rule (kappa, sigma):",
          {L: tuple(round(v, 3) for v in population_rule(Y, L)) for L in LS})

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(ids_all), size=args.n_truth, replace=False)
    starts = keep.set_index("app_id").loc[ids_all[pick], ["power_A", "power_alpha"]].to_numpy()
    items = [(int(ids_all[i]), Y[i], float(a), float(al)) for i, (a, al) in zip(pick, starts)]
    with ProcessPoolExecutor() as pool:
        truths = pd.DataFrame(list(pool.map(fit_truth, items, chunksize=10)))
    usable = truths[truths["ok"] & ~truths["at_bound"]]
    print(f"truths fitted: {len(truths)}, usable (C above bound): {len(usable)}")

    truth_list = usable.to_dict("records")
    synthetic = [generate(t) for t in truth_list]
    Y_syn = np.vstack([ys for ys, _ in synthetic])
    rules = {L: population_rule(Y_syn, L) for L in LS}   # the rule the synthetic population implies
    print("synthetic population rule (kappa, sigma):", {L: tuple(round(v, 3) for v in r) for L, r in rules.items()})
    with ProcessPoolExecutor() as pool:
        rows = [r for rows in pool.map(recover, [(t, ys, mu, rules) for t, (ys, mu) in zip(truth_list, synthetic)],
                                       chunksize=5) for r in rows]
    res = pd.DataFrame(rows)
    res.to_csv(args.processed / "recover_params.csv", index=False)
    truths.to_csv(args.processed / "recover_truths.csv", index=False)
    text = summarise(res, truths)
    print(text)
    (args.results / "recover_summary.txt").write_text(text, encoding="utf-8")
    plot(res, args.figures / "recover_params.png")
    print("figure saved")


if __name__ == "__main__":
    main()
