"""Trajectory types: cluster games under several descriptions and compare the results.

Each game is described in two families, both from its 104 weekly review counts:
  full-window (shape, summary, params)     use all 104 weeks. They describe a game but must not be
                                           used to judge prediction from early weeks: the first weeks
                                           are part of the definition of the type.
  future (shape_f, summary_f, params_f)    use only weeks 12..103, so a type can be predicted from
                                           weeks 0..11 without leakage.
Descriptions:
  shape    weekly counts divided by their sum, folded into four-week blocks
  summary  peak week, week with 50% of reviews, share of reviews in the first 8 weeks, log of total
           reviews, log ratio of the second to the first half (standardised)
  params   parameters of the hawkes50 fit: log B, log alpha, eta, rho, log r (standardised); the
           future version refits the model on weeks 12..103 only
Clustering: k-means and Ward hierarchical, k = 2..8, the same for every description.
Criteria that do not depend on the feature space:
  stability    mean ARI between the clustering of a bootstrap sample and the original clustering
  agreement    ARI between descriptions at the same method and k
  external     association of the type with first tag, price and positive-review ratio from
               games.csv / games_metadata.json (validation only, never an input)
  predictable  logistic regression from weeks 0..11 to the type, 5-fold cross-validation by game,
               against a most-frequent-class baseline (leak-free only for the future family)
Diagnostics on why types are weak: PCA of the shapes, a structureless control (Gaussian data with
the same mean and covariance, same k-means and stability), and parameters without games whose fit
sits on a bound.
Only 'main' games with the full 104-week window are used.

Reads processed/*.csv and the Anton games files; writes processed/cluster_assignments.csv,
cluster_criteria.csv, cluster_agreement.csv, fits_hawkes50_future.csv and
figures/cluster_criteria.png.

Usage:
    python cluster_types.py --data-dir datasets/kaggle/antonkozyriev-game-recommendations-on-steam
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, f1_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import fit_hawkes

PROCESSED = Path("processed")
FIG_DIR = Path("figures")
WINDOW = 104
BLOCK = 4
PREDICT_WEEKS = 12  # types in the future family are defined from this week on
EARLY_SHARE_WEEKS = 8
KS = list(range(2, 9))
METHODS = ["kmeans", "ward"]
FEATURES = ["shape", "summary", "params", "shape_f", "summary_f", "params_f"]
N_BOOT = 50
N_NULL_BOOT = 20
TOP_TAGS = 10
SEED = 0
ETA_CAP = 0.5  # hawkes50
SIL_SAMPLE = 2000


def load_games(data_dir: Path) -> pd.DataFrame:
    """External attributes per game (validation only)."""
    games = pd.read_csv(data_dir / "games.csv", usecols=["app_id", "price_original", "positive_ratio"])
    first_tag = {}
    with open(data_dir / "games_metadata.json", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("tags"):
                first_tag[record["app_id"]] = record["tags"][0]
    games["first_tag"] = games["app_id"].map(first_tag)
    top = games["first_tag"].value_counts().head(TOP_TAGS).index
    games["first_tag"] = games["first_tag"].where(games["first_tag"].isin(top) | games["first_tag"].isna(),
                                                  "other")
    return games.set_index("app_id")


def shape_matrix(counts: np.ndarray) -> np.ndarray:
    blocks = counts[:, :counts.shape[1] // BLOCK * BLOCK].reshape(len(counts), -1, BLOCK).sum(axis=2)
    return blocks / np.clip(counts.sum(axis=1, keepdims=True), 1, None)


def summary_matrix(counts: np.ndarray) -> np.ndarray:
    total = np.clip(counts.sum(axis=1), 1, None)
    cum = np.cumsum(counts, axis=1)
    windows = np.apply_along_axis(lambda v: np.convolve(v, np.ones(4), mode="valid"), 1, counts)
    half = counts.shape[1] // 2
    first, second = counts[:, :half].sum(axis=1), counts[:, half:].sum(axis=1)
    return np.column_stack([
        windows.argmax(axis=1),
        (cum >= 0.5 * total[:, None]).argmax(axis=1),
        counts[:, :EARLY_SHARE_WEEKS].sum(axis=1) / total,
        np.log1p(counts.sum(axis=1)),
        np.log((second + 1.0) / (first + 1.0))])


def refit_future(item: tuple) -> dict:
    app_id, y, B0, alpha0 = item
    row = {"app_id": app_id}
    row.update(fit_hawkes.fit_variant("hawkes50", ETA_CAP, y, B0, alpha0))
    return row


def params_matrix(h: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.log(h["hawkes50_B"]), np.log(h["hawkes50_alpha"]),
                            h["hawkes50_eta"], h["hawkes50_rho"], np.log(h["hawkes50_r"])])


def build_features(series: pd.DataFrame, hawkes: pd.DataFrame, app_ids: np.ndarray):
    """Feature matrices for the six descriptions, the early-weeks matrix and the future hawkes fits."""
    wide = series.pivot(index="app_id", columns="week", values="n_reviews").loc[app_ids]
    counts = wide.to_numpy(dtype=float)
    future = counts[:, PREDICT_WEEKS:]
    h = hawkes.set_index("app_id").loc[app_ids]

    print(f"refitting hawkes50 on weeks {PREDICT_WEEKS}..{WINDOW - 1} for {len(app_ids):,} games")
    items = [(a, future[i], h["hawkes50_B"].iloc[i], h["hawkes50_alpha"].iloc[i])
             for i, a in enumerate(app_ids)]
    with ProcessPoolExecutor() as pool:
        refits = pd.DataFrame(pool.map(refit_future, items, chunksize=50))
    refits.to_csv(PROCESSED / "fits_hawkes50_future.csv", index=False)
    refits = refits.set_index("app_id").loc[app_ids]

    raw = {"shape": shape_matrix(counts), "summary": summary_matrix(counts), "params": params_matrix(h),
           "shape_f": shape_matrix(future), "summary_f": summary_matrix(future),
           "params_f": params_matrix(refits)}
    features = {k: (v if k.startswith("shape") else StandardScaler().fit_transform(v))
                for k, v in raw.items()}
    early = np.log1p(counts[:, :PREDICT_WEEKS])
    assert early.shape[1] == PREDICT_WEEKS  # prediction uses weeks 0..11 only
    return features, early, h, refits


def cluster(X: np.ndarray, method: str, k_values: list) -> dict:
    """Labels for each k. Ward builds one tree and cuts it at every k."""
    if method == "kmeans":
        return {k: KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
                for k in k_values}
    tree = linkage(X, method="ward")
    return {k: fcluster(tree, k, criterion="maxclust") - 1 for k in k_values}


def bootstrap_task(args: tuple) -> tuple:
    feature, method, boot, X, reference = args
    rng = np.random.default_rng(SEED + boot)
    sample = np.unique(rng.integers(0, len(X), size=len(X)))
    labels = cluster(X[sample], method, KS)
    return feature, method, {k: adjusted_rand_score(reference[(feature, method, k)][sample], labels[k])
                             for k in KS}


def run_stability(tasks: list) -> dict:
    stability = {}
    with ProcessPoolExecutor() as pool:
        for feature, method, result in pool.map(bootstrap_task, tasks, chunksize=4):
            for k, ari in result.items():
                stability.setdefault((feature, method, k), []).append(ari)
    return stability


def cramers_v(table: pd.DataFrame) -> tuple:
    chi2, p, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), float(p)


def kruskal_effect(values: np.ndarray, labels: np.ndarray) -> tuple:
    keep = ~np.isnan(values)
    groups = [values[keep & (labels == g)] for g in np.unique(labels[keep])]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    return float((h - k + 1) / (n - k)), float(p)  # epsilon-squared effect size


def external_association(labels: np.ndarray, ext: pd.DataFrame) -> dict:
    tag = ext["first_tag"].notna().to_numpy()
    v, p_tag = cramers_v(pd.crosstab(labels[tag], ext["first_tag"][tag]))
    eps_price, p_price = kruskal_effect(ext["price_original"].to_numpy(dtype=float), labels)
    eps_ratio, p_ratio = kruskal_effect(ext["positive_ratio"].to_numpy(dtype=float), labels)
    return dict(tag_V=v, tag_p=p_tag, price_eps2=eps_price, price_p=p_price,
                ratio_eps2=eps_ratio, ratio_p=p_ratio)


def predictability(early: np.ndarray, labels: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X = StandardScaler().fit_transform(early)
    pred = cross_val_predict(LogisticRegression(max_iter=2000), X, labels, cv=cv)
    base = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, labels, cv=cv)
    return dict(macro_f1=f1_score(labels, pred, average="macro"),
                balanced_acc=balanced_accuracy_score(labels, pred),
                dummy_macro_f1=f1_score(labels, base, average="macro"),
                dummy_balanced_acc=balanced_accuracy_score(labels, base))


def plot_criteria(criteria: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    panels = [("stability_mean", "bootstrap stability (ARI)"),
              ("macro_f1", "macro-F1 from weeks 0-11"),
              ("tag_V", "association with first tag (Cramér's V)")]
    colors = {"shape": "C0", "summary": "C1", "params": "C2"}
    fig, axes = plt.subplots(len(METHODS), len(panels), figsize=(15, 7), sharex=True)
    for row, method in enumerate(METHODS):
        for col, (column, title) in enumerate(panels):
            ax = axes[row, col]
            for feature in FEATURES:
                part = criteria[(criteria["method"] == method) & (criteria["feature"] == feature)]
                base = feature.removesuffix("_f")
                ax.plot(part["k"], part[column], marker="o", color=colors[base],
                        linestyle="-" if feature.endswith("_f") else ":", label=feature)
            ax.set_title(f"{method}: {title}", fontsize=9)
            ax.set_xlabel("number of clusters k")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("solid = future description (leak-free), dotted = full-window description")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cluster_criteria.png", dpi=120)
    plt.close(fig)


def null_like(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Structureless control: Gaussian data with the same mean and covariance as X."""
    return rng.multivariate_normal(X.mean(axis=0), np.cov(X, rowvar=False), size=len(X),
                                   method="eigh")


def diagnostics(features: dict, h: pd.DataFrame, refits: pd.DataFrame, labels: dict,
                stability: dict) -> None:
    print("\n=== diagnostic 1: PCA of the shapes (components needed for 90% of the variance) ===")
    for name in ("shape", "shape_f"):
        ratios = PCA().fit(features[name]).explained_variance_ratio_
        need = int(np.searchsorted(np.cumsum(ratios), 0.9) + 1)
        print(f"{name}: first components explain {np.round(ratios[:4], 3).tolist()}, "
              f"{need} components for 90% (of {features[name].shape[1]})")

    print("\n=== diagnostic 2: real data vs structureless control (k-means, same space) ===")
    rng = np.random.default_rng(SEED)
    nulls = {f: null_like(features[f], rng) for f in FEATURES}
    null_refs = {(f"null_{f}", "kmeans", k): lab for f in FEATURES
                 for k, lab in cluster(nulls[f], "kmeans", KS).items()}
    tasks = [(f"null_{f}", "kmeans", b, nulls[f], null_refs) for f in FEATURES for b in range(N_NULL_BOOT)]
    null_stability = run_stability(tasks)
    rows = []
    for f in FEATURES:
        null_labels = {k: null_refs[(f"null_{f}", "kmeans", k)] for k in KS}
        for k in KS:
            rows.append(dict(
                feature=f, k=k,
                silhouette_real=silhouette_score(features[f], labels[(f, "kmeans")][k],
                                                 sample_size=SIL_SAMPLE, random_state=SEED),
                silhouette_null=silhouette_score(nulls[f], null_labels[k], sample_size=SIL_SAMPLE,
                                                 random_state=SEED),
                stability_real=np.mean(stability[(f, "kmeans", k)]),
                stability_null=np.mean(null_stability[(f"null_{f}", "kmeans", k)])))
    print(pd.DataFrame(rows).set_index(["feature", "k"]).round(3).to_string())

    print("\n=== diagnostic 3: parameters without games whose fit sits on a bound ===")
    for name, fits in (("params", h), ("params_f", refits)):
        eta, rho = fits["hawkes50_eta"].to_numpy(), fits["hawkes50_rho"].to_numpy()
        interior = ((eta > 1e-3) & (eta < ETA_CAP - 1e-3) &
                    (rho > fit_hawkes.RHO_BOUNDS[0] + 1e-3) & (rho < fit_hawkes.RHO_BOUNDS[1] - 1e-3))
        print(f"{name}: {interior.mean():.1%} of games are interior ({interior.sum():,})")
        X = features[name][interior]
        other = features[name.replace("params", "summary")][interior]
        rows = []
        for k in KS:
            lab = cluster(X, "kmeans", [k])[k]
            other_lab = cluster(other, "kmeans", [k])[k]
            boots = []
            for b in range(10):
                brng = np.random.default_rng(SEED + b)
                sample = np.unique(brng.integers(0, len(X), size=len(X)))
                boots.append(adjusted_rand_score(lab[sample],
                                                 cluster(X[sample], "kmeans", [k])[k]))
            rows.append(dict(k=k, stability=np.mean(boots),
                             ari_with_summary=adjusted_rand_score(lab, other_lab)))
        print(pd.DataFrame(rows).set_index("k").round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    series = pd.read_csv(PROCESSED / "weekly_series.csv")
    simple = pd.read_csv(PROCESSED / "fits_simple.csv")
    hawkes = pd.read_csv(PROCESSED / "fits_hawkes.csv")
    keep = simple[(simple["group"] == "main") & (simple["n_weeks"] == WINDOW)]["app_id"]
    keep = keep[keep.isin(hawkes.loc[hawkes["hawkes50_converged"].astype(bool), "app_id"])]
    app_ids = np.sort(keep.to_numpy())
    series = series[series["app_id"].isin(app_ids)]
    print(f"{len(app_ids):,} main games with the full {WINDOW}-week window and a converged hawkes50 fit")

    features, early, h, refits = build_features(series, hawkes, app_ids)
    ext = load_games(args.data_dir).reindex(app_ids)

    labels = {(f, m): cluster(features[f], m, KS) for f in FEATURES for m in METHODS}
    reference = {(f, m, k): labels[(f, m)][k] for f in FEATURES for m in METHODS for k in KS}

    print(f"bootstrap stability: {N_BOOT} resamples x {len(FEATURES)} descriptions x {len(METHODS)} methods")
    tasks = [(f, m, b, features[f], reference) for f in FEATURES for m in METHODS for b in range(N_BOOT)]
    stability = run_stability(tasks)

    rows = []
    for f in FEATURES:
        for m in METHODS:
            for k in KS:
                lab = labels[(f, m)][k]
                sizes = np.bincount(lab) / len(lab)
                row = dict(feature=f, method=m, k=k, leak_free=f.endswith("_f"),
                           smallest_cluster_share=sizes.min(),
                           stability_mean=np.mean(stability[(f, m, k)]),
                           stability_sd=np.std(stability[(f, m, k)]))
                row.update(external_association(lab, ext))
                row.update(predictability(early, lab))
                rows.append(row)
    criteria = pd.DataFrame(rows)
    criteria.to_csv(PROCESSED / "cluster_criteria.csv", index=False)

    agreement = []
    for m in METHODS:
        for k in KS:
            for i, a in enumerate(FEATURES):
                for b in FEATURES[i + 1:]:
                    agreement.append(dict(method=m, k=k, pair=f"{a} vs {b}",
                                          ari=adjusted_rand_score(labels[(a, m)][k], labels[(b, m)][k])))
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(PROCESSED / "cluster_agreement.csv", index=False)

    assignments = pd.DataFrame({"app_id": app_ids})
    for (f, m), per_k in labels.items():
        for k, lab in per_k.items():
            assignments[f"{f}_{m}_k{k}"] = lab
    assignments.to_csv(PROCESSED / "cluster_assignments.csv", index=False)
    plot_criteria(criteria)

    show = ["leak_free", "stability_mean", "macro_f1", "dummy_macro_f1", "tag_V", "ratio_eps2",
            "smallest_cluster_share"]
    for m in METHODS:
        print(f"\n== {m} ==")
        print(criteria[criteria["method"] == m].set_index(["feature", "k"])[show].round(3).to_string())
    print("\nagreement (ARI): future descriptions with each other, and each with its full-window version")
    wanted = ["shape_f vs summary_f", "shape_f vs params_f", "summary_f vs params_f",
              "shape vs shape_f", "summary vs summary_f", "params vs params_f"]
    print(agreement[agreement["pair"].isin(wanted)].pivot_table(
        index=["method", "k"], columns="pair", values="ari")[wanted].round(3).to_string())

    diagnostics(features, h, refits, labels, stability)


if __name__ == "__main__":
    main()
