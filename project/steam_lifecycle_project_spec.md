# PROJECT SPEC: Steam Game Lifecycle Analysis

## 0. Meta
- Domain: statistical modeling (university course project, 1 week)
- Deliverable: analysis report + reproducible pipeline (notebooks/scripts) + optional interpretable model
- Language for code: Python (pandas, numpy, scipy, statsmodels, scikit-learn, lifelines, matplotlib)
- Non-goals: no deep learning, no black-box models as primary result, no scraping at scale beyond a small validation sample

## 1. Research Question
Primary: What types of post-release popularity trajectories exist among Steam games, and what factors correlate with long-term audience retention vs short-lived hype?

Secondary (optional model): Given only the first 3 months of activity, can we classify a game's likely lifecycle type / predict its 12-month retention ratio?

## 2. Data Sources

| Source | Content | Access | Notes |
|---|---|---|---|
| Steam Reviews Dataset 2021 (Kaggle: `najzeko/steam-reviews-2021`) | review timestamp, voted_up (bool), playtime_forever, votes_helpful, votes_funny, steam_purchase, received_for_free, written_during_early_access, app_id | CSV download | ~21M reviews, ~300 games; non-random game selection |
| Steam Games Dataset — player count history (Mendeley `ycy3sy3vj2`) | player_count time series (5min/1h resolution), price, discount, genre, developer/publisher | CSV/ZIP download | ~2000 apps, Dec 2017–Aug 2020, selected by popularity on a fixed date |
| Steam Web API `appreviews/<appid>` | live review pull for spot-checks | REST, `filter=recent`, `cursor` pagination | rate-limited; use only for small validation set |
| SteamDB / SteamCharts | manual cross-check of known peaks/events | web pages | not for bulk ingestion |

## 3. Unit of Analysis
One row = (game_id, month_since_release), `t = 0` at release month.

## 4. Target Schema (post-ETL)

```
game_id            str
game_name          str
release_date       date
month_since_release int        # t
genre              str
early_access        bool
avg_players         float
median_players       float
peak_players         float
review_count         int
positive_review_share float   # in [0,1]
median_playtime      float    # hours
helpful_votes_sum    int
price                float
discount_pct          float
major_update_flag    bool
dlc_release_flag      bool
```

Missing values are allowed; document coverage % per field before analysis.

## 5. ETL Pipeline

1. Load raw review CSV → parse Unix timestamp → derive `month_since_release` using game's `release_date`.
2. Load player-count time series → resample to monthly avg/median/peak per game.
3. Join on `game_id` (map via Steam appid); inner join for games present in both sources; keep outer-joined subset separately for review-only analysis.
4. Drop/flag games with < 6 months of observed history.
5. Cap analysis window at 24 months since release (configurable).
6. Output: `panel.csv` (long format, one row per game-month) — this is the single source of truth for all downstream steps.

## 6. Derived Metrics (compute in a `metrics.py` module)

```
retention_ratio(g)      = median(P_13..T) / median(P_1..P_3)
peak_retention(g)       = median(P_13..T) / max_t(P_t)
decay_lambda(g)          # OLS fit: log(P_t+1) = alpha - lambda*t + eps, on monotonic-decline segment
time_to_20pct(g)         # first t where P_t < 0.2*P_initial for 3 consecutive months
tail_area(g)             = sum_t(P_t)                     # cumulative activity
polarization(t)          = 2*p_t*(1-p_t), p_t = positive_review_share_t
peak_count(g)             # local maxima where P_t > 2*median(P_{t-3..t-1})
review_per_player(g,t)    = review_count_t / avg_players_t
```

All ratio metrics computed on log-scale inputs where right-skew is present; use median not mean for playtime and player counts.

## 7. Analysis Modules

### 7.1 EDA
- Time series plots (log y-axis): players, reviews, per game.
- Distribution plots: peak_players, retention_ratio, time_to_20pct.
- Scatter: log(reviews) vs log(players), by month.

### 7.2 Clustering (lifecycle typology)
- Input: normalized vector `x_g = P_g,1..24 / max_t(P_g,t)` (or `/ sum_t`).
- Methods: hierarchical (ward, correlation distance) and k-means; compare.
- Model selection: silhouette score, elbow, stability under bootstrap resampling.
- Output labels (expected, not enforced): hype / sustained / late-growth / comeback / low-stable.
- Deliverable: `cluster_assignments.csv` (game_id, cluster_id, cluster_label).

### 7.3 Group Comparisons
- Kruskal-Wallis / Mann-Whitney across genre and cluster groups on: retention_ratio, time_to_20pct, decay_lambda.
- Spearman correlation matrix: peak_players, review_count, median_playtime, retention_ratio.
- Report effect sizes and CIs (bootstrap), not just p-values.

### 7.4 Survival Analysis
- Event: activity drops below 20% of initial for 3 consecutive months (censored if never observed).
- Kaplan-Meier curves by genre / early_access flag.
- Log-rank test between groups.
- Optional: Cox PH regression with genre, early_access, initial_peak as covariates.

### 7.5 Regression (long-term retention drivers)
```
Y_g = log(1 + median_players_after_month12)
Y_g ~ log(1+peak_players_first3mo) + positive_review_share + log(1+median_playtime) + genre + early_access
```
- Fit OLS with robust SE; report coefficients, CIs, R².
- Sensitivity check: rerun with median-based target and with retention_ratio as target.

### 7.6 Event Study (updates/DLC/discounts)
- Window: [-30d, event, +30d].
- Metric: `delta_P = mean(P_after)/mean(P_before) - 1`.
- Aggregate delta_P by event type (update / DLC / discount / early-access-exit).
- No causal claims — report as observational association only.

### 7.7 (Optional) Early-Signal Predictive Model
- Features: metrics computed ONLY from months 0–2 (strict — no leakage from month ≥3).
- Target A (classification): cluster_label from 7.2.
- Target B (regression): retention_ratio (12mo/3mo).
- Models: multinomial logistic regression (primary, interpretable) + random forest (comparison only).
- Baseline to beat: naive extrapolation `month3_activity / month1_activity`.
- Metrics: macro-F1, balanced accuracy, confusion matrix (classification); MAE, RMSE, R² vs baseline (regression).
- Explicit leakage check step required before reporting results.

## 8. Hypotheses to Test

| ID | Statement | Test |
|---|---|---|
| H1 | High peak_players does not guarantee high retention_ratio | Spearman correlation, scatter |
| H2 | reviews and players correlate but imperfectly | Spearman, regression R² |
| H3 | Genre affects time_to_20pct | Kruskal-Wallis |
| H4 | Updates/DLC/discounts associate with activity spikes | Event study delta_P |
| H5 | Higher median_playtime associates with higher retention | Regression coefficient sign/CI |
| H6 | Sustained-interest cluster has lower review-score volatility than hype cluster | Compare polarization variance across clusters |

## 9. Known Limitations (must appear in final report)
- Review count ≠ player count; review propensity varies by genre/audience.
- Two main datasets cover different, non-fully-overlapping time windows; document overlap before joining.
- Mendeley dataset selects apps popular as of Dec 2017 — non-random sample.
- Event dates for updates/DLC may be incomplete; only use events with confirmed dates.
- All regression/event-study results are associational, not causal.
- Small per-genre sample sizes → report group sizes and CIs, not just point estimates.

## 10. Deliverables Checklist
- [ ] `panel.csv` — cleaned long-format dataset
- [ ] `metrics.py` — reusable metric functions
- [ ] EDA notebook/plots
- [ ] `cluster_assignments.csv` + cluster interpretation writeup
- [ ] Group comparison tables (genre × cluster)
- [ ] KM curves + log-rank results
- [ ] Regression table (coefficients, CI, R²)
- [ ] Event-study summary table
- [ ] (Optional) predictive model results + leakage check documented
- [ ] Final report: intro, data, methods, results, limitations, conclusion

## 11. 7-Day Execution Plan

| Day | Task |
|---|---|
| 1 | Acquire data, inspect schemas, build ETL to `panel.csv` |
| 2 | Compute derived metrics, coverage checks |
| 3 | EDA plots, sanity-check outliers |
| 4 | Clustering + label interpretation |
| 5 | Hypothesis tests + regression |
| 6 | Survival analysis + event study; optional predictive model |
| 7 | Report writeup, limitations, final plots |

## 12. Agent Instructions Summary
When implementing with a coding agent:
1. Build ETL first and validate `panel.csv` schema against Section 4 before any analysis.
2. Implement Section 6 metrics as pure, testable functions — one function per metric, unit-tested on a synthetic toy series.
3. Each analysis module (7.1–7.7) should be a standalone script/notebook that reads only from `panel.csv` / `cluster_assignments.csv` — no hidden cross-dependencies.
4. Enforce the leakage rule in 7.7 explicitly in code (assert no month ≥ 3 columns used as features).
5. All plots use log-scale y-axis for player/review counts unless explicitly comparing shares/ratios.
