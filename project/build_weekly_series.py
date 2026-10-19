"""Build weekly review-count series per game from the Anton Kaggle dataset.

Reads games.csv and recommendations.csv (read-only), writes to processed/:
  games_groups.csv   one row per game with reviews: group, reason, first review date, review count
  weekly_series.csv  app_id, group, week, n_reviews, n_positive  (week 0 = week of first review)

Groups:
  main         first review 0..LATE_DAYS days after release
  censored     released before the review record starts, so the start of its life is not observed
Excluded (listed in games_groups.csv with a reason): first review before release,
first review more than LATE_DAYS days after release, fewer than MIN_REVIEWS reviews,
no catalog entry.

Usage:
    python build_weekly_series.py --data-dir datasets/kaggle/antonkozyriev-game-recommendations-on-steam
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_ROWS = 2_000_000


def first_pass(reviews_path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Per game: first review date and review count; also the last date in the file."""
    first = None
    counts = None
    last = pd.Timestamp.min
    for chunk in pd.read_csv(reviews_path, usecols=["app_id", "date"], parse_dates=["date"],
                             chunksize=CHUNK_ROWS):
        f = chunk.groupby("app_id")["date"].min()
        c = chunk["app_id"].value_counts()
        first = f if first is None else pd.concat([first, f]).groupby(level=0).min()
        counts = c if counts is None else counts.add(c, fill_value=0)
        last = max(last, chunk["date"].max())
    stats = pd.DataFrame({"first_review": first, "n_total": counts.astype("int64")})
    return stats, last


def assign_groups(games: pd.DataFrame, record_start: pd.Timestamp,
                  min_reviews: int, late_days: int) -> pd.DataFrame:
    """Apply the inclusion rules in order; the first rule that matches decides."""
    gap = (games["first_review"] - games["date_release"]).dt.days
    group = pd.Series("main", index=games.index)
    reason = pd.Series("first review within window of release", index=games.index)

    # Later assignments overwrite earlier ones, so list rules from least to most specific.
    late = gap > late_days
    group[late], reason[late] = "excluded", f"first review > {late_days} days after release"
    before = gap < 0
    group[before], reason[before] = "excluded", "first review before release"
    censored = games["date_release"] < record_start
    group[censored], reason[censored] = "censored", "released before review record starts"
    small = games["n_total"] < min_reviews
    group[small], reason[small] = "excluded", f"fewer than {min_reviews} reviews"
    nocat = games["date_release"].isna()
    group[nocat], reason[nocat] = "excluded", "no catalog entry"

    return games.assign(gap_days=gap, group=group, reason=reason)


def second_pass(reviews_path: Path, kept: pd.DataFrame, window_weeks: int) -> pd.DataFrame:
    """Weekly review and positive-review counts for kept games, weeks 0..window_weeks-1."""
    first_dates = kept["first_review"]
    parts = []
    for chunk in pd.read_csv(reviews_path, usecols=["app_id", "date", "is_recommended"],
                             parse_dates=["date"], chunksize=CHUNK_ROWS):
        chunk = chunk[chunk["app_id"].isin(first_dates.index)]
        week = (chunk["date"] - chunk["app_id"].map(first_dates)).dt.days // 7
        chunk = chunk.assign(week=week, positive=chunk["is_recommended"].astype(int))
        chunk = chunk[chunk["week"] < window_weeks]
        parts.append(chunk.groupby(["app_id", "week"]).agg(
            n_reviews=("positive", "size"), n_positive=("positive", "sum")))
    return pd.concat(parts).groupby(level=[0, 1]).sum()


def fill_missing_weeks(weekly: pd.DataFrame, kept: pd.DataFrame, record_end: pd.Timestamp,
                       window_weeks: int) -> pd.DataFrame:
    """Weeks with no reviews become explicit zeros, but only up to the end of observation."""
    observed = ((record_end - kept["first_review"]).dt.days // 7 + 1).clip(upper=window_weeks)
    grid = pd.MultiIndex.from_arrays(
        [np.repeat(observed.index, observed.values),
         np.concatenate([np.arange(n) for n in observed.values])],
        names=["app_id", "week"])
    full = weekly.reindex(grid, fill_value=0).reset_index()
    full.insert(1, "group", full["app_id"].map(kept["group"]))
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("processed"))
    parser.add_argument("--min-reviews", type=int, default=200)
    parser.add_argument("--late-days", type=int, default=30)
    parser.add_argument("--window-weeks", type=int, default=104)
    args = parser.parse_args()

    reviews_path = args.data_dir / "recommendations.csv"
    games = pd.read_csv(args.data_dir / "games.csv", usecols=["app_id", "date_release"],
                        parse_dates=["date_release"]).set_index("app_id")

    print("pass 1: first review date and review count per game")
    stats, record_end = first_pass(reviews_path)
    record_start = stats["first_review"].min()
    print(f"record: {record_start.date()} .. {record_end.date()}")

    table = stats.join(games, how="left")
    table = assign_groups(table, record_start, args.min_reviews, args.late_days)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(args.out_dir / "games_groups.csv", index=False)
    print(table.groupby(["group", "reason"]).size().to_string())

    kept = table[table["group"] != "excluded"]
    print(f"\npass 2: weekly counts for {len(kept):,} games")
    weekly = second_pass(reviews_path, kept, args.window_weeks)
    series = fill_missing_weeks(weekly, kept, record_end, args.window_weeks)
    series.to_csv(args.out_dir / "weekly_series.csv", index=False)
    print(f"weekly_series.csv: {len(series):,} rows")


if __name__ == "__main__":
    main()
