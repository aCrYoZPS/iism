"""Read-only first look at raw datasets: size, structure, time range, missing values, app_id overlap.

Usage:
    python inspect_data.py --reviews path/to/reviews.csv --players path/to/players_dir_or_csv
Either argument may be omitted. Nothing is written to disk.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

CHUNK_ROWS = 1_000_000
ID_PATTERN = re.compile(r"^(app_?id|steam_?app_?id|game_?id)$", re.IGNORECASE)
TIME_PATTERN = re.compile(r"(timestamp|date|time)", re.IGNORECASE)


def find_csvs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    return [path]


def to_datetime(series: pd.Series) -> pd.Series:
    """Parse a column that is either Unix seconds or a date string."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def profile_csv(path: Path) -> set | None:
    """Stream the file in chunks; print the report; return the set of app ids if an id column exists."""
    size_mb = path.stat().st_size / 1024**2
    print(f"\n=== {path} ({size_mb:.1f} MB) ===")

    rows = 0
    nulls = None
    dtypes = None
    ids: set = set()
    id_col = None
    time_ranges: dict[str, list] = {}

    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False):
        rows += len(chunk)
        dtypes = dtypes if dtypes is not None else chunk.dtypes
        chunk_nulls = chunk.isna().sum()
        nulls = chunk_nulls if nulls is None else nulls + chunk_nulls

        if id_col is None:
            id_col = next((c for c in chunk.columns if ID_PATTERN.match(c)), None)
        if id_col is not None:
            ids.update(chunk[id_col].dropna().unique())

        for col in chunk.columns:
            if TIME_PATTERN.search(col):
                parsed = to_datetime(chunk[col]).dropna()
                if parsed.empty:
                    continue
                lo, hi = time_ranges.get(col, [parsed.min(), parsed.max()])
                time_ranges[col] = [min(lo, parsed.min()), max(hi, parsed.max())]

    print(f"rows: {rows:,}")
    print("columns (dtype, % missing):")
    for col in dtypes.index:
        print(f"  {col:35s} {str(dtypes[col]):10s} {100 * nulls[col] / rows:6.2f}%")

    if time_ranges:
        print("time columns (min .. max):")
        for col, (lo, hi) in time_ranges.items():
            print(f"  {col}: {lo} .. {hi}")
    else:
        print("time columns: none detected by name")

    if id_col:
        print(f"id column '{id_col}': {len(ids):,} unique")
        return ids
    print("id column: none detected (check filenames or columns manually)")
    return None


def profile_source(label: str, path: Path) -> set:
    print(f"\n##### {label} #####")
    files = find_csvs(path)
    print(f"{len(files)} csv file(s)")
    all_ids: set = set()
    # Many small per-game files would flood the output; show the first few in detail.
    for f in files[:5]:
        file_ids = profile_csv(f)
        if file_ids:
            all_ids |= file_ids
    if len(files) > 5:
        print(f"\n(+{len(files) - 5} more files not profiled; first 5 shown)")
    return all_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--players", type=Path)
    args = parser.parse_args()

    ids_by_source = {}
    if args.reviews:
        ids_by_source["reviews"] = profile_source("REVIEWS", args.reviews)
    if args.players:
        ids_by_source["players"] = profile_source("PLAYERS", args.players)

    if len(ids_by_source) == 2 and all(ids_by_source.values()):
        a, b = ids_by_source["reviews"], ids_by_source["players"]
        print("\n##### OVERLAP #####")
        print(f"reviews ids: {len(a):,}  players ids: {len(b):,}  in both: {len(a & b):,}")


if __name__ == "__main__":
    main()
