"""
Create a representative sample of NYC Yellow Taxi trips.

The NYC TLC monthly parquet endpoint is currently protected by CloudFront bot
checks in some environments. This script uses the official NYC Open Data
yearly Yellow Taxi datasets (Socrata API) to build a stratified sample.

Sampling design:
1) Discover available yearly Yellow Taxi datasets.
2) Allocate rows proportionally by year.
3) Allocate rows proportionally by month within each year.
4) Pull rows from randomized pickup-time windows inside each month.

Usage:
    python src/data/sample_yellow_data.py --start-year 2009 --end-year 2025 \
        --sample-size 24000 --output-dir data/samples
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


CATALOG_URL = "https://api.us.socrata.com/api/catalog/v1"
RESOURCE_URL_TEMPLATE = "https://data.cityofnewyork.us/resource/{dataset_id}.json"
VIEWS_URL_TEMPLATE = "https://data.cityofnewyork.us/api/views/{dataset_id}.json"

# Known official yearly yellow taxi datasets from NYC Open Data.
KNOWN_YELLOW_DATASET_IDS = {
    2009: "f9tw-8p66",
    2010: "74wj-s5ij",
    2011: "uwyp-dntv",
    2012: "kerk-3eby",
    2013: "t7ny-aygi",
    2014: "gkne-dk5s",
    2015: "2yzn-sicd",
    2016: "uacg-pexx",
    2017: "biws-g3hs",
    2018: "t29m-gskq",
    2019: "2upf-qytp",
    2020: "kxp8-n2sj",
    2021: "m6nq-qud6",
    2022: "qp3b-zxtp",
    2023: "4b4i-vvec",
}

DESIRED_COLUMNS = [
    "vendorid",
    "tpep_pickup_datetime",
    "pickup_datetime",
    "tpep_dropoff_datetime",
    "dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "ratecodeid",
    "store_and_fwd_flag",
    "pulocationid",
    "dolocationid",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_location",
    "dropoff_location",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "total_amount",
]

NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "ratecodeid",
    "pulocationid",
    "dolocationid",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "total_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]


@dataclass
class YearDataset:
    year: int
    dataset_id: str
    row_count: int
    month_counts: Dict[int, int]
    available_columns: List[str]
    pickup_datetime_column: str
    dropoff_datetime_column: Optional[str]


def request_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, str]] = None,
    retries: int = 5,
    timeout_seconds: int = 120,
) -> List[Dict]:
    """Request JSON from Socrata with retries for transient failures."""
    wait_seconds = 1.0

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)

            if response.status_code == 200:
                return response.json()

            if response.status_code in {429, 500, 502, 503, 504}:
                logger.warning(
                    "Transient status %s for %s (attempt %s/%s). Retrying in %.1fs.",
                    response.status_code,
                    url,
                    attempt,
                    retries,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                wait_seconds *= 2
                continue

            response.raise_for_status()

        except requests.RequestException as exc:
            if attempt == retries:
                raise
            logger.warning(
                "Request error for %s (attempt %s/%s): %s. Retrying in %.1fs.",
                url,
                attempt,
                retries,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2

    raise RuntimeError(f"Failed to request URL after retries: {url}")


def discover_dataset_id(session: requests.Session, year: int) -> Optional[str]:
    """Find the yearly Yellow Taxi dataset ID for a given year."""
    if year in KNOWN_YELLOW_DATASET_IDS:
        return KNOWN_YELLOW_DATASET_IDS[year]

    dataset_name = f"{year} Yellow Taxi Trip Data"
    params = {
        "q": dataset_name,
        "domains": "data.cityofnewyork.us",
        "limit": "50",
    }
    results = request_json(session, CATALOG_URL, params=params)

    for item in results.get("results", []):
        name = item.get("resource", {}).get("name", "").strip()
        if name.lower() == dataset_name.lower():
            return item.get("resource", {}).get("id")

    return None


def get_available_columns(session: requests.Session, dataset_id: str) -> List[str]:
    """Fetch field names for a Socrata dataset."""
    url = VIEWS_URL_TEMPLATE.format(dataset_id=dataset_id)
    metadata = request_json(session, url)
    return [c.get("fieldName") for c in metadata.get("columns", []) if c.get("fieldName")]


def get_row_count(session: requests.Session, dataset_id: str) -> int:
    """Fetch total row count for a dataset."""
    url = RESOURCE_URL_TEMPLATE.format(dataset_id=dataset_id)
    rows = request_json(session, url, params={"$select": "count(*)"})
    return int(rows[0]["count"])


def get_month_counts(
    session: requests.Session,
    dataset_id: str,
    pickup_datetime_column: str,
) -> Dict[int, int]:
    """Fetch row counts by pickup month for a yearly dataset."""
    url = RESOURCE_URL_TEMPLATE.format(dataset_id=dataset_id)
    params = {
        "$select": f"date_extract_m({pickup_datetime_column}) as pickup_month, count(*) as n",
        "$where": f"{pickup_datetime_column} is not null",
        "$group": "pickup_month",
        "$order": "pickup_month",
    }
    rows = request_json(session, url, params=params)

    month_counts: Dict[int, int] = {}
    for row in rows:
        month = int(row["pickup_month"])
        month_counts[month] = int(row["n"])

    return month_counts


def build_uniform_month_counts(total_rows: int) -> Dict[int, int]:
    """Build a uniform month-count proxy when monthly aggregates are unavailable."""
    estimated_per_month = max(1, total_rows // 12)
    return {month: estimated_per_month for month in range(1, 13)}


def allocate_proportional(
    total_rows: int,
    counts: Dict[int, int],
    min_per_group: int = 0,
) -> Dict[int, int]:
    """Allocate integer targets proportionally to counts."""
    positive_keys = [k for k, v in counts.items() if v > 0]
    allocation = {k: 0 for k in counts}

    if total_rows <= 0 or not positive_keys:
        return allocation

    if min_per_group > 0 and total_rows >= len(positive_keys) * min_per_group:
        for key in positive_keys:
            allocation[key] = min_per_group
        total_rows -= len(positive_keys) * min_per_group

    if total_rows == 0:
        return allocation

    total_count = sum(counts[key] for key in positive_keys)
    raw = {
        key: total_rows * counts[key] / total_count
        for key in positive_keys
    }
    floor_alloc = {key: int(math.floor(value)) for key, value in raw.items()}
    for key, value in floor_alloc.items():
        allocation[key] += value

    remainder = total_rows - sum(floor_alloc.values())
    remainders = sorted(
        positive_keys,
        key=lambda key: raw[key] - floor_alloc[key],
        reverse=True,
    )
    for key in remainders[:remainder]:
        allocation[key] += 1

    return allocation


def detect_datetime_columns(columns: List[str]) -> Optional[tuple[str, Optional[str]]]:
    """Detect pickup/dropoff datetime column names for a dataset schema."""
    pickup_candidates = ["tpep_pickup_datetime", "pickup_datetime"]
    dropoff_candidates = ["tpep_dropoff_datetime", "dropoff_datetime"]

    pickup_column = next((c for c in pickup_candidates if c in columns), None)
    if not pickup_column:
        return None

    dropoff_column = next((c for c in dropoff_candidates if c in columns), None)
    return pickup_column, dropoff_column


def build_time_window_where(
    pickup_datetime_column: str,
    start_dt: datetime,
    end_dt: datetime,
) -> str:
    """Build Socrata where-clause for a pickup datetime window."""
    return (
        f"{pickup_datetime_column} >= '{start_dt:%Y-%m-%dT%H:%M:%S}' "
        f"and {pickup_datetime_column} < '{end_dt:%Y-%m-%dT%H:%M:%S}'"
    )


def fetch_month_sample(
    session: requests.Session,
    dataset_id: str,
    year: int,
    month: int,
    target_rows: int,
    month_row_count: int,
    selected_columns: List[str],
    pickup_datetime_column: str,
    rng: random.Random,
) -> pd.DataFrame:
    """Fetch a randomized month-level sample using pickup-time windows."""
    if target_rows <= 0:
        return pd.DataFrame()

    url = RESOURCE_URL_TEMPLATE.format(dataset_id=dataset_id)
    ####
    month_where = f"date_extract_m({pickup_datetime_column}) = {month}"
    selected = [col for col in selected_columns if col]
    select_expr = ":id as socrata_row_id"
    if selected:
        select_expr += "," + ",".join(selected)

    # days_in_month = monthrange(year, month)[1]
    # chunk_size = min(600, max(75, target_rows // 2))
    # max_attempts = max(8, int(math.ceil(target_rows / chunk_size) * 8))

    # collected_rows: List[Dict] = []
    # seen_ids = set()

    collected_rows: List[Dict] = []
    seen_ids = set()

    # Large samples are much faster via randomized offset batches than many
    # small time-window calls.
    if target_rows >= 5000:
        batch_limit = min(10000, max(5000, target_rows))
        max_batches = max(4, int(math.ceil(target_rows / batch_limit) * 5))

        for _ in range(max_batches):
            if len(collected_rows) >= target_rows:
                break

            offset_max = max(0, month_row_count - batch_limit)
            offset = rng.randint(0, offset_max) if offset_max > 0 else 0
            params = {
                "$select": select_expr,
                "$where": month_where,
                "$order": ":id",
                "$limit": str(batch_limit),
                "$offset": str(offset),
            }

            try:
                rows = request_json(session, url, params=params)
            except requests.RequestException as exc:
                logger.warning(
                    "Skipping failed offset batch for %s %s-%02d: %s",
                    dataset_id,
                    year,
                    month,
                    exc,
                )
                continue

            if not rows:
                continue

            rng.shuffle(rows)
            for row in rows:
                row_id = row.get("socrata_row_id")
                if row_id and row_id in seen_ids:
                    continue
                if row_id:
                    seen_ids.add(row_id)
                collected_rows.append(row)
                if len(collected_rows) >= target_rows:
                    break

    days_in_month = monthrange(year, month)[1]
    chunk_size = min(1200, max(150, target_rows // 2))
    max_attempts = max(10, int(math.ceil(target_rows / chunk_size) * 6))

    attempts = 0
    while len(collected_rows) < target_rows and attempts < max_attempts:
        attempts += 1
        day = rng.randint(1, days_in_month)
        hour = rng.randint(0, 23)
        window_hours = rng.choice([1, 2, 3, 4, 6])

        start_dt = datetime(year, month, day, hour, 0, 0)
        end_dt = start_dt + timedelta(hours=window_hours)
        where_clause = build_time_window_where(
            pickup_datetime_column=pickup_datetime_column,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        limit = min(chunk_size, target_rows - len(collected_rows) + 50)
        params = {
            "$select": select_expr,
            "$where": where_clause,
            "$order": ":id",
            "$limit": str(limit),
        }

        try:
            rows = request_json(session, url, params=params)
        except requests.RequestException as exc:
            logger.warning(
                "Skipping failed window request for %s %s-%02d: %s",
                dataset_id,
                year,
                month,
                exc,
            )
            continue

        if not rows:
            continue

        rng.shuffle(rows)
        for row in rows:
            row_id = row.get("socrata_row_id")
            if row_id and row_id in seen_ids:
                continue
            if row_id:
                seen_ids.add(row_id)
            collected_rows.append(row)
            if len(collected_rows) >= target_rows:
                break

    missing = target_rows - len(collected_rows)
    if missing > 0 and month_row_count > 0:
        logger.info(
            "Month %s-%02d missing %s rows after random windows; using offset fallback.",
            year,
            month,
            missing,
        )
        offset_max = max(0, month_row_count - min(5000, month_row_count))
        offset = rng.randint(0, offset_max) if offset_max > 0 else 0
        month_where = f"date_extract_m({pickup_datetime_column}) = {month}"
        params = {
            "$select": select_expr,
            "$where": month_where,
            "$order": ":id",
            "$limit": str(min(5000, missing * 3)),
            "$offset": str(offset),
        }
        try:
            rows = request_json(session, url, params=params)
            rng.shuffle(rows)
            for row in rows:
                row_id = row.get("socrata_row_id")
                if row_id and row_id in seen_ids:
                    continue
                if row_id:
                    seen_ids.add(row_id)
                collected_rows.append(row)
                if len(collected_rows) >= target_rows:
                    break
        except requests.RequestException as exc:
            logger.warning(
                "Offset fallback failed for %s %s-%02d: %s",
                dataset_id,
                year,
                month,
                exc,
            )

    if not collected_rows:
        return pd.DataFrame()

    sampled = pd.DataFrame(collected_rows).head(target_rows).copy()
    sampled["source_year"] = year
    sampled["source_month"] = month
    sampled["source_dataset_id"] = dataset_id
    return sampled


def normalize_and_derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema differences and add derived analytical features."""
    if df.empty:
        return df

    rename_map = {
        "pickup_datetime": "tpep_pickup_datetime",
        "dropoff_datetime": "tpep_dropoff_datetime",
        "vendorid": "vendor_id",
        "ratecodeid": "rate_code_id",
        "pulocationid": "pickup_location_id",
        "dolocationid": "dropoff_location_id",
    }
    df = df.rename(columns=rename_map)

    if "tpep_pickup_datetime" in df.columns:
        df["tpep_pickup_datetime"] = pd.to_datetime(
            df["tpep_pickup_datetime"],
            errors="coerce",
        )
    if "tpep_dropoff_datetime" in df.columns:
        df["tpep_dropoff_datetime"] = pd.to_datetime(
            df["tpep_dropoff_datetime"],
            errors="coerce",
        )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "tpep_pickup_datetime" in df.columns:
        df = df[df["tpep_pickup_datetime"].notna()].copy()

        df["pickup_year"] = df["tpep_pickup_datetime"].dt.year
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
        df["pickup_day"] = df["tpep_pickup_datetime"].dt.day
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["is_weekend"] = df["pickup_dayofweek"] >= 5

        df["time_of_day"] = pd.cut(
            df["pickup_hour"],
            bins=[-1, 5, 11, 17, 23],
            labels=["night", "morning", "afternoon", "evening"],
        )

        season_map = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "fall",
            10: "fall",
            11: "fall",
        }
        df["season"] = df["pickup_month"].map(season_map)

    if {"tpep_pickup_datetime", "tpep_dropoff_datetime"}.issubset(df.columns):
        duration = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds()
        df["trip_duration_seconds"] = duration
        df["trip_duration_minutes"] = duration / 60.0

        valid_duration = duration > 0
        if "trip_distance" in df.columns:
            df["speed_mph"] = None
            df.loc[valid_duration, "speed_mph"] = (
                df.loc[valid_duration, "trip_distance"]
                / (duration[valid_duration] / 3600.0)
            )

    df["vehicle_type"] = "yellow"
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a representative sample from NYC Yellow Taxi data.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year to consider (default: 2021).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year - 2,
        help="End year to consider (default: previous calendar year).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=24000,
        help="Target number of sampled rows (default: 24000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/samples",
        help="Output directory for sample parquet and metadata (default: data/samples).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="yellow_taxi_representative_sample.parquet",
        help="Output parquet filename (default: yellow_taxi_representative_sample.parquet).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--month-allocation",
        type=str,
        default="uniform",
        choices=["uniform", "observed"],
        help=(
            "How to allocate rows across months within each year. "
            "'uniform' is more reliable; 'observed' queries month counts from API."
        ),
    )
    parser.add_argument(
        "--app-token",
        type=str,
        default=os.getenv("SOCRATA_APP_TOKEN"),
        help="Optional Socrata app token (or set SOCRATA_APP_TOKEN).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    session = requests.Session()
    if args.app_token:
        session.headers.update({"X-App-Token": args.app_token})

    year_datasets: List[YearDataset] = []
    missing_years: List[int] = []

    for year in range(args.start_year, args.end_year + 1):
        dataset_id = discover_dataset_id(session, year)
        if not dataset_id:
            missing_years.append(year)
            continue

        logger.info("Preparing dataset for %s (%s)", year, dataset_id)
        available_columns = get_available_columns(session, dataset_id)
        datetime_columns = detect_datetime_columns(available_columns)
        if not datetime_columns:
            logger.warning(
                "Skipping year %s (%s): no pickup datetime column found.",
                year,
                dataset_id,
            )
            missing_years.append(year)
            continue
        pickup_datetime_column, dropoff_datetime_column = datetime_columns
        try:
            row_count = get_row_count(session, dataset_id)
        except requests.RequestException as exc:
            logger.warning(
                "Skipping year %s (%s): failed to get row count (%s).",
                year,
                dataset_id,
                exc,
            )
            missing_years.append(year)
            continue

        if args.month_allocation == "uniform":
            month_counts = build_uniform_month_counts(row_count)
        else:
            try:
                month_counts = get_month_counts(
                    session,
                    dataset_id,
                    pickup_datetime_column=pickup_datetime_column,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Month-count query failed for %s (%s): %s. Falling back to uniform month allocation.",
                    year,
                    dataset_id,
                    exc,
                )
                month_counts = build_uniform_month_counts(row_count)

        year_datasets.append(
            YearDataset(
                year=year,
                dataset_id=dataset_id,
                row_count=row_count,
                month_counts=month_counts,
                available_columns=available_columns,
                pickup_datetime_column=pickup_datetime_column,
                dropoff_datetime_column=dropoff_datetime_column,
            )
        )

    if not year_datasets:
        raise RuntimeError("No yearly yellow taxi datasets were discovered.")

    if missing_years:
        logger.warning(
            "No yearly yellow taxi dataset found for years: %s",
            ", ".join(map(str, missing_years)),
        )

    year_counts = {item.year: item.row_count for item in year_datasets}
    year_targets = allocate_proportional(
        total_rows=args.sample_size,
        counts=year_counts,
        min_per_group=1,
    )

    frames: List[pd.DataFrame] = []
    allocation_summary = []

    for item in year_datasets:
        year_target = year_targets.get(item.year, 0)
        if year_target <= 0:
            continue

        month_targets = allocate_proportional(
            total_rows=year_target,
            counts=item.month_counts,
            min_per_group=1,
        )

        selected_columns = [c for c in DESIRED_COLUMNS if c in item.available_columns]

        for month in sorted(item.month_counts):
            month_target = month_targets.get(month, 0)
            if month_target <= 0:
                continue

            logger.info(
                "Sampling %s rows for %s-%02d from %s",
                month_target,
                item.year,
                month,
                item.dataset_id,
            )
            sampled = fetch_month_sample(
                session=session,
                dataset_id=item.dataset_id,
                year=item.year,
                month=month,
                target_rows=month_target,
                month_row_count=item.month_counts[month],
                selected_columns=selected_columns,
                pickup_datetime_column=item.pickup_datetime_column,
                rng=rng,
            )

            if not sampled.empty:
                frames.append(sampled)

            allocation_summary.append(
                {
                    "year": item.year,
                    "month": month,
                    "dataset_id": item.dataset_id,
                    "population_rows": item.month_counts[month],
                    "target_sample_rows": month_target,
                    "actual_sample_rows": int(len(sampled)),
                }
            )

    if not frames:
        raise RuntimeError("Sampling completed but no rows were fetched.")

    sample_df = pd.concat(frames, ignore_index=True)

    # De-duplicate just in case overlapping windows produced repeated records.
    dedupe_keys = ["source_dataset_id", "socrata_row_id"]
    if set(dedupe_keys).issubset(sample_df.columns):
        sample_df = sample_df.drop_duplicates(subset=dedupe_keys, keep="first")

    sample_df = normalize_and_derive_features(sample_df)

    if len(sample_df) > args.sample_size:
        sample_df = sample_df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file
    summary_json_path = output_dir / output_path.with_suffix("").name
    summary_json_path = summary_json_path.with_name(summary_json_path.name + "_summary.json")
    summary_csv_path = output_dir / output_path.with_suffix("").name
    summary_csv_path = summary_csv_path.with_name(summary_csv_path.name + "_distribution.csv")

    sample_df.to_parquet(output_path, index=False)

    distribution = (
        sample_df.groupby(["pickup_year", "pickup_month"])
        .size()
        .reset_index(name="sample_rows")
        .sort_values(["pickup_year", "pickup_month"])
    )
    distribution.to_csv(summary_csv_path, index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sampling_method": "stratified_by_year_and_month_random_time_windows",
        "requested_year_range": [args.start_year, args.end_year],
        "missing_years": missing_years,
        "available_years": [item.year for item in year_datasets],
        "requested_sample_size": args.sample_size,
        "actual_sample_size": int(len(sample_df)),
        "output_parquet": str(output_path),
        "output_distribution_csv": str(summary_csv_path),
        "allocation_details": allocation_summary,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved sample parquet: %s", output_path)
    logger.info("Saved sample distribution: %s", summary_csv_path)
    logger.info("Saved sampling summary: %s", summary_json_path)
    logger.info("Done. Final sampled rows: %s", len(sample_df))


if __name__ == "__main__":
    main()
