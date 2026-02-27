"""
Borough-level yellow taxi demand forecasting (explainable model).

Model choice:
- Per-borough Poisson GLM with log link (interpretable coefficients).
- Hourly demand target.
- Recursive forecasting for the next N hours (default: 168 = 7 days).

Data scope:
- Uses all processed parquet files that match: data/processed/yellow_tripdata_*.parquet
- Builds borough demand via taxi zone lookup mapping.

Outputs:
- reports/forecasts/borough_hourly_demand_timeseries.parquet
- reports/forecasts/borough_7day_hourly_forecast.csv
- reports/forecasts/borough_validation_metrics.csv
- reports/forecasts/borough_model_coefficients.csv
- reports/forecasts/borough_forecast_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "log_lag_1",
    "log_lag_24",
    "log_lag_168",
    "log_roll_mean_24",
    "log_roll_mean_168",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train explainable borough demand forecasting models.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="data/processed/yellow_tripdata_*.parquet",
        help="Glob pattern for processed input files.",
    )
    parser.add_argument(
        "--zone-lookup",
        type=str,
        default="data/external/taxi_zone_lookup.csv",
        help="Path to taxi zone lookup CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/forecasts",
        help="Output directory for forecasts and model artifacts.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=168,
        help="Forecast horizon in hours (default: 168 = 7 days).",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=24 * 30,
        help="Minimum training rows required per borough.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive datetime filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional exclusive datetime filter (YYYY-MM-DD).",
    )
    return parser.parse_args()


def load_borough_lookup(path: Path) -> Dict[int, str]:
    lookup_df = pd.read_csv(path, usecols=["LocationID", "Borough"])
    lookup_df["LocationID"] = pd.to_numeric(lookup_df["LocationID"], errors="coerce").astype("Int64")
    lookup_df = lookup_df.dropna(subset=["LocationID"])
    lookup_df["Borough"] = lookup_df["Borough"].fillna("Unknown")
    return dict(zip(lookup_df["LocationID"].astype(int), lookup_df["Borough"]))


def aggregate_hourly_borough_demand(
    files: List[Path],
    location_to_borough: Dict[int, str],
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    grouped_frames: List[pd.DataFrame] = []

    for file_path in tqdm(files, desc="Aggregating parquet files"):
        try:
            df = pd.read_parquet(file_path, columns=["pickup_datetime", "pickup_location_id"])
        except Exception as exc:
            logger.warning("Skipping %s due to read error: %s", file_path.name, exc)
            continue

        if df.empty:
            continue

        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df = df.dropna(subset=["pickup_datetime"])

        if start_ts is not None:
            df = df[df["pickup_datetime"] >= start_ts]
        if end_ts is not None:
            df = df[df["pickup_datetime"] < end_ts]

        if df.empty:
            continue

        df["pickup_hour"] = df["pickup_datetime"].dt.floor("h")
        location_id = pd.to_numeric(df["pickup_location_id"], errors="coerce").astype("Int64")
        df["borough"] = location_id.map(location_to_borough).fillna("Unknown")

        grouped = (
            df.groupby(["pickup_hour", "borough"], as_index=False)
            .size()
            .rename(columns={"pickup_hour": "ds", "size": "trips"})
        )
        grouped_frames.append(grouped)

    if not grouped_frames:
        raise RuntimeError("No data could be aggregated from processed parquet files.")

    combined = pd.concat(grouped_frames, ignore_index=True)
    combined = combined.groupby(["ds", "borough"], as_index=False)["trips"].sum()
    combined = combined.sort_values(["borough", "ds"]).reset_index(drop=True)
    return combined


def make_complete_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    min_ts = hourly["ds"].min()
    max_ts = hourly["ds"].max()
    all_hours = pd.date_range(min_ts, max_ts, freq="h")
    boroughs = sorted(hourly["borough"].unique())

    panels = []
    for borough in boroughs:
        borough_df = hourly[hourly["borough"] == borough][["ds", "trips"]].copy()
        full = pd.DataFrame({"ds": all_hours})
        full["borough"] = borough
        full = full.merge(borough_df, on="ds", how="left")
        full["trips"] = full["trips"].fillna(0.0)
        panels.append(full)

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["borough", "ds"]).reset_index(drop=True)
    return panel


def infer_datetime_bounds_from_filenames(files: List[Path]) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Infer datetime bounds from yellow_tripdata_YYYY-MM filenames.

    Example:
    - min file: yellow_tripdata_2024-01.parquet -> start 2024-01-01
    - max file: yellow_tripdata_2025-11.parquet -> end 2025-12-01 (exclusive)
    """
    year_month = []
    pattern = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})", re.IGNORECASE)

    for file_path in files:
        match = pattern.search(file_path.name)
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        year_month.append((year, month))

    if not year_month:
        return None, None

    start_year, start_month = min(year_month)
    end_year, end_month = max(year_month)

    start_ts = pd.Timestamp(year=start_year, month=start_month, day=1)
    end_month_next = end_month + 1
    end_year_next = end_year
    if end_month_next == 13:
        end_month_next = 1
        end_year_next += 1
    end_ts = pd.Timestamp(year=end_year_next, month=end_month_next, day=1)
    return start_ts, end_ts


def get_holiday_dates(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DatetimeIndex:
    calendar = USFederalHolidayCalendar()
    return calendar.holidays(
        start=start_ts.normalize() - pd.Timedelta(days=14),
        end=end_ts.normalize() + pd.Timedelta(days=14),
    )


def add_time_and_lag_features(df: pd.DataFrame, holiday_dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = df.copy()
    ts = out["ds"]

    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    out["is_weekend"] = (dow >= 5).astype(int)
    out["is_holiday"] = ts.dt.normalize().isin(holiday_dates).astype(int)

    out["lag_1"] = out["trips"].shift(1)
    out["lag_24"] = out["trips"].shift(24)
    out["lag_168"] = out["trips"].shift(168)
    out["roll_mean_24"] = out["trips"].shift(1).rolling(24, min_periods=24).mean()
    out["roll_mean_168"] = out["trips"].shift(1).rolling(168, min_periods=168).mean()

    out["log_lag_1"] = np.log1p(out["lag_1"])
    out["log_lag_24"] = np.log1p(out["lag_24"])
    out["log_lag_168"] = np.log1p(out["lag_168"])
    out["log_roll_mean_24"] = np.log1p(out["roll_mean_24"])
    out["log_roll_mean_168"] = np.log1p(out["roll_mean_168"])

    out = out.dropna(subset=FEATURE_COLUMNS + ["trips"])
    return out


def error_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    wape = float(np.sum(np.abs(actual - pred)) / max(np.sum(actual), 1.0) * 100.0)
    mape = float(np.mean(np.abs(actual - pred) / np.maximum(actual, 1.0)) * 100.0)
    return {"mae": mae, "rmse": rmse, "wape_pct": wape, "mape_pct": mape}


def train_borough_model(
    borough_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
    horizon_hours: int,
    min_train_rows: int,
) -> Tuple[sm.GLM, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    feature_df = add_time_and_lag_features(borough_df, holiday_dates)
    if feature_df.empty:
        raise RuntimeError("No feature rows available after lag construction.")

    val_size = min(horizon_hours, max(24, len(feature_df) // 10))
    split_idx = len(feature_df) - val_size
    train_df = feature_df.iloc[:split_idx].copy()
    val_df = feature_df.iloc[split_idx:].copy()

    if len(train_df) < min_train_rows:
        raise RuntimeError(
            f"Insufficient train rows ({len(train_df)}), required at least {min_train_rows}."
        )

    x_train = sm.add_constant(train_df[FEATURE_COLUMNS], has_constant="add")
    y_train = train_df["trips"]
    glm = sm.GLM(y_train, x_train, family=sm.families.Poisson())
    result = glm.fit(maxiter=300, disp=0)

    x_val = sm.add_constant(val_df[FEATURE_COLUMNS], has_constant="add")
    val_pred = result.predict(x_val).clip(lower=0.0)

    baseline_pred = val_df["lag_24"].clip(lower=0.0)
    model_metrics = error_metrics(val_df["trips"].values, val_pred.values)
    baseline_metrics = error_metrics(val_df["trips"].values, baseline_pred.values)

    metrics = {
        "model_mae": model_metrics["mae"],
        "model_rmse": model_metrics["rmse"],
        "model_wape_pct": model_metrics["wape_pct"],
        "model_mape_pct": model_metrics["mape_pct"],
        "naive24_mae": baseline_metrics["mae"],
        "naive24_rmse": baseline_metrics["rmse"],
        "naive24_wape_pct": baseline_metrics["wape_pct"],
        "naive24_mape_pct": baseline_metrics["mape_pct"],
    }

    return result, feature_df, val_df, metrics


def build_single_feature_row(
    ts: pd.Timestamp,
    history_trips: pd.Series,
    holiday_dates: pd.DatetimeIndex,
) -> Dict[str, float]:
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month

    lag_1 = float(history_trips.iloc[-1])
    lag_24 = float(history_trips.iloc[-24])
    lag_168 = float(history_trips.iloc[-168])
    roll_mean_24 = float(history_trips.iloc[-24:].mean())
    roll_mean_168 = float(history_trips.iloc[-168:].mean())

    return {
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
        "is_weekend": int(dow >= 5),
        "is_holiday": int(ts.normalize() in holiday_dates),
        "log_lag_1": float(np.log1p(lag_1)),
        "log_lag_24": float(np.log1p(lag_24)),
        "log_lag_168": float(np.log1p(lag_168)),
        "log_roll_mean_24": float(np.log1p(roll_mean_24)),
        "log_roll_mean_168": float(np.log1p(roll_mean_168)),
    }


def recursive_forecast(
    result: sm.GLM,
    borough_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
    horizon_hours: int,
    borough: str,
) -> pd.DataFrame:
    history = borough_df[["ds", "trips"]].sort_values("ds").copy()
    history["trips"] = history["trips"].astype(float)

    if len(history) < 168:
        raise RuntimeError(f"Not enough history for recursive lags in borough {borough}.")

    rows = []
    for _ in range(horizon_hours):
        next_ts = history["ds"].iloc[-1] + pd.Timedelta(hours=1)
        row_features = build_single_feature_row(next_ts, history["trips"], holiday_dates)
        x = pd.DataFrame([row_features])
        x = sm.add_constant(x, has_constant="add")
        pred = float(result.predict(x).iloc[0])
        if not np.isfinite(pred):
            pred = float(history["trips"].iloc[-24])
        pred = max(0.0, pred)

        lower_95 = max(0.0, pred - 1.96 * np.sqrt(max(pred, 1e-6)))
        upper_95 = pred + 1.96 * np.sqrt(max(pred, 1e-6))

        rows.append(
            {
                "borough": borough,
                "ds": next_ts,
                "predicted_trips": pred,
                "lower_95": lower_95,
                "upper_95": upper_95,
            }
        )

        history = pd.concat(
            [history, pd.DataFrame([{"ds": next_ts, "trips": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


def coefficients_to_frame(result: sm.GLM, borough: str) -> pd.DataFrame:
    coef = result.params.reset_index()
    coef.columns = ["feature", "coefficient"]
    coef["borough"] = borough
    coef["rate_ratio_exp_coef"] = np.exp(coef["coefficient"])
    coef["abs_coefficient"] = coef["coefficient"].abs()
    return coef.sort_values("abs_coefficient", ascending=False)


def main() -> None:
    args = parse_args()

    input_files = sorted(Path().glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(f"No files matched input glob: {args.input_glob}")

    logger.info("Using %s processed files for training.", len(input_files))

    inferred_start, inferred_end = infer_datetime_bounds_from_filenames(input_files)
    start_ts = pd.to_datetime(args.start_date) if args.start_date else inferred_start
    end_ts = pd.to_datetime(args.end_date) if args.end_date else inferred_end
    if start_ts is not None or end_ts is not None:
        logger.info("Datetime filter applied: start=%s end=%s", start_ts, end_ts)

    borough_lookup = load_borough_lookup(Path(args.zone_lookup))
    hourly = aggregate_hourly_borough_demand(
        files=input_files,
        location_to_borough=borough_lookup,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    panel = make_complete_panel(hourly)

    horizon_hours = args.horizon_hours
    holiday_dates = get_holiday_dates(
        panel["ds"].min(),
        panel["ds"].max() + pd.Timedelta(hours=horizon_hours),
    )

    forecasts = []
    metrics_rows = []
    coef_rows = []

    boroughs = sorted(panel["borough"].unique())
    logger.info("Training models for boroughs: %s", ", ".join(boroughs))

    for borough in boroughs:
        borough_df = panel[panel["borough"] == borough].copy()
        logger.info("Training borough model: %s", borough)

        try:
            result, _, _, metrics = train_borough_model(
                borough_df=borough_df,
                holiday_dates=holiday_dates,
                horizon_hours=horizon_hours,
                min_train_rows=args.min_train_rows,
            )
            forecast_df = recursive_forecast(
                result=result,
                borough_df=borough_df,
                holiday_dates=holiday_dates,
                horizon_hours=horizon_hours,
                borough=borough,
            )
            coef_df = coefficients_to_frame(result, borough)
        except Exception as exc:
            logger.warning("Skipping borough %s due to model error: %s", borough, exc)
            continue

        forecasts.append(forecast_df)
        coef_rows.append(coef_df)
        metrics_rows.append({"borough": borough, **metrics})

    if not forecasts:
        raise RuntimeError("No borough models were trained successfully.")

    forecast_all = pd.concat(forecasts, ignore_index=True)
    metrics_all = pd.DataFrame(metrics_rows).sort_values("borough").reset_index(drop=True)
    coef_all = pd.concat(coef_rows, ignore_index=True).sort_values(
        ["borough", "abs_coefficient"], ascending=[True, False]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = output_dir / "borough_hourly_demand_timeseries.parquet"
    forecast_path = output_dir / "borough_7day_hourly_forecast.csv"
    metrics_path = output_dir / "borough_validation_metrics.csv"
    coef_path = output_dir / "borough_model_coefficients.csv"
    summary_path = output_dir / "borough_forecast_summary.json"

    panel.to_parquet(panel_path, index=False)
    forecast_all.to_csv(forecast_path, index=False)
    metrics_all.to_csv(metrics_path, index=False)
    coef_all.to_csv(coef_path, index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_type": "Poisson GLM (per borough)",
        "frequency": "hourly",
        "forecast_horizon_hours": horizon_hours,
        "input_glob": args.input_glob,
        "num_input_files": len(input_files),
        "num_rows_panel": int(len(panel)),
        "boroughs_modeled": sorted(metrics_all["borough"].unique().tolist()),
        "output_files": {
            "panel": str(panel_path),
            "forecast": str(forecast_path),
            "metrics": str(metrics_path),
            "coefficients": str(coef_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved panel: %s", panel_path)
    logger.info("Saved forecast: %s", forecast_path)
    logger.info("Saved metrics: %s", metrics_path)
    logger.info("Saved coefficients: %s", coef_path)
    logger.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
