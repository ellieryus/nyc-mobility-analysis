"""
Borough-level hourly demand forecasting with tree-based models.

Models:
- XGBoost regressor (Poisson objective)
- LightGBM regressor (Poisson objective)

Workflow:
1) Load borough hourly panel data.
2) Train each model per borough on train window.
3) Recursive 7-day (168h) validation forecast.
4) Compare metrics and save forecasts.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm import tqdm

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


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
        description="Train borough tree-based demand forecasting models."
    )
    parser.add_argument(
        "--panel-path",
        type=str,
        default="reports/forecasts/borough_hourly_demand_timeseries.parquet",
        help="Path to borough hourly panel parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/forecasts",
        help="Output directory.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=168,
        help="Validation forecast horizon in hours.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def get_holiday_dates(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DatetimeIndex:
    cal = USFederalHolidayCalendar()
    return cal.holidays(
        start=start_ts.normalize() - pd.Timedelta(days=14),
        end=end_ts.normalize() + pd.Timedelta(days=14),
    )


def add_features(df: pd.DataFrame, holiday_dates: pd.DatetimeIndex) -> pd.DataFrame:
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

    out = out.dropna(subset=FEATURE_COLUMNS + ["trips"]).copy()
    return out


def build_feature_row(
    ts: pd.Timestamp,
    history: pd.Series,
    holiday_dates: pd.DatetimeIndex,
) -> Dict[str, float]:
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month

    lag_1 = float(history.iloc[-1])
    lag_24 = float(history.iloc[-24])
    lag_168 = float(history.iloc[-168])
    roll_24 = float(history.iloc[-24:].mean())
    roll_168 = float(history.iloc[-168:].mean())

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
        "log_roll_mean_24": float(np.log1p(roll_24)),
        "log_roll_mean_168": float(np.log1p(roll_168)),
    }


def error_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    wape = float(np.sum(np.abs(actual - pred)) / max(np.sum(actual), 1.0) * 100.0)
    mape = float(np.mean(np.abs(actual - pred) / np.maximum(actual, 1.0)) * 100.0)
    return {"mae": mae, "rmse": rmse, "wape_pct": wape, "mape_pct": mape}


def make_models(seed: int) -> Dict[str, object]:
    return {
        "xgboost": XGBRegressor(
            objective="count:poisson",
            n_estimators=450,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            objective="poisson",
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def recursive_forecast(
    model: object,
    history_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
    horizon_hours: int,
    borough: str,
    model_name: str,
) -> pd.DataFrame:
    history = history_df[["ds", "trips"]].copy().sort_values("ds")
    history["trips"] = history["trips"].astype(float)

    rows = []
    for _ in range(horizon_hours):
        next_ts = history["ds"].iloc[-1] + pd.Timedelta(hours=1)
        row = build_feature_row(next_ts, history["trips"], holiday_dates)
        x = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        pred = float(model.predict(x)[0])
        pred = max(0.0, pred)

        rows.append(
            {
                "model": model_name,
                "borough": borough,
                "ds": next_ts,
                "predicted_trips": pred,
            }
        )
        history = pd.concat(
            [history, pd.DataFrame([{"ds": next_ts, "trips": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found: {panel_path}. Run forecast_borough_demand.py first."
        )

    panel = pd.read_parquet(panel_path)
    panel["ds"] = pd.to_datetime(panel["ds"])
    panel = panel.sort_values(["borough", "ds"]).reset_index(drop=True)

    horizon = args.horizon_hours
    holiday_dates = get_holiday_dates(panel["ds"].min(), panel["ds"].max() + pd.Timedelta(hours=horizon))

    models = make_models(args.random_seed)
    boroughs = sorted(panel["borough"].unique())

    forecast_rows: List[pd.DataFrame] = []
    metrics_rows: List[Dict[str, float]] = []

    for borough in tqdm(boroughs, desc="Training tree models by borough"):
        bdf = panel[panel["borough"] == borough].copy()
        bdf = bdf.sort_values("ds").reset_index(drop=True)

        if len(bdf) < (horizon + 200):
            logger.warning("Skipping %s due to short history (%s rows).", borough, len(bdf))
            continue

        train_raw = bdf.iloc[:-horizon].copy()
        val_raw = bdf.iloc[-horizon:].copy()
        train_feat = add_features(train_raw, holiday_dates)

        if train_feat.empty:
            logger.warning("Skipping %s due to empty features.", borough)
            continue

        x_train = train_feat[FEATURE_COLUMNS]
        y_train = train_feat["trips"]

        for model_name, model in models.items():
            try:
                model.fit(x_train, y_train)
                pred_df = recursive_forecast(
                    model=model,
                    history_df=train_raw,
                    holiday_dates=holiday_dates,
                    horizon_hours=horizon,
                    borough=borough,
                    model_name=model_name,
                )
            except Exception as exc:
                logger.warning("Model %s failed for %s: %s", model_name, borough, exc)
                continue

            actual = val_raw["trips"].values.astype(float)
            pred = pred_df["predicted_trips"].values.astype(float)
            metric = error_metrics(actual, pred)
            metrics_rows.append(
                {
                    "model": model_name,
                    "borough": borough,
                    **metric,
                }
            )
            forecast_rows.append(pred_df)

    if not metrics_rows:
        raise RuntimeError("No tree-based model runs succeeded.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["model", "borough"]).reset_index(drop=True)
    forecast_df = pd.concat(forecast_rows, ignore_index=True).sort_values(["model", "borough", "ds"])

    # Pick best model per borough using MAE.
    best_models = (
        metrics_df.sort_values(["borough", "mae"])
        .groupby("borough", as_index=False)
        .first()[["borough", "model"]]
    )
    best_forecast_df = forecast_df.merge(best_models, on=["borough", "model"], how="inner")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "tree_model_validation_metrics.csv"
    forecast_csv = out_dir / "tree_model_7day_hourly_forecast.csv"
    best_forecast_csv = out_dir / "tree_best_model_7day_hourly_forecast.csv"
    metrics_xlsx = out_dir / "tree_model_validation_metrics.xlsx"
    forecast_xlsx = out_dir / "tree_model_7day_hourly_forecast.xlsx"
    best_forecast_xlsx = out_dir / "tree_best_model_7day_hourly_forecast.xlsx"
    summary_json = out_dir / "tree_model_summary.json"

    metrics_df.to_csv(metrics_csv, index=False)
    forecast_df.to_csv(forecast_csv, index=False)
    best_forecast_df.to_csv(best_forecast_csv, index=False)

    metrics_df.to_excel(metrics_xlsx, index=False)
    forecast_df.to_excel(forecast_xlsx, index=False)
    best_forecast_df.to_excel(best_forecast_xlsx, index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "panel_path": str(panel_path),
        "models_evaluated": sorted(metrics_df["model"].unique().tolist()),
        "horizon_hours": horizon,
        "boroughs_evaluated": sorted(metrics_df["borough"].unique().tolist()),
        "best_model_by_borough": best_models.to_dict(orient="records"),
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "forecast_csv": str(forecast_csv),
            "best_forecast_csv": str(best_forecast_csv),
            "metrics_xlsx": str(metrics_xlsx),
            "forecast_xlsx": str(forecast_xlsx),
            "best_forecast_xlsx": str(best_forecast_xlsx),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved metrics: %s", metrics_csv)
    logger.info("Saved forecast: %s", forecast_csv)
    logger.info("Saved best forecast: %s", best_forecast_csv)
    logger.info("Saved summary: %s", summary_json)


if __name__ == "__main__":
    main()
