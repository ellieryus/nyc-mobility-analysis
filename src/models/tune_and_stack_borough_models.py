"""
Tune tree-based borough demand models and add stacking ensemble.

This script evaluates, per borough:
- Poisson GLM (interpretable baseline)
- Tuned XGBoost (Poisson objective)
- Tuned LightGBM (Poisson objective)
- Tuned CatBoost (Poisson objective)
- Prophet (calendar/seasonality baseline)
- Stacked ensemble (meta-linear model on tree + GLM predictions)

Then it selects the best model by MAE for each borough on the 7-day horizon.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from tqdm import tqdm
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

BASE_MODEL_COLUMNS = ["glm_pred", "xgb_pred", "lgb_pred", "cat_pred"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune XGBoost/LightGBM/CatBoost, run Prophet, and build stacking ensemble.",
    )
    parser.add_argument(
        "--panel-path",
        type=str,
        default="reports/forecasts/borough_hourly_demand_timeseries.parquet",
        help="Path to borough hourly panel data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/forecasts",
        help="Output directory for tuned/stacked artifacts.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=168,
        help="Forecast horizon (default: 168 = 7 days).",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=6,
        help="Random parameter samples per model and borough (default: 6).",
    )
    parser.add_argument(
        "--seed",
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

    out = out.dropna(subset=FEATURE_COLUMNS + ["trips"]).reset_index(drop=True)
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


def xgb_model(params: Dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="count:poisson",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=seed,
        n_jobs=-1,
    )


def lgb_model(params: Dict, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="poisson",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def cat_model(params: Dict, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="Poisson",
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def add_calendar_regressors_for_prophet(df: pd.DataFrame, holiday_dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = df.copy()
    out["is_weekend"] = (out["ds"].dt.dayofweek >= 5).astype(int)
    out["is_holiday"] = out["ds"].dt.normalize().isin(holiday_dates).astype(int)
    return out


def prophet_predict(
    train_raw: pd.DataFrame,
    future_ds: pd.Series,
    holiday_dates: pd.DatetimeIndex,
) -> np.ndarray:
    train_df = train_raw[["ds", "trips"]].rename(columns={"trips": "y"}).copy()
    train_df = add_calendar_regressors_for_prophet(train_df, holiday_dates)
    train_df["y"] = train_df["y"].astype(float)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        interval_width=0.95,
    )
    model.add_regressor("is_weekend")
    model.add_regressor("is_holiday")
    model.fit(train_df[["ds", "y", "is_weekend", "is_holiday"]])

    future = pd.DataFrame({"ds": pd.to_datetime(future_ds).values})
    future = add_calendar_regressors_for_prophet(future, holiday_dates)
    pred = model.predict(future[["ds", "is_weekend", "is_holiday"]])["yhat"].values
    pred = np.maximum(pred, 0.0)
    return pred


def tune_model_params(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    tune_iter: int,
    seed: int,
) -> Tuple[Dict, float]:
    if model_name == "xgboost":
        param_space = {
            "n_estimators": [250, 400, 550, 700],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [5, 7, 9],
            "min_child_weight": [1, 5, 10],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [1.0, 2.0, 5.0],
        }
    elif model_name == "lightgbm":
        param_space = {
            "n_estimators": [250, 400, 550, 700],
            "learning_rate": [0.03, 0.05, 0.08],
            "num_leaves": [31, 63, 95],
            "min_child_samples": [20, 40, 80],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [1.0, 2.0, 5.0],
        }
    elif model_name == "catboost":
        param_space = {
            "iterations": [300, 500, 700],
            "learning_rate": [0.03, 0.05, 0.08],
            "depth": [6, 8, 10],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    candidates = list(ParameterSampler(param_space, n_iter=tune_iter, random_state=seed))
    n_splits = 3 if len(x) >= 1200 else 2
    cv = TimeSeriesSplit(n_splits=n_splits)

    best_params = None
    best_score = float("inf")

    for params in candidates:
        fold_mae = []
        for train_idx, val_idx in cv.split(x):
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_name == "xgboost":
                model = xgb_model(params, seed)
            elif model_name == "lightgbm":
                model = lgb_model(params, seed)
            else:
                model = cat_model(params, seed)

            model.fit(x_train, y_train)
            pred = np.maximum(model.predict(x_val), 0.0)
            fold_mae.append(float(np.mean(np.abs(y_val.values - pred))))

        score = float(np.mean(fold_mae))
        if score < best_score:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError(f"Tuning failed for model {model_name}")

    return best_params, best_score


def fit_glm(x_train: pd.DataFrame, y_train: pd.Series):
    glm = sm.GLM(y_train, sm.add_constant(x_train, has_constant="add"), family=sm.families.Poisson())
    return glm.fit(maxiter=300, disp=0)


def fit_stacking_meta(
    train_feat: pd.DataFrame,
    best_xgb_params: Dict,
    best_lgb_params: Dict,
    best_cat_params: Dict,
    seed: int,
) -> Tuple[LinearRegression, Dict[str, float]]:
    x = train_feat[FEATURE_COLUMNS]
    y = train_feat["trips"]
    n_splits = 3 if len(train_feat) >= 1200 else 2
    cv = TimeSeriesSplit(n_splits=n_splits)

    oof_rows = []
    for train_idx, val_idx in cv.split(x):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        glm_res = fit_glm(x_train, y_train)
        xgb_res = xgb_model(best_xgb_params, seed)
        lgb_res = lgb_model(best_lgb_params, seed)
        cat_res = cat_model(best_cat_params, seed)

        xgb_res.fit(x_train, y_train)
        lgb_res.fit(x_train, y_train)
        cat_res.fit(x_train, y_train)

        glm_pred = np.maximum(glm_res.predict(sm.add_constant(x_val, has_constant="add")), 0.0)
        xgb_pred = np.maximum(xgb_res.predict(x_val), 0.0)
        lgb_pred = np.maximum(lgb_res.predict(x_val), 0.0)
        cat_pred = np.maximum(cat_res.predict(x_val), 0.0)

        oof_fold = pd.DataFrame(
            {
                "glm_pred": glm_pred,
                "xgb_pred": xgb_pred,
                "lgb_pred": lgb_pred,
                "cat_pred": cat_pred,
                "target": y_val.values,
            }
        )
        oof_rows.append(oof_fold)

    oof_df = pd.concat(oof_rows, ignore_index=True)
    meta = LinearRegression(positive=True)
    meta.fit(oof_df[BASE_MODEL_COLUMNS], oof_df["target"])

    weights = {
        "intercept": float(meta.intercept_),
        "w_glm": float(meta.coef_[0]),
        "w_xgboost": float(meta.coef_[1]),
        "w_lightgbm": float(meta.coef_[2]),
        "w_catboost": float(meta.coef_[3]),
    }
    return meta, weights


def recursive_forecast_glm(
    glm_res,
    history_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
    horizon: int,
) -> List[float]:
    history = history_df["trips"].astype(float).copy()
    preds = []
    ts = history_df["ds"].iloc[-1]
    for _ in range(horizon):
        ts = ts + pd.Timedelta(hours=1)
        row = build_feature_row(ts, history, holiday_dates)
        x = sm.add_constant(pd.DataFrame([row], columns=FEATURE_COLUMNS), has_constant="add")
        pred = float(np.maximum(glm_res.predict(x).iloc[0], 0.0))
        preds.append(pred)
        history = pd.concat([history, pd.Series([pred])], ignore_index=True)
    return preds


def recursive_forecast_tree(
    model,
    history_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
    horizon: int,
) -> List[float]:
    history = history_df["trips"].astype(float).copy()
    preds = []
    ts = history_df["ds"].iloc[-1]
    for _ in range(horizon):
        ts = ts + pd.Timedelta(hours=1)
        row = build_feature_row(ts, history, holiday_dates)
        x = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        pred = float(np.maximum(model.predict(x)[0], 0.0))
        preds.append(pred)
        history = pd.concat([history, pd.Series([pred])], ignore_index=True)
    return preds


def recursive_forecast_stack(
    glm_res,
    xgb_res,
    lgb_res,
    cat_res,
    meta_res: LinearRegression,
    history_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
    horizon: int,
) -> List[float]:
    history = history_df["trips"].astype(float).copy()
    preds = []
    ts = history_df["ds"].iloc[-1]

    for _ in range(horizon):
        ts = ts + pd.Timedelta(hours=1)
        row = build_feature_row(ts, history, holiday_dates)
        x = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        glm_pred = float(np.maximum(glm_res.predict(sm.add_constant(x, has_constant="add")).iloc[0], 0.0))
        xgb_pred = float(np.maximum(xgb_res.predict(x)[0], 0.0))
        lgb_pred = float(np.maximum(lgb_res.predict(x)[0], 0.0))
        cat_pred = float(np.maximum(cat_res.predict(x)[0], 0.0))

        base_vec = pd.DataFrame(
            [{"glm_pred": glm_pred, "xgb_pred": xgb_pred, "lgb_pred": lgb_pred, "cat_pred": cat_pred}],
            columns=BASE_MODEL_COLUMNS,
        )
        pred = float(np.maximum(meta_res.predict(base_vec)[0], 0.0))
        preds.append(pred)
        history = pd.concat([history, pd.Series([pred])], ignore_index=True)

    return preds


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found: {panel_path}. Run forecast_borough_demand.py first."
        )

    panel = pd.read_parquet(panel_path)
    panel["ds"] = pd.to_datetime(panel["ds"])
    panel = panel.sort_values(["borough", "ds"]).reset_index(drop=True)

    holiday_dates = get_holiday_dates(panel["ds"].min(), panel["ds"].max() + pd.Timedelta(hours=args.horizon_hours))
    horizon = args.horizon_hours

    forecast_rows = []
    metrics_rows = []
    tuning_rows = []
    stacking_rows = []

    boroughs = sorted(panel["borough"].unique())
    for borough in tqdm(boroughs, desc="Tuning + stacking by borough"):
        bdf = panel[panel["borough"] == borough].copy().sort_values("ds").reset_index(drop=True)
        if len(bdf) < (horizon + 300):
            logger.warning("Skipping %s due to insufficient rows (%s).", borough, len(bdf))
            continue

        train_raw = bdf.iloc[:-horizon].copy()
        test_raw = bdf.iloc[-horizon:].copy()
        train_feat = add_features(train_raw, holiday_dates)

        if len(train_feat) < 300:
            logger.warning("Skipping %s due to insufficient feature rows (%s).", borough, len(train_feat))
            continue

        x_train = train_feat[FEATURE_COLUMNS]
        y_train = train_feat["trips"]

        best_xgb_params, best_xgb_cv_mae = tune_model_params(
            model_name="xgboost",
            x=x_train,
            y=y_train,
            tune_iter=args.tune_iter,
            seed=args.seed,
        )
        best_lgb_params, best_lgb_cv_mae = tune_model_params(
            model_name="lightgbm",
            x=x_train,
            y=y_train,
            tune_iter=args.tune_iter,
            seed=args.seed + 1,
        )
        best_cat_params, best_cat_cv_mae = tune_model_params(
            model_name="catboost",
            x=x_train,
            y=y_train,
            tune_iter=args.tune_iter,
            seed=args.seed + 2,
        )

        tuning_rows.append(
            {
                "borough": borough,
                "xgboost_cv_mae": best_xgb_cv_mae,
                "xgboost_best_params": json.dumps(best_xgb_params),
                "lightgbm_cv_mae": best_lgb_cv_mae,
                "lightgbm_best_params": json.dumps(best_lgb_params),
                "catboost_cv_mae": best_cat_cv_mae,
                "catboost_best_params": json.dumps(best_cat_params),
            }
        )

        glm_res = fit_glm(x_train, y_train)
        xgb_res = xgb_model(best_xgb_params, args.seed)
        lgb_res = lgb_model(best_lgb_params, args.seed)
        cat_res = cat_model(best_cat_params, args.seed)
        xgb_res.fit(x_train, y_train)
        lgb_res.fit(x_train, y_train)
        cat_res.fit(x_train, y_train)

        meta_res, stack_weights = fit_stacking_meta(
            train_feat=train_feat,
            best_xgb_params=best_xgb_params,
            best_lgb_params=best_lgb_params,
            best_cat_params=best_cat_params,
            seed=args.seed,
        )
        stacking_rows.append({"borough": borough, **stack_weights})

        preds_glm = recursive_forecast_glm(glm_res, train_raw, holiday_dates, horizon)
        preds_xgb = recursive_forecast_tree(xgb_res, train_raw, holiday_dates, horizon)
        preds_lgb = recursive_forecast_tree(lgb_res, train_raw, holiday_dates, horizon)
        preds_cat = recursive_forecast_tree(cat_res, train_raw, holiday_dates, horizon)
        try:
            preds_prophet = prophet_predict(
                train_raw=train_raw,
                future_ds=test_raw["ds"],
                holiday_dates=holiday_dates,
            )
        except Exception as exc:
            logger.warning("Prophet failed for %s; skipping prophet model: %s", borough, exc)
            preds_prophet = None
        preds_stack = recursive_forecast_stack(
            glm_res=glm_res,
            xgb_res=xgb_res,
            lgb_res=lgb_res,
            cat_res=cat_res,
            meta_res=meta_res,
            history_df=train_raw,
            holiday_dates=holiday_dates,
            horizon=horizon,
        )

        actual = test_raw["trips"].values.astype(float)
        ds_values = test_raw["ds"].values
        model_preds = {
            "glm": preds_glm,
            "xgboost_tuned": preds_xgb,
            "lightgbm_tuned": preds_lgb,
            "catboost_tuned": preds_cat,
            "stacked": preds_stack,
        }
        if preds_prophet is not None:
            model_preds["prophet"] = preds_prophet

        for model_name, preds in model_preds.items():
            preds_arr = np.asarray(preds, dtype=float)
            metrics = error_metrics(actual, preds_arr)
            metrics_rows.append({"borough": borough, "model": model_name, **metrics})
            forecast_rows.append(
                pd.DataFrame(
                    {
                        "borough": borough,
                        "model": model_name,
                        "ds": ds_values,
                        "predicted_trips": preds_arr,
                    }
                )
            )

    if not metrics_rows:
        raise RuntimeError("No boroughs were successfully processed.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["borough", "mae"]).reset_index(drop=True)
    forecasts_df = pd.concat(forecast_rows, ignore_index=True).sort_values(["borough", "model", "ds"])
    tuning_df = pd.DataFrame(tuning_rows).sort_values("borough").reset_index(drop=True)
    stacking_df = pd.DataFrame(stacking_rows).sort_values("borough").reset_index(drop=True)

    best_model_df = metrics_df.groupby("borough", as_index=False).first()[["borough", "model", "mae", "rmse"]]
    best_forecasts_df = forecasts_df.merge(best_model_df[["borough", "model"]], on=["borough", "model"], how="inner")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "tuned_stacked_validation_metrics.csv"
    forecasts_csv = output_dir / "tuned_stacked_7day_hourly_forecast.csv"
    best_forecasts_csv = output_dir / "tuned_stacked_best_7day_hourly_forecast.csv"
    tuning_csv = output_dir / "tuned_hyperparameters_by_borough.csv"
    stacking_csv = output_dir / "stacking_weights_by_borough.csv"
    summary_json = output_dir / "tuned_stacked_model_summary.json"

    metrics_xlsx = output_dir / "tuned_stacked_validation_metrics.xlsx"
    forecasts_xlsx = output_dir / "tuned_stacked_7day_hourly_forecast.xlsx"
    best_forecasts_xlsx = output_dir / "tuned_stacked_best_7day_hourly_forecast.xlsx"
    tuning_xlsx = output_dir / "tuned_hyperparameters_by_borough.xlsx"
    stacking_xlsx = output_dir / "stacking_weights_by_borough.xlsx"

    metrics_df.to_csv(metrics_csv, index=False)
    forecasts_df.to_csv(forecasts_csv, index=False)
    best_forecasts_df.to_csv(best_forecasts_csv, index=False)
    tuning_df.to_csv(tuning_csv, index=False)
    stacking_df.to_csv(stacking_csv, index=False)

    metrics_df.to_excel(metrics_xlsx, index=False)
    forecasts_df.to_excel(forecasts_xlsx, index=False)
    best_forecasts_df.to_excel(best_forecasts_xlsx, index=False)
    tuning_df.to_excel(tuning_xlsx, index=False)
    stacking_df.to_excel(stacking_xlsx, index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "panel_path": str(panel_path),
        "horizon_hours": horizon,
        "tune_iter": args.tune_iter,
        "models_compared": sorted(metrics_df["model"].unique().tolist()),
        "best_model_by_borough": best_model_df.to_dict(orient="records"),
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "forecasts_csv": str(forecasts_csv),
            "best_forecasts_csv": str(best_forecasts_csv),
            "tuning_csv": str(tuning_csv),
            "stacking_csv": str(stacking_csv),
            "metrics_xlsx": str(metrics_xlsx),
            "forecasts_xlsx": str(forecasts_xlsx),
            "best_forecasts_xlsx": str(best_forecasts_xlsx),
            "tuning_xlsx": str(tuning_xlsx),
            "stacking_xlsx": str(stacking_xlsx),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved metrics: %s", metrics_csv)
    logger.info("Saved forecasts: %s", forecasts_csv)
    logger.info("Saved best forecasts: %s", best_forecasts_csv)
    logger.info("Saved tuning report: %s", tuning_csv)
    logger.info("Saved stacking report: %s", stacking_csv)
    logger.info("Saved summary: %s", summary_json)


if __name__ == "__main__":
    main()
