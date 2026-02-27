"""
FastAPI application for NYC Mobility Analysis.

This API provides endpoints for:
- Trip demand predictions
- Statistical analysis
- Data visualization
- Model information
"""

from typing import Dict, List, Optional
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import json
from io import BytesIO

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="NYC Mobility Analysis API",
    description="API for analyzing NYC transportation patterns and predicting trip demand",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TripPredictionRequest(BaseModel):
    """Request model for trip prediction."""
    
    pickup_datetime: datetime = Field(..., description="Pickup datetime")
    pickup_location_id: int = Field(..., description="Pickup location zone ID")
    vehicle_type: str = Field(..., description="Vehicle type (yellow, green, fhvhv)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pickup_datetime": "2024-01-15T14:30:00",
                "pickup_location_id": 161,
                "vehicle_type": "yellow"
            }
        }


class TripPredictionResponse(BaseModel):
    """Response model for trip prediction."""
    
    predicted_trips: float = Field(..., description="Predicted number of trips")
    confidence_interval: List[float] = Field(..., description="95% confidence interval")
    model_version: str = Field(..., description="Model version used")


class AnalyticsSummary(BaseModel):
    """Summary statistics response."""
    
    total_trips: int
    avg_trip_distance: float
    avg_fare: float
    peak_hour: int
    busiest_location: int


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT_ROOT / "reports" / "forecasts"
PROD_FORECAST_CSV = FORECAST_DIR / "production_borough_7day_hourly_forecast.csv"
PROD_BEST_MODELS_CSV = FORECAST_DIR / "production_best_model_by_borough.csv"
PROD_SUMMARY_JSON = FORECAST_DIR / "production_summary.json"
PROD_ACTUAL_PARQUET = FORECAST_DIR / "borough_hourly_demand_timeseries.parquet"


def _ensure_forecast_files():
    required = [PROD_FORECAST_CSV, PROD_BEST_MODELS_CSV, PROD_SUMMARY_JSON]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Forecast files not found. Missing: {missing}",
        )


@lru_cache(maxsize=1)
def load_summary_data() -> Dict:
    _ensure_forecast_files()
    with open(PROD_SUMMARY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_best_models_data() -> pd.DataFrame:
    _ensure_forecast_files()
    df = pd.read_csv(PROD_BEST_MODELS_CSV)
    return df.sort_values("borough").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_forecast_data() -> pd.DataFrame:
    _ensure_forecast_files()
    df = pd.read_csv(PROD_FORECAST_CSV, parse_dates=["ds"])
    df = df.sort_values(["borough", "ds"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def load_actual_data() -> pd.DataFrame:
    """
    Load borough-level hourly actual demand if available.
    """
    if not PROD_ACTUAL_PARQUET.exists():
        return pd.DataFrame(columns=["ds", "borough", "actual_trips"])

    df = pd.read_parquet(PROD_ACTUAL_PARQUET)
    if df.empty:
        return pd.DataFrame(columns=["ds", "borough", "actual_trips"])

    trip_col = "trips" if "trips" in df.columns else "actual_trips"
    keep = ["ds", "borough", trip_col]
    df = df[keep].copy()
    df = df.rename(columns={trip_col: "actual_trips"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["borough"] = df["borough"].astype(str)
    df["actual_trips"] = pd.to_numeric(df["actual_trips"], errors="coerce")
    return df.sort_values(["borough", "ds"]).reset_index(drop=True)


def _build_enriched_forecast_df() -> pd.DataFrame:
    """
    Build a forecast dataframe with uncertainty proxy and matched actuals.
    """
    forecast_df = load_forecast_data().copy()
    best_models_df = load_best_models_data().copy()

    mae_df = (
        best_models_df[["borough", "mae"]].copy()
        if "mae" in best_models_df.columns
        else pd.DataFrame(columns=["borough", "mae"])
    )
    mae_df["mae"] = pd.to_numeric(mae_df["mae"], errors="coerce")
    fallback_mae = float(mae_df["mae"].dropna().median()) if not mae_df.empty else 0.0

    enriched = forecast_df.merge(mae_df, on="borough", how="left")
    enriched["mae"] = pd.to_numeric(enriched["mae"], errors="coerce").fillna(fallback_mae)

    # Uncertainty proxy: +/- borough MAE from backtesting metrics.
    enriched["uncertainty_low"] = (enriched["predicted_trips"] - enriched["mae"]).clip(lower=0.0)
    enriched["uncertainty_high"] = enriched["predicted_trips"] + enriched["mae"]

    actual_df = load_actual_data().copy()
    if not actual_df.empty:
        min_ds = enriched["ds"].min()
        max_ds = enriched["ds"].max()
        actual_df = actual_df[(actual_df["ds"] >= min_ds) & (actual_df["ds"] <= max_ds)].copy()
        enriched = enriched.merge(actual_df, on=["borough", "ds"], how="left")
    else:
        enriched["actual_trips"] = pd.NA

    actual_numeric = pd.to_numeric(enriched["actual_trips"], errors="coerce")
    enriched["absolute_error"] = (actual_numeric - enriched["predicted_trips"]).abs()
    enriched["date"] = enriched["ds"].dt.strftime("%Y-%m-%d")
    enriched["day_of_week"] = enriched["ds"].dt.day_name()
    enriched["hour"] = enriched["ds"].dt.hour
    return enriched.sort_values(["borough", "ds"]).reset_index(drop=True)


def _apply_series_filters(
    df: pd.DataFrame,
    borough: Optional[str] = None,
    date: Optional[str] = None,
    day: Optional[str] = None,
    hour: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply common borough/date/day/hour filters to the provided dataframe.
    """
    filtered = df.copy()

    if borough:
        borough_filtered = filtered[filtered["borough"].str.lower() == borough.lower()].copy()
        if borough_filtered.empty:
            raise HTTPException(status_code=404, detail=f"Borough not found: {borough}")
        filtered = borough_filtered

    if date:
        try:
            normalized_date = pd.to_datetime(date).strftime("%Y-%m-%d")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {date}. Use YYYY-MM-DD.",
            ) from exc
        filtered = filtered[filtered["date"] == normalized_date].copy()

    if day:
        normalized_day = day.strip().lower()
        valid_days = {
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        }
        if normalized_day not in valid_days:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid day: {day}. Use full day name like Monday.",
            )
        filtered = filtered[filtered["day_of_week"].str.lower() == normalized_day].copy()

    if hour is not None:
        filtered = filtered[filtered["hour"] == int(hour)].copy()

    return filtered.sort_values(["borough", "ds"]).reset_index(drop=True)


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "message": "NYC Mobility Analysis API",
        "status": "healthy",
        "version": "0.1.0",
        "documentation": "/docs",
        "forecast_ui": "/forecast/ui",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "nyc-mobility-api",
        "models_loaded": True  # TODO: Actually check model loading
    }


# Prediction endpoints
@app.post("/predict/demand", response_model=TripPredictionResponse, tags=["Predictions"])
async def predict_trip_demand(request: TripPredictionRequest):
    """
    Predict trip demand for a given time and location.
    
    Args:
        request: Trip prediction request with datetime, location, and vehicle type
    
    Returns:
        Predicted number of trips with confidence interval
    """
    # TODO: Implement actual prediction logic
    # This is a placeholder
    return TripPredictionResponse(
        predicted_trips=125.5,
        confidence_interval=[110.2, 140.8],
        model_version="v0.1.0"
    )


@app.get("/analytics/summary", response_model=AnalyticsSummary, tags=["Analytics"])
async def get_analytics_summary(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    vehicle_type: Optional[str] = Query(None, description="Vehicle type filter")
):
    """
    Get summary statistics for a date range.
    
    Args:
        start_date: Optional start date
        end_date: Optional end date
        vehicle_type: Optional vehicle type filter
    
    Returns:
        Summary statistics
    """
    # TODO: Implement actual analytics logic
    return AnalyticsSummary(
        total_trips=1234567,
        avg_trip_distance=3.5,
        avg_fare=18.50,
        peak_hour=18,
        busiest_location=161
    )


@app.get("/analytics/trends", tags=["Analytics"])
async def get_trends(
    metric: str = Query(..., description="Metric to analyze (trips, distance, fare)"),
    granularity: str = Query("daily", description="Time granularity (hourly, daily, weekly, monthly)")
):
    """
    Get trend data for visualization.
    
    Args:
        metric: Metric to analyze
        granularity: Time granularity
    
    Returns:
        Time series data
    """
    # TODO: Implement actual trend analysis
    return {
        "metric": metric,
        "granularity": granularity,
        "data": [
            {"timestamp": "2024-01-01", "value": 50000},
            {"timestamp": "2024-01-02", "value": 52000},
            {"timestamp": "2024-01-03", "value": 48000},
        ]
    }


@app.get("/locations/popular", tags=["Locations"])
async def get_popular_locations(
    limit: int = Query(10, description="Number of locations to return"),
    vehicle_type: Optional[str] = Query(None, description="Vehicle type filter")
):
    """
    Get most popular pickup/dropoff locations.
    
    Args:
        limit: Number of locations to return
        vehicle_type: Optional vehicle type filter
    
    Returns:
        List of popular locations with trip counts
    """
    # TODO: Implement actual location analysis
    return {
        "locations": [
            {"location_id": 161, "name": "Midtown Center", "trip_count": 45000},
            {"location_id": 237, "name": "Upper East Side South", "trip_count": 38000},
            {"location_id": 162, "name": "Clinton East", "trip_count": 35000},
        ][:limit]
    }


@app.get("/models/info", tags=["Models"])
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns:
        Model metadata and performance metrics
    """
    # TODO: Implement actual model info retrieval
    return {
        "models": [
            {
                "name": "trip_demand_forecaster",
                "version": "v0.1.0",
                "type": "XGBoost",
                "metrics": {
                    "mae": 12.5,
                    "rmse": 18.3,
                    "r2": 0.87
                },
                "last_trained": "2024-01-15T10:30:00"
            }
        ]
    }


@app.get("/forecast/summary", tags=["Forecast"])
async def get_forecast_summary():
    """Return production forecast summary metadata."""
    return load_summary_data()


@app.get("/forecast/best-models", tags=["Forecast"])
async def get_best_models():
    """Return best model per borough from production outputs."""
    df = load_best_models_data().copy()
    return {"best_models": df.to_dict(orient="records")}


@app.get("/forecast/boroughs", tags=["Forecast"])
async def get_forecast_boroughs():
    """Return borough list available in production forecast file."""
    df = load_forecast_data()
    boroughs = sorted(df["borough"].dropna().astype(str).unique().tolist())
    return {"boroughs": boroughs}


@app.get("/forecast/borough-totals", tags=["Forecast"])
async def get_forecast_borough_totals():
    """
    Return borough-level totals across the 7-day horizon with forecast, actual, and WAPE.
    """
    df = _build_enriched_forecast_df()
    grouped = (
        df.groupby("borough", as_index=False)
        .agg(
            forecast_trips=("predicted_trips", "sum"),
            actual_trips=("actual_trips", "sum"),
            abs_error=("absolute_error", "sum"),
        )
        .rename(columns={"borough": "borough"})
    )

    grouped["forecast_trips"] = pd.to_numeric(grouped["forecast_trips"], errors="coerce")
    grouped["actual_trips"] = pd.to_numeric(grouped["actual_trips"], errors="coerce")
    grouped["abs_error"] = pd.to_numeric(grouped["abs_error"], errors="coerce")
    grouped["wape_pct"] = np.where(
        grouped["actual_trips"] > 0,
        grouped["abs_error"] / grouped["actual_trips"] * 100.0,
        np.nan,
    )

    best_df = load_best_models_data()[["borough", "model", "mae"]].copy()
    best_df["mae"] = pd.to_numeric(best_df["mae"], errors="coerce")
    grouped = grouped.merge(best_df, on="borough", how="left")
    grouped = grouped.sort_values("forecast_trips", ascending=False).reset_index(drop=True)

    output_cols = [
        "borough",
        "model",
        "mae",
        "forecast_trips",
        "actual_trips",
        "abs_error",
        "wape_pct",
    ]
    out = grouped[output_cols].copy()
    out = out.where(pd.notnull(out), None)

    return {
        "borough_totals": out.to_dict(orient="records"),
        "units": "Trips across forecast horizon",
    }


@app.get("/forecast/series", tags=["Forecast"])
async def get_forecast_series(
    borough: Optional[str] = Query(None, description="Optional borough filter"),
    date: Optional[str] = Query(None, description="Optional date filter (YYYY-MM-DD)"),
    day: Optional[str] = Query(None, description="Optional day-of-week filter (e.g. Monday)"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Optional hour filter (0-23)"),
):
    """
    Return hourly forecast series, optionally filtered by borough/date/day/hour.
    """
    df = _build_enriched_forecast_df()
    df = _apply_series_filters(df, borough=borough, date=date, day=day, hour=hour)

    response_cols = [
        "borough",
        "model",
        "ds",
        "date",
        "day_of_week",
        "hour",
        "predicted_trips",
        "uncertainty_low",
        "uncertainty_high",
        "actual_trips",
        "absolute_error",
    ]
    output_df = df[response_cols].copy()
    output_df["ds"] = output_df["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_df = output_df.where(pd.notnull(output_df), None)
    return {
        "series": output_df.to_dict(orient="records"),
        "uncertainty_note": "Uncertainty band uses prediction +/- borough MAE from backtesting.",
    }


@app.get("/forecast/export", tags=["Forecast"])
async def export_forecast_series(
    borough: Optional[str] = Query(None, description="Optional borough filter"),
    date: Optional[str] = Query(None, description="Optional date filter (YYYY-MM-DD)"),
    day: Optional[str] = Query(None, description="Optional day-of-week filter (e.g. Monday)"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Optional hour filter (0-23)"),
    file_type: str = Query("csv", pattern="^(csv|xlsx)$", description="Export format"),
):
    """
    Export filtered forecast rows (with uncertainty and actuals) as CSV or XLSX.
    """
    df = _build_enriched_forecast_df()
    df = _apply_series_filters(df, borough=borough, date=date, day=day, hour=hour)

    export_cols = [
        "borough",
        "model",
        "ds",
        "date",
        "day_of_week",
        "hour",
        "predicted_trips",
        "uncertainty_low",
        "uncertainty_high",
        "actual_trips",
        "absolute_error",
    ]
    export_df = df[export_cols].copy()
    export_df["ds"] = export_df["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"borough_forecast_filtered_{ts}"

    if file_type == "csv":
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        return StreamingResponse(
            BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{base}.csv"'},
        )

    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="forecast")
    xlsx_buffer.seek(0)
    return StreamingResponse(
        xlsx_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{base}.xlsx"'},
    )


@app.get("/forecast/ui", response_class=HTMLResponse, tags=["Forecast"])
async def forecast_ui():
    """
    Small interactive UI for production borough forecast outputs.
    """
    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NYC Borough Demand Forecast</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    crossorigin=""
  />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <style>
    :root {
      --taxi-yellow: #ffcf00;
      --taxi-yellow-soft: #ffe56a;
      --taxi-black: #0d0d0d;
      --panel: #ffffff;
      --line: #2a2a2a;
      --text: #ffffff;
      --muted: #6f6f6f;
      --shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 0%, rgba(255, 207, 0, 0.22) 0%, rgba(255, 207, 0, 0) 44%),
        radial-gradient(circle at 82% 6%, rgba(255, 207, 0, 0.14) 0%, rgba(255, 207, 0, 0) 45%),
        linear-gradient(180deg, #050505 0%, #111111 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }
    .hero {
      background: linear-gradient(135deg, #ffd93f 0%, var(--taxi-yellow) 52%, #ffc000 100%);
      border: 2px solid #ffefae;
      border-radius: 16px;
      padding: 16px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      box-shadow: var(--shadow);
    }
    .hero-left { display: flex; align-items: center; gap: 14px; }
    .hero h1 {
      margin: 0;
      font-size: clamp(1.25rem, 2.8vw, 1.9rem);
      letter-spacing: 0.3px;
      color: #111111;
    }
    .hero p {
      margin: 3px 0 0;
      color: #1b1b1b;
      font-size: 0.93rem;
    }
    .taxi-icon {
      width: 68px;
      height: 46px;
      flex: 0 0 auto;
      border-radius: 10px;
      background: rgba(0, 0, 0, 0.1);
      display: grid;
      place-items: center;
      border: 1px solid rgba(0, 0, 0, 0.2);
    }
    .hero-right {
      font-weight: 700;
      font-size: 0.9rem;
      color: #ffffff;
      background: #0d0d0d;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #ffd84d;
      white-space: nowrap;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .card {
      background: var(--panel);
      border: 1px solid #d9d9d9;
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
    }
    .label { color: #8a6c00; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .value { margin-top: 6px; font-size: 1.28rem; font-weight: 800; color: #111111; }
    .controls {
      margin: 14px 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      align-items: end;
    }
    .field {
      background: var(--panel);
      border: 1px solid #d9d9d9;
      border-radius: 10px;
      padding: 8px 10px;
    }
    .field label {
      display: block;
      font-size: 0.73rem;
      color: #8a6c00;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 5px;
      font-weight: 700;
    }
    select, button {
      width: 100%;
      border: 1px solid #d1d1d1;
      background: #ffffff;
      border-radius: 8px;
      padding: 8px 9px;
      font-size: 0.92rem;
      color: #111111;
    }
    button {
      background: var(--taxi-yellow);
      color: #111111;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.08s ease, opacity 0.15s ease;
    }
    .actions {
      display: grid;
      gap: 6px;
    }
    .btn-secondary {
      background: #111111;
      color: #ffffff;
      border-color: #111111;
    }
    button:hover { opacity: 0.92; }
    button:active { transform: translateY(1px); }
    .chart-wrap {
      background: var(--panel);
      border: 1px solid #d9d9d9;
      border-radius: 14px;
      padding: 8px;
      box-shadow: var(--shadow);
    }
    #chart {
      width: 100%;
      height: 320px;
      background: #ffffff;
      border-radius: 10px;
    }
    .map-wrap {
      margin-top: 12px;
      background: var(--panel);
      border: 1px solid #d9d9d9;
      border-radius: 14px;
      padding: 10px;
      box-shadow: var(--shadow);
    }
    .map-title {
      font-weight: 800;
      color: #111111;
      font-size: 1.05rem;
      margin-bottom: 4px;
    }
    .map-sub {
      color: #666666;
      font-size: 0.85rem;
      margin-bottom: 8px;
    }
    #boroughMap {
      width: 100%;
      height: 380px;
      border: 1px solid #e1e1e1;
      border-radius: 10px;
      overflow: hidden;
    }
    .map-tooltip {
      font-size: 12px;
      line-height: 1.35;
    }
    .muted { color: #f5d86c; font-size: 0.79rem; margin: 8px 4px 0; }
    .table-wrap {
      margin-top: 12px;
      background: var(--panel);
      border: 1px solid #d9d9d9;
      border-radius: 14px;
      padding: 6px;
      box-shadow: var(--shadow);
      max-height: 350px;
      overflow: auto;
    }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 8px 9px; font-size: 0.85rem; text-align: left; border-bottom: 1px solid #e7e7e7; color: #111111; }
    th { background: #ffcf00; color: #101010; position: sticky; top: 0; z-index: 1; }
    tbody tr:nth-child(odd) { background: #ffffff; }
    tbody tr:nth-child(even) { background: #f8f8f8; }
    .empty {
      text-align: center;
      color: var(--muted);
      padding: 16px;
      font-size: 0.88rem;
    }
    @media (max-width: 650px) {
      .hero { align-items: flex-start; flex-direction: column; }
      .hero-right { align-self: flex-start; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="hero-left">
        <div class="taxi-icon" aria-hidden="true">
          <svg viewBox="0 0 120 70" width="56" height="34">
            <rect x="12" y="22" width="95" height="25" rx="7" fill="#1f1f1f"></rect>
            <rect x="28" y="13" width="44" height="14" rx="4" fill="#1f1f1f"></rect>
            <rect x="34" y="16" width="13" height="8" rx="2" fill="#f7c600"></rect>
            <rect x="50" y="16" width="17" height="8" rx="2" fill="#f7c600"></rect>
            <circle cx="33" cy="50" r="9" fill="#1f1f1f"></circle>
            <circle cx="86" cy="50" r="9" fill="#1f1f1f"></circle>
            <circle cx="33" cy="50" r="4" fill="#f7c600"></circle>
            <circle cx="86" cy="50" r="4" fill="#f7c600"></circle>
            <rect x="79" y="28" width="19" height="8" rx="2" fill="#f7c600"></rect>
            <rect x="57" y="8" width="14" height="4" rx="2" fill="#1f1f1f"></rect>
          </svg>
        </div>
        <div>
          <h1>NYC Yellow Taxi Demand Forecast</h1>
          <p>Production 7-day hourly outlook with borough-level model selection and drill-down filters.</p>
        </div>
      </div>
      <div class="hero-right">Forecast Dashboard</div>
    </div>

    <div class="row">
      <div class="card"><div class="label">Forecast Window</div><div id="window" class="value">-</div></div>
      <div class="card"><div class="label">Best Borough Model</div><div id="bestModel" class="value">-</div></div>
      <div class="card"><div class="label">Predicted Trips (Filtered)</div><div id="totalTrips" class="value">-</div></div>
      <div class="card"><div class="label">Visible Rows</div><div id="rowsCount" class="value">-</div></div>
    </div>

    <div class="controls">
      <div class="field">
        <label for="boroughSelect">Borough</label>
        <select id="boroughSelect"></select>
      </div>
      <div class="field">
        <label for="dateSelect">Date</label>
        <select id="dateSelect"></select>
      </div>
      <div class="field">
        <label for="daySelect">Day</label>
        <select id="daySelect"></select>
      </div>
      <div class="field">
        <label for="hourSelect">Hour</label>
        <select id="hourSelect"></select>
      </div>
      <div class="field">
        <label for="resetBtn">Actions</label>
        <div class="actions">
          <button id="resetBtn" type="button">Reset Filters</button>
          <button id="exportCsvBtn" type="button" class="btn-secondary">Export CSV</button>
          <button id="exportXlsxBtn" type="button" class="btn-secondary">Export XLSX</button>
        </div>
      </div>
    </div>

    <div class="chart-wrap">
      <svg id="chart" viewBox="0 0 1200 320" preserveAspectRatio="none"></svg>
    </div>
    <div id="uncertaintyNote" class="muted">Black line: forecast, orange line: actual, yellow band: uncertainty proxy.</div>

    <div class="map-wrap">
      <div class="map-title">Geographical Forecast Map (Borough)</div>
      <div class="map-sub">Each borough has its own color. Hover a borough for Forecast vs Actual and model details.</div>
      <div id="boroughMap"></div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Date Time</th><th>Day</th><th>Hour</th><th>Predicted</th><th>Band Low</th><th>Band High</th><th>Actual</th><th>Abs Error</th></tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const state = {
      summary: null,
      bestModels: [],
      boroughs: [],
      rawSeries: [],
      filteredSeries: [],
      uncertaintyNote: "",
      boroughTotals: [],
      boroughTotalsByName: {},
      mapHasFit: false,
      map: null,
      mapLayer: null,
    };
    const WEEK_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    const NYC_BOROUGH_GEOJSON_URL = "https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson";
    const BOROUGH_COLORS = {
      "Manhattan": "#f4c20d",
      "Brooklyn": "#1f77b4",
      "Queens": "#2ca02c",
      "Bronx": "#ff7f0e",
      "Staten Island": "#9467bd",
      "EWR": "#d62728",
      "Unknown": "#7f7f7f",
    };

    function fmtNum(n) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 }); }
    function numOrDash(v) { return Number.isFinite(Number(v)) ? fmtNum(v) : "-"; }
    function byId(id) { return document.getElementById(id); }

    async function loadJSON(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Request failed: ${url}`);
      return res.json();
    }

    function setCards(borough, series) {
      const bm = state.bestModels.find(r => r.borough === borough);
      const label = bm ? `${bm.model} (MAE ${Number(bm.mae).toFixed(2)})` : "n/a";
      byId("bestModel").textContent = label;
      const total = series.reduce((s, r) => s + Number(r.predicted_trips || 0), 0);
      byId("totalTrips").textContent = fmtNum(total);
      byId("rowsCount").textContent = series.length.toLocaleString();
      const w = state.summary?.forecast_window;
      byId("window").textContent = w ? `${w.start} to ${w.end}` : "-";
    }

    function renderTable(series) {
      const tbody = byId("tableBody");
      tbody.innerHTML = "";
      if (!Array.isArray(series)) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td class="empty" colspan="8">Table data is not available.</td>`;
        tbody.appendChild(tr);
        return;
      }
      if (!series.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td class="empty" colspan="8">No rows for this filter combination.</td>`;
        tbody.appendChild(tr);
        return;
      }

      for (const row of series.slice(0, 336)) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${row.ds}</td><td>${row.day_of_week}</td><td>${row.hour}</td><td>${numOrDash(row.predicted_trips)}</td><td>${numOrDash(row.uncertainty_low)}</td><td>${numOrDash(row.uncertainty_high)}</td><td>${numOrDash(row.actual_trips)}</td><td>${numOrDash(row.absolute_error)}</td>`;
        tbody.appendChild(tr);
      }
    }

    function renderChart(series) {
      const svg = byId("chart");
      const width = 1200, height = 320, padL = 54, padR = 20, padT = 16, padB = 36;
      svg.innerHTML = "";

      if (!series.length) {
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", width / 2);
        text.setAttribute("y", height / 2);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("fill", "#111111");
        text.setAttribute("font-size", "20");
        text.textContent = "No data for selected filters";
        svg.appendChild(text);
        return;
      }

      const pred = series.map(r => Number(r.predicted_trips || 0));
      const low = series.map((r, i) => Number.isFinite(Number(r.uncertainty_low)) ? Number(r.uncertainty_low) : pred[i]);
      const high = series.map((r, i) => Number.isFinite(Number(r.uncertainty_high)) ? Number(r.uncertainty_high) : pred[i]);
      const actual = series.map(r => Number.isFinite(Number(r.actual_trips)) ? Number(r.actual_trips) : null);
      const yAll = [...pred, ...low, ...high, ...actual.filter(v => v !== null)];
      const minY = Math.min(...yAll);
      const maxY = Math.max(...yAll);
      const rangeY = Math.max(maxY - minY, 1);

      function xAt(i) {
        if (series.length === 1) return (width - padL - padR) / 2 + padL;
        return padL + (i / (series.length - 1)) * (width - padL - padR);
      }
      function yAt(y) {
        return height - padB - ((y - minY) / rangeY) * (height - padT - padB);
      }

      for (let i = 0; i <= 4; i += 1) {
        const ratio = i / 4;
        const y = padT + ratio * (height - padT - padB);
        const value = maxY - ratio * rangeY;

        const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
        grid.setAttribute("x1", padL);
        grid.setAttribute("x2", width - padR);
        grid.setAttribute("y1", y);
        grid.setAttribute("y2", y);
        grid.setAttribute("stroke", "#e7e7e7");
        grid.setAttribute("stroke-width", "1");
        svg.appendChild(grid);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", 8);
        label.setAttribute("y", y + 4);
        label.setAttribute("fill", "#555555");
        label.setAttribute("font-size", "12");
        label.textContent = Math.round(value).toLocaleString();
        svg.appendChild(label);
      }

      const predPoints = pred.map((y, i) => `${xAt(i)},${yAt(y)}`).join(" ");
      const highPoints = high.map((y, i) => `${xAt(i)},${yAt(y)}`);
      const lowPoints = low.map((y, i) => `${xAt(i)},${yAt(y)}`);
      const bandPoints = `${highPoints.join(" ")} ${lowPoints.reverse().join(" ")}`;

      const band = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      band.setAttribute("points", bandPoints);
      band.setAttribute("fill", "rgba(255, 207, 0, 0.30)");
      band.setAttribute("stroke", "none");
      svg.appendChild(band);

      const predLine = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      predLine.setAttribute("points", predPoints);
      predLine.setAttribute("fill", "none");
      predLine.setAttribute("stroke", "#111111");
      predLine.setAttribute("stroke-width", "2.5");
      svg.appendChild(predLine);

      const actualPoints = actual
        .map((v, i) => (v === null ? null : `${xAt(i)},${yAt(v)}`))
        .filter(Boolean)
        .join(" ");
      if (actualPoints) {
        const actualLine = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        actualLine.setAttribute("points", actualPoints);
        actualLine.setAttribute("fill", "none");
        actualLine.setAttribute("stroke", "#f08c00");
        actualLine.setAttribute("stroke-width", "2.2");
        actualLine.setAttribute("stroke-dasharray", "6 4");
        svg.appendChild(actualLine);
      }

      const axisY = document.createElementNS("http://www.w3.org/2000/svg", "line");
      axisY.setAttribute("x1", padL);
      axisY.setAttribute("x2", padL);
      axisY.setAttribute("y1", padT);
      axisY.setAttribute("y2", height - padB);
      axisY.setAttribute("stroke", "#111111");
      axisY.setAttribute("stroke-width", "1.2");
      svg.appendChild(axisY);

      const axisX = document.createElementNS("http://www.w3.org/2000/svg", "line");
      axisX.setAttribute("x1", padL);
      axisX.setAttribute("x2", width - padR);
      axisX.setAttribute("y1", height - padB);
      axisX.setAttribute("y2", height - padB);
      axisX.setAttribute("stroke", "#111111");
      axisX.setAttribute("stroke-width", "1.2");
      svg.appendChild(axisX);

      const labelIdx = [0, Math.floor((series.length - 1) / 2), series.length - 1]
        .filter((idx, pos, arr) => arr.indexOf(idx) === pos);
      for (const idx of labelIdx) {
        const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("x", xAt(idx));
        t.setAttribute("y", height - 10);
        t.setAttribute("text-anchor", idx === 0 ? "start" : (idx === series.length - 1 ? "end" : "middle"));
        t.setAttribute("fill", "#444444");
        t.setAttribute("font-size", "12");
        t.textContent = series[idx].ds.slice(5, 16);
        svg.appendChild(t);
      }

      const legendX = width - 285;
      const legendY = 18;
      const legendW = 265;
      const legendH = 74;

      const legendBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      legendBg.setAttribute("x", legendX);
      legendBg.setAttribute("y", legendY);
      legendBg.setAttribute("width", legendW);
      legendBg.setAttribute("height", legendH);
      legendBg.setAttribute("rx", "8");
      legendBg.setAttribute("fill", "rgba(255, 255, 255, 0.90)");
      legendBg.setAttribute("stroke", "#dddddd");
      svg.appendChild(legendBg);

      function legendLine(y, color, label, dashed = false) {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", legendX + 14);
        line.setAttribute("x2", legendX + 54);
        line.setAttribute("y1", y);
        line.setAttribute("y2", y);
        line.setAttribute("stroke", color);
        line.setAttribute("stroke-width", "3");
        if (dashed) line.setAttribute("stroke-dasharray", "6 4");
        svg.appendChild(line);

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", legendX + 62);
        text.setAttribute("y", y + 4);
        text.setAttribute("fill", "#222222");
        text.setAttribute("font-size", "12");
        text.textContent = label;
        svg.appendChild(text);
      }

      const bandSwatch = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bandSwatch.setAttribute("x", legendX + 14);
      bandSwatch.setAttribute("y", legendY + 12);
      bandSwatch.setAttribute("width", "40");
      bandSwatch.setAttribute("height", "10");
      bandSwatch.setAttribute("fill", "rgba(255, 207, 0, 0.30)");
      bandSwatch.setAttribute("stroke", "#d8b11f");
      svg.appendChild(bandSwatch);
      const bandLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      bandLabel.setAttribute("x", legendX + 62);
      bandLabel.setAttribute("y", legendY + 21);
      bandLabel.setAttribute("fill", "#222222");
      bandLabel.setAttribute("font-size", "12");
      bandLabel.textContent = "Uncertainty (+/- MAE)";
      svg.appendChild(bandLabel);

      legendLine(legendY + 40, "#111111", "Forecast");
      legendLine(legendY + 58, "#f08c00", "Actual", true);
    }

    async function loadSeries(borough) {
      const data = await loadJSON(`/forecast/series?borough=${encodeURIComponent(borough)}`);
      state.uncertaintyNote = data.uncertainty_note || "";
      return data.series || [];
    }

    function normalizeBoroughName(name) {
      if (!name) return "";
      const text = String(name).trim();
      if (text.toLowerCase() === "the bronx") return "Bronx";
      return text;
    }

    async function loadBoroughTotals() {
      const data = await loadJSON("/forecast/borough-totals");
      state.boroughTotals = data.borough_totals || [];
      state.boroughTotalsByName = {};
      for (const row of state.boroughTotals) {
        const key = normalizeBoroughName(row.borough);
        state.boroughTotalsByName[key] = row;
      }
    }

    function mapColor(borough) {
      const key = normalizeBoroughName(borough);
      return BOROUGH_COLORS[key] || "#bdbdbd";
    }

    function styleFeature(feature) {
      const borough = normalizeBoroughName(feature?.properties?.BoroName);
      return {
        fillColor: mapColor(borough),
        weight: 1.4,
        opacity: 1,
        color: "#202020",
        dashArray: "",
        fillOpacity: 0.76,
      };
    }

    function applySelectedBoroughStyle() {
      if (!state.mapLayer) return;
      const selected = normalizeBoroughName(byId("boroughSelect")?.value || "");
      state.mapLayer.eachLayer(layer => {
        const borough = normalizeBoroughName(layer.feature?.properties?.BoroName);
        layer.setStyle(styleFeature(layer.feature));
        if (borough === selected) {
          layer.setStyle({ weight: 3.2, color: "#000000", fillOpacity: 0.88 });
        }
      });
    }

    async function renderBoroughMap() {
      if (!state.map) {
        state.map = L.map("boroughMap", { zoomControl: true }).setView([40.72, -73.94], 10);
        L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
          attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
          maxZoom: 18,
        }).addTo(state.map);
      }

      const geo = await loadJSON(NYC_BOROUGH_GEOJSON_URL);
      if (state.mapLayer) {
        state.map.removeLayer(state.mapLayer);
      }

      state.mapLayer = L.geoJSON(geo, {
        style: styleFeature,
        onEachFeature: (feature, layer) => {
          const borough = normalizeBoroughName(feature?.properties?.BoroName);
          const row = state.boroughTotalsByName[borough] || {};
          const forecast = Number.isFinite(Number(row.forecast_trips)) ? fmtNum(row.forecast_trips) : "n/a";
          const actual = Number.isFinite(Number(row.actual_trips)) ? fmtNum(row.actual_trips) : "n/a";
          const wape = Number.isFinite(Number(row.wape_pct)) ? `${Number(row.wape_pct).toFixed(1)}%` : "n/a";
          const model = row.model || "n/a";

          layer.bindTooltip(
            `<div class="map-tooltip"><strong>${borough}</strong><br/>Forecast: ${forecast}<br/>Actual: ${actual}<br/>WAPE: ${wape}<br/>Model: ${model}</div>`,
            { sticky: true }
          );

          layer.on({
            mouseover: e => {
              const l = e.target;
              l.setStyle({ weight: 2.8, color: "#111111", fillOpacity: 0.9 });
              if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
                l.bringToFront();
              }
            },
            mouseout: e => {
              state.mapLayer.resetStyle(e.target);
              applySelectedBoroughStyle();
            },
            click: async () => {
              if (state.boroughs.includes(borough)) {
                byId("boroughSelect").value = borough;
                await onBoroughChange();
              }
            },
          });
        },
      }).addTo(state.map);

      if (!state.mapHasFit) {
        state.map.fitBounds(state.mapLayer.getBounds(), { padding: [12, 12] });
        state.mapHasFit = true;
      }
      applySelectedBoroughStyle();
    }

    function buildFilterParams() {
      const params = new URLSearchParams();
      const borough = byId("boroughSelect").value;
      const selectedDate = byId("dateSelect").value;
      const selectedDay = byId("daySelect").value;
      const selectedHour = byId("hourSelect").value;

      if (borough) params.set("borough", borough);
      if (selectedDate !== "all") params.set("date", selectedDate);
      if (selectedDay !== "all") params.set("day", selectedDay);
      if (selectedHour !== "all") params.set("hour", Number(selectedHour));
      return params;
    }

    function downloadFiltered(fileType) {
      const params = buildFilterParams();
      params.set("file_type", fileType);
      window.location.href = `/forecast/export?${params.toString()}`;
    }

    function setSelectOptions(id, values, allLabel) {
      const select = byId(id);
      select.innerHTML = "";
      const all = document.createElement("option");
      all.value = "all";
      all.textContent = allLabel;
      select.appendChild(all);

      for (const value of values) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }
    }

    function prepareFilterOptions() {
      const dates = [...new Set(state.rawSeries.map(r => r.date))].sort();
      const days = WEEK_DAYS.filter(d => state.rawSeries.some(r => r.day_of_week === d));
      setSelectOptions("dateSelect", dates, "All Dates");
      setSelectOptions("daySelect", days, "All Days");
      setSelectOptions("hourSelect", Array.from({ length: 24 }, (_, h) => String(h).padStart(2, "0")), "All Hours");
    }

    function applyFilters() {
      const selectedDate = byId("dateSelect").value;
      const selectedDay = byId("daySelect").value;
      const selectedHour = byId("hourSelect").value;

      state.filteredSeries = state.rawSeries.filter(row => {
        if (selectedDate !== "all" && row.date !== selectedDate) return false;
        if (selectedDay !== "all" && row.day_of_week !== selectedDay) return false;
        if (selectedHour !== "all" && Number(row.hour) !== Number(selectedHour)) return false;
        return true;
      });

      const borough = byId("boroughSelect").value;
      setCards(borough, state.filteredSeries);
      renderTable(state.filteredSeries);
      try {
        renderChart(state.filteredSeries);
      } catch (err) {
        console.error("Chart render failed:", err);
      }
      byId("uncertaintyNote").textContent = `Black line: forecast, orange line: actual, yellow band: ${state.uncertaintyNote || "uncertainty proxy"}`;
    }

    async function onBoroughChange() {
      const borough = byId("boroughSelect").value;
      state.rawSeries = await loadSeries(borough);
      prepareFilterOptions();
      applyFilters();
      applySelectedBoroughStyle();
    }

    async function init() {
      const [summary, best, boroughs] = await Promise.all([
        loadJSON("/forecast/summary"),
        loadJSON("/forecast/best-models"),
        loadJSON("/forecast/boroughs"),
      ]);

      state.summary = summary;
      state.bestModels = best.best_models || [];
      state.boroughs = boroughs.boroughs || [];

      const boroughSelect = byId("boroughSelect");
      for (const b of state.boroughs) {
        const opt = document.createElement("option");
        opt.value = b;
        opt.textContent = b;
        boroughSelect.appendChild(opt);
      }

      boroughSelect.addEventListener("change", onBoroughChange);
      byId("dateSelect").addEventListener("change", applyFilters);
      byId("daySelect").addEventListener("change", applyFilters);
      byId("hourSelect").addEventListener("change", applyFilters);
      byId("exportCsvBtn").addEventListener("click", () => downloadFiltered("csv"));
      byId("exportXlsxBtn").addEventListener("click", () => downloadFiltered("xlsx"));
      byId("resetBtn").addEventListener("click", () => {
        byId("dateSelect").value = "all";
        byId("daySelect").value = "all";
        byId("hourSelect").value = "all";
        applyFilters();
      });

      await loadBoroughTotals();
      await renderBoroughMap();

      if (state.boroughs.length) {
        boroughSelect.value = state.boroughs[0];
        await onBoroughChange();
      }
    }

    init().catch(err => {
      console.error(err);
      alert("Failed to load forecast UI data. Check forecast files and API logs.");
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

