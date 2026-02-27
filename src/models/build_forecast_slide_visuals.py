"""
Build slide-ready forecast visualizations:
1) NYC taxi-zone demand hotspot map from borough forecasts.
2) Model comparison chart from production metrics.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path
from typing import Iterable, List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

try:
    import contextily as ctx
except Exception:
    ctx = None


TAXI_ZONES_ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
MONTHLY_FILE_PATTERN = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create forecast map and model comparison visuals for slides."
    )
    parser.add_argument(
        "--forecast-csv",
        type=Path,
        default=Path("reports/forecasts/production_borough_7day_hourly_forecast.csv"),
        help="Forecast CSV with borough, ds, predicted_trips.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("reports/forecasts/production_all_model_metrics.csv"),
        help="Model metrics CSV with borough/model MAE/RMSE.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed monthly parquet files.",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("data/external"),
        help="Directory used to cache external geo data.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Output directory for slide visuals.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=60,
        help="Recent history window used for zone-share allocation.",
    )
    parser.add_argument(
        "--top-zones",
        type=int,
        default=20,
        help="Number of top forecast zones to export as CSV.",
    )
    return parser.parse_args()


def ensure_taxi_zones_shapefile(external_dir: Path) -> Path:
    """
    Download NYC TLC taxi zone shapefile if missing and return shapefile path.
    """
    target_dir = external_dir / "taxi_zones"
    target_dir.mkdir(parents=True, exist_ok=True)

    shapefile = target_dir / "taxi_zones.shp"
    if shapefile.exists():
        return shapefile

    response = requests.get(TAXI_ZONES_ZIP_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(target_dir)

    if not shapefile.exists():
        shp_candidates = list(target_dir.rglob("*.shp"))
        if not shp_candidates:
            raise FileNotFoundError(
                f"No shapefile found after extracting taxi zones into {target_dir}."
            )
        shapefile = shp_candidates[0]

    return shapefile


def monthly_files_in_window(
    processed_dir: Path, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> List[Path]:
    """
    Return processed monthly files that overlap the [window_start, window_end] window.
    """
    files: List[Path] = []
    for path in sorted(processed_dir.glob("yellow_tripdata_*.parquet")):
        match = MONTHLY_FILE_PATTERN.search(path.name)
        if not match:
            continue
        year = int(match.group(1))
        month = int(match.group(2))
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
        if month_end >= window_start and month_start <= window_end:
            files.append(path)
    return files


def load_recent_zone_counts(
    processed_dir: Path, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> pd.DataFrame:
    """
    Aggregate pickup trip counts by pickup_location_id over the lookback window.
    """
    files = monthly_files_in_window(processed_dir, window_start, window_end)
    if not files:
        raise FileNotFoundError(
            f"No processed monthly files found in {processed_dir} for window "
            f"{window_start} to {window_end}."
        )

    chunks: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_parquet(path, columns=["pickup_datetime", "pickup_location_id"])
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["pickup_location_id"] = pd.to_numeric(df["pickup_location_id"], errors="coerce")
        df = df.dropna(subset=["pickup_datetime", "pickup_location_id"])
        df = df[
            (df["pickup_datetime"] >= window_start)
            & (df["pickup_datetime"] <= window_end)
        ].copy()
        if df.empty:
            continue

        grp = (
            df.groupby("pickup_location_id", as_index=False)
            .size()
            .rename(columns={"pickup_location_id": "LocationID", "size": "recent_trips"})
        )
        grp["LocationID"] = grp["LocationID"].astype(int)
        chunks.append(grp)

    if not chunks:
        raise ValueError(
            "No recent pickup rows found in the selected lookback window. "
            "Try increasing --lookback-days."
        )

    zone_counts = (
        pd.concat(chunks, ignore_index=True)
        .groupby("LocationID", as_index=False)["recent_trips"]
        .sum()
    )
    return zone_counts


def build_zone_forecast_allocation(
    forecast_csv: Path,
    zones_gdf: gpd.GeoDataFrame,
    zone_counts: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Allocate borough forecast totals to zones using recent zone share.
    """
    forecast = pd.read_csv(forecast_csv, parse_dates=["ds"])
    borough_totals = (
        forecast.groupby("borough", as_index=False)["predicted_trips"]
        .sum()
        .rename(columns={"borough": "Borough", "predicted_trips": "forecast_7d_trips"})
    )

    z = zones_gdf.copy()
    if "Borough" not in z.columns and "borough" in z.columns:
        z = z.rename(columns={"borough": "Borough"})
    if "zone" not in z.columns and "Zone" in z.columns:
        z = z.rename(columns={"Zone": "zone"})

    required_cols = {"LocationID", "Borough", "zone"}
    missing = required_cols - set(z.columns)
    if missing:
        raise KeyError(f"Taxi zone shapefile missing columns: {sorted(missing)}")

    z["LocationID"] = pd.to_numeric(z["LocationID"], errors="coerce").astype("Int64")
    z = z.dropna(subset=["LocationID"]).copy()
    z["LocationID"] = z["LocationID"].astype(int)

    demand = z.merge(zone_counts, on="LocationID", how="left")
    demand["recent_trips"] = demand["recent_trips"].fillna(0.0)

    borough_recent = (
        demand.groupby("Borough", as_index=False)["recent_trips"]
        .sum()
        .rename(columns={"recent_trips": "borough_recent_trips"})
    )
    demand = demand.merge(borough_recent, on="Borough", how="left")
    demand["zone_share"] = np.where(
        demand["borough_recent_trips"] > 0,
        demand["recent_trips"] / demand["borough_recent_trips"],
        0.0,
    )
    demand = demand.merge(borough_totals, on="Borough", how="left")
    demand["forecast_7d_trips"] = demand["forecast_7d_trips"].fillna(0.0)
    demand["zone_forecast_7d_trips"] = demand["forecast_7d_trips"] * demand["zone_share"]

    return demand


def plot_forecast_hotspot_map(zone_forecast_gdf: gpd.GeoDataFrame, out_png: Path) -> None:
    """
    Plot and save NYC taxi-zone forecast demand map.
    """
    plot_gdf = zone_forecast_gdf[
        ~zone_forecast_gdf["Borough"].isin(["Unknown", "N/A", None])
    ].copy()

    fig, ax = plt.subplots(figsize=(13, 13))
    plot_gdf = plot_gdf.to_crs(epsg=3857)

    plot_gdf.plot(
        column="zone_forecast_7d_trips",
        cmap="YlOrRd",
        linewidth=0.22,
        edgecolor="#3a3a3a",
        legend=True,
        ax=ax,
        legend_kwds={"label": "Forecast Trips (Next 7 Days)", "shrink": 0.72},
    )

    if ctx is not None:
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                alpha=0.65,
            )
        except Exception:
            pass

    ax.set_title(
        "NYC Yellow Taxi Forecast Demand Hotspots (Next 7 Days)\n"
        "Zone-level allocation from borough forecasts using recent pickup shares",
        fontsize=17,
        weight="bold",
        pad=14,
    )
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ranked_hotspots_slide(
    zone_forecast_gdf: gpd.GeoDataFrame, top_n: int, out_png: Path
) -> None:
    """
    Plot presentation map with ranked hotspot markers + side table of forecast trips.
    """
    plot_gdf = zone_forecast_gdf[
        ~zone_forecast_gdf["Borough"].isin(["Unknown", "N/A", None])
    ].copy()
    plot_gdf = plot_gdf.to_crs(epsg=3857)

    hotspots = (
        plot_gdf.sort_values("zone_forecast_7d_trips", ascending=False)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    hotspots["rank"] = np.arange(1, len(hotspots) + 1)
    reps = hotspots.geometry.representative_point()
    hotspots["x"] = reps.x
    hotspots["y"] = reps.y

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.8], wspace=0.02)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    plot_gdf.plot(
        column="zone_forecast_7d_trips",
        cmap="YlOrRd",
        linewidth=0.2,
        edgecolor="#3a3a3a",
        legend=True,
        ax=ax_map,
        legend_kwds={"label": "Forecast Trips (Next 7 Days)", "shrink": 0.68},
    )

    if ctx is not None:
        try:
            ctx.add_basemap(ax_map, source=ctx.providers.CartoDB.Positron, alpha=0.62)
        except Exception:
            pass

    ax_map.scatter(
        hotspots["x"],
        hotspots["y"],
        s=115,
        c="#111111",
        edgecolors="#ffffff",
        linewidths=1.0,
        zorder=8,
    )
    for _, row in hotspots.iterrows():
        ax_map.text(
            row["x"],
            row["y"],
            str(int(row["rank"])),
            color="white",
            fontsize=8.5,
            ha="center",
            va="center",
            weight="bold",
            zorder=9,
        )

    ax_map.set_title(
        "NYC Forecast Demand Hotspots (Next 7 Days)\n"
        "Numbered spots = highest predicted pickup zones",
        fontsize=16,
        weight="bold",
        pad=10,
    )
    ax_map.set_axis_off()

    table_df = hotspots[["rank", "zone", "Borough", "zone_forecast_7d_trips"]].copy()
    table_df["zone"] = table_df["zone"].astype(str).str.slice(0, 28)
    table_df["zone_forecast_7d_trips"] = table_df["zone_forecast_7d_trips"].round(0).astype(int)
    table_df = table_df.rename(
        columns={
            "rank": "#",
            "zone": "Zone",
            "Borough": "Borough",
            "zone_forecast_7d_trips": "Forecast Trips",
        }
    )

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        loc="center",
        colColours=["#f7c600"] * len(table_df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.42)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="#111111")
        else:
            cell.set_facecolor("#fffdf4" if r % 2 else "#f7f7f7")

    ax_tbl.set_title(
        f"Top {top_n} Forecast Spots",
        fontsize=13,
        weight="bold",
        pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_csv: Path, out_png: Path, out_summary_csv: Path) -> None:
    """
    Plot slide-ready model comparison summary.
    """
    metrics = pd.read_csv(metrics_csv)
    metrics["mae"] = pd.to_numeric(metrics["mae"], errors="coerce")
    metrics["rmse"] = pd.to_numeric(metrics["rmse"], errors="coerce")
    metrics = metrics.dropna(subset=["model", "borough", "mae", "rmse"]).copy()

    summary = (
        metrics.groupby("model", as_index=False)[["mae", "rmse"]]
        .mean()
        .sort_values("mae")
        .reset_index(drop=True)
    )

    winners = (
        metrics.sort_values("mae")
        .groupby("borough", as_index=False)
        .first()[["borough", "model", "mae"]]
    )
    win_counts = (
        winners.groupby("model", as_index=False)
        .size()
        .rename(columns={"size": "best_borough_count"})
    )

    baseline = metrics[metrics["model"].str.lower() == "glm"][["borough", "mae"]].rename(
        columns={"mae": "glm_mae"}
    )
    best = (
        metrics.groupby("borough", as_index=False)["mae"]
        .min()
        .rename(columns={"mae": "best_mae"})
    )
    improv = best.merge(baseline, on="borough", how="left")
    improv["mae_improvement_pct_vs_glm"] = np.where(
        improv["glm_mae"] > 0,
        (improv["glm_mae"] - improv["best_mae"]) / improv["glm_mae"] * 100.0,
        np.nan,
    )
    avg_improvement = float(improv["mae_improvement_pct_vs_glm"].mean(skipna=True))

    summary_out = summary.merge(win_counts, on="model", how="left").fillna(
        {"best_borough_count": 0}
    )
    summary_out["best_borough_count"] = summary_out["best_borough_count"].astype(int)
    summary_out.to_csv(out_summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    axes[0].bar(
        summary["model"],
        summary["mae"],
        color="#f7c600",
        edgecolor="#212121",
        linewidth=1.0,
    )
    axes[0].set_title("Average MAE by Model (Lower is Better)", weight="bold")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].grid(axis="y", alpha=0.28)

    wins_plot = win_counts.merge(summary[["model"]], on="model", how="right").fillna(0)
    wins_plot["best_borough_count"] = wins_plot["best_borough_count"].astype(int)
    axes[1].bar(
        wins_plot["model"],
        wins_plot["best_borough_count"],
        color="#212121",
        edgecolor="#f7c600",
        linewidth=1.0,
    )
    axes[1].set_title("Best-Model Wins by Borough (MAE)", weight="bold")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Borough Win Count")
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].grid(axis="y", alpha=0.28)

    best_model = summary.iloc[0]
    fig.suptitle(
        "Forecast Model Comparison Summary",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"Best average MAE model: {best_model['model']} "
        f"(MAE {best_model['mae']:.2f}). "
        f"Mean MAE improvement vs GLM baseline: {avg_improvement:.1f}%",
        ha="center",
        fontsize=10.5,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_top_zone_table(
    zone_forecast_gdf: gpd.GeoDataFrame, top_n: int, out_csv: Path
) -> None:
    """
    Export top-N zones by forecasted 7-day trips.
    """
    cols = ["Borough", "zone", "LocationID", "zone_forecast_7d_trips", "zone_share"]
    top = (
        zone_forecast_gdf[cols]
        .sort_values("zone_forecast_7d_trips", ascending=False)
        .head(top_n)
        .copy()
    )
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["zone_forecast_7d_trips"] = top["zone_forecast_7d_trips"].round(2)
    top["zone_share"] = top["zone_share"].round(4)
    top.to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.external_dir.mkdir(parents=True, exist_ok=True)

    if not args.forecast_csv.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {args.forecast_csv}. "
            "Run borough forecasting pipeline first."
        )
    if not args.metrics_csv.exists():
        raise FileNotFoundError(
            f"Model metrics file not found: {args.metrics_csv}. "
            "Run model comparison pipeline first."
        )

    forecast_df = pd.read_csv(args.forecast_csv, parse_dates=["ds"])
    if forecast_df.empty:
        raise ValueError(f"Forecast file is empty: {args.forecast_csv}")

    forecast_start = forecast_df["ds"].min()
    window_end = forecast_start - pd.Timedelta(seconds=1)
    window_start = forecast_start - pd.Timedelta(days=args.lookback_days)

    shp_path = ensure_taxi_zones_shapefile(args.external_dir)
    zones = gpd.read_file(shp_path)

    zone_counts = load_recent_zone_counts(args.processed_dir, window_start, window_end)
    zone_forecast = build_zone_forecast_allocation(args.forecast_csv, zones, zone_counts)

    map_png = args.out_dir / "forecast_zone_demand_hotspots_map.png"
    ranked_map_png = args.out_dir / "forecast_zone_hotspots_ranked_slide.png"
    model_png = args.out_dir / "forecast_model_comparison_slide.png"
    top_csv = args.out_dir / "forecast_zone_top_areas.csv"
    summary_csv = args.out_dir / "forecast_model_comparison_summary.csv"

    plot_forecast_hotspot_map(zone_forecast, map_png)
    plot_ranked_hotspots_slide(zone_forecast, args.top_zones, ranked_map_png)
    plot_model_comparison(args.metrics_csv, model_png, summary_csv)
    save_top_zone_table(zone_forecast, args.top_zones, top_csv)

    print("Created slide visuals:")
    print(f"- {map_png}")
    print(f"- {ranked_map_png}")
    print(f"- {model_png}")
    print(f"- {top_csv}")
    print(f"- {summary_csv}")


if __name__ == "__main__":
    main()
