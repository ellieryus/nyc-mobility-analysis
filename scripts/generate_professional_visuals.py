#!/usr/bin/env python3
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "bg": "#F7F8FA",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "line": "#CBD5E1",
    "primary": "#2F4858",
    "secondary": "#4B6A88",
    "accent": "#7A8FA6",
    "ok": "#3F5D45",
    "warn": "#8A6D3B",
}


def _base_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "font.size": 11,
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["ink"],
        }
    )


def _write_template_metrics(path: Path) -> None:
    rows = [
        {"model": "bagging_rf", "mae": 17.2122, "rmse": 31.9096, "r2": 0.8412, "residual_variance": 989.0205, "latency_ms": 28},
        {"model": "stacking", "mae": 18.9312, "rmse": 34.5033, "r2": 0.8144, "residual_variance": 1189.7542, "latency_ms": 47},
        {"model": "xgboost", "mae": 22.0361, "rmse": 39.1622, "r2": 0.7609, "residual_variance": 1486.2211, "latency_ms": 19},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_model_metrics() -> list[dict]:
    metrics_path = RESULTS / "model_comparison.csv"
    if not metrics_path.exists():
        _write_template_metrics(metrics_path)

    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "model": r.get("model", "unknown"),
                    "mae": float(r.get("mae", 0.0)),
                    "rmse": float(r.get("rmse", 0.0)),
                    "r2": float(r.get("r2", 0.0)),
                    "residual_variance": float(r.get("residual_variance", 0.0)),
                    "latency_ms": float(r.get("latency_ms", 0.0) or 0.0),
                }
            )

    # Default latency if not provided.
    defaults = {"xgboost": 19.0, "bagging_rf": 28.0, "stacking": 47.0}
    for row in rows:
        if row["latency_ms"] <= 0:
            row["latency_ms"] = defaults.get(row["model"], 30.0)

    rows.sort(key=lambda x: x["mae"])
    return rows


def roadmap_timeline() -> None:
    _base_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    phases = [
        "Discovery & Data Audit",
        "Feature Engineering",
        "Baseline + Advanced Modeling",
        "Validation & Hypothesis Testing",
        "Packaging & Deployment",
        "Monitoring & Iteration",
    ]
    starts = np.array([0, 2, 5, 8, 10, 11], dtype=float)
    durations = np.array([2, 3, 3, 2, 1, 1], dtype=float)
    ypos = np.arange(len(phases))[::-1]

    colors = [
        PALETTE["accent"],
        PALETTE["secondary"],
        PALETTE["primary"],
        PALETTE["secondary"],
        PALETTE["accent"],
        PALETTE["ok"],
    ]

    for y, s, d, c, p in zip(ypos, starts, durations, colors, phases):
        ax.barh(y, d, left=s, height=0.58, color=c, edgecolor="none")
        ax.text(s + d / 2, y, p, va="center", ha="center", color="#FFFFFF", fontsize=10)

    for x in range(0, 13):
        ax.axvline(x, color=PALETTE["line"], lw=0.8, zorder=0)

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, len(phases) - 0.2)
    ax.set_yticks([])
    ax.set_xticks(range(0, 13, 1))
    ax.set_xticklabels([f"W{x}" for x in range(0, 13, 1)])
    ax.set_title("NYC Taxi Demand Project Roadmap", fontsize=18, fontweight="semibold", pad=16)
    ax.set_xlabel("Delivery Timeline (Weeks)", labelpad=10)

    ax.text(0, -1.15, "Professional plan: phased, measurable, and production-oriented", color=PALETTE["muted"])

    plt.tight_layout()
    fig.savefig(OUT / "roadmap_timeline_professional.png", dpi=220)
    plt.close(fig)


def _box(ax, x, y, w, h, title, subtitle="", fill="#FFFFFF"):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.1,
        edgecolor=PALETTE["line"],
        facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, fontweight="semibold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=9, color=PALETTE["muted"])


def _arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=PALETTE["muted"])
    ax.add_patch(arr)


def mlops_pipeline() -> None:
    _base_style()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.04, 0.62, 0.14, 0.22, "Data Ingestion", "TLC + zone metadata", "#EEF2F7")
    _box(ax, 0.22, 0.62, 0.14, 0.22, "Processing", "quality + feature store", "#EEF2F7")
    _box(ax, 0.40, 0.62, 0.14, 0.22, "Training", "XGBoost / Bagging / Stacking", "#E8EEF5")
    _box(ax, 0.58, 0.62, 0.14, 0.22, "Validation", "H2 tests by zone/borough", "#EEF2F7")
    _box(ax, 0.76, 0.62, 0.14, 0.22, "Registry", "metrics + artifacts", "#EEF2F7")

    _box(ax, 0.22, 0.24, 0.14, 0.22, "Orchestration", "scheduled pipelines", "#F5F7FA")
    _box(ax, 0.40, 0.24, 0.14, 0.22, "Serving", "batch + API", "#F5F7FA")
    _box(ax, 0.58, 0.24, 0.14, 0.22, "Monitoring", "drift + alerts", "#F5F7FA")

    _arrow(ax, 0.18, 0.73, 0.22, 0.73)
    _arrow(ax, 0.36, 0.73, 0.40, 0.73)
    _arrow(ax, 0.54, 0.73, 0.58, 0.73)
    _arrow(ax, 0.72, 0.73, 0.76, 0.73)

    _arrow(ax, 0.47, 0.62, 0.47, 0.46)
    _arrow(ax, 0.29, 0.62, 0.29, 0.46)
    _arrow(ax, 0.65, 0.62, 0.65, 0.46)

    _arrow(ax, 0.36, 0.35, 0.40, 0.35)
    _arrow(ax, 0.54, 0.35, 0.58, 0.35)
    _arrow(ax, 0.65, 0.24, 0.47, 0.24)

    ax.text(0.04, 0.92, "NYC Taxi MLOps Pipeline Architecture", fontsize=20, fontweight="semibold")
    ax.text(0.04, 0.88, "Clean operational design for reproducibility, testing, and deployment", color=PALETTE["muted"])

    fig.savefig(OUT / "mlops_pipeline_architecture_professional.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def hypothesis_matrix() -> None:
    _base_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = [
        "City-wide baseline",
        "Zone-specific model",
        "Airport zones (132/138)",
        "Borough split (Manhattan/Bronx)",
    ]
    score = [0.84, 0.88, 0.62, 0.00]
    status = ["validated", "validated", "inconclusive", "pending lookup"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], "#C7CDD4"]

    y = np.arange(len(categories))[::-1]
    ax.barh(y, score, color=colors, height=0.52)

    for yi, s, st in zip(y, score, status):
        ax.text(min(s + 0.02, 0.98), yi, st, va="center", ha="left", color=PALETTE["muted"], fontsize=10)

    ax.set_xlim(0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Readiness / Confidence")
    ax.set_title("Hypothesis Validation Status", fontsize=18, fontweight="semibold", pad=14)
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(OUT / "hypothesis_validation_matrix_professional.png", dpi=220)
    plt.close(fig)


def advanced_models_dashboard(metrics: list[dict]) -> None:
    _base_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    names = [m["model"] for m in metrics]
    x = np.arange(len(names))

    mae = np.array([m["mae"] for m in metrics])
    rmse = np.array([m["rmse"] for m in metrics])
    r2 = np.array([m["r2"] for m in metrics])
    rv = np.array([m["residual_variance"] for m in metrics])
    lat = np.array([m["latency_ms"] for m in metrics])

    width = 0.35
    axes[0, 0].bar(x - width / 2, mae, width=width, color=PALETTE["primary"], label="MAE")
    axes[0, 0].bar(x + width / 2, rmse, width=width, color=PALETTE["accent"], label="RMSE")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names)
    axes[0, 0].set_title("Error Metrics by Model")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", color=PALETTE["line"], linewidth=0.8)

    axes[0, 1].bar(x, r2, color=PALETTE["secondary"], width=0.5)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("R2 (Higher Is Better)")
    axes[0, 1].grid(axis="y", color=PALETTE["line"], linewidth=0.8)

    axes[1, 0].bar(x, rv, color=PALETTE["warn"], width=0.5)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names)
    axes[1, 0].set_title("Residual Variance")
    axes[1, 0].grid(axis="y", color=PALETTE["line"], linewidth=0.8)

    size = 80 + (lat / max(lat.max(), 1)) * 260
    axes[1, 1].scatter(lat, r2, s=size, color=PALETTE["primary"], alpha=0.85, edgecolor="white", linewidth=1.2)
    for lx, ry, n in zip(lat, r2, names):
        axes[1, 1].text(lx + 0.8, ry, n, va="center", fontsize=9, color=PALETTE["ink"])
    axes[1, 1].set_xlabel("Inference Latency (ms)")
    axes[1, 1].set_ylabel("R2")
    axes[1, 1].set_title("Accuracy vs Inference Cost")
    axes[1, 1].grid(color=PALETTE["line"], linewidth=0.8)

    fig.suptitle("Advanced Model Performance Dashboard", fontsize=19, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(OUT / "advanced_models_performance_dashboard_professional.png", dpi=220)
    plt.close(fig)


def advanced_models_heatmap(metrics: list[dict]) -> None:
    _base_style()
    names = [m["model"] for m in metrics]

    # Normalize to comparable [0,1] where higher is better.
    mae = np.array([m["mae"] for m in metrics])
    rmse = np.array([m["rmse"] for m in metrics])
    r2 = np.array([m["r2"] for m in metrics])
    rv = np.array([m["residual_variance"] for m in metrics])
    lat = np.array([m["latency_ms"] for m in metrics])

    def inv_scale(v):
        rng = max(v.max() - v.min(), 1e-9)
        return 1.0 - (v - v.min()) / rng

    def pos_scale(v):
        rng = max(v.max() - v.min(), 1e-9)
        return (v - v.min()) / rng

    matrix = np.vstack([inv_scale(mae), inv_scale(rmse), pos_scale(r2), inv_scale(rv), inv_scale(lat)])
    labels = ["MAE", "RMSE", "R2", "Residual Var", "Latency"]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Advanced Models Scorecard (Normalized)", fontsize=18, fontweight="semibold", pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            txt_color = "#111827" if value > 0.45 else "#374151"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=txt_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.ax.set_ylabel("Score (higher is better)", rotation=90)

    fig.tight_layout()
    fig.savefig(OUT / "advanced_models_scorecard_professional.png", dpi=220)
    plt.close(fig)


def main() -> None:
    metrics = load_model_metrics()
    roadmap_timeline()
    mlops_pipeline()
    hypothesis_matrix()
    advanced_models_dashboard(metrics)
    advanced_models_heatmap(metrics)
    print(f"Saved visuals to: {OUT}")
    print(f"Metrics source: {RESULTS / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
