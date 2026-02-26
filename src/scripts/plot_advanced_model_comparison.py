#!/usr/bin/env python3
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "results" / "model_comparison.csv"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "bg": "#F7F8FA",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "boosting": "#355C7D",
    "bagging": "#4E6E58",
    "stacking": "#6C7A89",
}

NAME_MAP = {
    "xgboost": "Boosting (XGBoost)",
    "bagging_rf": "Bagging (Random Forest)",
    "stacking": "Stacking Ensemble",
}
COLOR_MAP = {
    "xgboost": PALETTE["boosting"],
    "bagging_rf": PALETTE["bagging"],
    "stacking": PALETTE["stacking"],
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "axes.edgecolor": PALETTE["line"],
            "font.size": 11,
        }
    )


def load_metrics() -> list[dict]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    rows = []
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            model = r.get("model", "")
            if model not in NAME_MAP:
                continue
            rows.append(
                {
                    "model": model,
                    "display": NAME_MAP[model],
                    "mae": float(r.get("mae", 0.0)),
                    "rmse": float(r.get("rmse", 0.0)),
                    "r2": float(r.get("r2", 0.0)),
                    "residual_variance": float(r.get("residual_variance", 0.0)),
                    "latency_ms": float(r.get("latency_ms", 0.0) or 0.0),
                    "color": COLOR_MAP[model],
                }
            )

    if len(rows) < 3:
        raise ValueError("Need metrics for xgboost, bagging_rf, and stacking.")

    return rows


def plot_metric_bars(rows: list[dict]) -> None:
    style()
    labels = [r["display"] for r in rows]
    colors = [r["color"] for r in rows]

    mae = [r["mae"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    r2 = [r["r2"] for r in rows]

    x = np.arange(len(labels))
    w = 0.28

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.bar(x - w / 2, mae, width=w, color=colors, alpha=0.95, label="MAE")
    ax1.bar(x + w / 2, rmse, width=w, color=colors, alpha=0.55, label="RMSE")
    ax1.set_ylabel("Error (lower is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", color=PALETTE["line"], linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(x, r2, color=PALETTE["ink"], marker="o", linewidth=2, label="R2")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("R2 (higher is better)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False)

    ax1.set_title("Advanced Tree-Based Models: Performance Comparison", fontsize=18, fontweight="semibold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "advanced_tree_models_metric_comparison.png", dpi=240)
    plt.close(fig)


def plot_tradeoff(rows: list[dict]) -> None:
    style()
    fig, ax = plt.subplots(figsize=(12, 7))

    for r in rows:
        ax.scatter(
            r["latency_ms"],
            r["rmse"],
            s=220,
            color=r["color"],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.95,
        )
        ax.text(r["latency_ms"] + 0.9, r["rmse"], f"{r['display']}\nR2={r['r2']:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Model Trade-off: Prediction Error vs Inference Cost", fontsize=18, fontweight="semibold", pad=12)
    ax.grid(color=PALETTE["line"], linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "advanced_tree_models_tradeoff.png", dpi=240)
    plt.close(fig)


def plot_scorecard(rows: list[dict]) -> None:
    style()
    labels = [r["display"] for r in rows]

    mae = np.array([r["mae"] for r in rows])
    rmse = np.array([r["rmse"] for r in rows])
    r2 = np.array([r["r2"] for r in rows])
    rv = np.array([r["residual_variance"] for r in rows])
    lat = np.array([r["latency_ms"] for r in rows])

    def inv_scale(v: np.ndarray) -> np.ndarray:
        span = max(float(v.max() - v.min()), 1e-9)
        return 1 - (v - v.min()) / span

    def pos_scale(v: np.ndarray) -> np.ndarray:
        span = max(float(v.max() - v.min()), 1e-9)
        return (v - v.min()) / span

    matrix = np.vstack([
        inv_scale(mae),
        inv_scale(rmse),
        pos_scale(r2),
        inv_scale(rv),
        inv_scale(lat),
    ])

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap="Greys", aspect="auto", vmin=0, vmax=1)

    ylabels = ["MAE", "RMSE", "R2", "Residual Var", "Latency"]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title("Advanced Tree-Based Models: Normalized Scorecard", fontsize=18, fontweight="semibold", pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=PALETTE["ink"])

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Score (higher is better)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "advanced_tree_models_scorecard.png", dpi=240)
    plt.close(fig)


def main() -> None:
    rows = load_metrics()
    plot_metric_bars(rows)
    plot_tradeoff(rows)
    plot_scorecard(rows)
    print(f"Saved advanced model comparison plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
