#!/usr/bin/env python3
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results" / "hypothesis_testing2"
OUT = ROOT / "reports" / "figures" / "updated_plots"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "bg": "#F7F8FA",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "c1": "#355C7D",
    "c2": "#4E6E58",
    "c3": "#6C7A89",
    "ok": "#2E5941",
    "warn": "#8B6A3E",
    "bad": "#8A4F4F",
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


def model_performance() -> None:
    path = RESULTS / "model_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("mae")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = [PALETTE["c1"], PALETTE["c2"], PALETTE["c3"]][: len(df)]

    axes[0].bar(df["model"], df["mae"], color=colors)
    axes[0].set_title("MAE")
    axes[0].set_ylabel("Lower is better")
    axes[0].grid(axis="y", color=PALETTE["line"], linewidth=0.8)

    axes[1].bar(df["model"], df["rmse"], color=colors)
    axes[1].set_title("RMSE")
    axes[1].grid(axis="y", color=PALETTE["line"], linewidth=0.8)

    axes[2].bar(df["model"], df["r2"], color=colors)
    axes[2].set_title("R2")
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", color=PALETTE["line"], linewidth=0.8)

    fig.suptitle("Updated Plots - Advanced Model Performance", fontsize=15, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(OUT / "updated_plots_model_performance.png", dpi=220)
    plt.close(fig)


def pairwise_tests() -> None:
    path = RESULTS / "pairwise_model_stat_tests.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    labels = df["model_a"] + " vs " + df["model_b"]
    pvals = df["p_permutation"].astype(float)
    effects = df["mean_abs_error_diff_a_minus_b"].astype(float)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    bars = ax1.bar(labels, pvals, color=PALETTE["c3"], alpha=0.8)
    ax1.axhline(0.05, color=PALETTE["bad"], linestyle="--", linewidth=1.4, label="alpha=0.05")
    ax1.set_ylabel("Permutation p-value")
    ax1.set_title("Updated Plots - Pairwise Statistical Significance")
    ax1.grid(axis="y", color=PALETTE["line"], linewidth=0.8)
    ax1.tick_params(axis="x", rotation=15)

    ax2 = ax1.twinx()
    ax2.plot(labels, effects, color=PALETTE["c1"], marker="o", linewidth=2, label="mean abs error diff")
    ax2.set_ylabel("Mean Abs Error Diff (A-B)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "updated_plots_pairwise_tests.png", dpi=220)
    plt.close(fig)


def prediction_error_distribution() -> None:
    path = RESULTS / "test_predictions.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [PALETTE["c1"], PALETTE["c2"], PALETTE["c3"], PALETTE["warn"]]

    for i, c in enumerate(pred_cols):
        err = (df["demand"] - df[c]).astype(float)
        ax.hist(err, bins=40, density=True, alpha=0.35, color=colors[i % len(colors)], label=c.replace("pred_", ""))

    ax.set_title("Updated Plots - Prediction Error Distributions")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color=PALETTE["line"], linewidth=0.8)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "updated_plots_error_distribution.png", dpi=220)
    plt.close(fig)


def roadmap_status_plot() -> None:
    path = RESULTS / "roadmap_execution_report.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return

    labels = [d["stage"] for d in data]
    status = [d["status"] for d in data]
    score = [1 if s == "done" else 0.5 if s == "in_progress" else 0 for s in status]
    colors = [PALETTE["ok"] if s == "done" else PALETTE["warn"] if s == "in_progress" else PALETTE["bad"] for s in status]

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.barh(y, score, color=colors, height=0.55)

    for yi, s in zip(y, status):
        ax.text(1.02, yi, s, va="center", fontsize=10, color=PALETTE["muted"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Execution completion")
    ax.set_title("Updated Plots - Roadmap Execution Status")
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUT / "updated_plots_roadmap_status.png", dpi=220)
    plt.close(fig)


def zone_and_borough_plots() -> None:
    zone_path = RESULTS / "zone_vs_city_comparison.csv"
    if zone_path.exists():
        z = pd.read_csv(zone_path)
        if not z.empty and "mae_gain" in z.columns:
            z = z.sort_values("mae_gain", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(11, 6))
            colors = [PALETTE["ok"] if v > 0 else PALETTE["bad"] for v in z["mae_gain"]]
            ax.bar(range(len(z)), z["mae_gain"], color=colors)
            ax.axhline(0, color=PALETTE["line"], linewidth=1.2)
            ax.set_title("Updated Plots - Zone-Specific vs City-Wide MAE Gain (Top 20)")
            ax.set_ylabel("MAE gain (city - zone)")
            ax.set_xlabel("Ranked zones")
            ax.grid(axis="y", color=PALETTE["line"], linewidth=0.8)
            fig.tight_layout()
            fig.savefig(OUT / "updated_plots_zone_gain.png", dpi=220)
            plt.close(fig)

    borough_path = RESULTS / "borough_xgb_metrics.csv"
    if borough_path.exists():
        b = pd.read_csv(borough_path)
        if not b.empty and {"Borough", "mae", "rmse"}.issubset(b.columns):
            b = b.sort_values("mae")
            x = np.arange(len(b))
            fig, ax = plt.subplots(figsize=(10, 5.5))
            w = 0.35
            ax.bar(x - w / 2, b["mae"], width=w, color=PALETTE["c1"], label="MAE")
            ax.bar(x + w / 2, b["rmse"], width=w, color=PALETTE["c3"], label="RMSE")
            ax.set_xticks(x)
            ax.set_xticklabels(b["Borough"], rotation=15)
            ax.set_title("Updated Plots - Borough Model Metrics")
            ax.set_ylabel("Error")
            ax.grid(axis="y", color=PALETTE["line"], linewidth=0.8)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(OUT / "updated_plots_borough_metrics.png", dpi=220)
            plt.close(fig)


def main() -> None:
    style()
    model_performance()
    pairwise_tests()
    prediction_error_distribution()
    roadmap_status_plot()
    zone_and_borough_plots()
    print(f"Saved updated plots in: {OUT}")


if __name__ == "__main__":
    main()
