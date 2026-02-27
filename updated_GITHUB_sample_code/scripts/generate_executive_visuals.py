#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "executive_visuals"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "bg": "#F5F6F8",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "line": "#CBD5E1",
    "slate": "#415A77",
    "steel": "#6C7A89",
    "sage": "#5E7D6A",
    "sand": "#8A7A66",
    "soft": "#E9EEF3",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.edgecolor": PALETTE["line"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "font.size": 11,
        }
    )


def roadmap_visual() -> None:
    style()
    phases = [
        "Data Intake",
        "Feature Engineering",
        "Tree/Ensemble Modeling",
        "Statistical Validation",
        "Artifact Packaging",
        "Review & Iteration",
    ]
    starts = np.array([0, 1.5, 3.5, 5.5, 7.2, 8.5])
    durations = np.array([1.5, 2.0, 2.0, 1.7, 1.3, 1.2])

    y = np.arange(len(phases))[::-1]
    colors = [PALETTE["steel"], PALETTE["slate"], PALETTE["sage"], PALETTE["sand"], PALETTE["steel"], PALETTE["sage"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    for yi, s, d, c, p in zip(y, starts, durations, colors, phases):
        ax.barh(yi, d, left=s, color=c, edgecolor="none", height=0.56)
        ax.text(s + d / 2, yi, p, ha="center", va="center", color="white", fontsize=10)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.8, len(phases) - 0.2)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 10.5, 1))
    ax.set_xticklabels([f"Phase {i}" if i > 0 else "Start" for i in range(0, 11)])
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)
    ax.set_title("Modeling Roadmap - High Level Delivery Plan", fontsize=18, fontweight="semibold", pad=14)
    ax.set_xlabel("Delivery Sequence")

    fig.tight_layout()
    fig.savefig(OUT / "roadmap_high_level.png", dpi=240)
    plt.close(fig)


def _box(ax, x, y, w, h, title, subtitle="", fill=None):
    if fill is None:
        fill = "#FFFFFF"
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=PALETTE["line"],
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontweight="semibold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=9, color=PALETTE["muted"])


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.2, color=PALETTE["muted"]))


def pipeline_architecture() -> None:
    style()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.03, 0.62, 0.16, 0.22, "Input Layer", "Teammate sample CSV", PALETTE["soft"])
    _box(ax, 0.23, 0.62, 0.16, 0.22, "Preparation", "temporal features + lags", PALETTE["soft"])
    _box(ax, 0.43, 0.62, 0.16, 0.22, "Modeling", "boosting, bagging, stacking", "#E6EDF5")
    _box(ax, 0.63, 0.62, 0.16, 0.22, "Validation", "time split + paired tests", PALETTE["soft"])
    _box(ax, 0.83, 0.62, 0.14, 0.22, "Outputs", "metrics, plots, reports", PALETTE["soft"])

    _box(ax, 0.23, 0.24, 0.16, 0.22, "MLOps Controls", "config + scripts + CI")
    _box(ax, 0.43, 0.24, 0.16, 0.22, "Hypothesis Engine", "p-values + CI + decisions")
    _box(ax, 0.63, 0.24, 0.16, 0.22, "Communication", "executive visuals + summary")

    _arrow(ax, 0.19, 0.73, 0.23, 0.73)
    _arrow(ax, 0.39, 0.73, 0.43, 0.73)
    _arrow(ax, 0.59, 0.73, 0.63, 0.73)
    _arrow(ax, 0.79, 0.73, 0.83, 0.73)

    _arrow(ax, 0.51, 0.62, 0.51, 0.46)
    _arrow(ax, 0.31, 0.62, 0.31, 0.46)
    _arrow(ax, 0.71, 0.62, 0.71, 0.46)

    _arrow(ax, 0.39, 0.35, 0.43, 0.35)
    _arrow(ax, 0.59, 0.35, 0.63, 0.35)

    ax.text(0.03, 0.92, "Updated Sample Pipeline Architecture", fontsize=20, fontweight="semibold")
    ax.text(0.03, 0.88, "Professional high-level view of modeling + hypothesis workflow", color=PALETTE["muted"])

    fig.savefig(OUT / "pipeline_architecture_high_level.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def algorithms_and_steps() -> None:
    style()
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [
        "Boosting\n(XGBoost)",
        "Bagging\n(Random Forest)",
        "Stacking\n(Ensemble Meta-Model)",
    ]
    step_depth = [4, 4, 6]
    expected_stability = [0.78, 0.82, 0.85]

    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, step_depth, width=w, color=PALETTE["steel"], label="Pipeline complexity level")
    ax.bar(x + w / 2, expected_stability, width=w, color=PALETTE["sage"], label="Expected stability index")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 6.6)
    ax.set_ylabel("Relative scale")
    ax.set_title("Algorithm Stack and Operational Steps", fontsize=18, fontweight="semibold", pad=12)
    ax.grid(axis="y", color=PALETTE["line"], linewidth=0.8)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "algorithms_and_steps_overview.png", dpi=240)
    plt.close(fig)


def hypothesis_framework() -> None:
    style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")

    sections = [
        (0.04, 0.65, 0.28, 0.24, "Hypothesis Inputs", "Temporal split, model predictions, residuals", PALETTE["soft"]),
        (0.36, 0.65, 0.28, 0.24, "Statistical Tests", "Permutation, Wilcoxon, Bootstrap CI", "#E6EDF5"),
        (0.68, 0.65, 0.28, 0.24, "Decision Layer", "Support / Not Support by alpha + direction", PALETTE["soft"]),
        (0.20, 0.28, 0.28, 0.24, "H1 Approach", "Seasonality check (Kruskal-Wallis)", "#FFFFFF"),
        (0.52, 0.28, 0.28, 0.24, "H2 Approach", "Ensemble vs boosting comparative inference", "#FFFFFF"),
    ]

    for x, y, w, h, t, s, f in sections:
        _box(ax, x, y, w, h, t, s, f)

    _arrow(ax, 0.32, 0.77, 0.36, 0.77)
    _arrow(ax, 0.64, 0.77, 0.68, 0.77)
    _arrow(ax, 0.50, 0.65, 0.34, 0.52)
    _arrow(ax, 0.50, 0.65, 0.66, 0.52)

    ax.text(0.04, 0.94, "Hypothesis Validation Framework", fontsize=19, fontweight="semibold")
    ax.text(0.04, 0.90, "Traceable statistical approach for executive-level decisioning", color=PALETTE["muted"])

    fig.savefig(OUT / "hypothesis_approach_framework.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    roadmap_visual()
    pipeline_architecture()
    algorithms_and_steps()
    hypothesis_framework()
    print(f"Saved executive visuals to: {OUT}")


if __name__ == "__main__":
    main()
