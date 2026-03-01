#!/usr/bin/env python3
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
OUT = ROOT / "reports" / "figures" / "insightful_visuals"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "bg": "#F6F7F9",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "c1": "#3D5A80",
    "c2": "#5C7C6F",
    "c3": "#7B8794",
    "good": "#4D7C5A",
    "warn": "#8A7A58",
    "bad": "#8A5A5A",
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


def plot_model_rankings() -> None:
    p = RES / "model_comparison.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("mae")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.barh(df["model"], df["mae"], color=[PALETTE["c1"], PALETTE["c2"], PALETTE["c3"]][: len(df)])
    ax.invert_yaxis()
    ax.set_title("Model Ranking by MAE (Lower is Better)", fontsize=15, fontweight="semibold")
    ax.set_xlabel("MAE")
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)

    for bar, v in zip(bars, df["mae"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(OUT / "insight_model_ranking_mae.png", dpi=220)
    plt.close(fig)


def plot_pairwise_ci() -> None:
    p = RES / "pairwise_stat_tests.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return

    labels = (df["model_a"] + " vs " + df["model_b"]).tolist()
    center = df["mean_abs_error_diff_a_minus_b"].to_numpy(dtype=float)
    lo = df["ci95_low"].to_numpy(dtype=float)
    hi = df["ci95_high"].to_numpy(dtype=float)

    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    for yi, c, l, h in zip(y, center, lo, hi):
        color = PALETTE["good"] if (l > 0 or h < 0) else PALETTE["warn"]
        ax.plot([l, h], [yi, yi], color=color, linewidth=3)
        ax.scatter([c], [yi], color=PALETTE["c1"], s=45, zorder=3)

    ax.axvline(0, color=PALETTE["bad"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Abs Error Difference (A - B), with 95% CI")
    ax.set_title("Pairwise Statistical Effect Sizes", fontsize=15, fontweight="semibold")
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUT / "insight_pairwise_confidence_intervals.png", dpi=220)
    plt.close(fig)


def plot_residual_box() -> None:
    p = RES / "test_predictions.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        return

    labels, values = [], []
    for c in pred_cols:
        labels.append(c.replace("pred_", ""))
        values.append((df["sample_rows"] - df[c]).to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bp = ax.boxplot(values, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], [PALETTE["c1"], PALETTE["c2"], PALETTE["c3"]][: len(values)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    ax.axhline(0, color=PALETTE["bad"], linestyle="--", linewidth=1.2)
    ax.set_title("Residual Spread by Model", fontsize=15, fontweight="semibold")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.grid(axis="y", color=PALETTE["line"], linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUT / "insight_residual_spread_boxplot.png", dpi=220)
    plt.close(fig)


def plot_roadmap_health() -> None:
    p = RES / "roadmap_execution.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "status" not in df.columns:
        return

    score_map = {"done": 1.0, "in_progress": 0.6, "pending": 0.2}
    scores = df["status"].map(score_map).fillna(0.2).to_numpy(dtype=float)
    colors = [PALETTE["good"] if s >= 1 else PALETTE["warn"] if s >= 0.6 else PALETTE["bad"] for s in scores]

    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.barh(y, scores, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(df["stage"].tolist())
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Completion score")
    ax.set_title("Roadmap Execution Health", fontsize=15, fontweight="semibold")
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)

    for yi, st in zip(y, df["status"].tolist()):
        ax.text(1.02, yi, st, va="center", color=PALETTE["muted"])

    fig.tight_layout()
    fig.savefig(OUT / "insight_roadmap_health.png", dpi=220)
    plt.close(fig)


def plot_hypothesis_status() -> None:
    p = RES / "hypothesis_summary.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    h = d.get("hypothesis_tests", {})

    rows = []
    for k, v in h.items():
        if isinstance(v, dict):
            tested = bool(v.get("tested", False))
            support = bool(v.get("support", False)) if tested else False
            if tested and support:
                score = 1.0
            elif tested and not support:
                score = 0.45
            else:
                score = 0.2
            rows.append((k, score, tested, support))

    if not rows:
        return

    labels = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    colors = [PALETTE["good"] if s >= 1 else PALETTE["warn"] if s >= 0.45 else PALETTE["bad"] for s in scores]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.barh(y, scores, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Hypothesis evidence score")
    ax.set_title("Hypothesis Testing Status Dashboard", fontsize=15, fontweight="semibold")
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)

    for yi, (_, _, tested, support) in zip(y, rows):
        label = "supported" if tested and support else "not supported" if tested else "not tested"
        ax.text(1.02, yi, label, va="center", color=PALETTE["muted"])

    fig.tight_layout()
    fig.savefig(OUT / "insight_hypothesis_status_dashboard.png", dpi=220)
    plt.close(fig)


def main() -> None:
    style()
    plot_model_rankings()
    plot_pairwise_ci()
    plot_residual_box()
    plot_roadmap_health()
    plot_hypothesis_status()
    print(f"Saved insightful visuals to: {OUT}")


if __name__ == "__main__":
    main()
