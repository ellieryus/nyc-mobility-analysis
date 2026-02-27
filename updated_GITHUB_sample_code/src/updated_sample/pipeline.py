from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kruskal, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_config(project_root: Path) -> dict:
    cfg_path = project_root / "configs" / "modeling.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"pickup_year", "pickup_month", "sample_rows"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["pickup_year"] = pd.to_numeric(df["pickup_year"], errors="coerce").astype("Int64")
    df["pickup_month"] = pd.to_numeric(df["pickup_month"], errors="coerce").astype("Int64")
    df["sample_rows"] = pd.to_numeric(df["sample_rows"], errors="coerce")
    df = df.dropna(subset=["pickup_year", "pickup_month", "sample_rows"]).copy()

    df["date"] = pd.to_datetime(
        df["pickup_year"].astype(int).astype(str) + "-" + df["pickup_month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["year"] = x["date"].dt.year
    x["month"] = x["date"].dt.month
    x["quarter"] = x["date"].dt.quarter
    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12)

    x["lag_1"] = x["sample_rows"].shift(1)
    x["lag_2"] = x["sample_rows"].shift(2)
    x["lag_3"] = x["sample_rows"].shift(3)
    x["rolling_mean_3"] = x["sample_rows"].shift(1).rolling(3, min_periods=1).mean()
    x["rolling_std_3"] = x["sample_rows"].shift(1).rolling(3, min_periods=1).std().fillna(0.0)

    x = x.dropna().reset_index(drop=True)
    return x


def temporal_split(df: pd.DataFrame) -> SplitData:
    uniq = np.array(sorted(df["date"].unique()))
    n = len(uniq)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_set = set(uniq[:train_end])
    valid_set = set(uniq[train_end:valid_end])
    test_set = set(uniq[valid_end:])

    return SplitData(
        train=df[df["date"].isin(train_set)].copy(),
        valid=df[df["date"].isin(valid_set)].copy(),
        test=df[df["date"].isin(test_set)].copy(),
    )


def make_models(seed: int) -> Dict[str, object]:
    xgb = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
    )

    bagging = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=1,
    )

    stacking = StackingRegressor(
        estimators=[
            ("xgb", xgb),
            ("rf", RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=1)),
            ("hgb", HistGradientBoostingRegressor(max_depth=6, random_state=seed)),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
        passthrough=True,
        cv=3,
        n_jobs=1,
    )

    return {
        "boosting_xgboost": xgb,
        "bagging_random_forest": bagging,
        "stacking_ensemble": stacking,
    }


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_variance": float(np.var(y_true - y_pred)),
    }


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = a - b
    obs = float(np.mean(d))
    idx = rng.integers(0, len(d), size=(n_bootstrap, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return obs, float(lo), float(hi)


def permutation_p_value(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    d = a - b
    obs = abs(d.mean())
    signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    perm = np.abs((d * signs).mean(axis=1))
    return float((np.sum(perm >= obs) + 1) / (n_perm + 1))


def pairwise_stat_tests(y_true: np.ndarray, pred_df: pd.DataFrame, n_bootstrap: int, n_perm: int, seed: int) -> pd.DataFrame:
    models = list(pred_df.columns)
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            ea = np.abs(y_true - pred_df[a].to_numpy())
            eb = np.abs(y_true - pred_df[b].to_numpy())

            diff, lo, hi = bootstrap_ci_mean_diff(ea, eb, n_bootstrap=n_bootstrap, seed=seed)
            p_perm = permutation_p_value(ea, eb, n_perm=n_perm, seed=seed)
            try:
                p_wil = float(wilcoxon(ea, eb).pvalue)
            except Exception:
                p_wil = float("nan")

            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "mean_abs_error_diff_a_minus_b": diff,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "p_permutation": p_perm,
                    "p_wilcoxon": p_wil,
                }
            )
    return pd.DataFrame(rows)


def hypothesis_tests(df_feat: pd.DataFrame, model_results: pd.DataFrame, pairwise_df: pd.DataFrame, alpha: float) -> dict:
    # H1: seasonality exists across months (sample_rows distribution differs by month)
    month_groups = [g["sample_rows"].to_numpy() for _, g in df_feat.groupby("month") if len(g) > 0]
    if len(month_groups) >= 3:
        k = kruskal(*month_groups)
        h1 = {
            "tested": True,
            "p_value": float(k.pvalue),
            "support": bool(k.pvalue < alpha),
            "description": "Monthly seasonality effect exists in sample_rows.",
        }
    else:
        h1 = {"tested": False, "reason": "insufficient monthly groups"}

    # H2: tree-based ensembles outperform pure boosting on this sample
    try:
        boost_mae = float(model_results.loc[model_results["model"] == "boosting_xgboost", "mae"].iloc[0])
        bag_mae = float(model_results.loc[model_results["model"] == "bagging_random_forest", "mae"].iloc[0])
        stack_mae = float(model_results.loc[model_results["model"] == "stacking_ensemble", "mae"].iloc[0])
    except Exception:
        return {"h1_seasonality": h1, "h2_ensemble_better_than_boosting": {"tested": False, "reason": "model rows missing"}}

    better_by_metric = (bag_mae < boost_mae) or (stack_mae < boost_mae)

    subset = pairwise_df[
        ((pairwise_df["model_a"].isin(["bagging_random_forest", "stacking_ensemble"])) & (pairwise_df["model_b"] == "boosting_xgboost"))
        | ((pairwise_df["model_b"].isin(["bagging_random_forest", "stacking_ensemble"])) & (pairwise_df["model_a"] == "boosting_xgboost"))
    ].copy()

    # Significant if any pair has p_perm < alpha and CI excludes 0 in favor of ensemble.
    sig = False
    details = []
    for _, r in subset.iterrows():
        a, b = r["model_a"], r["model_b"]
        lo, hi = float(r["ci95_low"]), float(r["ci95_high"])
        p = float(r["p_permutation"])
        # diff = err(a)-err(b); ensemble better means diff<0 if a is ensemble vs boosting,
        # or diff>0 if a is boosting vs ensemble.
        if a in ["bagging_random_forest", "stacking_ensemble"] and b == "boosting_xgboost":
            direction_ok = hi < 0
        elif a == "boosting_xgboost" and b in ["bagging_random_forest", "stacking_ensemble"]:
            direction_ok = lo > 0
        else:
            direction_ok = False
        passed = (p < alpha) and direction_ok
        sig = sig or passed
        details.append({"pair": f"{a} vs {b}", "p_permutation": p, "ci95": [lo, hi], "direction_support": bool(direction_ok), "passed": bool(passed)})

    h2 = {
        "tested": True,
        "support": bool(better_by_metric and sig),
        "rule": "At least one ensemble beats boosting on MAE and shows significant paired error difference.",
        "boosting_mae": boost_mae,
        "bagging_mae": bag_mae,
        "stacking_mae": stack_mae,
        "pairwise_details": details,
    }

    return {
        "h1_seasonality": h1,
        "h2_ensemble_better_than_boosting": h2,
    }


def make_plots(model_df: pd.DataFrame, pred_df: pd.DataFrame, y_true: np.ndarray, out_fig_dir: Path) -> None:
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # Model metrics plot
    x = np.arange(len(model_df))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#355C7D", "#4E6E58", "#6C7A89"]

    m = model_df.sort_values("mae").reset_index(drop=True)
    axes[0].bar(m["model"], m["mae"], color=colors[: len(m)])
    axes[0].set_title("MAE")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(m["model"], m["rmse"], color=colors[: len(m)])
    axes[1].set_title("RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(m["model"], m["r2"], color=colors[: len(m)])
    axes[2].set_title("R2")
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Updated Sample - Tree/Ensemble Model Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_fig_dir / "model_metrics.png", dpi=220)
    plt.close(fig)

    # Residual distributions
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, color in zip(pred_df.columns, colors):
        residual = y_true - pred_df[col].to_numpy()
        ax.hist(residual, bins=10, density=True, alpha=0.35, label=col, color=color)
    ax.set_title("Residual Distributions")
    ax.set_xlabel("actual - predicted")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig_dir / "error_distributions.png", dpi=220)
    plt.close(fig)

    # Tradeoff plot (latency proxy + rmse)
    latency_proxy = {
        "boosting_xgboost": 18,
        "bagging_random_forest": 28,
        "stacking_ensemble": 45,
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in model_df.iterrows():
        name = row["model"]
        xlat = latency_proxy.get(name, 30)
        yrmse = row["rmse"]
        ax.scatter(xlat, yrmse, s=180)
        ax.text(xlat + 0.6, yrmse, name)
    ax.set_title("RMSE vs Inference Cost (Latency Proxy)")
    ax.set_xlabel("Latency (ms, proxy)")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig_dir / "model_tradeoff.png", dpi=220)
    plt.close(fig)


def run_pipeline(project_root: Path) -> None:
    cfg = load_config(project_root)
    input_csv = project_root / cfg["input_csv"]
    out_results = project_root / cfg["output_results_dir"]
    out_figures = project_root / cfg["output_figures_dir"]

    out_results.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    df = load_data(input_csv)
    feat = feature_engineering(df)
    split = temporal_split(feat)

    features = ["year", "month", "quarter", "month_sin", "month_cos", "lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3"]
    target = "sample_rows"

    models = make_models(seed=int(cfg["seed"]))
    rows = []
    pred_df = pd.DataFrame(index=split.test.index)

    for name, model in models.items():
        model.fit(split.train[features], split.train[target])
        pred = model.predict(split.test[features])
        pred_df[name] = pred
        m = eval_metrics(split.test[target].to_numpy(), pred)
        m["model"] = name
        rows.append(m)

    model_df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    pairwise_df = pairwise_stat_tests(
        y_true=split.test[target].to_numpy(),
        pred_df=pred_df,
        n_bootstrap=int(cfg["n_bootstrap"]),
        n_perm=int(cfg["n_permutations"]),
        seed=int(cfg["seed"]),
    )

    hypothesis = hypothesis_tests(
        df_feat=feat,
        model_results=model_df,
        pairwise_df=pairwise_df,
        alpha=float(cfg["alpha"]),
    )

    summary = {
        "data_info": {
            "rows_raw": int(len(df)),
            "rows_featured": int(len(feat)),
            "split_sizes": {
                "train": int(len(split.train)),
                "validation": int(len(split.valid)),
                "test": int(len(split.test)),
            },
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
        },
        "best_model": str(model_df.iloc[0]["model"]),
        "hypothesis_tests": hypothesis,
    }

    model_df.to_csv(out_results / "model_comparison.csv", index=False)
    pairwise_df.to_csv(out_results / "pairwise_stat_tests.csv", index=False)
    split.test[["date", target]].assign(**{f"pred_{c}": pred_df[c].values for c in pred_df.columns}).to_csv(
        out_results / "test_predictions.csv", index=False
    )
    pd.DataFrame(
        [
            {"stage": "data_loading", "status": "done"},
            {"stage": "feature_engineering", "status": "done"},
            {"stage": "temporal_split_train_valid_test", "status": "done"},
            {"stage": "model_training_tree_and_ensemble", "status": "done"},
            {"stage": "statistical_hypothesis_testing", "status": "done"},
            {"stage": "reporting_and_visualization", "status": "done"},
        ]
    ).to_csv(out_results / "roadmap_execution.csv", index=False)

    with (out_results / "hypothesis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(model_df=model_df, pred_df=pred_df, y_true=split.test[target].to_numpy(), out_fig_dir=out_figures)

    with (out_results / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# updated_GITHUB_sample_code Summary\n\n")
        f.write(f"- Best model: **{summary['best_model']}**\n")
        f.write(f"- Train/Validation/Test: {summary['data_info']['split_sizes']}\n\n")
        f.write("## Model Comparison\n\n")
        f.write(model_df.to_string(index=False))
        f.write("\n\n## Hypotheses\n\n")
        f.write(json.dumps(hypothesis, indent=2))

    print(f"Pipeline completed. Results: {out_results}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    run_pipeline(project_root)


if __name__ == "__main__":
    main()
