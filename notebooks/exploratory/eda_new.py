from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)


def resolve_project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[0],
    ]
    for candidate in candidates:
        if (candidate / "data" / "processed").exists():
            return candidate
    return Path.cwd()


def resolve_data_path(project_root: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    preferred = project_root / "data" / "processed" / "Yellow_Taxi_24Months_Complete.parquet"
    if preferred.exists():
        return preferred

    processed_dir = project_root / "data" / "processed"
    parquet_files = sorted(processed_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {processed_dir}")
    return parquet_files[-1]


def infer_attribute_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if non_null.nunique() <= 20:
            return "categorical-discrete"
        return "numeric"
    if pd.api.types.is_object_dtype(series):
        sample = non_null.astype(str).head(200)
        if sample.empty:
            return "text"
        structured_ratio = sample.str.strip().str.startswith(("{", "[", "(")).mean()
        if structured_ratio >= 0.6:
            return "structured"
        avg_len = sample.str.len().mean()
        cardinality_ratio = non_null.nunique() / max(len(non_null), 1)
        if avg_len > 40:
            return "text"
        if non_null.nunique() <= 40 or cardinality_ratio < 0.05:
            return "categorical"
        return "text"
    return "unknown"


def infer_distribution(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "n/a"
    clean = series.dropna().astype(float)
    if clean.empty:
        return "empty"
    if clean.nunique() <= 20:
        return "discrete"
    skew = clean.skew()
    kurtosis = clean.kurtosis()
    if abs(skew) < 0.5 and abs(kurtosis) < 1.0:
        return "approximately Gaussian"
    q10, q50, q90 = clean.quantile([0.10, 0.50, 0.90]).tolist()
    width = q90 - q10
    if width > 0:
        lower = q50 - q10
        upper = q90 - q50
        if abs(lower - upper) / width < 0.15:
            return "roughly symmetric/uniform-like"
    if skew > 1.0:
        return "right-skewed"
    if skew < -1.0:
        return "left-skewed"
    return "non-Gaussian"


def profile_attributes(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    total_rows = len(df)
    for col in df.columns:
        s = df[col]
        missing_pct = s.isna().mean() * 100
        records.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "inferred_type": infer_attribute_type(s),
                "distribution": infer_distribution(s),
                "n_unique": int(s.nunique(dropna=True)),
                "missing_pct": round(float(missing_pct), 3),
                "non_null_rows": int(total_rows - s.isna().sum()),
            }
        )
    return pd.DataFrame(records).sort_values(["inferred_type", "column"]).reset_index(drop=True)


def statistical_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.DataFrame(), pd.DataFrame(), df.isna().mean() * 100
    descriptive = numeric_df.describe().T
    quantiles = numeric_df.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    return descriptive, quantiles, missing_pct


def assess_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna().astype(float)
        if len(s) < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            iqr_mask = (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))
            iqr_outlier_pct = iqr_mask.mean() * 100
        else:
            iqr_outlier_pct = 0.0

        std = s.std(ddof=0)
        if std > 0:
            z = ((s - s.mean()) / std).abs()
            z_outlier_pct = (z > 3).mean() * 100
            cv = std / max(abs(s.mean()), 1e-12)
        else:
            z_outlier_pct = 0.0
            cv = 0.0

        rounded_integer_pct = np.isclose(s, np.round(s), atol=1e-9).mean() * 100
        rounded_half_pct = np.isclose((s * 2) % 1, 0, atol=1e-9).mean() * 100

        noise_flags: List[str] = []
        if iqr_outlier_pct > 2:
            noise_flags.append("outliers")
        if cv > 2 and z_outlier_pct > 1:
            noise_flags.append("stochastic_noise")
        if s.nunique() > 30 and rounded_integer_pct > 70:
            noise_flags.append("possible_rounding")
        if not noise_flags:
            noise_flags.append("low_noise")

        rows.append(
            {
                "column": col,
                "iqr_outlier_pct": round(float(iqr_outlier_pct), 3),
                "zscore_outlier_pct": round(float(z_outlier_pct), 3),
                "coef_variation": round(float(cv), 3),
                "rounded_integer_pct": round(float(rounded_integer_pct), 3),
                "rounded_half_pct": round(float(rounded_half_pct), 3),
                "noise_type": ", ".join(noise_flags),
            }
        )
    return pd.DataFrame(rows).sort_values("iqr_outlier_pct", ascending=False).reset_index(drop=True)


def identify_target_candidates(df: pd.DataFrame) -> pd.DataFrame:
    candidates: List[Dict[str, str]] = []
    if "total_amount" in df.columns:
        candidates.append(
            {
                "candidate_target": "total_amount",
                "task_type": "regression",
                "reason": "Direct business KPI for trip revenue prediction",
            }
        )
    if "trip_duration_seconds" in df.columns:
        candidates.append(
            {
                "candidate_target": "trip_duration_seconds",
                "task_type": "regression",
                "reason": "Useful for ETA or trip-time prediction",
            }
        )
    if "fare_amount" in df.columns:
        candidates.append(
            {
                "candidate_target": "fare_amount",
                "task_type": "regression",
                "reason": "Core monetary component of each trip",
            }
        )
    if "payment_type" in df.columns:
        candidates.append(
            {
                "candidate_target": "payment_type",
                "task_type": "classification",
                "reason": "Discrete label for payment behavior modeling",
            }
        )
    if "pickup_location_id" in df.columns and "dropoff_location_id" in df.columns:
        candidates.append(
            {
                "candidate_target": "dropoff_location_id",
                "task_type": "multiclass_classification",
                "reason": "Destination prediction from pickup/time context",
            }
        )
    if not candidates:
        candidates.append(
            {
                "candidate_target": "None identified",
                "task_type": "unsupervised",
                "reason": "Dataset can still support clustering/anomaly detection",
            }
        )
    return pd.DataFrame(candidates)


def study_correlations(df: pd.DataFrame, threshold: float = 0.9) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(), [], pd.DataFrame()

    corr = numeric_df.corr(numeric_only=True)
    pairs: List[Dict[str, object]] = []
    columns = corr.columns.tolist()
    missing = df.isna().mean()

    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if np.isnan(value) or abs(value) < threshold:
                continue
            pairs.append(
                {
                    "feature_a": col_a,
                    "feature_b": col_b,
                    "corr": round(float(value), 4),
                }
            )

    high_corr_df = pd.DataFrame(pairs).sort_values("corr", key=lambda s: s.abs(), ascending=False) if pairs else pd.DataFrame()

    reject: List[str] = []
    if not high_corr_df.empty:
        for _, row in high_corr_df.iterrows():
            a = row["feature_a"]
            b = row["feature_b"]
            if a in reject or b in reject:
                continue
            reject_col = a if missing.get(a, 0) > missing.get(b, 0) else b
            reject.append(str(reject_col))

    return corr, reject, high_corr_df


def visualize_data(df: pd.DataFrame, corr: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return

    # 1) Missingness
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    top_missing = missing_pct.head(25)
    sns.barplot(x=top_missing.index, y=top_missing.values, color="#4C78A8")
    plt.xticks(rotation=75, ha="right")
    plt.title("Top 25 Columns by Missing Percentage")
    plt.ylabel("Missing (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "missingness_top25.png", dpi=150)
    plt.close()

    # 2) Distributions for top-variance numeric columns
    variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
    hist_cols = variances.head(min(6, len(variances))).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, col in enumerate(hist_cols):
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=axes[i], color="#F58518")
        axes[i].set_title(col)
    for j in range(len(hist_cols), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_distributions.png", dpi=150)
    plt.close(fig)

    # 3) Correlation heatmap
    if corr.shape[0] >= 2:
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    # 4) Dimensionality reduction (PCA / t-SNE / UMAP)
    num_data = df[numeric_cols].copy()
    num_data = num_data.fillna(num_data.median(numeric_only=True))
    if num_data.shape[0] > 0 and num_data.shape[1] >= 2:
        sample_size = min(5000, len(num_data))
        sample = num_data.sample(sample_size, random_state=42) if len(num_data) > sample_size else num_data
        scaled = StandardScaler().fit_transform(sample)

        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(scaled)
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], s=8, alpha=0.5, color="#54A24B")
        plt.title("PCA (2D) of Numeric Features")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_2d.png", dpi=150)
        plt.close()

        tsne_size = min(2000, sample.shape[0])
        tsne_input = scaled[:tsne_size]
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_2d = tsne.fit_transform(tsne_input)
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], s=8, alpha=0.5, color="#E45756")
        plt.title("t-SNE (2D) of Numeric Features")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_2d.png", dpi=150)
        plt.close()

        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=42)
            umap_2d = reducer.fit_transform(scaled)
            plt.figure(figsize=(8, 6))
            plt.scatter(umap_2d[:, 0], umap_2d[:, 1], s=8, alpha=0.5, color="#B279A2")
            plt.title("UMAP (2D) of Numeric Features")
            plt.tight_layout()
            plt.savefig(output_dir / "umap_2d.png", dpi=150)
            plt.close()


def simulate_manual_solution(df: pd.DataFrame) -> pd.DataFrame:
    playbook = [
        "Estimate expected demand manually by averaging historical trips by pickup_hour, pickup_dayofweek, and pickup_location_id.",
        "Estimate fare manually using fare_amount baseline + distance adjustment + rush-hour surcharge proxy.",
        "Flag anomalous trips manually when trip_duration_seconds is very high/low for similar trip_distance and pickup hour.",
        "Assign service zones manually by grouping high-volume pickup/dropoff location IDs into hotspot tiers.",
    ]
    return pd.DataFrame({"manual_solution_playbook": playbook})


def identify_transformations(df: pd.DataFrame, high_corr_df: pd.DataFrame) -> pd.DataFrame:
    suggestions: List[Dict[str, str]] = []
    numeric_df = df.select_dtypes(include=np.number)

    if "pickup_hour" in df.columns:
        suggestions.append(
            {
                "transformation": "Cyclical encoding for pickup_hour",
                "why": "Preserves circular time relationships (23 and 0 are adjacent).",
            }
        )
    if "pickup_dayofweek" in df.columns:
        suggestions.append(
            {
                "transformation": "Cyclical encoding for pickup_dayofweek",
                "why": "Captures weekly periodicity in travel behavior.",
            }
        )

    if not numeric_df.empty:
        skewed_cols = [c for c in numeric_df.columns if abs(numeric_df[c].dropna().skew()) > 1.0]
        if skewed_cols:
            suggestions.append(
                {
                    "transformation": f"log1p transform for skewed columns ({', '.join(skewed_cols[:6])})",
                    "why": "Reduces heavy-tail effects and stabilizes model training.",
                }
            )

    if not high_corr_df.empty:
        high_corr_cols = sorted(set(high_corr_df["feature_a"]).union(set(high_corr_df["feature_b"])))
        suggestions.append(
            {
                "transformation": f"Drop or combine high-correlation variables ({', '.join(high_corr_cols[:8])})",
                "why": "Reduces multicollinearity and feature redundancy.",
            }
        )

    suggestions.extend(
        [
            {
                "transformation": "Robust scaling for fare/distance/duration features",
                "why": "Mitigates impact of outliers.",
            },
            {
                "transformation": "Frequency encoding for high-cardinality location IDs",
                "why": "Retains signal while controlling feature dimensionality.",
            },
            {
                "transformation": "Add weather and special-event features",
                "why": "External context often explains demand and duration shifts.",
            },
        ]
    )

    return pd.DataFrame(suggestions)


def run_eda(df: pd.DataFrame, project_root: Path) -> None:
    print("\n=== 1) Profile Each Attribute ===")
    attr_profile = profile_attributes(df)
    print(attr_profile.to_string(index=False))

    print("\n=== 2) Statistical Analysis ===")
    desc, quantiles, missing_pct = statistical_analysis(df)
    if not desc.empty:
        print("\nDescriptive statistics (numeric):")
        print(desc.round(4).to_string())
        print("\nQuantiles (numeric):")
        print(quantiles.round(4).to_string())
    print("\nMissing values (%):")
    print(missing_pct.round(3).to_string())

    print("\n=== 3) Data Quality Assessment ===")
    quality_df = assess_data_quality(df)
    if quality_df.empty:
        print("No numeric columns found for quality checks.")
    else:
        print(quality_df.to_string(index=False))
    print(f"\nDuplicate rows: {int(df.duplicated().sum())}")

    print("\n=== 4) Identify Target Candidates ===")
    targets = identify_target_candidates(df)
    print(targets.to_string(index=False))

    print("\n=== 6) Study Correlations ===")
    corr, reject_cols, high_corr_df = study_correlations(df, threshold=0.9)
    if high_corr_df.empty:
        print("No variable pairs above correlation threshold (|r| >= 0.9).")
    else:
        print(high_corr_df.to_string(index=False))
        print(f"\nSuggested candidates to reject due to high correlation: {reject_cols}")

    print("\n=== 7) Simulate Manual Solutions ===")
    manual_df = simulate_manual_solution(df)
    print(manual_df.to_string(index=False))

    print("\n=== 8) Identify Promising Transformations / Extra Data ===")
    transform_df = identify_transformations(df, high_corr_df)
    print(transform_df.to_string(index=False))

    print("\n=== 5) Create Visualizations ===")
    fig_dir = project_root / "notebooks" / "exploratory" / "figures" / "eda_new"
    visualize_data(df, corr, fig_dir)
    print(f"Saved plots to: {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA workflow for processed NYC mobility data.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to processed parquet file. Defaults to the latest file in data/processed.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()
    data_path = resolve_data_path(project_root, args.data)

    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded dataframe: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    run_eda(df, project_root)


if __name__ == "__main__":
    main()
