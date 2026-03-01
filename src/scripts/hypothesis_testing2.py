#!/usr/bin/env python3
"""
Comprehensive hypothesis testing pipeline with roadmap execution tracking,
advanced model comparison, and formal statistical tests.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import levene, mannwhitneyu, wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
AIRPORT_ZONE_IDS = {132, 138}


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def excel_col_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    n = 0
    for c in letters:
        n = n * 26 + (ord(c) - 64)
    return n - 1


def parse_xlsx_sheet(xlsx_path: Path, sheet_name: str = "Sample_Data") -> pd.DataFrame:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared_strings.append("".join((t.text or "") for t in si.findall(".//m:t", NS)))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        target = None
        for sh in wb.findall("m:sheets/m:sheet", NS):
            if sh.attrib["name"] == sheet_name:
                rid = sh.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                target = rid_to_target[rid].lstrip("/")
                if not target.startswith("xl/"):
                    target = f"xl/{target}"
                break

        if target is None:
            raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")

        ws = ET.fromstring(zf.read(target))
        rows = ws.findall("m:sheetData/m:row", NS)

        parsed_rows: List[Dict[int, str]] = []
        max_col = 0
        for row in rows:
            vals: Dict[int, str] = {}
            for c in row.findall("m:c", NS):
                ref = c.attrib.get("r", "")
                col = excel_col_idx(ref) if ref else len(vals)
                ctype = c.attrib.get("t")
                v = c.find("m:v", NS)
                value = v.text if v is not None else ""
                if ctype == "s" and value != "":
                    value = shared_strings[int(value)]
                elif ctype == "inlineStr":
                    t = c.find("m:is/m:t", NS)
                    value = t.text if t is not None else ""
                vals[col] = value
                max_col = max(max_col, col)
            parsed_rows.append(vals)

    records = [[row.get(i, "") for i in range(max_col + 1)] for row in parsed_rows]
    header = records[0]
    body = records[1:]

    df = pd.DataFrame(body, columns=header)
    df.columns = [str(c).strip() if str(c).strip() else f"col_{i}" for i, c in enumerate(df.columns)]

    dedup = {}
    for c in df.columns:
        key = c.lower()
        if key not in dedup:
            dedup[key] = c
    return df[[dedup[k] for k in dedup]]


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime("1899-12-30") + pd.to_timedelta(series, unit="D")


def prepare_zone_hour_data(raw: pd.DataFrame, fill_full_grid: bool = False) -> pd.DataFrame:
    required = {"tpep_pickup_datetime", "PULocationID"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw = coerce_numeric(raw.copy(), ["tpep_pickup_datetime", "PULocationID"])
    raw = raw.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
    raw["PULocationID"] = raw["PULocationID"].astype(int)
    raw["pickup_dt"] = excel_serial_to_datetime(raw["tpep_pickup_datetime"])
    raw["pickup_hour"] = raw["pickup_dt"].dt.floor("h")

    demand = (
        raw.groupby(["pickup_hour", "PULocationID"], as_index=False)
        .size()
        .rename(columns={"size": "demand"})
    )

    if fill_full_grid:
        all_hours = pd.date_range(demand["pickup_hour"].min(), demand["pickup_hour"].max(), freq="h")
        all_zones = np.sort(demand["PULocationID"].unique())
        grid = pd.MultiIndex.from_product([all_hours, all_zones], names=["pickup_hour", "PULocationID"])
        demand = demand.set_index(["pickup_hour", "PULocationID"]).reindex(grid, fill_value=0).reset_index()

    demand["hour"] = demand["pickup_hour"].dt.hour
    demand["dow"] = demand["pickup_hour"].dt.dayofweek
    demand["month"] = demand["pickup_hour"].dt.month
    demand["is_weekend"] = (demand["dow"] >= 5).astype(int)
    demand["hour_sin"] = np.sin(2 * np.pi * demand["hour"] / 24)
    demand["hour_cos"] = np.cos(2 * np.pi * demand["hour"] / 24)

    demand = demand.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)
    grp = demand.groupby("PULocationID")

    demand["lag_1"] = grp["demand"].shift(1)
    demand["lag_2"] = grp["demand"].shift(2)
    demand["lag_24"] = grp["demand"].shift(24)
    demand["roll_mean_24"] = grp["demand"].shift(1).rolling(24, min_periods=1).mean().reset_index(level=0, drop=True)

    if fill_full_grid:
        demand["lag_168"] = grp["demand"].shift(168)
        demand["roll_mean_168"] = grp["demand"].shift(1).rolling(168, min_periods=1).mean().reset_index(level=0, drop=True)
    else:
        demand["lag_168"] = demand["lag_24"]
        demand["roll_mean_168"] = demand["roll_mean_24"]

    lag_cols = ["lag_1", "lag_2", "lag_24", "lag_168", "roll_mean_24", "roll_mean_168"]
    demand[lag_cols] = demand[lag_cols].fillna(0.0)
    return demand.reset_index(drop=True)


def temporal_split(df: pd.DataFrame, time_col: str = "pickup_hour") -> SplitData:
    uniq = np.array(sorted(df[time_col].unique()))
    n = len(uniq)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_t = set(uniq[:train_end])
    valid_t = set(uniq[train_end:valid_end])
    test_t = set(uniq[valid_end:])

    return SplitData(
        train=df[df[time_col].isin(train_t)].copy(),
        valid=df[df[time_col].isin(valid_t)].copy(),
        test=df[df[time_col].isin(test_t)].copy(),
    )


def base_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_variance": float(np.var(y_true - y_pred)),
    }


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Returns (mean_diff, ci_low, ci_high) for mean(a - b)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    observed = float(np.mean(diff))
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return observed, float(low), float(high)


def permutation_pvalue_mean_diff(a: np.ndarray, b: np.ndarray, n_perm: int = 4000, seed: int = 42) -> float:
    """Two-sided permutation p-value for paired mean difference."""
    rng = np.random.default_rng(seed)
    diff = a - b
    observed = abs(diff.mean())
    signs = rng.choice([-1, 1], size=(n_perm, len(diff)))
    perm = np.abs((diff * signs).mean(axis=1))
    return float((np.sum(perm >= observed) + 1) / (n_perm + 1))


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    try:
        stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(stat.pvalue)
    except Exception:
        return float("nan")


def roadmap_template() -> List[dict]:
    return [
        {"stage": "data_ingestion", "status": "pending"},
        {"stage": "data_quality_and_feature_engineering", "status": "pending"},
        {"stage": "model_training_and_comparison", "status": "pending"},
        {"stage": "statistical_hypothesis_tests", "status": "pending"},
        {"stage": "artifacts_and_reporting", "status": "pending"},
    ]


def set_stage(roadmap: List[dict], stage: str, status: str) -> None:
    for row in roadmap:
        if row["stage"] == stage:
            row["status"] = status
            return


def get_models(seed: int = 42) -> Dict[str, object]:
    return {
        "xgboost": XGBRegressor(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=2,
        ),
        "bagging_rf": RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=2,
        ),
        "stacking": StackingRegressor(
            estimators=[
                ("xgb", XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=seed,
                    n_jobs=1,
                )),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=2)),
                ("hgb", HistGradientBoostingRegressor(max_depth=6, random_state=seed)),
                ("etr", ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=2)),
            ],
            final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
            passthrough=True,
            cv=3,
            n_jobs=1,
        ),
    }


def fit_eval_model(model, train: pd.DataFrame, test: pd.DataFrame, features: List[str], target: str):
    model.fit(train[features], train[target])
    pred = model.predict(test[features])
    m = base_metrics(test[target].to_numpy(), pred)
    return m, pred


def load_zone_lookup(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    z = pd.read_csv(path)
    need = {"LocationID", "Borough"}
    if not need.issubset(z.columns):
        return None
    z["LocationID"] = pd.to_numeric(z["LocationID"], errors="coerce")
    z = z.dropna(subset=["LocationID", "Borough"]).copy()
    z["LocationID"] = z["LocationID"].astype(int)
    return z[["LocationID", "Borough"]]


def zone_specific_xgb(split: SplitData, features: List[str], target: str, min_train_rows: int = 120, max_zones: int = 80) -> pd.DataFrame:
    candidates = (
        split.train.groupby("PULocationID").size().sort_values(ascending=False).head(max_zones).index.tolist()
    )
    for z in AIRPORT_ZONE_IDS:
        if z in set(split.train["PULocationID"]) and z not in candidates:
            candidates.append(z)

    rows = []
    for zone in candidates:
        tr = split.train[split.train["PULocationID"] == zone]
        te = split.test[split.test["PULocationID"] == zone]
        min_rows = 24 if zone in AIRPORT_ZONE_IDS else min_train_rows
        if len(tr) < min_rows or len(te) == 0:
            continue

        model = XGBRegressor(
            n_estimators=90,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
        model.fit(tr[features], tr[target])
        pred = model.predict(te[features])
        m = base_metrics(te[target].to_numpy(), pred)
        m.update({"PULocationID": int(zone), "n_test": int(len(te))})
        rows.append(m)

    return pd.DataFrame(rows)


def plot_model_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#355C7D", "#4E6E58", "#6C7A89"]

    df = df.sort_values("mae")
    axes[0].bar(df["model"], df["mae"], color=colors[: len(df)])
    axes[0].set_title("MAE")
    axes[0].set_ylabel("Lower is better")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(df["model"], df["rmse"], color=colors[: len(df)])
    axes[1].set_title("RMSE")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(df["model"], df["r2"], color=colors[: len(df)])
    axes[2].set_title("R2")
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Hypothesis Testing2 - Advanced Model Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "hypothesis_testing2_model_metrics.png", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roadmap = roadmap_template()

    set_stage(roadmap, "data_ingestion", "in_progress")
    raw = parse_xlsx_sheet(Path(args.input_xlsx), sheet_name="Sample_Data")
    set_stage(roadmap, "data_ingestion", "done")

    set_stage(roadmap, "data_quality_and_feature_engineering", "in_progress")
    demand = prepare_zone_hour_data(raw, fill_full_grid=args.fill_full_grid)
    split = temporal_split(demand)
    set_stage(roadmap, "data_quality_and_feature_engineering", "done")

    feature_cols = [
        "PULocationID", "hour", "dow", "month", "is_weekend",
        "hour_sin", "hour_cos", "lag_1", "lag_2", "lag_24", "lag_168", "roll_mean_24", "roll_mean_168",
    ]
    target = "demand"

    set_stage(roadmap, "model_training_and_comparison", "in_progress")
    models = get_models(seed=args.seed)
    model_rows = []
    pred_table = split.test[["pickup_hour", "PULocationID", target]].copy().reset_index(drop=True)

    for name, model in models.items():
        m, pred = fit_eval_model(model, split.train, split.test, feature_cols, target)
        model_rows.append({"model": name, **m})
        pred_table[f"pred_{name}"] = pred

    model_results = pd.DataFrame(model_rows).sort_values("mae").reset_index(drop=True)
    best_model = model_results.iloc[0]["model"]
    set_stage(roadmap, "model_training_and_comparison", "done")

    set_stage(roadmap, "statistical_hypothesis_tests", "in_progress")
    # Pairwise statistical comparison on absolute errors.
    pairwise = []
    model_names = model_results["model"].tolist()
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            e1 = np.abs(pred_table[target] - pred_table[f"pred_{m1}"]).to_numpy()
            e2 = np.abs(pred_table[target] - pred_table[f"pred_{m2}"]).to_numpy()

            mean_diff, ci_low, ci_high = bootstrap_ci_mean_diff(e1, e2, n_boot=args.n_boot, seed=args.seed)
            p_perm = permutation_pvalue_mean_diff(e1, e2, n_perm=args.n_perm, seed=args.seed)
            p_wil = safe_wilcoxon(e1, e2)

            pairwise.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "mean_abs_error_diff_a_minus_b": mean_diff,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_permutation": p_perm,
                    "p_wilcoxon": p_wil,
                }
            )

    pairwise_df = pd.DataFrame(pairwise)

    # Zone-specific hypothesis (zone model vs city-wide xgboost)
    zone_df = zone_specific_xgb(split, feature_cols, target, min_train_rows=args.min_zone_train_rows, max_zones=args.max_zones)
    city_abs = np.abs(pred_table[target] - pred_table["pred_xgboost"]).rename("abs_err_city_xgb")
    city_zone_mae = pred_table.assign(abs_err_city_xgb=city_abs).groupby("PULocationID")["abs_err_city_xgb"].mean().rename("city_xgb_zone_mae")

    if len(zone_df):
        zone_cmp = zone_df.merge(city_zone_mae, on="PULocationID", how="left")
        zone_cmp["mae_gain"] = zone_cmp["city_xgb_zone_mae"] - zone_cmp["mae"]
        z = zone_cmp["mae_gain"].dropna().to_numpy()
        if len(z) >= 5:
            z_mean, z_low, z_high = bootstrap_ci_mean_diff(z, np.zeros_like(z), n_boot=args.n_boot, seed=args.seed)
            z_p = permutation_pvalue_mean_diff(z, np.zeros_like(z), n_perm=args.n_perm, seed=args.seed)
            zone_test = {
                "tested": True,
                "mean_mae_gain": float(z_mean),
                "ci95": [float(z_low), float(z_high)],
                "p_value": float(z_p),
                "support": bool(z_mean > 0 and z_p < args.alpha),
            }
        else:
            zone_test = {"tested": False, "reason": "insufficient zones for stable inference"}
    else:
        zone_cmp = pd.DataFrame()
        zone_test = {"tested": False, "reason": "no zone-level models built"}

    # Airport vs non-airport residual variance test.
    if len(zone_df):
        zone_df = zone_df.copy()
        zone_df["is_airport"] = zone_df["PULocationID"].isin(AIRPORT_ZONE_IDS)
        a = zone_df.loc[zone_df["is_airport"], "residual_variance"].dropna().to_numpy()
        n = zone_df.loc[~zone_df["is_airport"], "residual_variance"].dropna().to_numpy()
        if len(a) >= 2 and len(n) >= 2:
            stat_lev = levene(a, n, center="median")
            stat_mw = mannwhitneyu(a, n, alternative="two-sided")
            diff_mean, diff_low, diff_high = bootstrap_ci_mean_diff(a, n, n_boot=args.n_boot, seed=args.seed)
            airport_test = {
                "tested": True,
                "airport_mean_var": float(np.mean(a)),
                "non_airport_mean_var": float(np.mean(n)),
                "mean_diff_airport_minus_nonairport": float(diff_mean),
                "ci95": [float(diff_low), float(diff_high)],
                "p_levene": float(stat_lev.pvalue),
                "p_mannwhitney": float(stat_mw.pvalue),
                "support": bool(np.mean(a) < np.mean(n) and stat_mw.pvalue < args.alpha),
            }
        else:
            airport_test = {"tested": False, "reason": "insufficient airport/non-airport zone residual samples"}
    else:
        airport_test = {"tested": False, "reason": "zone-level output unavailable"}

    # Borough Manhattan vs Bronx.
    borough_test = {"tested": False, "reason": "zone lookup not provided"}
    borough_df = pd.DataFrame()
    zone_lookup = load_zone_lookup(Path(args.zone_lookup) if args.zone_lookup else None)
    if zone_lookup is not None:
        zmap = zone_lookup.rename(columns={"LocationID": "PULocationID"})
        tr = split.train.merge(zmap, on="PULocationID", how="inner")
        te = split.test.merge(zmap, on="PULocationID", how="inner")

        rows = []
        borough_abs_errors = {}
        for borough, btr in tr.groupby("Borough"):
            bte = te[te["Borough"] == borough]
            if len(btr) < 300 or len(bte) < 80:
                continue
            model = XGBRegressor(
                n_estimators=140,
                max_depth=5,
                learning_rate=0.07,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=args.seed,
                n_jobs=1,
            )
            model.fit(btr[feature_cols], btr[target])
            pred = model.predict(bte[feature_cols])
            m = base_metrics(bte[target].to_numpy(), pred)
            m.update({"Borough": borough, "n_test": int(len(bte))})
            rows.append(m)
            borough_abs_errors[borough] = np.abs(bte[target].to_numpy() - pred)

        borough_df = pd.DataFrame(rows)
        if {"Manhattan", "Bronx"}.issubset(set(borough_df["Borough"])):
            m_err = borough_abs_errors["Manhattan"]
            b_err = borough_abs_errors["Bronx"]
            mean_diff, ci_low, ci_high = bootstrap_ci_mean_diff(b_err, m_err, n_boot=args.n_boot, seed=args.seed)
            p_perm = permutation_pvalue_mean_diff(b_err, m_err, n_perm=args.n_perm, seed=args.seed)
            m_mae = float(borough_df.loc[borough_df["Borough"] == "Manhattan", "mae"].iloc[0])
            b_mae = float(borough_df.loc[borough_df["Borough"] == "Bronx", "mae"].iloc[0])
            rel_gain = (b_mae - m_mae) / max(b_mae, 1e-9)
            borough_test = {
                "tested": True,
                "manhattan_mae": m_mae,
                "bronx_mae": b_mae,
                "relative_gain": float(rel_gain),
                "ci95_bronx_minus_manhattan_abs_err": [float(ci_low), float(ci_high)],
                "p_permutation": float(p_perm),
                "support": bool(rel_gain >= args.min_relative_gain and p_perm < args.alpha and mean_diff > 0),
            }
        else:
            borough_test = {"tested": False, "reason": "Manhattan/Bronx not both available after filtering"}

    set_stage(roadmap, "statistical_hypothesis_tests", "done")

    set_stage(roadmap, "artifacts_and_reporting", "in_progress")
    plot_model_metrics(model_results, out_dir)

    summary = {
        "best_citywide_model": best_model,
        "alpha": args.alpha,
        "roadmap": roadmap,
        "statistical_tests": {
            "pairwise_model_error_tests": pairwise,
            "h2_zone_specific_outperform_citywide": zone_test,
            "h2_airport_predictability": airport_test,
            "h2_borough_consistency_manhattan_vs_bronx": borough_test,
        },
    }

    model_results.to_csv(out_dir / "model_comparison.csv", index=False)
    pred_table.to_csv(out_dir / "test_predictions.csv", index=False)
    pairwise_df.to_csv(out_dir / "pairwise_model_stat_tests.csv", index=False)
    zone_df.to_csv(out_dir / "zone_specific_xgb_metrics.csv", index=False)
    zone_cmp.to_csv(out_dir / "zone_vs_city_comparison.csv", index=False)
    if len(borough_df):
        borough_df.to_csv(out_dir / "borough_xgb_metrics.csv", index=False)

    (out_dir / "hypothesis_testing2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "roadmap_execution_report.json").write_text(json.dumps(roadmap, indent=2), encoding="utf-8")

    summary_md = [
        "# Hypothesis Testing2 Report",
        "",
        f"- Best city-wide model: **{best_model}**",
        f"- Significance level: `{args.alpha}`",
        "",
        "## Model Comparison",
        "```",
        model_results.to_string(index=False),
        "```",
        "",
        "## Statistical Verdicts",
        f"- Zone-specific vs city-wide tested: `{zone_test.get('tested', False)}`",
        f"- Zone-specific support: `{zone_test.get('support', False)}`",
        f"- Airport predictability tested: `{airport_test.get('tested', False)}`",
        f"- Airport support: `{airport_test.get('support', False)}`",
        f"- Borough consistency tested: `{borough_test.get('tested', False)}`",
        f"- Borough support: `{borough_test.get('support', False)}`",
        "",
        "## Roadmap",
    ]
    for row in roadmap:
        summary_md.append(f"- {row['stage']}: {row['status']}")

    (out_dir / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    set_stage(roadmap, "artifacts_and_reporting", "done")

    print(f"Saved Hypothesis Testing2 outputs to: {out_dir}")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Comprehensive Hypothesis Testing2 pipeline")
    p.add_argument("--input-xlsx", default=str(PROJECT_ROOT / "data" / "raw" / "yellow_taxi_24months_complete.xlsx"))
    p.add_argument("--zone-lookup", default=str(PROJECT_ROOT / "data" / "external" / "taxi_zone_lookup.csv"))
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "results" / "hypothesis_testing2"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=1500)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--min-zone-train-rows", type=int, default=120)
    p.add_argument("--max-zones", type=int, default=80)
    p.add_argument("--min-relative-gain", type=float, default=0.05)
    p.add_argument("--fill-full-grid", action="store_true")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
