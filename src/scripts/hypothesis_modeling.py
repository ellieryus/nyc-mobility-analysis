#!/usr/bin/env python3
"""
Hypothesis testing pipeline for NYC yellow taxi demand predictability.

Focus:
1) XGBoost baseline for H2-related checks.
2) Advanced ensembles (bagging + stacking).
3) Explicit support / not-support verdicts for:
   - Manhattan vs Bronx consistency
   - Airport predictability (JFK/LGA)
   - Zone-specific vs city-wide modeling grain
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
AIRPORT_ZONE_IDS = {132, 138}
COL_RE = re.compile(r"([A-Z]+)([0-9]+)")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
            row_vals: Dict[int, str] = {}
            for c in row.findall("m:c", NS):
                ref = c.attrib.get("r", "")
                col = excel_col_idx(ref) if ref else len(row_vals)
                ctype = c.attrib.get("t")
                v = c.find("m:v", NS)
                val = v.text if v is not None else ""
                if ctype == "s" and val != "":
                    val = shared_strings[int(val)]
                elif ctype == "inlineStr":
                    t = c.find("m:is/m:t", NS)
                    val = t.text if t is not None else ""
                row_vals[col] = val
                max_col = max(max_col, col)
            parsed_rows.append(row_vals)

    records = []
    for row in parsed_rows:
        rec = [row.get(i, "") for i in range(max_col + 1)]
        records.append(rec)

    header = records[0]
    body = records[1:]
    df = pd.DataFrame(body, columns=header)
    df.columns = [str(c).strip() if str(c).strip() else f"col_{i}" for i, c in enumerate(df.columns)]

    # Keep the first non-empty version when duplicated columns appear (airport_fee/Airport_fee).
    dedup = {}
    for c in df.columns:
        key = c.lower()
        if key not in dedup:
            dedup[key] = c
    df = df[[dedup[k] for k in dedup]]
    return df


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    # Excel date serial origin.
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
    demand = demand.reset_index(drop=True)
    return demand


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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_variance": float(np.var(y_true - y_pred)),
    }


def get_models(seed: int = 42) -> Dict[str, object]:
    xgb = XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=2,
    )
    bagging = RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=2,
    )
    stack = StackingRegressor(
        estimators=[
            ("xgb", xgb),
            ("rf", RandomForestRegressor(n_estimators=80, random_state=seed, n_jobs=2)),
            ("hgb", HistGradientBoostingRegressor(max_depth=6, random_state=seed)),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
        passthrough=True,
        n_jobs=1,
        cv=3,
    )
    return {"xgboost": xgb, "bagging_rf": bagging, "stacking": stack}


def fit_eval_model(model, train: pd.DataFrame, test: pd.DataFrame, features: List[str], target: str) -> Tuple[object, Dict[str, float], np.ndarray]:
    model.fit(train[features], train[target])
    preds = model.predict(test[features])
    return model, metrics(test[target].to_numpy(), preds), preds


def zone_specific_xgb(
    split: SplitData,
    features: List[str],
    target: str,
    min_train_rows: int = 120,
    max_zones: int = 80,
) -> pd.DataFrame:
    rows = []
    candidate_zones = (
        split.train.groupby("PULocationID")
        .size()
        .sort_values(ascending=False)
        .head(max_zones)
        .index.tolist()
    )
    for airport_zone in AIRPORT_ZONE_IDS:
        if airport_zone in set(split.train["PULocationID"]) and airport_zone not in candidate_zones:
            candidate_zones.append(airport_zone)
    for zone in candidate_zones:
        ztr = split.train[split.train["PULocationID"] == zone]
        zte = split.test[split.test["PULocationID"] == zone]
        min_rows = 24 if zone in AIRPORT_ZONE_IDS else min_train_rows
        if len(ztr) < min_rows or len(zte) == 0:
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
        model.fit(ztr[features], ztr[target])
        pred = model.predict(zte[features])
        m = metrics(zte[target].to_numpy(), pred)
        m["PULocationID"] = int(zone)
        m["n_test"] = int(len(zte))
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=["PULocationID", "mae", "rmse", "r2", "residual_variance", "n_test"])
    return pd.DataFrame(rows)


def load_zone_lookup(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    z = pd.read_csv(path)
    z = z.rename(columns={c: c.strip() for c in z.columns})
    need = {"LocationID", "Borough"}
    if not need.issubset(z.columns):
        return None
    z["LocationID"] = pd.to_numeric(z["LocationID"], errors="coerce")
    z = z.dropna(subset=["LocationID", "Borough"]).copy()
    z["LocationID"] = z["LocationID"].astype(int)
    return z[["LocationID", "Borough"]]


def borough_xgb_test(split: SplitData, features: List[str], target: str, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    zmap = zone_lookup.rename(columns={"LocationID": "PULocationID"})
    tr = split.train.merge(zmap, on="PULocationID", how="inner")
    te = split.test.merge(zmap, on="PULocationID", how="inner")

    out = []
    for borough, btr in tr.groupby("Borough"):
        bte = te[te["Borough"] == borough]
        if len(btr) < 300 or len(bte) < 80:
            continue
        model = XGBRegressor(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )
        model.fit(btr[features], btr[target])
        pred = model.predict(bte[features])
        m = metrics(bte[target].to_numpy(), pred)
        m["Borough"] = borough
        m["n_test"] = int(len(bte))
        out.append(m)

    return pd.DataFrame(out)


def verdict(flag: bool, msg_if_true: str, msg_if_false: str) -> Dict[str, str]:
    return {"status": "support" if flag else "not_support", "reason": msg_if_true if flag else msg_if_false}


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = parse_xlsx_sheet(Path(args.input_xlsx), sheet_name="Sample_Data")
    demand = prepare_zone_hour_data(raw, fill_full_grid=args.fill_full_grid)
    split = temporal_split(demand)

    feature_cols = [
        "PULocationID",
        "hour",
        "dow",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "lag_1",
        "lag_2",
        "lag_24",
        "lag_168",
        "roll_mean_24",
        "roll_mean_168",
    ]
    target = "demand"

    models = get_models(seed=args.seed)
    model_rows = []
    preds = {}
    for name, model in models.items():
        fitted, m, pred = fit_eval_model(model, split.train, split.test, feature_cols, target)
        _ = fitted
        model_rows.append({"model": name, **m})
        preds[name] = pred

    model_results = pd.DataFrame(model_rows).sort_values("mae")
    best_model_name = model_results.iloc[0]["model"]

    # Zone-specific XGB
    zone_results = zone_specific_xgb(
        split,
        feature_cols,
        target,
        min_train_rows=args.min_zone_train_rows,
        max_zones=args.max_zones,
    )

    # Compare zone-specific average MAE vs city-wide XGBoost MAE on same zones.
    city_xgb_pred = preds["xgboost"]
    city_test = split.test.copy()
    city_test["city_xgb_pred"] = city_xgb_pred
    city_test["abs_err_city_xgb"] = (city_test[target] - city_test["city_xgb_pred"]).abs()
    city_zone_mae = city_test.groupby("PULocationID")["abs_err_city_xgb"].mean().rename("city_xgb_zone_mae")
    zone_cmp = zone_results.merge(city_zone_mae, on="PULocationID", how="left")
    zone_cmp["mae_gain"] = zone_cmp["city_xgb_zone_mae"] - zone_cmp["mae"]

    zone_tested = len(zone_cmp) > 0 and np.isfinite(zone_cmp["mae_gain"].mean())
    zone_outperform = float(zone_cmp["mae_gain"].mean()) > 0.0 if zone_tested else False

    # Airport predictability from zone-specific residual variance.
    airport_stats = None
    airport_support = False
    airport_tested = False
    if len(zone_results):
        zone_results = zone_results.copy()
        zone_results["is_airport"] = zone_results["PULocationID"].isin(AIRPORT_ZONE_IDS)
        airport_grp = zone_results.groupby("is_airport")["residual_variance"].mean()
        airport_var = float(airport_grp.get(True, np.nan))
        non_airport_var = float(airport_grp.get(False, np.nan))
        if np.isfinite(airport_var) and np.isfinite(non_airport_var):
            airport_tested = True
            airport_support = airport_var < non_airport_var
        airport_stats = {
            "airport_residual_variance": airport_var,
            "non_airport_residual_variance": non_airport_var,
            "metric": "zone_model_residual_variance",
        }
    else:
        # Fallback: compare variance of observed hourly demand in test split.
        tmp = split.test.copy()
        tmp["is_airport"] = tmp["PULocationID"].isin(AIRPORT_ZONE_IDS)
        var_grp = tmp.groupby("is_airport")["demand"].var()
        airport_var = float(var_grp.get(True, np.nan))
        non_airport_var = float(var_grp.get(False, np.nan))
        if np.isfinite(airport_var) and np.isfinite(non_airport_var):
            airport_tested = True
            airport_support = airport_var < non_airport_var
            airport_stats = {
                "airport_demand_variance": airport_var,
                "non_airport_demand_variance": non_airport_var,
                "metric": "observed_hourly_demand_variance_fallback",
            }

    # Borough Manhattan vs Bronx with optional zone lookup.
    borough_lookup = load_zone_lookup(Path(args.zone_lookup) if args.zone_lookup else None)
    borough_results = pd.DataFrame()
    borough_verdict = {
        "status": "not_tested",
        "reason": "Provide --zone-lookup taxi_zone_lookup.csv to run Manhattan vs Bronx consistency test.",
    }
    if borough_lookup is not None:
        borough_results = borough_xgb_test(split, feature_cols, target, borough_lookup)
        if {"Manhattan", "Bronx"}.issubset(set(borough_results["Borough"])):
            m_mae = float(borough_results.loc[borough_results["Borough"] == "Manhattan", "mae"].iloc[0])
            b_mae = float(borough_results.loc[borough_results["Borough"] == "Bronx", "mae"].iloc[0])
            rel_gain = (b_mae - m_mae) / max(b_mae, 1e-9)
            borough_verdict = verdict(
                rel_gain >= args.min_relative_gain,
                f"Manhattan MAE is {rel_gain:.1%} lower than Bronx (>= {args.min_relative_gain:.0%} threshold).",
                f"Manhattan MAE is only {rel_gain:.1%} lower than Bronx (< {args.min_relative_gain:.0%} threshold).",
            )
        else:
            borough_verdict = {
                "status": "not_tested",
                "reason": "Manhattan/Bronx rows unavailable after merging with zone lookup and split filters.",
            }

    if zone_tested:
        zone_verdict = verdict(
            zone_outperform,
            "Average zone-level XGBoost MAE is lower than city-wide XGBoost MAE on matched zones.",
            "Average zone-level XGBoost MAE is not lower than city-wide XGBoost MAE on matched zones.",
        )
    else:
        zone_verdict = {
            "status": "not_tested",
            "reason": "Insufficient overlapping zone-level metrics to compare zone-specific vs city-wide models.",
        }

    if airport_tested:
        airport_verdict = verdict(
            airport_support,
            "Airport zones (132, 138) show lower residual variance than non-airport zones.",
            "Airport zones (132, 138) do not show lower residual variance than non-airport zones.",
        )
    else:
        airport_verdict = {
            "status": "not_tested",
            "reason": "Insufficient airport/non-airport comparison data in this split/model set.",
        }

    hypothesis_summary = {
        "best_citywide_model": best_model_name,
        "h2_zone_specific_outperform_citywide": zone_verdict,
        "h2_airport_predictability": airport_verdict,
        "h2_borough_consistency_manhattan_vs_bronx": borough_verdict,
        "airport_stats": airport_stats,
    }

    model_results.to_csv(out_dir / "model_comparison.csv", index=False)
    zone_results.sort_values("mae").to_csv(out_dir / "zone_specific_xgb_metrics.csv", index=False)
    zone_cmp.sort_values("mae_gain", ascending=False).to_csv(out_dir / "zone_vs_city_comparison.csv", index=False)
    if len(borough_results):
        borough_results.sort_values("mae").to_csv(out_dir / "borough_xgb_metrics.csv", index=False)

    summary_md = [
        "# NYC Hypothesis Testing Summary",
        "",
        f"- Input: `{args.input_xlsx}`",
        f"- Output directory: `{out_dir}`",
        f"- Best city-wide model: **{best_model_name}**",
        "",
        "## Model Comparison (City-Wide)",
        "```",
        model_results.to_string(index=False),
        "```",
        "",
        "## Hypothesis Verdicts",
        f"- Zone-specific > city-wide: **{hypothesis_summary['h2_zone_specific_outperform_citywide']['status']}**",
        f"  - {hypothesis_summary['h2_zone_specific_outperform_citywide']['reason']}",
        f"- Airport predictability (132/138): **{hypothesis_summary['h2_airport_predictability']['status']}**",
        f"  - {hypothesis_summary['h2_airport_predictability']['reason']}",
        f"- Borough consistency (Manhattan vs Bronx): **{hypothesis_summary['h2_borough_consistency_manhattan_vs_bronx']['status']}**",
        f"  - {hypothesis_summary['h2_borough_consistency_manhattan_vs_bronx']['reason']}",
        "",
        "## Notes",
        "- Borough test requires `taxi_zone_lookup.csv` with `LocationID,Borough` columns.",
        "- Stacking + bagging are included in city-wide comparison to extend beyond pure XGBoost.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    (out_dir / "hypothesis_summary.json").write_text(json.dumps(hypothesis_summary, indent=2), encoding="utf-8")

    print(f"Saved results to: {out_dir}")
    print(f"Best city-wide model: {best_model_name}")
    print(json.dumps(hypothesis_summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NYC yellow taxi H2 modeling pipeline")
    p.add_argument(
        "--input-xlsx",
        default=str(PROJECT_ROOT / "data" / "raw" / "yellow_taxi_24months_complete.xlsx"),
        help="Path to Excel file",
    )
    p.add_argument(
        "--zone-lookup",
        default=None,
        help="Optional taxi zone lookup CSV path (LocationID,Borough required)",
    )
    p.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "results"),
        help="Directory for generated outputs",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-zone-train-rows", type=int, default=120)
    p.add_argument("--max-zones", type=int, default=80)
    p.add_argument("--min-relative-gain", type=float, default=0.05)
    p.add_argument("--fill-full-grid", action="store_true", help="Expand to all hour x zone pairs with zeros.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
