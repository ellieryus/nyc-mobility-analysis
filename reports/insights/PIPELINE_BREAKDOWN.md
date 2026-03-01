# Pipeline Breakdown

## End-to-End Flow
1. Data ingestion from teammate sample CSV
2. Temporal feature engineering and lag construction
3. Strict chronological split into train/validation/test (70/15/15)
4. Model training:
   - Boosting (`XGBoost`)
   - Bagging (`RandomForestRegressor`)
   - Stacking (meta-ensemble)
5. Statistical inference:
   - Pairwise permutation tests
   - Wilcoxon signed-rank tests
   - Bootstrap confidence intervals
   - H1 seasonality (Kruskal-Wallis)
6. Reporting and visualization outputs

## Data Contract
- Required columns: `pickup_year`, `pickup_month`, `sample_rows`
- Derived timestamp key: first day of each month
- Modeling target: `sample_rows`

## Core Artifacts
- `reports/results/model_comparison.csv`
- `reports/results/pairwise_stat_tests.csv`
- `reports/results/hypothesis_summary.json`
- `reports/results/roadmap_execution.csv`

## Governance Checks
- Time leakage avoided by chronological split
- Statistical decision threshold controlled by alpha
- Reproducibility ensured by deterministic seed
