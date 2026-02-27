# updated_GITHUB_sample_code

End-to-end MLOps-ready project built from scratch on teammate sample dataset:
`data/raw/yellow_taxi_representative_sample_2021_2023_distribution.csv`

## What this project includes
- Time-based train/validation/test split (70/15/15 by month)
- Tree/ensemble models:
  - Boosting: XGBoost
  - Bagging: RandomForestRegressor
  - Stacking: XGBoost + RandomForest + HistGradientBoosting -> RidgeCV
- Statistical hypothesis testing:
  - Paired permutation tests on absolute errors
  - Wilcoxon signed-rank tests
  - Bootstrap confidence intervals
  - Seasonality test over month groups (Kruskal-Wallis)
- Reproducible pipeline + outputs + plots

## Quick run
```bash
cd updated_GITHUB_sample_code
python -m updated_sample.pipeline
```

## Main outputs
- `reports/results/model_comparison.csv`
- `reports/results/pairwise_stat_tests.csv`
- `reports/results/hypothesis_summary.json`
- `reports/figures/model_metrics.png`
- `reports/figures/error_distributions.png`
- `reports/figures/model_tradeoff.png`
