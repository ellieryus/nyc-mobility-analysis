"""Prophet forecasting pipeline exported from notebooks/exploratory/modeling/02_prophet.ipynb.

This is a script-form copy of the notebook code cells, kept primarily for submission
and code review. It preserves the notebook step structure and plotting logic.
"""


# Step 0: Pre-aggregation data inspection (run before Step 1)
from pathlib import Path
import re
import pandas as pd

cwd = Path.cwd().resolve()
candidates = [cwd, *cwd.parents]
repo_root_probe = next((p for p in candidates if (p / 'data' / 'processed').exists()), None)
if repo_root_probe is None:
    raise FileNotFoundError('Cannot find repo root with data/processed')

processed_dir_probe = repo_root_probe / 'data' / 'processed'
pattern = re.compile(r'^yellow_tripdata_(\d{4}-\d{2})\.parquet$')
files_probe = []
for f in sorted(processed_dir_probe.glob('yellow_tripdata_*.parquet')):
    m = pattern.match(f.name)
    if m and '2024-01' <= m.group(1) <= '2025-12':
        files_probe.append(f)

assert all(f.name.startswith('yellow_tripdata_') for f in files_probe), 'Non-yellow file detected'

if not files_probe:
    raise FileNotFoundError(f'No monthly files found in {processed_dir_probe}')

file_summary = []
sample_frames = []
for fp in files_probe:
    tmp = pd.read_parquet(fp, columns=['pickup_datetime', 'dropoff_datetime', 'pickup_location_id'])
    tmp['pickup_datetime'] = pd.to_datetime(tmp['pickup_datetime'], errors='coerce')
    tmp['dropoff_datetime'] = pd.to_datetime(tmp['dropoff_datetime'], errors='coerce')
    row = {
        'file': fp.name,
        'rows': len(tmp),
        'pickup_min': tmp['pickup_datetime'].min(),
        'pickup_max': tmp['pickup_datetime'].max(),
        'null_pickup_pct': float(tmp['pickup_datetime'].isna().mean() * 100),
        'null_zone_pct': float(tmp['pickup_location_id'].isna().mean() * 100),
    }
    file_summary.append(row)
    sample_frames.append(tmp[['pickup_datetime', 'dropoff_datetime', 'pickup_location_id']].sample(min(5000, len(tmp)), random_state=42))

file_summary_df = pd.DataFrame(file_summary)
sample_df = pd.concat(sample_frames, ignore_index=True)
sample_df['trip_duration_hours'] = (sample_df['dropoff_datetime'] - sample_df['pickup_datetime']).dt.total_seconds() / 3600
global_checks = pd.DataFrame([
    {'check': 'sample_rows', 'value': int(len(sample_df))},
    {'check': 'sample_pickup_min', 'value': sample_df['pickup_datetime'].min()},
    {'check': 'sample_pickup_max', 'value': sample_df['pickup_datetime'].max()},
    {'check': 'sample_years', 'value': sorted(sample_df['pickup_datetime'].dropna().dt.year.unique().tolist())},
    {'check': 'negative_duration_pct', 'value': float((sample_df['trip_duration_hours'] < 0).mean() * 100)},
    {'check': 'gt_24h_duration_pct', 'value': float((sample_df['trip_duration_hours'] > 24).mean() * 100)},
])

print('Repo root:', repo_root_probe)
print('Files found:', len(files_probe))
print('First file:', files_probe[0].name)
print('Last file :', files_probe[-1].name)
print()
display(file_summary_df.head(10))
display(global_checks)


# Step 1: Project setup and path resolution
from pathlib import Path
import re
import pandas as pd

# Robust project-root detection (works even if notebook launches from another cwd)
cwd = Path.cwd().resolve()
candidates = [cwd, *cwd.parents]
repo_root = next((p for p in candidates if (p / "data" / "processed").exists()), None)

if repo_root is None:
    raise FileNotFoundError(
        "Could not locate project root with data/processed. "
        "Set repo_root manually."
    )

processed_dir = repo_root / "data" / "processed"
output_dir = repo_root / "reports" / "results" / "prophet"
output_dir.mkdir(parents=True, exist_ok=True)

# Only monthly files in requested window
start_month = "2024-01"
end_month = "2025-12"  # safe upper bound; your current data ends at 2025-11
pattern = re.compile(r"^yellow_tripdata_(\d{4}-\d{2})\.parquet$")

files = []
for f in sorted(processed_dir.glob("yellow_tripdata_*.parquet")):
    m = pattern.match(f.name)
    if not m:
        continue
    month_key = m.group(1)
    if start_month <= month_key <= end_month:
        files.append(f)

if not files:
    raise FileNotFoundError(
        f"No monthly files found in {processed_dir} for [{start_month}, {end_month}]"
    )

print("cwd:", cwd)
print("repo_root:", repo_root)
print("processed_dir exists:", processed_dir.exists())
print(f"Using {len(files)} files")
print("first:", files[0].name)
print("last :", files[-1].name)


# Step 1A: Aggregate trip-level rows to hourly demand
# Aggregate trip-level rows to hourly demand (city + zone)
city_chunks = []
zone_chunks = []

for fp in files:
    df = pd.read_parquet(fp, columns=['pickup_datetime', 'pickup_location_id'])
    df = df.dropna(subset=['pickup_datetime'])
    df['ds'] = pd.to_datetime(df['pickup_datetime']).dt.floor('h')

    # Keep only the intended modeling window to avoid bad historical timestamps
    lower = pd.Timestamp("2024-01-01 00:00:00")
    upper = pd.Timestamp("2025-12-31 23:59:59")
    df = df[(df["ds"] >= lower) & (df["ds"] <= upper)]

    city_part = df.groupby('ds', as_index=False).size().rename(columns={'size': 'y'})
    city_chunks.append(city_part)

    zone_part = (
        df.dropna(subset=['pickup_location_id'])
        .assign(zone_id=lambda x: x['pickup_location_id'].astype('int64'))
        .groupby(['zone_id', 'ds'], as_index=False)
        .size()
        .rename(columns={'size': 'y'})
    )
    zone_chunks.append(zone_part)

city_hourly = (
    pd.concat(city_chunks, ignore_index=True)
    .groupby('ds', as_index=False)['y']
    .sum()
    .sort_values('ds')
    .reset_index(drop=True)
)

zone_hourly = (
    pd.concat(zone_chunks, ignore_index=True)
    .groupby(['zone_id', 'ds'], as_index=False)['y']
    .sum()
    .sort_values(['zone_id', 'ds'])
    .reset_index(drop=True)
)

print('City hourly rows:', len(city_hourly))
print('Zone hourly rows:', len(zone_hourly))
print('City date range:', city_hourly['ds'].min(), 'to', city_hourly['ds'].max())
city_hourly.head()


# Step 1B: Leakage-safe time splits
# Strict time splits to avoid leakage
train_start = pd.Timestamp('2024-01-01 00:00:00')
train_end = pd.Timestamp('2024-12-31 23:59:59')
val_start = pd.Timestamp('2025-01-01 00:00:00')
val_end = pd.Timestamp('2025-06-30 23:59:59')
test_start = pd.Timestamp('2025-07-01 00:00:00')

def split_by_time(df):
    train_df = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)].copy()
    val_df = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)].copy()
    test_df = df[df['ds'] >= test_start].copy()
    return train_df, val_df, test_df

city_train, city_val, city_test = split_by_time(city_hourly)
zone_train, zone_val, zone_test = split_by_time(zone_hourly)

# Leakage guards: non-overlap + chronological order
assert not city_train.empty and not city_val.empty and not city_test.empty
assert city_train['ds'].max() < city_val['ds'].min()
assert city_val['ds'].max() < city_test['ds'].min()

print('City split sizes:', len(city_train), len(city_val), len(city_test))
print('Zone split sizes:', len(zone_train), len(zone_val), len(zone_test))
print('City train range:', city_train['ds'].min(), 'to', city_train['ds'].max())
print('City val range  :', city_val['ds'].min(), 'to', city_val['ds'].max())
print('City test range :', city_test['ds'].min(), 'to', city_test['ds'].max())


# Step 1C: Persist aggregates and split outputs
# Persist full aggregates and split datasets
city_hourly.to_parquet(output_dir / 'yellow_city_hourly.parquet', index=False)
zone_hourly.to_parquet(output_dir / 'yellow_zone_hourly.parquet', index=False)

city_train[['ds', 'y']].to_parquet(output_dir / 'yellow_city_train.parquet', index=False)
city_val[['ds', 'y']].to_parquet(output_dir / 'yellow_city_val.parquet', index=False)
city_test[['ds', 'y']].to_parquet(output_dir / 'yellow_city_test.parquet', index=False)

zone_train.to_parquet(output_dir / 'yellow_zone_train.parquet', index=False)
zone_val.to_parquet(output_dir / 'yellow_zone_val.parquet', index=False)
zone_test.to_parquet(output_dir / 'yellow_zone_test.parquet', index=False)

print('Saved Prophet datasets to:', output_dir)

# Prophet-ready frames for city model
prophet_city_train = city_train[['ds', 'y']].copy()
prophet_city_val = city_val[['ds', 'y']].copy()
prophet_city_test = city_test[['ds', 'y']].copy()
prophet_city_train.head()


# Step 1 diagnostics: aggregation and split summary
summary_rows = [
    {"dataset": "city_hourly", "rows": len(city_hourly), "start": city_hourly['ds'].min(), "end": city_hourly['ds'].max()},
    {"dataset": "city_train", "rows": len(city_train), "start": city_train['ds'].min(), "end": city_train['ds'].max()},
    {"dataset": "city_val", "rows": len(city_val), "start": city_val['ds'].min(), "end": city_val['ds'].max()},
    {"dataset": "city_test", "rows": len(city_test), "start": city_test['ds'].min(), "end": city_test['ds'].max()},
    {"dataset": "zone_hourly", "rows": len(zone_hourly), "start": zone_hourly['ds'].min(), "end": zone_hourly['ds'].max()},
    {"dataset": "zone_train", "rows": len(zone_train), "start": zone_train['ds'].min(), "end": zone_train['ds'].max()},
    {"dataset": "zone_val", "rows": len(zone_val), "start": zone_val['ds'].min(), "end": zone_val['ds'].max()},
    {"dataset": "zone_test", "rows": len(zone_test), "start": zone_test['ds'].min(), "end": zone_test['ds'].max()},
]
summary_df = pd.DataFrame(summary_rows)
print('Trip-level to time-series transformation')
print('city_hourly has one row per hour with target y = trip count')
print('zone_hourly has one row per zone_id per hour with target y = trip count')
print()
print('No-leakage checks')
print('city_train max < city_val min:', city_train['ds'].max() < city_val['ds'].min())
print('city_val max < city_test min:', city_val['ds'].max() < city_test['ds'].min())
print('zone_train max < zone_val min:', zone_train['ds'].max() < zone_val['ds'].min())
print('zone_val max < zone_test min:', zone_val['ds'].max() < zone_test['ds'].min())
print()
summary_df


# Step 2: Train city-level Prophet on train split, evaluate on validation split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from prophet import Prophet
except ImportError as exc:
    raise ImportError('prophet is not installed in this kernel. Run: pip install prophet') from exc

def add_time_regressors(df):
    out = df.copy()
    out['hour'] = out['ds'].dt.hour
    out['dow'] = out['ds'].dt.dayofweek
    out['is_weekend'] = (out['dow'] >= 5).astype(int)
    out['is_rush_am'] = out['hour'].between(7, 9).astype(int)
    out['is_rush_pm'] = out['hour'].between(17, 19).astype(int)
    out['is_late_night'] = ((out['hour'] >= 22) | (out['hour'] <= 3)).astype(int)
    return out

train_df = add_time_regressors(prophet_city_train[['ds', 'y']])
val_df = add_time_regressors(prophet_city_val[['ds', 'y']])

regressors = ['is_weekend', 'is_rush_am', 'is_rush_pm', 'is_late_night']

m = Prophet(
    growth='linear',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative'
)
m.add_country_holidays(country_name='US')
for r in regressors:
    m.add_regressor(r)

m.fit(train_df[['ds', 'y'] + regressors])

# Forecast only through validation horizon (no test exposure yet)
val_future = val_df[['ds'] + regressors].copy()
val_forecast = m.predict(val_future)

val_pred = (
    val_df[['ds', 'y']]
    .merge(val_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
)
val_pred[['yhat', 'yhat_lower', 'yhat_upper']] = val_pred[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

mae = mean_absolute_error(val_pred['y'], val_pred['yhat'])
rmse = mean_squared_error(val_pred['y'], val_pred['yhat']) ** 0.5
mape = (np.abs((val_pred['y'] - val_pred['yhat']) / val_pred['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100

metrics = pd.DataFrame([{
    'split': 'validation',
    'model': 'prophet_city_h1h2_time_regressors',
    'mae': float(mae),
    'rmse': float(rmse),
    'mape_pct': float(mape),
    'n_rows': int(len(val_pred))
}])

val_pred.to_parquet(output_dir / 'prophet_city_val_predictions.parquet', index=False)
metrics.to_csv(output_dir / 'prophet_city_val_metrics.csv', index=False)

print(metrics.to_string(index=False))
val_pred.head()


# Step 2Z: Zone-level data view (before zone modeling)
import matplotlib.pyplot as plt

if zone_train.empty:
    raise ValueError('zone_train is empty. Run Step 1A and Step 1B first, and confirm the date filters are correct.')

zone_summary = pd.DataFrame([
    {'split': 'train', 'rows': len(zone_train), 'zones': int(zone_train['zone_id'].nunique()), 'start': zone_train['ds'].min(), 'end': zone_train['ds'].max()},
    {'split': 'val', 'rows': len(zone_val), 'zones': int(zone_val['zone_id'].nunique()), 'start': zone_val['ds'].min(), 'end': zone_val['ds'].max()},
    {'split': 'test', 'rows': len(zone_test), 'zones': int(zone_test['zone_id'].nunique()), 'start': zone_test['ds'].min(), 'end': zone_test['ds'].max()},
])
print('Zone-level split summary')
display(zone_summary)

top_zone_counts = (
    zone_train.groupby('zone_id', as_index=False)['y']
    .sum()
    .sort_values('y', ascending=False)
    .rename(columns={'y': 'train_total_demand'})
)
print('Top zones by training-period demand')
display(top_zone_counts.head(15))

selected_zone_id = int(top_zone_counts.iloc[0]['zone_id'])
zone_view = pd.concat([
    zone_train[zone_train['zone_id'] == selected_zone_id][['ds', 'y']].assign(split='train'),
    zone_val[zone_val['zone_id'] == selected_zone_id][['ds', 'y']].assign(split='val'),
    zone_test[zone_test['zone_id'] == selected_zone_id][['ds', 'y']].assign(split='test'),
], ignore_index=True).sort_values('ds')

plt.figure(figsize=(16, 5))
for split, color in [('train', '#334155'), ('val', '#0ea5e9'), ('test', '#f97316')]:
    part = zone_view[zone_view['split'] == split]
    plt.plot(part['ds'], part['y'], label=f'{split} actual', linewidth=1.0, alpha=0.9, color=color)

plt.title(f'Zone-level hourly demand for zone_id={selected_zone_id}')
plt.xlabel('Datetime')
plt.ylabel('Trips per hour')
plt.legend()
plt.tight_layout()
plt.show()

zone_view.tail()


# Step 2A: Validation baseline (seasonal naive lag 24)
# Baseline: Seasonal naive (24-hour lag) on validation split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Build baseline predictions: predict each hour using demand from 24 hours earlier
full_city = city_hourly[['ds', 'y']].sort_values('ds').copy()
full_city['baseline_yhat'] = full_city['y'].shift(24)

val_baseline = (
    prophet_city_val[['ds', 'y']]
    .merge(full_city[['ds', 'baseline_yhat']], on='ds', how='left')
    .dropna(subset=['baseline_yhat'])
)

baseline_val_mae = mean_absolute_error(val_baseline['y'], val_baseline['baseline_yhat'])
baseline_val_rmse = mean_squared_error(val_baseline['y'], val_baseline['baseline_yhat']) ** 0.5

# Prophet validation metrics from Step 2
prophet_val_mae = mean_absolute_error(val_pred['y'], val_pred['yhat'])
prophet_val_rmse = mean_squared_error(val_pred['y'], val_pred['yhat']) ** 0.5

val_compare = pd.DataFrame([
    {'model': 'seasonal_naive_lag24', 'split': 'validation', 'mae': float(baseline_val_mae), 'rmse': float(baseline_val_rmse), 'n_rows': int(len(val_baseline))},
    {'model': 'prophet_city_h1h2_time_regressors', 'split': 'validation', 'mae': float(prophet_val_mae), 'rmse': float(prophet_val_rmse), 'n_rows': int(len(val_pred))},
])

val_compare.to_csv(output_dir / 'city_val_model_comparison.csv', index=False)
print(val_compare.to_string(index=False))
val_compare


# Step 2.5: Pre-test setup and model freeze (required before Step 3)
# Purpose: avoid accidental test peeking and lock selection from validation only

if 'val_compare' not in globals():
    raise ValueError('Run Step 2 and baseline validation cell first so val_compare is available.')

validation_ranking = val_compare.sort_values('rmse').reset_index(drop=True)
champion_model = validation_ranking.loc[0, 'model']

frozen_setup = {
    'train_window': '2024-01-01 to 2024-12-31',
    'val_window': '2025-01-01 to 2025-06-30',
    'test_window': '2025-07-01 onward',
    'selection_metric': 'rmse',
    'champion_from_validation': champion_model,
}

RUN_TEST = True
print('Validation ranking')
print(validation_ranking.to_string(index=False))
print()
print('Frozen setup')
print(frozen_setup)
print()
print('Set RUN_TEST = True in this cell after you review frozen setup, then run Step 3.')


# Step 3: Final fit and test evaluation (leakage-safe)
# Model selection is based on validation only. Then evaluate once on held-out test.
import numpy as np
# Hard gate to prevent unintended test evaluation
if 'RUN_TEST' not in globals():
    raise ValueError('Run Step 2.5 first.')
if not RUN_TEST:
    raise ValueError('RUN_TEST is False. Review Step 2.5 and set RUN_TEST = True before running Step 3.')


best_model = val_compare.sort_values('rmse').iloc[0]['model']
print('Selected by validation RMSE:', best_model)

test_df = prophet_city_test[['ds', 'y']].copy()

# Baseline test prediction
full_city_for_test = city_hourly[['ds', 'y']].sort_values('ds').copy()
full_city_for_test['baseline_yhat'] = full_city_for_test['y'].shift(24)
test_baseline = (
    test_df.merge(full_city_for_test[['ds', 'baseline_yhat']], on='ds', how='left')
    .dropna(subset=['baseline_yhat'])
)
baseline_test_mae = mean_absolute_error(test_baseline['y'], test_baseline['baseline_yhat'])
baseline_test_rmse = mean_squared_error(test_baseline['y'], test_baseline['baseline_yhat']) ** 0.5
baseline_test_mape = (np.abs((test_baseline['y'] - test_baseline['baseline_yhat']) / test_baseline['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100

# Prophet test prediction: refit on train + val only, then predict test
train_val_df = pd.concat([prophet_city_train[['ds', 'y']], prophet_city_val[['ds', 'y']]], ignore_index=True).sort_values('ds')
train_val_df = add_time_regressors(train_val_df)
test_reg = add_time_regressors(test_df)

m_final = Prophet(
    growth='linear',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative'
)
m_final.add_country_holidays(country_name='US')
for r in regressors:
    m_final.add_regressor(r)

m_final.fit(train_val_df[['ds', 'y'] + regressors])
test_forecast = m_final.predict(test_reg[['ds'] + regressors])
test_prophet = test_df.merge(test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
test_prophet[['yhat', 'yhat_lower', 'yhat_upper']] = test_prophet[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
prophet_test_mae = mean_absolute_error(test_prophet['y'], test_prophet['yhat'])
prophet_test_rmse = mean_squared_error(test_prophet['y'], test_prophet['yhat']) ** 0.5
prophet_test_mape = (np.abs((test_prophet['y'] - test_prophet['yhat']) / test_prophet['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100

test_compare = pd.DataFrame([
    {'model': 'seasonal_naive_lag24', 'split': 'test', 'mae': float(baseline_test_mae), 'rmse': float(baseline_test_rmse), 'mape_pct': float(baseline_test_mape), 'n_rows': int(len(test_baseline))},
    {'model': 'prophet_city_h1h2_time_regressors', 'split': 'test', 'mae': float(prophet_test_mae), 'rmse': float(prophet_test_rmse), 'mape_pct': float(prophet_test_mape), 'n_rows': int(len(test_prophet))},
])

test_compare.to_csv(output_dir / 'city_test_model_comparison.csv', index=False)
test_prophet.to_parquet(output_dir / 'prophet_city_test_predictions.parquet', index=False)
test_baseline.to_parquet(output_dir / 'baseline_city_test_predictions.parquet', index=False)

print(test_compare.to_string(index=False))
test_compare


# Step 4: Segment setup for borough and zone models
zone_lookup_path = repo_root / 'data' / 'external' / 'taxi_zone_lookup.csv'
zone_lookup = pd.read_csv(zone_lookup_path).rename(columns={'LocationID': 'zone_id'})

zone_lookup['is_airport'] = zone_lookup['zone_id'].isin([1, 132, 138]).astype(int)
zone_lookup['is_business'] = ((zone_lookup['Borough'] == 'Manhattan') & (zone_lookup['is_airport'] == 0)).astype(int)
zone_lookup['is_residential'] = (zone_lookup['service_zone'] == 'Boro Zone').astype(int)
zone_lookup['zone_type'] = np.where(zone_lookup['is_airport'] == 1, 'airport', 'non_airport')

zone_train_enriched = zone_train.merge(zone_lookup[['zone_id', 'Borough', 'zone_type', 'is_airport', 'is_business', 'is_residential']], on='zone_id', how='left')
zone_val_enriched = zone_val.merge(zone_lookup[['zone_id', 'Borough', 'zone_type', 'is_airport', 'is_business', 'is_residential']], on='zone_id', how='left')
zone_test_enriched = zone_test.merge(zone_lookup[['zone_id', 'Borough', 'zone_type', 'is_airport', 'is_business', 'is_residential']], on='zone_id', how='left')

def add_time_regs(df):
    out = df.copy()
    out['hour'] = out['ds'].dt.hour
    out['dow'] = out['ds'].dt.dayofweek
    out['is_weekend'] = (out['dow'] >= 5).astype(int)
    out['is_rush_am'] = out['hour'].between(7, 9).astype(int)
    out['is_rush_pm'] = out['hour'].between(17, 19).astype(int)
    out['is_late_night'] = ((out['hour'] >= 22) | (out['hour'] <= 3)).astype(int)
    return out

def fit_predict_prophet(train_seg, pred_seg, static_regressors=None):
    static_regressors = static_regressors or []
    reg_cols = ['is_weekend', 'is_rush_am', 'is_rush_pm', 'is_late_night'] + static_regressors

    tr = add_time_regs(train_seg[['ds', 'y'] + static_regressors]).copy()
    pr = add_time_regs(pred_seg[['ds', 'y'] + static_regressors]).copy()

    m = Prophet(
        growth='linear',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    m.add_country_holidays(country_name='US')
    for r in reg_cols:
        m.add_regressor(r)

    m.fit(tr[['ds', 'y'] + reg_cols])
    fc = m.predict(pr[['ds'] + reg_cols])
    out = pr[['ds', 'y']].merge(fc[['ds', 'yhat']], on='ds', how='left')
    out['yhat'] = out['yhat'].clip(lower=0)
    return out, m

def metric_frame(df, model_name, split_name, group_cols=None):
    group_cols = group_cols or []
    work = df.copy()
    if group_cols:
        rows = []
        for k, part in work.groupby(group_cols):
            key = k if isinstance(k, tuple) else (k,)
            row = {c: v for c, v in zip(group_cols, key)}
            row.update({
                'model': model_name,
                'split': split_name,
                'mae': float(mean_absolute_error(part['y'], part['yhat'])),
                'rmse': float(mean_squared_error(part['y'], part['yhat']) ** 0.5),
                'mape_pct': float((np.abs((part['y'] - part['yhat']) / part['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100),
                'n_rows': int(len(part)),
            })
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame([{
        'model': model_name,
        'split': split_name,
        'mae': float(mean_absolute_error(work['y'], work['yhat'])),
        'rmse': float(mean_squared_error(work['y'], work['yhat']) ** 0.5),
        'mape_pct': float((np.abs((work['y'] - work['yhat']) / work['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100),
        'n_rows': int(len(work)),
    }])

print('Prepared enriched zone splits with borough and zone flags')
zone_lookup.head()


# Step 4A: Borough-level Prophet models
borough_train = zone_train_enriched.groupby(['Borough', 'ds'], as_index=False)['y'].sum()
borough_val = zone_val_enriched.groupby(['Borough', 'ds'], as_index=False)['y'].sum()
borough_test = zone_test_enriched.groupby(['Borough', 'ds'], as_index=False)['y'].sum()

borough_models = {}
borough_val_preds = []
borough_test_preds = []

for b in sorted(borough_train['Borough'].dropna().unique()):
    tr = borough_train[borough_train['Borough'] == b][['ds', 'y']].copy()
    va = borough_val[borough_val['Borough'] == b][['ds', 'y']].copy()
    te = borough_test[borough_test['Borough'] == b][['ds', 'y']].copy()

    if tr.empty or va.empty:
        continue

    pred_val, model_b = fit_predict_prophet(tr, va, static_regressors=[])
    pred_val['Borough'] = b
    borough_models[b] = model_b
    borough_val_preds.append(pred_val)

    if not te.empty:
        trv = pd.concat([tr, va], ignore_index=True).sort_values('ds')
        pred_test, _ = fit_predict_prophet(trv, te, static_regressors=[])
        pred_test['Borough'] = b
        borough_test_preds.append(pred_test)

borough_val_pred = pd.concat(borough_val_preds, ignore_index=True) if borough_val_preds else pd.DataFrame(columns=['ds', 'y', 'yhat', 'Borough'])
borough_test_pred = pd.concat(borough_test_preds, ignore_index=True) if borough_test_preds else pd.DataFrame(columns=['ds', 'y', 'yhat', 'Borough'])

borough_val_metrics = metric_frame(borough_val_pred, 'prophet_borough', 'validation')
borough_test_metrics = metric_frame(borough_test_pred, 'prophet_borough', 'test')

borough_val_pred.to_parquet(output_dir / 'prophet_borough_val_predictions.parquet', index=False)
borough_test_pred.to_parquet(output_dir / 'prophet_borough_test_predictions.parquet', index=False)
pd.concat([borough_val_metrics, borough_test_metrics], ignore_index=True).to_csv(output_dir / 'prophet_borough_metrics.csv', index=False)

print('Borough models trained:', len(borough_models))
pd.concat([borough_val_metrics, borough_test_metrics], ignore_index=True)


# Step 4A2: Borough baseline (seasonal naive lag 24)
from sklearn.metrics import mean_absolute_error, mean_squared_error

borough_full = (
    zone_hourly.merge(zone_lookup[['zone_id', 'Borough']], on='zone_id', how='left')
    .groupby(['Borough', 'ds'], as_index=False)['y']
    .sum()
    .sort_values(['Borough', 'ds'])
)
borough_full['yhat_baseline'] = borough_full.groupby('Borough')['y'].shift(24)

borough_val_base = (
    borough_val[['Borough', 'ds', 'y']]
    .merge(borough_full[['Borough', 'ds', 'yhat_baseline']], on=['Borough', 'ds'], how='left')
    .dropna(subset=['yhat_baseline'])
)
borough_test_base = (
    borough_test[['Borough', 'ds', 'y']]
    .merge(borough_full[['Borough', 'ds', 'yhat_baseline']], on=['Borough', 'ds'], how='left')
    .dropna(subset=['yhat_baseline'])
)

def baseline_metrics(df, split_name):
    return pd.DataFrame([{
        'model': 'seasonal_naive_borough_lag24',
        'split': split_name,
        'mae': float(mean_absolute_error(df['y'], df['yhat_baseline'])),
        'rmse': float(mean_squared_error(df['y'], df['yhat_baseline']) ** 0.5),
        'mape_pct': float((np.abs((df['y'] - df['yhat_baseline']) / df['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100),
        'n_rows': int(len(df)),
    }])

borough_baseline_metrics = pd.concat([
    baseline_metrics(borough_val_base, 'validation'),
    baseline_metrics(borough_test_base, 'test'),
], ignore_index=True)

borough_val_base.to_parquet(output_dir / 'baseline_borough_val_predictions.parquet', index=False)
borough_test_base.to_parquet(output_dir / 'baseline_borough_test_predictions.parquet', index=False)
borough_baseline_metrics.to_csv(output_dir / 'baseline_borough_metrics.csv', index=False)

display(borough_baseline_metrics)


# Step 4B: Top-N zone Prophet models by train volume
top_n = 30
min_train_rows = 24 * 90

zone_rank = zone_train_enriched.groupby('zone_id', as_index=False)['y'].sum().sort_values('y', ascending=False)
candidate_zones = zone_rank['zone_id'].tolist()

top_zone_models = {}
top_zone_ids = []
top_zone_val_preds = []
top_zone_test_preds = []

for z in candidate_zones:
    if len(top_zone_ids) >= top_n:
        break

    tr = zone_train_enriched[zone_train_enriched['zone_id'] == z][['ds', 'y', 'is_airport', 'is_business', 'is_residential']].copy()
    va = zone_val_enriched[zone_val_enriched['zone_id'] == z][['ds', 'y', 'is_airport', 'is_business', 'is_residential']].copy()
    te = zone_test_enriched[zone_test_enriched['zone_id'] == z][['ds', 'y', 'is_airport', 'is_business', 'is_residential']].copy()

    if len(tr) < min_train_rows or va.empty:
        continue

    pred_val, model_z = fit_predict_prophet(tr, va, static_regressors=['is_airport', 'is_business', 'is_residential'])
    pred_val['zone_id'] = z
    top_zone_val_preds.append(pred_val)
    top_zone_models[z] = model_z
    top_zone_ids.append(z)

    if not te.empty:
        trv = pd.concat([tr, va], ignore_index=True).sort_values('ds')
        pred_test, _ = fit_predict_prophet(trv, te, static_regressors=['is_airport', 'is_business', 'is_residential'])
        pred_test['zone_id'] = z
        top_zone_test_preds.append(pred_test)

top_zone_val_pred = pd.concat(top_zone_val_preds, ignore_index=True) if top_zone_val_preds else pd.DataFrame(columns=['ds', 'y', 'yhat', 'zone_id'])
top_zone_test_pred = pd.concat(top_zone_test_preds, ignore_index=True) if top_zone_test_preds else pd.DataFrame(columns=['ds', 'y', 'yhat', 'zone_id'])

top_zone_val_pred.to_parquet(output_dir / 'prophet_top_zone_val_predictions.parquet', index=False)
top_zone_test_pred.to_parquet(output_dir / 'prophet_top_zone_test_predictions.parquet', index=False)

print('Top-zone models trained:', len(top_zone_ids))
print('Sample zone IDs:', top_zone_ids[:10])
top_zone_val_pred.head()


# Step 4B2: Zone baselines (seasonal naive lag 24)
from sklearn.metrics import mean_absolute_error, mean_squared_error

zone_full = zone_hourly[['zone_id', 'ds', 'y']].sort_values(['zone_id', 'ds']).copy()
zone_full['yhat_baseline'] = zone_full.groupby('zone_id')['y'].shift(24)

top_zone_set = set(top_zone_ids) if 'top_zone_ids' in globals() else set()
zone_val_top_base = (
    zone_val[zone_val['zone_id'].isin(top_zone_set)][['zone_id', 'ds', 'y']]
    .merge(zone_full[['zone_id', 'ds', 'yhat_baseline']], on=['zone_id', 'ds'], how='left')
    .dropna(subset=['yhat_baseline'])
)
zone_test_top_base = (
    zone_test[zone_test['zone_id'].isin(top_zone_set)][['zone_id', 'ds', 'y']]
    .merge(zone_full[['zone_id', 'ds', 'yhat_baseline']], on=['zone_id', 'ds'], how='left')
    .dropna(subset=['yhat_baseline'])
)
zone_val_all_base = (
    zone_val[['zone_id', 'ds', 'y']]
    .merge(zone_full[['zone_id', 'ds', 'yhat_baseline']], on=['zone_id', 'ds'], how='left')
    .dropna(subset=['yhat_baseline'])
)
zone_test_all_base = (
    zone_test[['zone_id', 'ds', 'y']]
    .merge(zone_full[['zone_id', 'ds', 'yhat_baseline']], on=['zone_id', 'ds'], how='left')
    .dropna(subset=['yhat_baseline'])
)

def zone_baseline_metrics(df, split_name, model_name):
    return pd.DataFrame([{
        'model': model_name,
        'split': split_name,
        'mae': float(mean_absolute_error(df['y'], df['yhat_baseline'])),
        'rmse': float(mean_squared_error(df['y'], df['yhat_baseline']) ** 0.5),
        'mape_pct': float((np.abs((df['y'] - df['yhat_baseline']) / df['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100),
        'n_rows': int(len(df)),
    }])

zone_top_base_metrics = pd.concat([
    zone_baseline_metrics(zone_val_top_base, 'validation', 'seasonal_naive_top_zone_lag24'),
    zone_baseline_metrics(zone_test_top_base, 'test', 'seasonal_naive_top_zone_lag24'),
], ignore_index=True)
zone_all_base_metrics = pd.concat([
    zone_baseline_metrics(zone_val_all_base, 'validation', 'seasonal_naive_all_zone_lag24'),
    zone_baseline_metrics(zone_test_all_base, 'test', 'seasonal_naive_all_zone_lag24'),
], ignore_index=True)

zone_top_base_metrics.to_csv(output_dir / 'baseline_top_zone_metrics.csv', index=False)
zone_all_base_metrics.to_csv(output_dir / 'baseline_all_zone_metrics.csv', index=False)
zone_val_top_base.to_parquet(output_dir / 'baseline_top_zone_val_predictions.parquet', index=False)
zone_test_top_base.to_parquet(output_dir / 'baseline_top_zone_test_predictions.parquet', index=False)
zone_val_all_base.to_parquet(output_dir / 'baseline_all_zone_val_predictions.parquet', index=False)
zone_test_all_base.to_parquet(output_dir / 'baseline_all_zone_test_predictions.parquet', index=False)

display(pd.concat([zone_top_base_metrics, zone_all_base_metrics], ignore_index=True))


# Step 4C: Fallback hierarchy and zone-type metrics
# Hierarchy: top-zone model -> borough model -> city model

if 'test_prophet' not in globals():
    raise ValueError('Run Step 3 first so city-level test predictions are available in test_prophet')

city_test_pred = test_prophet[['ds', 'yhat']].copy().rename(columns={'yhat': 'yhat_city'})

# Build validation fallback predictions
val_base = zone_val_enriched[['zone_id', 'Borough', 'zone_type', 'ds', 'y']].copy()
val_base = val_base.merge(top_zone_val_pred[['zone_id', 'ds', 'yhat']].rename(columns={'yhat': 'yhat_zone'}), on=['zone_id', 'ds'], how='left')
val_base = val_base.merge(borough_val_pred[['Borough', 'ds', 'yhat']].rename(columns={'yhat': 'yhat_borough'}), on=['Borough', 'ds'], how='left')
val_base = val_base.merge(val_pred[['ds', 'yhat']].rename(columns={'yhat': 'yhat_city'}), on='ds', how='left')
val_base['yhat'] = val_base['yhat_zone'].combine_first(val_base['yhat_borough']).combine_first(val_base['yhat_city'])
val_base['yhat'] = val_base['yhat'].clip(lower=0)
val_base['source_model'] = np.where(val_base['yhat_zone'].notna(), 'zone', np.where(val_base['yhat_borough'].notna(), 'borough', 'city'))

# Build test fallback predictions
test_base = zone_test_enriched[['zone_id', 'Borough', 'zone_type', 'ds', 'y']].copy()
test_base = test_base.merge(top_zone_test_pred[['zone_id', 'ds', 'yhat']].rename(columns={'yhat': 'yhat_zone'}), on=['zone_id', 'ds'], how='left')
test_base = test_base.merge(borough_test_pred[['Borough', 'ds', 'yhat']].rename(columns={'yhat': 'yhat_borough'}), on=['Borough', 'ds'], how='left')
test_base = test_base.merge(city_test_pred, on='ds', how='left')
test_base['yhat'] = test_base['yhat_zone'].combine_first(test_base['yhat_borough']).combine_first(test_base['yhat_city'])
test_base['yhat'] = test_base['yhat'].clip(lower=0)
test_base['source_model'] = np.where(test_base['yhat_zone'].notna(), 'zone', np.where(test_base['yhat_borough'].notna(), 'borough', 'city'))

val_overall = metric_frame(val_base[['y', 'yhat']], 'hierarchical_prophet', 'validation')
test_overall = metric_frame(test_base[['y', 'yhat']], 'hierarchical_prophet', 'test')
val_by_zone_type = metric_frame(val_base[['zone_type', 'y', 'yhat']], 'hierarchical_prophet', 'validation', group_cols=['zone_type'])
test_by_zone_type = metric_frame(test_base[['zone_type', 'y', 'yhat']], 'hierarchical_prophet', 'test', group_cols=['zone_type'])

model_source_mix_val = val_base['source_model'].value_counts(normalize=True).rename('pct').reset_index().rename(columns={'index': 'source_model'})
model_source_mix_test = test_base['source_model'].value_counts(normalize=True).rename('pct').reset_index().rename(columns={'index': 'source_model'})

val_base.to_parquet(output_dir / 'hierarchical_zone_val_predictions.parquet', index=False)
test_base.to_parquet(output_dir / 'hierarchical_zone_test_predictions.parquet', index=False)
pd.concat([val_overall, test_overall, val_by_zone_type, test_by_zone_type], ignore_index=True).to_csv(output_dir / 'hierarchical_prophet_metrics.csv', index=False)
model_source_mix_val.to_csv(output_dir / 'hierarchical_source_mix_val.csv', index=False)
model_source_mix_test.to_csv(output_dir / 'hierarchical_source_mix_test.csv', index=False)

print('Validation and test hierarchical metrics saved')
pd.concat([val_overall, test_overall, val_by_zone_type, test_by_zone_type], ignore_index=True)


# Step 4D: Rolling time-based backtests (no random split)
# Runs on city and borough segments for reproducible temporal validation

def rolling_backtest(segment_df, segment_id='city', horizon_hours=24*14, n_folds=4):
    segment_df = segment_df[['ds', 'y']].sort_values('ds').drop_duplicates('ds').reset_index(drop=True)
    cutoffs = pd.date_range(start=segment_df['ds'].min() + pd.Timedelta(days=180), end=segment_df['ds'].max() - pd.Timedelta(hours=horizon_hours), periods=n_folds)

    rows = []
    for cutoff in cutoffs:
        train_fold = segment_df[segment_df['ds'] < cutoff].copy()
        test_fold = segment_df[(segment_df['ds'] >= cutoff) & (segment_df['ds'] < cutoff + pd.Timedelta(hours=horizon_hours))].copy()
        if len(train_fold) < 24 * 90 or test_fold.empty:
            continue

        pred_fold, _ = fit_predict_prophet(train_fold, test_fold, static_regressors=[])
        rows.append({
            'segment_id': segment_id,
            'cutoff': cutoff,
            'mae': float(mean_absolute_error(pred_fold['y'], pred_fold['yhat'])),
            'rmse': float(mean_squared_error(pred_fold['y'], pred_fold['yhat']) ** 0.5),
            'mape_pct': float((np.abs((pred_fold['y'] - pred_fold['yhat']) / pred_fold['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100),
            'n_rows': int(len(pred_fold)),
        })

    return pd.DataFrame(rows)

city_series_for_bt = pd.concat([prophet_city_train[['ds', 'y']], prophet_city_val[['ds', 'y']]], ignore_index=True)
city_bt = rolling_backtest(city_series_for_bt, segment_id='city', horizon_hours=24*14, n_folds=4)

borough_bt_frames = []
for b in sorted(borough_train['Borough'].dropna().unique()):
    seg = borough_train[borough_train['Borough'] == b][['ds', 'y']].copy()
    if len(seg) < 24 * 120:
        continue
    bt = rolling_backtest(seg, segment_id=f'borough:{b}', horizon_hours=24*14, n_folds=3)
    borough_bt_frames.append(bt)

borough_bt = pd.concat(borough_bt_frames, ignore_index=True) if borough_bt_frames else pd.DataFrame(columns=['segment_id', 'cutoff', 'mae', 'rmse', 'mape_pct', 'n_rows'])
rolling_bt = pd.concat([city_bt, borough_bt], ignore_index=True)
rolling_bt.to_csv(output_dir / 'rolling_backtest_metrics.csv', index=False)
rolling_bt.head()


# Step 4E: Deployment hooks (retrain cadence and drift monitor scaffolding)
deployment_plan = {
    'model_family': 'Prophet hierarchical (zone -> borough -> city)',
    'retrain_cadence': 'weekly for top-zones, monthly for borough/city',
    'serving_priority': ['zone', 'borough', 'city'],
    'monitoring': {
        'metrics': ['mae', 'rmse', 'mape_pct'],
        'slices': ['zone_id', 'Borough', 'zone_type', 'hour'],
        'alert_rule': 'alert if weekly MAE increases > 20 percent vs trailing 4-week mean'
    }
}

# Drift report from hierarchical test predictions (template for production scoring logs)
if (output_dir / 'hierarchical_zone_test_predictions.parquet').exists():
    drift_base = pd.read_parquet(output_dir / 'hierarchical_zone_test_predictions.parquet')
    drift_base['abs_err'] = (drift_base['y'] - drift_base['yhat']).abs()
    drift_base['hour'] = drift_base['ds'].dt.hour

    drift_weekly = (
        drift_base.assign(week=drift_base['ds'].dt.to_period('W').astype(str))
        .groupby(['week', 'zone_type', 'hour'], as_index=False)['abs_err']
        .mean()
        .rename(columns={'abs_err': 'mae'})
    )
    drift_weekly.to_csv(output_dir / 'drift_weekly_zone_hour_mae.csv', index=False)

print(deployment_plan)
print('Saved deployment/drift artifacts under', output_dir)


# Step 5: Visualization setup
import matplotlib.pyplot as plt
import seaborn as sns

# Yellow + black brand style
sns.set_theme(context='talk', style='darkgrid')
plt.rcParams.update({
    'figure.facecolor': '#0b0b0b',
    'axes.facecolor': '#111111',
    'axes.edgecolor': '#facc15',
    'axes.labelcolor': '#fde68a',
    'axes.titlecolor': '#fde047',
    'xtick.color': '#fef3c7',
    'ytick.color': '#fef3c7',
    'grid.color': '#3a3a3a',
    'text.color': '#fde68a',
    'legend.facecolor': '#111111',
    'legend.edgecolor': '#facc15',
})

BRAND = {
    'yellow': '#facc15',
    'gold': '#fde047',
    'amber': '#f59e0b',
    'ink': '#0b0b0b',
    'slate': '#1f1f1f',
    'light': '#fef3c7',
    'blue': '#38bdf8',
    'orange': '#fb923c',
}

OMBR_YELLOW = ['#fef9c3', '#fde047', '#facc15', '#eab308', '#ca8a04']
OMBR_MIX = ['#fde047', '#f59e0b', '#f97316', '#fb7185', '#38bdf8']

plot_dir = output_dir / 'plots'
plot_dir.mkdir(parents=True, exist_ok=True)
print('Plot output directory:', plot_dir)

def stylize_ax(ax, title):
    ax.set_title(title, pad=12, fontweight='bold')
    ax.set_facecolor(BRAND['slate'])
    for spine in ax.spines.values():
        spine.set_color(BRAND['yellow'])
        spine.set_alpha(0.6)

def ombre_fill(ax, x, y, color):
    # Soft layered fill to mimic ombre glow
    for alpha, shrink in [(0.18, 1.00), (0.12, 0.95), (0.08, 0.90), (0.05, 0.85)]:
        ax.fill_between(x, y * shrink, y, color=color, alpha=alpha)


# Step 5A: City test model comparison plots
city_test_cmp = pd.read_csv(output_dir / 'city_test_model_comparison.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=city_test_cmp, x='model', y='mae', ax=axes[0], palette='Blues_r')
axes[0].set_title('Test MAE by Model')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=20)

sns.barplot(data=city_test_cmp, x='model', y='rmse', ax=axes[1], palette='Greens_r')
axes[1].set_title('Test RMSE by Model')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=20)

sns.barplot(data=city_test_cmp, x='model', y='mape_pct', ax=axes[2], palette='Oranges_r')
axes[2].set_title('Test MAPE by Model')
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(plot_dir / 'city_test_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
city_test_cmp


# Step 5B: Hierarchical metrics by split and zone type
hier_metrics = pd.read_csv(output_dir / 'hierarchical_prophet_metrics.csv')
zone_slice = hier_metrics[hier_metrics['zone_type'].notna()].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=zone_slice, x='zone_type', y='mae', hue='split', ax=axes[0], palette='Set2')
axes[0].set_title('Hierarchical MAE by Zone Type')

sns.barplot(data=zone_slice, x='zone_type', y='rmse', hue='split', ax=axes[1], palette='Set2')
axes[1].set_title('Hierarchical RMSE by Zone Type')

plt.tight_layout()
plt.savefig(plot_dir / 'hierarchical_zone_type_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
zone_slice


# Step 5C: Fallback source mix visualization
source_mix_val = pd.read_csv(output_dir / 'hierarchical_source_mix_val.csv')
source_mix_test = pd.read_csv(output_dir / 'hierarchical_source_mix_test.csv')
source_mix_val['split'] = 'validation'
source_mix_test['split'] = 'test'
source_mix = pd.concat([source_mix_val, source_mix_test], ignore_index=True)

plt.figure(figsize=(10, 5))
sns.barplot(data=source_mix, x='source_model', y='pct', hue='split', palette='Set1')
plt.title('Fallback Hierarchy Usage Mix')
plt.ylabel('Share of predictions')
plt.xlabel('')
plt.tight_layout()
plt.savefig(plot_dir / 'fallback_source_mix.png', dpi=150, bbox_inches='tight')
plt.show()
source_mix


# Step 5D: Rolling backtest trend visualization
bt = pd.read_csv(output_dir / 'rolling_backtest_metrics.csv')
bt['cutoff'] = pd.to_datetime(bt['cutoff'])

plt.figure(figsize=(12, 5))
for sid, part in bt.groupby('segment_id'):
    plt.plot(part['cutoff'], part['rmse'], marker='o', label=sid)
plt.title('Rolling Backtest RMSE by Segment')
plt.xlabel('Cutoff date')
plt.ylabel('RMSE')
plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(plot_dir / 'rolling_backtest_rmse_trend.png', dpi=150, bbox_inches='tight')
plt.show()
bt.head()


# Step 5E: Drift heatmap by hour and week
drift = pd.read_csv(output_dir / 'drift_weekly_zone_hour_mae.csv')
drift_air = drift[drift['zone_type'] == 'airport'].copy()
drift_non = drift[drift['zone_type'] == 'non_airport'].copy()

def plot_heatmap(df, title, file_name):
    if df.empty:
        print('No data for', title)
        return
    pivot = df.pivot_table(index='week', columns='hour', values='mae', aggfunc='mean')
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title(title)
    plt.xlabel('Hour of day')
    plt.ylabel('Week')
    plt.tight_layout()
    plt.savefig(plot_dir / file_name, dpi=150, bbox_inches='tight')
    plt.show()

plot_heatmap(drift_air, 'Airport Drift Weekly MAE Heatmap', 'drift_airport_heatmap.png')
plot_heatmap(drift_non, 'Non-Airport Drift Weekly MAE Heatmap', 'drift_non_airport_heatmap.png')


# Step 5F: Historical actual vs forecast overlay
city_actual = city_hourly[['ds', 'y']].copy().sort_values('ds')

val_fc = pd.read_parquet(output_dir / 'prophet_city_val_predictions.parquet')[['ds', 'yhat']].copy()
test_fc = pd.read_parquet(output_dir / 'prophet_city_test_predictions.parquet')[['ds', 'yhat']].copy()
city_fc = pd.concat([val_fc, test_fc], ignore_index=True).drop_duplicates('ds').sort_values('ds')
plot_df = city_actual.merge(city_fc, on='ds', how='left')

# Full period view
fig, ax = plt.subplots(figsize=(16, 6), facecolor='white')
ax.set_facecolor('white')
ax.plot(plot_df['ds'], plot_df['y'], label='Actual demand', color='#1f2937', linewidth=1.4)
ax.plot(plot_df['ds'], plot_df['yhat'], label='Prophet forecast', color='#f59e0b', linewidth=1.8)
ax.axvline(pd.Timestamp('2025-01-01'), color='#92400e', linestyle='--', linewidth=1.2, label='Validation start')
ax.axvline(pd.Timestamp('2025-07-01'), color='#b45309', linestyle='--', linewidth=1.2, label='Test start')
ax.set_title('Citywide Hourly Demand: Actual vs Prophet Forecast', pad=12, fontweight='bold', color='#111111')
ax.set_xlabel('Datetime', color='#111111')
ax.set_ylabel('Trips per hour', color='#111111')
ax.tick_params(colors='#111111')
ax.grid(color='#d4d4d4', alpha=0.6)
for spine in ax.spines.values():
    spine.set_color('#f59e0b')
    spine.set_alpha(0.8)
leg = ax.legend(loc='upper left', ncol=2, frameon=True)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('#f59e0b')
for txt in leg.get_texts():
    txt.set_color('#111111')
plt.tight_layout()
plt.savefig(plot_dir / 'city_actual_vs_forecast_overlay.png', dpi=170, bbox_inches='tight', facecolor='white')
plt.show()

# Test-only view for results slide
test_zoom = plot_df[plot_df['ds'] >= pd.Timestamp('2025-07-01')].copy()
fig, ax = plt.subplots(figsize=(16, 5), facecolor='white')
ax.set_facecolor('white')
ax.plot(test_zoom['ds'], test_zoom['y'], label='Actual test demand', color='#111827', linewidth=1.4)
ax.plot(test_zoom['ds'], test_zoom['yhat'], label='Prophet test forecast', color='#f59e0b', linewidth=1.9)
ax.fill_between(test_zoom['ds'], 0, test_zoom['yhat'].fillna(0), color='#fde68a', alpha=0.25)
ax.set_title('Test Period: Actual vs Prophet Forecast', pad=12, fontweight='bold', color='#111111')
ax.set_xlabel('Datetime', color='#111111')
ax.set_ylabel('Trips per hour', color='#111111')
ax.tick_params(colors='#111111')
ax.grid(color='#d4d4d4', alpha=0.6)
for spine in ax.spines.values():
    spine.set_color('#f59e0b')
    spine.set_alpha(0.8)
leg = ax.legend(frameon=True)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('#f59e0b')
for txt in leg.get_texts():
    txt.set_color('#111111')
plt.tight_layout()
plt.savefig(plot_dir / 'city_test_actual_vs_forecast_overlay.png', dpi=170, bbox_inches='tight', facecolor='white')
plt.show()

plot_df.tail()


# Step 5G: City test baseline vs Prophet (RMSE and MAE)
city_cmp_path = output_dir / 'city_test_model_comparison.csv'
if not city_cmp_path.exists():
    raise FileNotFoundError(f'Missing {city_cmp_path}. Run Step 3 first.')

city_cmp = pd.read_csv(city_cmp_path).copy()
city_cmp = city_cmp[city_cmp['split'] == 'test'].copy()
if city_cmp.empty:
    raise ValueError('No test rows in city_test_model_comparison.csv')

label_map = {
    'seasonal_naive_lag24': 'Naive City 24h',
    'prophet_city_h1h2_time_regressors': 'Prophet City H1/H2',
}
city_cmp['model_display'] = city_cmp['model'].map(label_map).fillna(city_cmp['model'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

for ax in axes:
    ax.set_facecolor('white')
    ax.grid(color='#d4d4d4', alpha=0.6, axis='y')
    for spine in ax.spines.values():
        spine.set_color('#f59e0b')
        spine.set_alpha(0.8)
    ax.tick_params(colors='#111111', labelsize=11)

sns.barplot(data=city_cmp, x='model_display', y='rmse', palette=['#fde68a', '#f59e0b'], ax=axes[0])
axes[0].set_title('City Test RMSE: Naive vs Prophet', color='#111111', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('RMSE', color='#111111')
axes[0].tick_params(axis='x', rotation=10)

sns.barplot(data=city_cmp, x='model_display', y='mae', palette=['#fcd34d', '#b45309'], ax=axes[1])
axes[1].set_title('City Test MAE: Naive vs Prophet', color='#111111', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('MAE', color='#111111')
axes[1].tick_params(axis='x', rotation=10)

# Value labels inside bars
for ax in axes:
    for patch in ax.patches:
        h = patch.get_height()
        if h == h and h >= 1.0:
            y_mid = patch.get_y() + h / 2.0
            ax.annotate(
                f'{h:.1f}',
                (patch.get_x() + patch.get_width() / 2.0, y_mid),
                ha='center',
                va='center',
                fontsize=11,
                color='#111111',
                fontweight='bold'
            )

plt.tight_layout()
plt.savefig(plot_dir / 'city_test_naive_vs_prophet_rmse_mae.png', dpi=170, bbox_inches='tight', facecolor='white')
plt.show()

city_cmp[['model_display','rmse','mae','mape_pct','n_rows']].rename(columns={'model_display':'model'})


# Step 5H: Zone actual vs Prophet forecast overlay
zone_pred_path = output_dir / 'prophet_top_zone_test_predictions.parquet'
if not zone_pred_path.exists():
    raise FileNotFoundError(f'Missing {zone_pred_path}. Run Step 4B first.')

zone_pred = pd.read_parquet(zone_pred_path)
if zone_pred.empty:
    raise ValueError('Top-zone prediction file is empty.')

zone_counts = zone_pred.groupby('zone_id').size().sort_values(ascending=False)
plot_zone_id = int(zone_counts.index[0])
zone_plot = zone_pred[zone_pred['zone_id'] == plot_zone_id].sort_values('ds').copy()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(zone_plot['ds'], zone_plot['y'], label='Actual zone demand', color=BRAND['light'], linewidth=1.3)
ax.plot(zone_plot['ds'], zone_plot['yhat'], label='Prophet zone forecast', color=BRAND['yellow'], linewidth=1.8)
ombre_fill(ax, zone_plot['ds'], zone_plot['yhat'].fillna(0), BRAND['yellow'])
stylize_ax(ax, f'Zone Actual vs Prophet Forecast (zone_id={plot_zone_id})')
ax.set_xlabel('Datetime')
ax.set_ylabel('Trips per hour')
ax.legend()
plt.tight_layout()
plt.savefig(plot_dir / f'zone_actual_vs_prophet_zone_{plot_zone_id}.png', dpi=170, bbox_inches='tight')
plt.show()

zone_plot.tail()


# Step 5I: Borough actual vs Prophet forecast overlay
borough_pred_path = output_dir / 'prophet_borough_test_predictions.parquet'
if not borough_pred_path.exists():
    raise FileNotFoundError(f'Missing {borough_pred_path}. Run Step 4A first.')

borough_pred = pd.read_parquet(borough_pred_path)
if borough_pred.empty:
    raise ValueError('Borough prediction file is empty.')

borough_counts = borough_pred.groupby('Borough').size().sort_values(ascending=False)
plot_borough = borough_counts.index[0]
borough_plot = borough_pred[borough_pred['Borough'] == plot_borough].sort_values('ds').copy()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(borough_plot['ds'], borough_plot['y'], label='Actual borough demand', color=BRAND['light'], linewidth=1.3)
ax.plot(borough_plot['ds'], borough_plot['yhat'], label='Prophet borough forecast', color=BRAND['amber'], linewidth=1.8)
ombre_fill(ax, borough_plot['ds'], borough_plot['yhat'].fillna(0), BRAND['amber'])
stylize_ax(ax, f'Borough Actual vs Prophet Forecast ({plot_borough})')
ax.set_xlabel('Datetime')
ax.set_ylabel('Trips per hour')
ax.legend()
plt.tight_layout()
plt.savefig(plot_dir / f'borough_actual_vs_prophet_{plot_borough}.png', dpi=170, bbox_inches='tight')
plt.show()

borough_plot.tail()


# Step 5J: Borough baseline vs Prophet comparison
borough_base_path = output_dir / 'baseline_borough_metrics.csv'
borough_prophet_path = output_dir / 'prophet_borough_metrics.csv'

if not borough_base_path.exists() or not borough_prophet_path.exists():
    raise FileNotFoundError('Run Step 4A and Step 4A2 first to generate borough metrics files')

borough_base = pd.read_csv(borough_base_path)
borough_prophet_seg = pd.read_csv(borough_prophet_path)
borough_prophet = borough_prophet_seg.groupby('split', as_index=False)[['mae','rmse','mape_pct','n_rows']].mean()
borough_prophet['model'] = 'prophet_borough_mean_over_segments'

comp = pd.concat([
    borough_base[['model','split','mae','rmse','mape_pct','n_rows']],
    borough_prophet[['model','split','mae','rmse','mape_pct','n_rows']],
], ignore_index=True)

# friendly labels
label_map = {
    'seasonal_naive_borough_lag24': 'Naive Borough 24h',
    'prophet_borough_mean_over_segments': 'Prophet Borough',
}
comp['model_display'] = comp['model'].map(label_map).fillna(comp['model'])

test_comp = comp[comp['split'] == 'test'].copy().sort_values('rmse')

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#f59e0b')
        spine.set_alpha(0.8)
    ax.tick_params(colors='#111111')
    ax.grid(color='#d4d4d4', alpha=0.6, axis='y')

sns.barplot(data=test_comp, x='model_display', y='rmse', palette=['#fcd34d','#f59e0b'], ax=axes[0])
axes[0].set_title('Borough Test RMSE: Baseline vs Prophet', color='#111111', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('RMSE', color='#111111')
axes[0].tick_params(axis='x', rotation=10)

sns.barplot(data=test_comp, x='model_display', y='mae', palette=['#fde68a','#b45309'], ax=axes[1])
axes[1].set_title('Borough Test MAE: Baseline vs Prophet', color='#111111', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('MAE', color='#111111')
axes[1].tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig(plot_dir / 'borough_baseline_vs_prophet_test.png', dpi=170, bbox_inches='tight', facecolor='white')
plt.show()

test_comp[['model_display','rmse','mae','mape_pct','n_rows']]


# Step 5K: Top-zone baseline vs Prophet comparison
top_base_path = output_dir / 'baseline_top_zone_metrics.csv'
top_pred_path = output_dir / 'prophet_top_zone_test_predictions.parquet'

if not top_base_path.exists() or not top_pred_path.exists():
    raise FileNotFoundError('Run Step 4B and Step 4B2 first to generate top-zone metrics files')

top_base = pd.read_csv(top_base_path)
top_pred = pd.read_parquet(top_pred_path)

top_prophet = pd.DataFrame([{
    'model': 'prophet_top_zone',
    'split': 'test',
    'mae': float((top_pred['y'] - top_pred['yhat']).abs().mean()),
    'rmse': float(((top_pred['y'] - top_pred['yhat'])**2).mean() ** 0.5),
    'mape_pct': float((((top_pred['y'] - top_pred['yhat']).abs() / top_pred['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100),
    'n_rows': int(len(top_pred)),
}])

comp = pd.concat([
    top_base[['model','split','mae','rmse','mape_pct','n_rows']],
    top_prophet[['model','split','mae','rmse','mape_pct','n_rows']],
], ignore_index=True)

label_map = {
    'seasonal_naive_top_zone_lag24': 'Naive Top-Zones 24h',
    'prophet_top_zone': 'Prophet Top-Zones',
}
comp['model_display'] = comp['model'].map(label_map).fillna(comp['model'])

test_comp = comp[comp['split'] == 'test'].copy().sort_values('rmse')

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#f59e0b')
        spine.set_alpha(0.8)
    ax.tick_params(colors='#111111')
    ax.grid(color='#d4d4d4', alpha=0.6, axis='y')

sns.barplot(data=test_comp, x='model_display', y='rmse', palette=['#fcd34d','#f59e0b'], ax=axes[0])
axes[0].set_title('Top-Zone Test RMSE: Baseline vs Prophet', color='#111111', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('RMSE', color='#111111')
axes[0].tick_params(axis='x', rotation=10)

sns.barplot(data=test_comp, x='model_display', y='mae', palette=['#fde68a','#b45309'], ax=axes[1])
axes[1].set_title('Top-Zone Test MAE: Baseline vs Prophet', color='#111111', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('MAE', color='#111111')
axes[1].tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig(plot_dir / 'topzone_baseline_vs_prophet_test.png', dpi=170, bbox_inches='tight', facecolor='white')
plt.show()

test_comp[['model_display','rmse','mae','mape_pct','n_rows']]


# Step 6: Unified model leaderboard (city, borough, zone, hierarchical)
from sklearn.metrics import mean_absolute_error, mean_squared_error

city_cmp = pd.read_csv(output_dir / 'city_val_model_comparison.csv')
city_test_cmp = pd.read_csv(output_dir / 'city_test_model_comparison.csv')
city_all = pd.concat([city_cmp, city_test_cmp], ignore_index=True)
city_all['scope'] = 'city'
city_all['coverage_pct'] = 100.0

borough_metrics = pd.read_csv(output_dir / 'prophet_borough_metrics.csv')
borough_leader = borough_metrics.groupby(['split'], as_index=False)[['mae', 'rmse', 'mape_pct', 'n_rows']].mean()
borough_leader['model'] = 'prophet_borough_mean_over_segments'
borough_leader['scope'] = 'borough'
borough_leader['coverage_pct'] = 100.0

borough_baseline = pd.read_csv(output_dir / 'baseline_borough_metrics.csv')
borough_baseline['scope'] = 'borough'
borough_baseline['coverage_pct'] = 100.0

zone_top_baseline = pd.read_csv(output_dir / 'baseline_top_zone_metrics.csv')
zone_all_baseline = pd.read_csv(output_dir / 'baseline_all_zone_metrics.csv')

def agg_metrics_from_pred(path, split_name, model_name):
    df = pd.read_parquet(path)
    return pd.DataFrame([{
        'model': model_name,
        'split': split_name,
        'mae': float(mean_absolute_error(df['y'], df['yhat'])),
        'rmse': float(mean_squared_error(df['y'], df['yhat']) ** 0.5),
        'mape_pct': float((np.abs((df['y'] - df['yhat']) / df['y']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100),
        'n_rows': int(len(df)),
    }])

top_zone_val = agg_metrics_from_pred(output_dir / 'prophet_top_zone_val_predictions.parquet', 'validation', 'prophet_top_zone')
top_zone_test = agg_metrics_from_pred(output_dir / 'prophet_top_zone_test_predictions.parquet', 'test', 'prophet_top_zone')
top_zone_leader = pd.concat([top_zone_val, top_zone_test], ignore_index=True)
top_zone_leader['scope'] = 'top_zones_only'

zone_val_rows = len(zone_val)
zone_test_rows = len(zone_test)
top_zone_leader['coverage_pct'] = top_zone_leader.apply(lambda r: 100.0 * r['n_rows'] / (zone_val_rows if r['split'] == 'validation' else zone_test_rows), axis=1)
zone_top_baseline['scope'] = 'top_zones_only'
zone_top_baseline['coverage_pct'] = zone_top_baseline.apply(lambda r: 100.0 * r['n_rows'] / (zone_val_rows if r['split'] == 'validation' else zone_test_rows), axis=1)
zone_all_baseline['scope'] = 'all_zones'
zone_all_baseline['coverage_pct'] = zone_all_baseline.apply(lambda r: 100.0 * r['n_rows'] / (zone_val_rows if r['split'] == 'validation' else zone_test_rows), axis=1)

hier = pd.read_csv(output_dir / 'hierarchical_prophet_metrics.csv')
hier_overall = hier[hier['zone_type'].isna()][['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows']].copy()
hier_overall['scope'] = 'all_zones_with_fallback'
hier_overall['coverage_pct'] = 100.0

leaderboard = pd.concat([
    city_all[['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows', 'scope', 'coverage_pct']],
    borough_baseline[['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows', 'scope', 'coverage_pct']],
    borough_leader[['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows', 'scope', 'coverage_pct']],
    zone_top_baseline[['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows', 'scope', 'coverage_pct']],
    top_zone_leader[['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows', 'scope', 'coverage_pct']],
    zone_all_baseline[['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows', 'scope', 'coverage_pct']],
    hier_overall[['model', 'split', 'mae', 'rmse', 'mape_pct', 'n_rows', 'scope', 'coverage_pct']],
], ignore_index=True)

leaderboard = leaderboard.sort_values(['split', 'scope', 'rmse']).reset_index(drop=True)
leaderboard.to_csv(output_dir / 'unified_model_leaderboard.csv', index=False)
display(leaderboard)

for sp in ['validation', 'test']:
    print('')
    print('Split:', sp)
    display(leaderboard[leaderboard['split'] == sp][['model', 'scope', 'rmse', 'mae', 'mape_pct', 'coverage_pct', 'n_rows']])


# Step 6A: Leaderboard comparison chart
from pathlib import Path

# Self-contained path and style fallbacks
if 'repo_root' not in globals():
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    repo_root = next((p for p in candidates if (p / 'data' / 'processed').exists()), None)
    if repo_root is None:
        raise FileNotFoundError('Could not resolve repo_root')

if 'output_dir' not in globals():
    output_dir = repo_root / 'reports' / 'results' / 'prophet'
if 'plot_dir' not in globals():
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

leaderboard_path = output_dir / 'unified_model_leaderboard.csv'
if not leaderboard_path.exists():
    raise FileNotFoundError(f'Missing {leaderboard_path}. Run Step 6 first.')

lb = pd.read_csv(leaderboard_path)
required_cols = {'model','scope','split','rmse','mae','mape_pct','coverage_pct','n_rows'}
missing = required_cols - set(lb.columns)
if missing:
    raise ValueError(f'Leaderboard missing columns: {missing}')

lb_test = lb[lb['split'] == 'test'].copy().sort_values('rmse')
if lb_test.empty:
    raise ValueError('No test rows in unified_model_leaderboard.csv')

# Human-friendly shorter labels for presentation
model_name_map = {
    'seasonal_naive_lag24': 'Naive City 24h',
    'prophet_city_h1h2_time_regressors': 'Prophet City H1/H2',
    'seasonal_naive_borough_lag24': 'Naive Borough 24h',
    'prophet_borough_mean_over_segments': 'Prophet Borough',
    'seasonal_naive_top_zone_lag24': 'Naive Top-Zones 24h',
    'prophet_top_zone': 'Prophet Top-Zones',
    'seasonal_naive_all_zone_lag24': 'Naive All-Zones 24h',
    'hierarchical_prophet': 'Prophet Hierarchical',
}

scope_name_map = {
    'city': 'City',
    'borough': 'Borough',
    'top_zones_only': 'Top Zones',
    'all_zones': 'All Zones',
    'all_zones_with_fallback': 'All Zones Fallback',
}

lb_test['model_display'] = lb_test['model'].map(model_name_map).fillna(lb_test['model'])
lb_test['scope_display'] = lb_test['scope'].map(scope_name_map).fillna(lb_test['scope'])

# Remove hierarchical row from chart as requested
lb_test_plot = lb_test[lb_test['model_display'] != 'Prophet Hierarchical'].copy()

lb_test_plot['label'] = lb_test_plot['model_display']
lb_test_plot = lb_test_plot.sort_values('rmse', ascending=True).reset_index(drop=True)

palette_scope = {
    'City': '#fde68a',
    'Borough': '#fcd34d',
    'Top Zones': '#f59e0b',
    'All Zones': '#b45309',
    'All Zones Fallback': '#92400e',
}

fig_h = max(5, 0.55 * len(lb_test_plot) + 2)
fig, ax = plt.subplots(figsize=(12, fig_h), facecolor='white')
ax.set_facecolor('white')

sns.barplot(data=lb_test_plot, y='label', x='rmse', hue='scope_display', palette=palette_scope, ax=ax)

ax.set_title('Test RMSE Comparison Across Forecasting Scopes', pad=12, fontweight='bold', color='#111111')
ax.set_xlabel('RMSE', color='#111111')
ax.set_ylabel('Model', color='#111111')
ax.tick_params(axis='x', colors='#111111', labelsize=11)
ax.tick_params(axis='y', colors='#111111', labelsize=11)
ax.grid(color='#d4d4d4', alpha=0.6, axis='x')
for spine in ax.spines.values():
    spine.set_color('#f59e0b')
    spine.set_alpha(0.8)

# Bring back legend box with dark text
leg = ax.legend(title='Scope', loc='upper right', frameon=True, fontsize=10, title_fontsize=10)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('#f59e0b')
for txt in leg.get_texts():
    txt.set_color('#111111')
leg.get_title().set_color('#111111')

# Highlight Naive models with hatch + thicker edge
naive_labels = set(lb_test_plot[lb_test_plot['model_display'].str.contains('Naive')]['label'])
for patch, label in zip(ax.patches, lb_test_plot['label'].tolist()):
    if label in naive_labels:
        patch.set_hatch('////')
        patch.set_edgecolor('#111111')
        patch.set_linewidth(1.8)

# Bigger value labels
for patch in ax.patches:
    w = patch.get_width()
    if w == w and w >= 1.0:
        y = patch.get_y() + patch.get_height() / 2.0
        ax.annotate(f'{w:.1f}', (w, y), ha='left', va='center', textcoords='offset points', xytext=(5, 0), fontsize=12, color='#111111', fontweight='bold')

plt.tight_layout()
plt.savefig(plot_dir / 'test_rmse_comparison_by_model_scope.png', dpi=170, bbox_inches='tight', facecolor='white')
plt.show()

lb_test_plot[['model_display','scope_display','rmse','mae','mape_pct','coverage_pct','n_rows']].rename(columns={'model_display':'model','scope_display':'scope'})


# Step 6B: NYC borough map color-coded by RMSE (with labels)
import geopandas as gpd
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Self-contained paths
if 'repo_root' not in globals():
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    repo_root = next((p for p in candidates if (p / 'data' / 'processed').exists()), None)
    if repo_root is None:
        raise FileNotFoundError('Could not resolve repo_root')

if 'output_dir' not in globals():
    output_dir = repo_root / 'reports' / 'results' / 'prophet'
if 'plot_dir' not in globals():
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

shape_path = repo_root / 'data' / 'external' / 'taxi_zones' / 'taxi_zones' / 'taxi_zones.shp'
if not shape_path.exists():
    raise FileNotFoundError(f'Missing shapefile: {shape_path}')

gdf = gpd.read_file(shape_path)

# Normalize columns from shapefile
rename_map = {}
if 'LocationID' in gdf.columns:
    rename_map['LocationID'] = 'zone_id'
if 'borough' in gdf.columns:
    rename_map['borough'] = 'Borough'
if 'zone' in gdf.columns:
    rename_map['zone'] = 'Zone'
gdf = gdf.rename(columns=rename_map)

required = {'zone_id', 'Borough', 'geometry'}
missing = required - set(gdf.columns)
if missing:
    raise ValueError(f'Shapefile missing required columns: {missing}')

borough_pred_path = output_dir / 'prophet_borough_test_predictions.parquet'
if not borough_pred_path.exists():
    raise FileNotFoundError(f'Missing {borough_pred_path}. Run Step 4A first.')
borough_pred = pd.read_parquet(borough_pred_path)
if borough_pred.empty:
    raise ValueError('Borough prediction file is empty.')

borough_rmse = (
    borough_pred.groupby('Borough')
    .apply(lambda d: float(mean_squared_error(d['y'], d['yhat']) ** 0.5))
    .rename('rmse')
    .reset_index()
)

# Dissolve taxi zones into borough polygons for clean labels
borough_map = (
    gdf.merge(borough_rmse, on='Borough', how='left')
    .dissolve(by='Borough', aggfunc={'rmse': 'mean'})
    .reset_index()
)

fig, ax = plt.subplots(1, 1, figsize=(11, 10), facecolor='white')
ax.set_facecolor('white')
borough_map.plot(
    column='rmse',
    cmap='YlOrBr',
    linewidth=1.2,
    edgecolor='#f59e0b',
    legend=True,
    ax=ax,
    missing_kwds={'color': '#f3f4f6', 'label': 'No RMSE'}
)

# Ensure colorbar ticks/labels are dark
if len(fig.axes) > 1:
    cax = fig.axes[-1]
    cax.tick_params(colors='#111111', labelsize=10)
    for spine in cax.spines.values():
        spine.set_color('#f59e0b')

ax.set_title('Borough-Level Forecast Error in NYC (Test RMSE, Prophet)', pad=12, fontweight='bold', color='#111111')
ax.axis('off')
for spine in ax.spines.values():
    spine.set_color('#f59e0b')
    spine.set_alpha(0.8)

# Add borough labels at polygon representative points
for _, row in borough_map.iterrows():
    if row.geometry is None or row.geometry.is_empty:
        continue
    pt = row.geometry.representative_point()
    ax.text(
        pt.x,
        pt.y,
        f"{row['Borough']}\nRMSE: {row['rmse']:.1f}",
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='#111111',
        bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor='#f59e0b', alpha=0.95),
    )

plt.tight_layout()
plt.savefig(plot_dir / 'nyc_borough_rmse_map.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()

display(borough_rmse.sort_values('rmse'))


# Step 6C: Borough error scale-normalization view
# Shows why Manhattan can have highest absolute RMSE but not highest relative error

from sklearn.metrics import mean_squared_error

borough_pred_path = output_dir / 'prophet_borough_test_predictions.parquet'
if not borough_pred_path.exists():
    raise FileNotFoundError(f'Missing {borough_pred_path}. Run Step 4A/Step 3 test flow first.')

bp = pd.read_parquet(borough_pred_path).copy()
if bp.empty:
    raise ValueError('Borough test prediction table is empty.')

summary = (
    bp.groupby('Borough', as_index=False)
      .apply(lambda d: pd.Series({
          'rmse': float(mean_squared_error(d['y'], d['yhat']) ** 0.5),
          'mae': float((d['y'] - d['yhat']).abs().mean()),
          'mean_demand': float(d['y'].mean()),
          'n_rows': int(len(d)),
      }))
)

summary['rmse_over_mean'] = summary['rmse'] / summary['mean_demand']
summary = summary.sort_values('rmse', ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')
    ax.grid(color='#d4d4d4', alpha=0.6, axis='x')
    for spine in ax.spines.values():
        spine.set_color('#f59e0b')
        spine.set_alpha(0.8)
    ax.tick_params(colors='#111111', labelsize=11)

# Absolute RMSE
left_df = summary.sort_values('rmse', ascending=True)
sns.barplot(data=left_df, y='Borough', x='rmse', palette='YlOrBr', ax=axes[0])
axes[0].set_title('Absolute Error by Borough (Test RMSE)', color='#111111', fontweight='bold')
axes[0].set_xlabel('RMSE', color='#111111')
axes[0].set_ylabel('Borough', color='#111111')
for pch in axes[0].patches:
    w = pch.get_width()
    y_mid = pch.get_y() + pch.get_height()/2
    axes[0].annotate(f'{w:.1f}', (w, y_mid), xytext=(4,0), textcoords='offset points', va='center', color='#111111', fontsize=10)

# Scale-normalized RMSE
right_df = summary.sort_values('rmse_over_mean', ascending=True)
sns.barplot(data=right_df, y='Borough', x='rmse_over_mean', palette='YlOrBr', ax=axes[1])
axes[1].set_title('Scale-Normalized Error by Borough (RMSE / Mean Demand)', color='#111111', fontweight='bold')
axes[1].set_xlabel('RMSE / Mean Demand', color='#111111')
axes[1].set_ylabel('')
for pch in axes[1].patches:
    w = pch.get_width()
    y_mid = pch.get_y() + pch.get_height()/2
    axes[1].annotate(f'{w:.3f}', (w, y_mid), xytext=(4,0), textcoords='offset points', va='center', color='#111111', fontsize=10)

plt.tight_layout()
plt.savefig(plot_dir / 'borough_rmse_vs_normalized_rmse.png', dpi=170, bbox_inches='tight', facecolor='white')
plt.show()

summary.sort_values('rmse_over_mean', ascending=False)


# Step 6D: Top 15 zones by RMSE (Prophet)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = Path.cwd().resolve()

def find_repo_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for p in candidates:
        if (p / 'data').exists() and (p / 'notebooks').exists():
            return p
    for p in candidates:
        if (p / '.git').exists() or (p / 'setup.py').exists():
            return p
    return start

repo_root = find_repo_root(cwd)
output_dir = repo_root / 'reports' / 'results' / 'prophet'
plot_dir = output_dir / 'plots'
plot_dir.mkdir(parents=True, exist_ok=True)

pred_path = output_dir / 'prophet_top_zone_test_predictions.parquet'
lookup_path = repo_root / 'data' / 'external' / 'taxi_zone_lookup.csv'

if not pred_path.exists():
    raise FileNotFoundError(f'Missing: {pred_path}')
if not lookup_path.exists():
    raise FileNotFoundError(f'Missing: {lookup_path}')

pred = pd.read_parquet(pred_path).copy()
need_cols = {'zone_id', 'y', 'yhat'}
missing = need_cols - set(pred.columns)
if missing:
    raise ValueError(f'Missing columns in top-zone predictions: {sorted(missing)}')

pred['zone_id'] = pd.to_numeric(pred['zone_id'], errors='coerce')
pred['y'] = pd.to_numeric(pred['y'], errors='coerce')
pred['yhat'] = pd.to_numeric(pred['yhat'], errors='coerce').clip(lower=0)
pred = pred.dropna(subset=['zone_id', 'y', 'yhat']).copy()
pred['zone_id'] = pred['zone_id'].astype(int)

zone_metrics = (
    pred.assign(se=(pred['y'] - pred['yhat']) ** 2)
        .groupby('zone_id', as_index=False)
        .agg(rmse=('se', lambda s: float(np.sqrt(np.mean(s)))))
)

lookup = pd.read_csv(lookup_path)
lookup_cols = {c.lower(): c for c in lookup.columns}
zone_col = lookup_cols.get('locationid', lookup_cols.get('zone_id', None))
name_col = lookup_cols.get('zone', None)
if zone_col is None or name_col is None:
    raise ValueError('taxi_zone_lookup.csv must contain LocationID and Zone columns')

lookup = lookup.rename(columns={zone_col: 'zone_id', name_col: 'zone_name'})
lookup['zone_id'] = pd.to_numeric(lookup['zone_id'], errors='coerce').astype('Int64')
lookup = lookup.dropna(subset=['zone_id']).copy()
lookup['zone_id'] = lookup['zone_id'].astype(int)

zone_metrics = zone_metrics.merge(lookup[['zone_id', 'zone_name']], on='zone_id', how='left')
zone_metrics['zone_name'] = zone_metrics['zone_name'].fillna('Zone ' + zone_metrics['zone_id'].astype(str))

# Keep only top-15 highest RMSE zones and sort for horizontal bars
top15 = zone_metrics.sort_values('rmse', ascending=False).head(15).copy()
top15 = top15.sort_values('rmse', ascending=True)

# Styling
bg = 'white'
text_dark = '#1a1a1a'
frame = '#e59f20'
c1 = '#f2db89'
c2 = '#d9982a'

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

colors = [c1 if i < 7 else c2 for i in range(len(top15))]
bars = ax.barh(top15['zone_name'], top15['rmse'], color=colors, edgecolor=frame, linewidth=1.0)

max_v = float(top15['rmse'].max()) if len(top15) else 0.0
for b in bars:
    v = b.get_width()
    x = max(v * 0.55, v - max_v * 0.06)
    ax.text(x, b.get_y() + b.get_height()/2, f'{v:.1f}', va='center', ha='center',
            color=text_dark, fontsize=12, fontweight='bold')

ax.set_title('Top 15 Zones by Test RMSE (Prophet)', color=text_dark, fontsize=18, fontweight='bold', pad=12)
ax.set_xlabel('RMSE', color=text_dark, fontsize=13)
ax.set_ylabel('Zone', color=text_dark, fontsize=13)
ax.tick_params(colors=text_dark, labelsize=11)

# Make grid very light
ax.grid(False)

for s in ax.spines.values():
    s.set_edgecolor(frame)
    s.set_linewidth(1.4)

plt.tight_layout()
out_path = plot_dir / 'zone_top15_rmse_prophet.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=bg)
plt.show()

print('Saved:', out_path)
