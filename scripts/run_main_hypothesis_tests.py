#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

ROOT = Path('/Users/heliamahmoodzadeh/Desktop/HELIA/FINAL_DELIVARY')
DATA_FILE = ROOT / 'data' / 'raw' / 'yellow_taxi_representative_sample_2021_2023_distribution.csv'
RESULTS_DIR = ROOT / 'reports' / 'results'
FIGURES_DIR = ROOT / 'reports' / 'figures'
INSIGHTS_DIR = ROOT / 'reports' / 'insights'


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(perm[:n_a].mean() - perm[n_a:].mean())
        if diff >= observed:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def load_existing_summary() -> dict:
    summary_path = RESULTS_DIR / 'hypothesis_summary.json'
    if summary_path.exists():
        with summary_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def rename_existing_h2(existing: dict) -> dict:
    tests = existing.get('hypothesis_tests', {})
    old_h2 = tests.get('h2_ensemble_better_than_boosting', {})
    renamed = {
        'tested': bool(old_h2.get('tested', False)),
        'support': bool(old_h2.get('support', False)),
        'p_value': None,
        'details': {
            'definition': 'ML hypothesis: ensemble variants outperform standalone boosting',
            'boosting_mae': old_h2.get('boosting_mae'),
            'bagging_mae': old_h2.get('bagging_mae'),
            'stacking_mae': old_h2.get('stacking_mae'),
        },
    }

    model_cmp_path = RESULTS_DIR / 'model_comparison.csv'
    if model_cmp_path.exists():
        cmp = pd.read_csv(model_cmp_path)
        if {'model', 'mae'} <= set(cmp.columns):
            idx = dict(zip(cmp['model'], cmp['mae']))
            boost = idx.get('boosting_xgboost')
            bag = idx.get('bagging_random_forest')
            stack = idx.get('stacking_ensemble')
            if boost is not None and bag is not None and stack is not None:
                renamed['tested'] = True
                renamed['support'] = bool((bag < boost) and (stack < boost))
                renamed['details']['boosting_mae'] = float(boost)
                renamed['details']['bagging_mae'] = float(bag)
                renamed['details']['stacking_mae'] = float(stack)

    pairwise_path = RESULTS_DIR / 'pairwise_stat_tests.csv'
    if pairwise_path.exists():
        pairwise = pd.read_csv(pairwise_path)
        if 'p_permutation' in pairwise.columns:
            renamed['p_value'] = float(pairwise['p_permutation'].min())
            renamed['details']['p_value_source'] = 'min_pairwise_permutation_p'

    return renamed


def test_slide_h2_spatial(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    required_sets = {
        'manhattan_dominance': {'borough', 'zone_id', 'trips'},
        'airport_predictability': {'location_id', 'pickup_hour', 'trips'},
        'district_specific_timing': {'zone_type', 'is_weekend', 'pickup_hour', 'trips'},
    }

    rows = []
    for component, req in required_sets.items():
        missing = sorted(req - set(df.columns))
        rows.append(
            {
                'component': component,
                'tested': len(missing) == 0,
                'support': False,
                'p_value': np.nan,
                'reason': '' if len(missing) == 0 else f"Missing required columns: {', '.join(missing)}",
            }
        )

    detail_df = pd.DataFrame(rows)
    summary = {
        'tested': bool(detail_df['tested'].all()),
        'support': bool(detail_df['support'].all()) if bool(detail_df['tested'].all()) else False,
        'p_value': None,
        'reason': 'Insufficient granularity for full spatial H2 on this sample dataset' if not bool(detail_df['tested'].all()) else '',
    }
    return summary, detail_df


def test_slide_h3_seasonal(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    work = df.copy()

    if {'pickup_year', 'pickup_month', 'sample_rows'} <= set(work.columns):
        work['month'] = work['pickup_month'].astype(int)
        work['is_summer'] = work['month'].isin([6, 7, 8])
        summer = work.loc[work['is_summer'], 'sample_rows'].to_numpy(dtype=float)
        nonsummer = work.loc[~work['is_summer'], 'sample_rows'].to_numpy(dtype=float)

        if len(summer) >= 2 and len(nonsummer) >= 2:
            p_perm = permutation_pvalue(summer, nonsummer)
            _, p_mwu = mannwhitneyu(summer, nonsummer, alternative='two-sided')
            support = bool((summer.mean() > nonsummer.mean()) and (p_perm < 0.05))
            tourism_row = {
                'component': 'tourism_season_impact',
                'tested': True,
                'support': support,
                'p_value': float(p_perm),
                'p_value_secondary': float(p_mwu),
                'effect_summer_minus_nonsummer': float(summer.mean() - nonsummer.mean()),
                'reason': '',
            }
        else:
            tourism_row = {
                'component': 'tourism_season_impact',
                'tested': False,
                'support': False,
                'p_value': np.nan,
                'p_value_secondary': np.nan,
                'effect_summer_minus_nonsummer': np.nan,
                'reason': 'Not enough monthly points for tourism season test',
            }
    else:
        tourism_row = {
            'component': 'tourism_season_impact',
            'tested': False,
            'support': False,
            'p_value': np.nan,
            'p_value_secondary': np.nan,
            'effect_summer_minus_nonsummer': np.nan,
            'reason': 'Missing pickup_year/pickup_month/sample_rows columns',
        }

    # Weather and events are not available in this sample schema.
    weather_row = {
        'component': 'weather_driven_demand',
        'tested': False,
        'support': False,
        'p_value': np.nan,
        'p_value_secondary': np.nan,
        'effect_summer_minus_nonsummer': np.nan,
        'reason': 'Missing weather fields (e.g., precipitation, snowfall, temperature)',
    }
    event_row = {
        'component': 'event_based_spikes',
        'tested': False,
        'support': False,
        'p_value': np.nan,
        'p_value_secondary': np.nan,
        'effect_summer_minus_nonsummer': np.nan,
        'reason': 'Missing event flags/calendar linkage for venue/event dates',
    }

    detail_df = pd.DataFrame([tourism_row, weather_row, event_row])
    tested_all = bool(detail_df['tested'].all())
    supported_all = bool(detail_df['support'].all())

    summary = {
        'tested': tested_all,
        'support': supported_all if tested_all else False,
        'p_value': float(tourism_row['p_value']) if tourism_row['tested'] else None,
        'reason': 'Only tourism sub-test was statistically testable on this sample dataset',
    }

    return summary, detail_df


def make_status_plot(summary: dict) -> None:
    labels = ['H1_temporal', 'ML_hypothesis', 'H2_spatial_slide', 'H3_seasonal_slide']
    vals = [
        int(summary['hypothesis_tests'].get('h1_temporal_patterns', {}).get('support', False)),
        int(summary['hypothesis_tests'].get('ml_hypothesis', {}).get('support', False)),
        int(summary['hypothesis_tests'].get('h2_spatial_patterns_slide', {}).get('support', False)),
        int(summary['hypothesis_tests'].get('h3_seasonal_effects_slide', {}).get('support', False)),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#2E7D32' if v == 1 else '#B0BEC5' for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Support (1=True, 0=False)')
    ax.set_title('Hypothesis Validation Status (Current Sample)')
    ax.grid(axis='y', alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, 'TRUE' if v == 1 else 'FALSE', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'hypothesis_validation_status_v2.png', dpi=180)
    plt.close(fig)


def write_markdown(summary: dict, h2_detail: pd.DataFrame, h3_detail: pd.DataFrame) -> None:
    out = INSIGHTS_DIR / 'H2_H3_VALIDATION_SUMMARY.md'
    with out.open('w', encoding='utf-8') as f:
        f.write('# Main Slide H2/H3 Validation Summary\n\n')
        f.write('## Final True/False\n')
        f.write(f"- `ml_hypothesis`: **{'TRUE' if summary['hypothesis_tests']['ml_hypothesis']['support'] else 'FALSE'}**\n")
        f.write(f"- `h2_spatial_patterns_slide`: **{'TRUE' if summary['hypothesis_tests']['h2_spatial_patterns_slide']['support'] else 'FALSE'}**\n")
        f.write(f"- `h3_seasonal_effects_slide`: **{'TRUE' if summary['hypothesis_tests']['h3_seasonal_effects_slide']['support'] else 'FALSE'}**\n\n")
        f.write('## Why H2/H3 are limited on this sample\n')
        f.write('- Dataset contains monthly aggregate rows only (`pickup_year`, `pickup_month`, `sample_rows`).\n')
        f.write('- No zone, borough, airport, weather, or event-level columns are present.\n\n')
        f.write('## H2 component status\n')
        f.write('```csv\n')
        f.write(h2_detail.to_csv(index=False))
        f.write('```\n\n')
        f.write('## H3 component status\n')
        f.write('```csv\n')
        f.write(h3_detail.to_csv(index=False))
        f.write('```\n')


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA_FILE)

    existing = load_existing_summary()
    h1_old = existing.get('hypothesis_tests', {}).get('h1_seasonality', {})

    ml_h = rename_existing_h2(existing)
    h2_summary, h2_detail = test_slide_h2_spatial(df)
    h3_summary, h3_detail = test_slide_h3_seasonal(df)

    combined = {
        'best_model': existing.get('best_model', 'boosting_xgboost'),
        'hypothesis_tests': {
            'h1_temporal_patterns': {
                'tested': bool(h1_old.get('tested', False)),
                'p_value': h1_old.get('p_value', None),
                'support': bool(h1_old.get('support', False)),
            },
            'ml_hypothesis': ml_h,
            'h2_spatial_patterns_slide': h2_summary,
            'h3_seasonal_effects_slide': h3_summary,
        },
    }

    (RESULTS_DIR / 'h2_slide_component_tests.csv').write_text(h2_detail.to_csv(index=False), encoding='utf-8')
    (RESULTS_DIR / 'h3_slide_component_tests.csv').write_text(h3_detail.to_csv(index=False), encoding='utf-8')

    with (RESULTS_DIR / 'hypothesis_summary_v2.json').open('w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)

    # keep backward-compatible main summary path updated for your push flow
    with (RESULTS_DIR / 'hypothesis_summary.json').open('w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)

    make_status_plot(combined)
    write_markdown(combined, h2_detail, h3_detail)

    print('Generated:')
    print(RESULTS_DIR / 'hypothesis_summary.json')
    print(RESULTS_DIR / 'hypothesis_summary_v2.json')
    print(RESULTS_DIR / 'h2_slide_component_tests.csv')
    print(RESULTS_DIR / 'h3_slide_component_tests.csv')
    print(FIGURES_DIR / 'hypothesis_validation_status_v2.png')
    print(INSIGHTS_DIR / 'H2_H3_VALIDATION_SUMMARY.md')


if __name__ == '__main__':
    main()
