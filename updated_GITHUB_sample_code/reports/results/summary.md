# updated_GITHUB_sample_code Summary

- Best model: **boosting_xgboost**
- Train/Validation/Test: {'train': 23, 'validation': 5, 'test': 5}

## Model Comparison

     mae     rmse         r2  residual_variance                 model
1.212091 1.275717  -9.171595           0.158290      boosting_xgboost
2.689041 2.715370 -45.082722           0.142296     stacking_ensemble
3.116000 3.139414 -60.599500           0.146464 bagging_random_forest

## Hypotheses

{
  "h1_seasonality": {
    "tested": true,
    "p_value": 0.9520819621960708,
    "support": false,
    "description": "Monthly seasonality effect exists in sample_rows."
  },
  "h2_ensemble_better_than_boosting": {
    "tested": true,
    "support": false,
    "rule": "At least one ensemble beats boosting on MAE and shows significant paired error difference.",
    "boosting_mae": 1.212091088294983,
    "bagging_mae": 3.115999999999997,
    "stacking_mae": 2.689040556464363,
    "pairwise_details": [
      {
        "pair": "boosting_xgboost vs bagging_random_forest",
        "p_permutation": 0.0619876024795041,
        "ci95": [
          -1.9254858398437478,
          -1.8823320312499958
        ],
        "direction_support": false,
        "passed": false
      },
      {
        "pair": "boosting_xgboost vs stacking_ensemble",
        "p_permutation": 0.1239752049590082,
        "ci95": [
          -1.8834384639044686,
          -0.7016211287617693
        ],
        "direction_support": false,
        "passed": false
      }
    ]
  }
}