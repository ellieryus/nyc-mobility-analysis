# Hypothesis Testing2 Report

- Best city-wide model: **bagging_rf**
- Significance level: `0.05`

## Model Comparison
```
     model       mae      rmse       r2  residual_variance
bagging_rf 16.802721 31.710695 0.843204         983.142501
   xgboost 21.810591 39.079037 0.761871        1485.594847
  stacking 22.638014 37.544896 0.780201        1363.032003
```

## Statistical Verdicts
- Zone-specific vs city-wide tested: `False`
- Zone-specific support: `False`
- Airport predictability tested: `False`
- Airport support: `False`
- Borough consistency tested: `False`
- Borough support: `False`

## Roadmap
- data_ingestion: done
- data_quality_and_feature_engineering: done
- model_training_and_comparison: done
- statistical_hypothesis_tests: done
- artifacts_and_reporting: in_progress