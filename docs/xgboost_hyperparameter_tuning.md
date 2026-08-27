# XGBoost Hyperparameter Tuning

## Purpose

RandomizedSearchCV is used to improve the weighted XGBoost model while keeping the search space computationally controlled.

## Optimization Metric

**Macro F1**

Macro F1 is selected because the defect classes are imbalanced and performance on minority classes is important.

## Cross-validation

- Stratified 3-fold CV
- 20 randomized parameter combinations
- Training split only

## Best Cross-validation Score

0.5745

## Best Parameters

| parameter        |   value |
|:-----------------|--------:|
| subsample        |    0.8  |
| reg_lambda       |    2    |
| reg_alpha        |    0.3  |
| n_estimators     |  350    |
| min_child_weight |    1    |
| max_depth        |    6    |
| learning_rate    |    0.05 |
| colsample_bytree |    0.7  |

## Validation Performance

- Accuracy: 0.5720
- Macro Precision: 0.5396
- Macro Recall: 0.6307
- Macro F1: 0.5709
- Weighted F1: 0.5665

## Model Comparison

| metric          |   baseline_xgboost |   weighted_xgboost |   tuned_weighted_xgboost |
|:----------------|-------------------:|-------------------:|-------------------------:|
| accuracy        |           0.596663 |           0.570548 |                 0.571999 |
| macro_precision |           0.59386  |           0.537527 |                 0.539639 |
| macro_recall    |           0.537917 |           0.612439 |                 0.630729 |
| macro_f1        |           0.54961  |           0.56561  |                 0.570915 |
| weighted_f1     |           0.585769 |           0.56671  |                 0.5665   |

## Per-class Performance

| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| Bumps        |    0.528373 | 0.586835 |   0.556072 |       714 |
| Dirtiness    |    0.279661 | 0.452055 |   0.34555  |        73 |
| K_Scatch     |    0.891589 | 0.933464 |   0.912046 |       511 |
| Other_Faults |    0.566225 | 0.348624 |   0.431546 |       981 |
| Pastry       |    0.291545 | 0.454545 |   0.35524  |       220 |
| Stains       |    0.714286 | 0.882353 |   0.789474 |        85 |
| Z_Scratch    |    0.505792 | 0.757225 |   0.606481 |       173 |

## Test Set Policy

The held-out test set remains untouched. Final test evaluation will be performed only after model selection.
