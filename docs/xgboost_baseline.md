# XGBoost Baseline

## Purpose

XGBoost is evaluated as a gradient-boosted tree model using the same fixed train and validation datasets as the previous models.

## Model Configuration

- n_estimators: 400
- learning_rate: 0.05
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- class weighting: None

## Validation Performance

- Accuracy: 0.5967
- Macro Precision: 0.5939
- Macro Recall: 0.5379
- Macro F1: 0.5496
- Weighted F1: 0.5858

## Per-class Performance

| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| Bumps        |    0.53495  | 0.52521  |   0.530035 |       714 |
| Dirtiness    |    0.461538 | 0.164384 |   0.242424 |        73 |
| K_Scatch     |    0.900383 | 0.919765 |   0.909971 |       511 |
| Other_Faults |    0.502613 | 0.588175 |   0.542039 |       981 |
| Pastry       |    0.347368 | 0.15     |   0.209524 |       220 |
| Stains       |    0.792683 | 0.764706 |   0.778443 |        85 |
| Z_Scratch    |    0.617486 | 0.653179 |   0.634831 |       173 |

## Recall Observations

Highest recall: `K_Scatch` (0.9198)

Lowest recall: `Pastry` (0.1500)

## Preliminary Feature Importance

| feature               |   importance |
|:----------------------|-------------:|
| Log_X_Index           |    0.157293  |
| TypeOfSteel_A300      |    0.0865923 |
| TypeOfSteel_A400      |    0.0783002 |
| Outside_X_Index       |    0.0645745 |
| Steel_Plate_Thickness |    0.0626085 |
| Pixels_Areas          |    0.0576041 |
| LogOfAreas            |    0.0568755 |
| Outside_Global_Index  |    0.0388356 |
| Length_of_Conveyer    |    0.0384661 |
| Orientation_Index     |    0.0376669 |

## Interpretation

Feature importance describes how the XGBoost model uses available predictors. It does not establish causal manufacturing relationships.

The test set remains reserved and has not been used for model evaluation.
