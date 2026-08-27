# Random Forest Baseline

## Purpose

Random Forest is evaluated as a nonlinear tree-based model using the same fixed training and validation split as the Logistic Regression baseline.

## Model Configuration

- Trees: 400
- max_depth: None
- max_features: sqrt
- class_weight: None

Class weighting is intentionally disabled during this stage so imbalance handling can be evaluated separately.

## Validation Performance

- Accuracy: 0.5992
- Macro Precision: 0.5997
- Macro Recall: 0.5105
- Macro F1: 0.5273
- Weighted F1: 0.5840

## Per-class Performance

| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| Bumps        |    0.546547 | 0.509804 |   0.527536 |       714 |
| Dirtiness    |    0.444444 | 0.109589 |   0.175824 |        73 |
| K_Scatch     |    0.893536 | 0.919765 |   0.906461 |       511 |
| Other_Faults |    0.501597 | 0.640163 |   0.562472 |       981 |
| Pastry       |    0.424242 | 0.127273 |   0.195804 |       220 |
| Stains       |    0.768293 | 0.741176 |   0.754491 |        85 |
| Z_Scratch    |    0.619048 | 0.526012 |   0.56875  |       173 |

## Recall Observations

Highest recall: `K_Scatch` (0.9198)

Lowest recall: `Dirtiness` (0.1096)

## Preliminary Feature Importance

| feature               |   importance |
|:----------------------|-------------:|
| Outside_X_Index       |    0.0626821 |
| Pixels_Areas          |    0.0597421 |
| Log_X_Index           |    0.0577407 |
| LogOfAreas            |    0.0546813 |
| X_Perimeter           |    0.0482494 |
| Sum_of_Luminosity     |    0.0457883 |
| SigmoidOfAreas        |    0.0436178 |
| Length_of_Conveyer    |    0.0428647 |
| Minimum_of_Luminosity |    0.0424495 |
| X_Minimum             |    0.0415283 |

## Interpretation

Random Forest feature importance describes how the trained model uses features to split the training data. It does not establish manufacturing causality.

The test set remains reserved and is not evaluated during this stage.
