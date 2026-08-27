# Global Feature Importance

## Purpose

This stage evaluates which structured quality features the selected champion XGBoost model relies on most strongly across all defect predictions.

## Champion Model

**Tuned Weighted XGBoost**

## Top 15 Features

|   rank | feature               | feature_group       |   importance |   importance_share |
|-------:|:----------------------|:--------------------|-------------:|-------------------:|
|      1 | TypeOfSteel_A400      | Steel / Production  |    0.130418  |          0.130418  |
|      2 | Log_X_Index           | Log Transformation  |    0.0987863 |          0.0987863 |
|      3 | TypeOfSteel_A300      | Steel / Production  |    0.0968532 |          0.0968532 |
|      4 | Outside_Global_Index  | Shape / Edge Index  |    0.0953352 |          0.0953352 |
|      5 | Steel_Plate_Thickness | Steel / Production  |    0.0623036 |          0.0623036 |
|      6 | LogOfAreas            | Log Transformation  |    0.0573276 |          0.0573276 |
|      7 | Pixels_Areas          | Geometry / Position |    0.0468715 |          0.0468715 |
|      8 | Length_of_Conveyer    | Steel / Production  |    0.0451048 |          0.0451048 |
|      9 | Y_Perimeter           | Geometry / Position |    0.0447503 |          0.0447503 |
|     10 | Outside_X_Index       | Shape / Edge Index  |    0.0402212 |          0.0402212 |
|     11 | Orientation_Index     | Shape / Edge Index  |    0.0362894 |          0.0362894 |
|     12 | Log_Y_Index           | Log Transformation  |    0.0243582 |          0.0243582 |
|     13 | Square_Index          | Shape / Edge Index  |    0.0211199 |          0.0211199 |
|     14 | Edges_Y_Index         | Shape / Edge Index  |    0.0211065 |          0.0211065 |
|     15 | X_Maximum             | Geometry / Position |    0.0194838 |          0.0194838 |

## Feature Group Importance

|   rank | feature_group       |   feature_count |   total_importance |   importance_share |   mean_feature_importance |
|-------:|:--------------------|----------------:|-------------------:|-------------------:|--------------------------:|
|      1 | Steel / Production  |               4 |          0.33468   |          0.33468   |                 0.08367   |
|      2 | Shape / Edge Index  |               8 |          0.251692  |          0.251692  |                 0.0314615 |
|      3 | Log Transformation  |               3 |          0.180472  |          0.180472  |                 0.0601574 |
|      4 | Geometry / Position |               7 |          0.158266  |          0.158266  |                 0.0226094 |
|      5 | Luminosity          |               4 |          0.0616071 |          0.0616071 |                 0.0154018 |
|      6 | Area Transformation |               1 |          0.0132827 |          0.0132827 |                 0.0132827 |

## Main Observations

- Highest-ranked individual feature: `TypeOfSteel_A400`.
- Highest-ranked feature group: `Steel / Production`.

## Interpretation Guardrail

XGBoost native feature importance measures how strongly the model uses available predictors during classification.

**Feature importance does not establish manufacturing causality or root cause.**

Correlated or derived features can distribute importance across related variables, so this analysis should be treated as a global model overview rather than a causal explanation.

## Next Stage

Stage B14 applies SHAP to provide more detailed global, per-class, and individual prediction explanations.
