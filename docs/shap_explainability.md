# SHAP Explainability

## Purpose

SHAP is used to explain how the selected Tuned Weighted XGBoost model uses structured quality features when classifying steel defects.

## Global SHAP Importance

|   rank | feature               |   mean_abs_shap |   importance_share |
|-------:|:----------------------|----------------:|-------------------:|
|      1 | Steel_Plate_Thickness |        0.574158 |          0.145057  |
|      2 | Length_of_Conveyer    |        0.330603 |          0.0835246 |
|      3 | Orientation_Index     |        0.197145 |          0.0498074 |
|      4 | Pixels_Areas          |        0.184514 |          0.0466162 |
|      5 | Edges_Y_Index         |        0.177971 |          0.0449632 |
|      6 | Minimum_of_Luminosity |        0.17741  |          0.0448214 |
|      7 | Luminosity_Index      |        0.176544 |          0.0446027 |
|      8 | Outside_X_Index       |        0.167393 |          0.0422908 |
|      9 | LogOfAreas            |        0.151835 |          0.0383601 |
|     10 | Log_X_Index           |        0.147461 |          0.0372549 |
|     11 | Log_Y_Index           |        0.146977 |          0.0371328 |
|     12 | Edges_Index           |        0.129997 |          0.0328429 |
|     13 | TypeOfSteel_A300      |        0.127015 |          0.0320896 |
|     14 | Empty_Index           |        0.125133 |          0.0316141 |
|     15 | Y_Perimeter           |        0.123942 |          0.0313132 |

## Top Features by Defect Class

| class        |   rank | feature               |   mean_abs_shap |
|:-------------|-------:|:----------------------|----------------:|
| Bumps        |      1 | Log_Y_Index           |       0.264705  |
| Bumps        |      2 | TypeOfSteel_A300      |       0.193273  |
| Bumps        |      3 | Length_of_Conveyer    |       0.157418  |
| Bumps        |      4 | Minimum_of_Luminosity |       0.150916  |
| Bumps        |      5 | Steel_Plate_Thickness |       0.136764  |
| Dirtiness    |      1 | Minimum_of_Luminosity |       0.360493  |
| Dirtiness    |      2 | Orientation_Index     |       0.32313   |
| Dirtiness    |      3 | Square_Index          |       0.294443  |
| Dirtiness    |      4 | Steel_Plate_Thickness |       0.269303  |
| Dirtiness    |      5 | X_Perimeter           |       0.248327  |
| K_Scatch     |      1 | Steel_Plate_Thickness |       1.28941   |
| K_Scatch     |      2 | Log_X_Index           |       0.521651  |
| K_Scatch     |      3 | Outside_X_Index       |       0.358422  |
| K_Scatch     |      4 | Minimum_of_Luminosity |       0.273828  |
| K_Scatch     |      5 | TypeOfSteel_A300      |       0.237092  |
| Other_Faults |      1 | Length_of_Conveyer    |       0.184665  |
| Other_Faults |      2 | Steel_Plate_Thickness |       0.151106  |
| Other_Faults |      3 | Edges_Y_Index         |       0.0947044 |
| Other_Faults |      4 | Orientation_Index     |       0.0776735 |
| Other_Faults |      5 | Edges_Index           |       0.0657135 |
| Pastry       |      1 | Edges_Y_Index         |       0.634901  |
| Pastry       |      2 | Orientation_Index     |       0.565856  |
| Pastry       |      3 | Outside_X_Index       |       0.316153  |
| Pastry       |      4 | Length_of_Conveyer    |       0.300008  |
| Pastry       |      5 | Outside_Global_Index  |       0.205163  |
| Stains       |      1 | Steel_Plate_Thickness |       1.07114   |
| Stains       |      2 | Pixels_Areas          |       0.762688  |
| Stains       |      3 | LogOfAreas            |       0.677291  |
| Stains       |      4 | Luminosity_Index      |       0.634029  |
| Stains       |      5 | Y_Perimeter           |       0.437466  |
| Z_Scratch    |      1 | Length_of_Conveyer    |       1.0777    |
| Z_Scratch    |      2 | Steel_Plate_Thickness |       1.03582   |
| Z_Scratch    |      3 | TypeOfSteel_A300      |       0.236685  |
| Z_Scratch    |      4 | Empty_Index           |       0.216744  |
| Z_Scratch    |      5 | X_Maximum             |       0.200681  |

## Individual Prediction Examples

|    id | actual_defect   | predicted_defect   |   confidence | top_supporting_features                                      | top_opposing_features   |
|------:|:----------------|:-------------------|-------------:|:-------------------------------------------------------------|:------------------------|
| 11226 | Bumps           | Bumps              |     0.90052  | Length_of_Conveyer, TypeOfSteel_A300, Square_Index           |                         |
|  4231 | Dirtiness       | Dirtiness          |     0.962594 | Orientation_Index, Steel_Plate_Thickness, Square_Index       | Log_X_Index             |
|  5158 | K_Scatch        | K_Scatch           |     0.994691 | Log_X_Index, Outside_X_Index, Steel_Plate_Thickness          |                         |
| 10080 | Other_Faults    | Other_Faults       |     0.935845 | Steel_Plate_Thickness, Length_of_Conveyer, Orientation_Index |                         |
| 11673 | Pastry          | Pastry             |     0.939863 | Orientation_Index, Length_of_Conveyer, Log_Y_Index           |                         |
|  8196 | Stains          | Stains             |     0.992746 | Pixels_Areas, Steel_Plate_Thickness, LogOfAreas              | Luminosity_Index        |
|   442 | Z_Scratch       | Z_Scratch          |     0.96533  | Length_of_Conveyer, Steel_Plate_Thickness, X_Maximum         | Edges_X_Index           |

## Interpretation Guardrail

Positive SHAP values indicate that a feature pushes the model output toward the explained class, while negative SHAP values push the model output away from that class.

**SHAP explains model behavior, not physical manufacturing causality.**

A high SHAP contribution must therefore be interpreted as predictive evidence used by the model, not as confirmed root-cause evidence.

## Intended System Use

These explanations will later support the `explain_prediction()` backend tool so the AI Copilot can provide grounded model evidence to manufacturing engineers.
