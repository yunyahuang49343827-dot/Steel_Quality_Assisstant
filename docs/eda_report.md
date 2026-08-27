# Exploratory Data Analysis Report

## Purpose

EDA is used to understand class imbalance, feature distributions, class-level differences, skewness, correlations, and previously identified geometry-consistency warnings.

## Class Imbalance

The largest class is `Other_Faults` with 35.58% of modeling samples.

The smallest class is `Dirtiness` with 2.64%.

Because the classes are imbalanced, later model evaluation will include Macro F1 and per-class Recall rather than relying on Accuracy alone.

## Feature Skewness

The most strongly skewed analysed feature is `Pixels_Areas` with skewness 7.054.

Strongly skewed features are not automatically removed. Tree-based models can often handle non-normal feature distributions.

## Class Feature Patterns

Median feature values were compared across defect classes to reduce the influence of extreme values.

| defect_type   |   Pixels_Areas |   X_Perimeter |   Y_Perimeter |   Steel_Plate_Thickness |   Luminosity_Index |   Edges_Index |   Empty_Index |   Square_Index |
|:--------------|---------------:|--------------:|--------------:|------------------------:|-------------------:|--------------:|--------------:|---------------:|
| Bumps         |            121 |            19 |            17 |                      70 |           -0.1447  |       0.4468  |       0.3571  |         0.7    |
| Dirtiness     |            194 |            25 |            30 |                      70 |           -0.1043  |       0.4589  |       0.4022  |         0.2857 |
| K_Scatch      |           6432 |           275 |           138 |                      40 |           -0.1828  |       0.0585  |       0.4563  |         0.3943 |
| Other_Faults  |            140 |            22 |            21 |                      70 |           -0.1261  |       0.33555 |       0.4086  |         0.5625 |
| Pastry        |            169 |            19 |            26 |                      70 |           -0.1705  |       0.1765  |       0.3391  |         0.375  |
| Stains        |             17 |             7 |             5 |                      50 |           -0.0068  |       0.732   |       0.36055 |         0.75   |
| Z_Scratch     |            162 |            26 |            23 |                      70 |           -0.14905 |       0.15625 |       0.446   |         0.5714 |

## Geometry Consistency Warnings

The highest observed geometry-warning rate is in `K_Scatch` at 4.95%.

These records are retained as warnings rather than automatically removed because this competition dataset is synthetic and the inconsistencies do not necessarily represent corrupted records.

## Interpretation Principle

EDA identifies associations and distributional patterns. These observations must not be presented as proof of manufacturing causality.
