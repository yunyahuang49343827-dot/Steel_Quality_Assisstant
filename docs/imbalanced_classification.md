# Imbalanced Classification Evaluation

## Purpose

The baseline models showed weak recall for minority defect classes. This experiment tests balanced sample weighting using XGBoost.

## Why Sample Weighting

Minority-class training samples receive higher loss weights so classification errors on rare defect types have greater influence during model training.

Synthetic oversampling such as SMOTE is not used in this stage because the dataset contains binary and derived geometric features, and the competition dataset itself is synthetic.

## Training Class Weights

|   class_id | defect_type   |   class_weight |
|-----------:|:--------------|---------------:|
|          0 | Bumps         |       0.551455 |
|          1 | Dirtiness     |       5.42183  |
|          2 | K_Scatch      |       0.769682 |
|          3 | Other_Faults  |       0.401485 |
|          4 | Pastry        |       1.79317  |
|          5 | Stains        |       4.61809  |
|          6 | Z_Scratch     |       2.28323  |

## Weighted Validation Performance

- Accuracy: 0.5705
- Macro Precision: 0.5375
- Macro Recall: 0.6124
- Macro F1: 0.5656
- Weighted F1: 0.5667

## Baseline vs Weighted

| metric          |   baseline_xgboost |   weighted_xgboost |   difference |
|:----------------|-------------------:|-------------------:|-------------:|
| accuracy        |           0.596663 |           0.570548 |   -0.0261153 |
| macro_precision |           0.59386  |           0.537527 |   -0.0563337 |
| macro_recall    |           0.537917 |           0.612439 |    0.0745222 |
| macro_f1        |           0.54961  |           0.56561  |    0.016     |
| weighted_f1     |           0.585769 |           0.56671  |   -0.0190588 |

## Per-class Performance

| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| Bumps        |    0.53129  | 0.582633 |   0.555778 |       714 |
| Dirtiness    |    0.282828 | 0.383562 |   0.325581 |        73 |
| K_Scatch     |    0.889306 | 0.927593 |   0.908046 |       511 |
| Other_Faults |    0.546547 | 0.37105  |   0.442016 |       981 |
| Pastry       |    0.269113 | 0.4      |   0.321755 |       220 |
| Stains       |    0.721154 | 0.882353 |   0.793651 |        85 |
| Z_Scratch    |    0.522449 | 0.739884 |   0.61244  |       173 |

## Decision Principle

The weighted model is not automatically better because minority recall increases. Model selection must consider the trade-off between missed defects and false-positive inspection workload.

The test set remains reserved and is not used during this experiment.
