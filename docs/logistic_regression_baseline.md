# Logistic Regression Baseline

## Purpose

This model establishes a simple linear classification baseline before evaluating tree-based models.

## Dataset Split

- Train: 12,866
- Validation: 2,757
- Test: 2,757

The split is stratified by defect class. The test set is reserved and is not used for model comparison.

## Pipeline

StandardScaler

→ Logistic Regression

## Validation Performance

- Accuracy: 0.5709
- Macro Precision: 0.6119
- Macro Recall: 0.4740
- Macro F1: 0.4878
- Weighted F1: 0.5555

## Per-class Recall

| class        |   precision |    recall |   f1-score |   support |
|:-------------|------------:|----------:|-----------:|----------:|
| Bumps        |    0.488372 | 0.5       |  0.494118  |       714 |
| Dirtiness    |    0.666667 | 0.0273973 |  0.0526316 |        73 |
| K_Scatch     |    0.889101 | 0.90998   |  0.89942   |       511 |
| Other_Faults |    0.480627 | 0.594292  |  0.531449  |       981 |
| Pastry       |    0.442857 | 0.140909  |  0.213793  |       220 |
| Stains       |    0.769231 | 0.705882  |  0.736196  |        85 |
| Z_Scratch    |    0.546763 | 0.439306  |  0.487179  |       173 |

## Recall Observations

Highest recall: `K_Scatch` (0.9100)

Lowest recall: `Dirtiness` (0.0274)

## Interpretation

Because the defect classes are imbalanced, Macro F1 and per-class Recall are emphasized alongside Accuracy.

This model does not use class weighting. It serves as the unweighted linear baseline for later model comparison.
