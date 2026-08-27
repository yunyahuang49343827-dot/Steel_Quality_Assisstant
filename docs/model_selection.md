# Model Selection and Final Holdout Evaluation

## Selection Policy

All model selection and hyperparameter tuning were completed using training and validation data only.

The held-out test set was evaluated only after the champion model had been selected.

Primary selection metric: **Validation Macro F1**

Macro Recall is used as a secondary consideration because minority defect detection is important.

## Validation Model Ranking

| model               |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |
|:--------------------|-----------:|------------------:|---------------:|-----------:|--------------:|
| xgboost_tuned       |   0.571999 |          0.539639 |       0.630729 |   0.570915 |      0.5665   |
| xgboost_weighted    |   0.570548 |          0.537527 |       0.612439 |   0.56561  |      0.56671  |
| xgboost_baseline    |   0.596663 |          0.59386  |       0.537917 |   0.54961  |      0.585769 |
| random_forest       |   0.599202 |          0.599672 |       0.51054  |   0.527334 |      0.583999 |
| logistic_regression |   0.57091  |          0.611945 |       0.473967 |   0.487827 |      0.555492 |

## Champion Model

**xgboost_tuned**

## Final Holdout Test Performance

- Accuracy: 0.5851
- Macro Precision: 0.5615
- Macro Recall: 0.6471
- Macro F1: 0.5904
- Weighted F1: 0.5798

## Final Per-class Performance

| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| Bumps        |    0.534527 | 0.585434 |   0.558824 |       714 |
| Dirtiness    |    0.301887 | 0.438356 |   0.357542 |        73 |
| K_Scatch     |    0.893058 | 0.929688 |   0.911005 |       512 |
| Other_Faults |    0.582651 | 0.362895 |   0.447236 |       981 |
| Pastry       |    0.331551 | 0.563636 |   0.417508 |       220 |
| Stains       |    0.765306 | 0.882353 |   0.819672 |        85 |
| Z_Scratch    |    0.521739 | 0.767442 |   0.621176 |       172 |

## Deployment Interpretation

The selected model is treated as a quality triage and decision-support model rather than an autonomous product acceptance or rejection system.

Production deployment would require validation on real manufacturing data, operating-threshold definition, domain-shift testing, monitoring, and engineer-in-the-loop review.
