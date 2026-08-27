# Data Quality Report

## Dataset

- Raw rows: 19,219
- Columns: 35

## Structural Checks

- Missing cells: 0
- Duplicate rows: 0
- Duplicate IDs: 0
- Infinite numeric values: 0

## Binary Encoding Checks

- Invalid target binary values: 0
- Invalid steel-type binary values: 0

## Target Structure

- Exactly one positive defect label: 18,380
- Zero positive defect labels: 818
- Multiple positive defect labels: 21

## Steel Type Encoding

- Exactly one steel type: 19,198
- No steel type indicator: 20
- Multiple steel type indicators: 1

## Modeling Eligibility

- Eligible modeling rows: 18,380
- Excluded target exceptions: 839

The primary ML task is single-label multiclass classification. Samples without exactly one positive target label are excluded from the primary modeling dataset rather than silently converted into a class.

## Class Distribution

| Defect | Count | Percentage |
|---|---:|---:|
| Other_Faults | 6,540 | 35.58% |
| Bumps | 4,761 | 25.90% |
| K_Scatch | 3,411 | 18.56% |
| Pastry | 1,465 | 7.97% |
| Z_Scratch | 1,150 | 6.26% |
| Stains | 568 | 3.09% |
| Dirtiness | 485 | 2.64% |

## Outlier Policy

Extreme numerical observations are not automatically removed. In manufacturing and defect data, unusual values may represent genuine defect characteristics rather than data errors.

Outliers will be analysed during EDA and model evaluation before any removal or transformation decision is made.

## Modeling Decision

The cleaned modeling dataset retains only records containing exactly one defect label.

The original Kaggle training dataset remains unchanged and serves as the raw source.
