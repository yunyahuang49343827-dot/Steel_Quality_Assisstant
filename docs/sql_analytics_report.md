# SQL Quality Analytics Report

## Purpose

This stage uses PostgreSQL to perform descriptive quality analytics on the model-ready steel defect dataset.

SQL is used as the factual analytics layer. The results will later support FastAPI endpoints and LLM function-calling tools.

## 1. Defect Distribution

| defect_type | sample_count | percentage |
| --- | --- | --- |
| Other_Faults | 6540 | 35.58 |
| Bumps | 4761 | 25.90 |
| K_Scatch | 3411 | 18.56 |
| Pastry | 1465 | 7.97 |
| Z_Scratch | 1150 | 6.26 |
| Stains | 568 | 3.09 |
| Dirtiness | 485 | 2.64 |

### Key Observation

The most common defect category is `Other_Faults` with 6540 samples (35.58%).

## 2. Defect Feature Summary

| defect_type | sample_count | avg_plate_thickness | avg_pixels_area | avg_x_perimeter | avg_y_perimeter | avg_luminosity_index |
| --- | --- | --- | --- | --- | --- | --- |
| Other_Faults | 6540 | 91.56 | 637.10 | 47.97 | 38.63 | -0.1323 |
| Bumps | 4761 | 77.02 | 227.71 | 28.00 | 24.10 | -0.1501 |
| K_Scatch | 3411 | 41.16 | 7271.08 | 351.68 | 199.78 | -0.1363 |
| Pastry | 1465 | 91.26 | 364.96 | 30.95 | 41.22 | -0.1817 |
| Z_Scratch | 1150 | 73.95 | 477.93 | 46.01 | 39.25 | -0.1616 |
| Stains | 568 | 51.57 | 28.05 | 9.44 | 6.10 | -0.0231 |
| Dirtiness | 485 | 80.16 | 442.55 | 40.19 | 51.98 | -0.1093 |

These values describe average feature characteristics by defect category. They should not be interpreted as causal manufacturing relationships.

## 3. Thickness Group Analysis

| thickness_group | defect_type | sample_count |
| --- | --- | --- |
| Medium | Other_Faults | 2777 |
| Medium | Bumps | 2419 |
| Medium | Z_Scratch | 953 |
| Medium | Pastry | 680 |
| Medium | Dirtiness | 231 |
| Medium | K_Scatch | 21 |
| Medium | Stains | 17 |
| Thick | Other_Faults | 1167 |
| Thick | Bumps | 366 |
| Thick | Pastry | 254 |
| Thick | Z_Scratch | 70 |
| Thick | Dirtiness | 33 |
| Thick | K_Scatch | 17 |
| Thick | Stains | 1 |
| Thin | K_Scatch | 3373 |
| Thin | Other_Faults | 2596 |
| Thin | Bumps | 1976 |
| Thin | Stains | 550 |
| Thin | Pastry | 531 |
| Thin | Dirtiness | 221 |
| Thin | Z_Scratch | 127 |

### Important Note

Thin / Medium / Thick are exploratory data segmentation groups created for this project. They are not presented as official manufacturing thickness standards.

## 4. Luminosity Analysis

| defect_type | sample_count | avg_min_luminosity | avg_max_luminosity | avg_luminosity_index |
| --- | --- | --- | --- | --- |
| Pastry | 1465 | 84.37 | 124.39 | -0.1817 |
| Z_Scratch | 1150 | 91.79 | 124.32 | -0.1616 |
| Bumps | 4761 | 91.55 | 126.91 | -0.1501 |
| K_Scatch | 3411 | 48.58 | 132.31 | -0.1363 |
| Other_Faults | 6540 | 93.32 | 128.45 | -0.1323 |
| Dirtiness | 485 | 98.66 | 130.45 | -0.1093 |
| Stains | 568 | 112.36 | 138.07 | -0.0231 |

## 5. Fault Area Analysis

| defect_type | sample_count | avg_pixels_area | min_pixels_area | max_pixels_area | avg_log_area |
| --- | --- | --- | --- | --- | --- |
| K_Scatch | 3411 | 7271.08 | 16.00 | 152655.00 | 3.6886 |
| Other_Faults | 6540 | 637.10 | 6.00 | 37334.00 | 2.2624 |
| Z_Scratch | 1150 | 477.93 | 17.00 | 15704.00 | 2.3224 |
| Dirtiness | 485 | 442.55 | 18.00 | 6615.00 | 2.3924 |
| Pastry | 1465 | 364.96 | 12.00 | 17810.00 | 2.2909 |
| Bumps | 4761 | 227.71 | 8.00 | 21054.00 | 2.1456 |
| Stains | 568 | 28.05 | 6.00 | 212.00 | 1.3362 |

### Key Observation

`K_Scatch` has the highest average Pixels_Areas in this dataset, with an average of 7271.08.

## Interpretation Principle

SQL analytics describe patterns observed in the available dataset. Differences between defect classes do not establish causal process relationships.
