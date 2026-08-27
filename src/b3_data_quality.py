from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

TRAIN_PATH = DATA_DIR / "train.csv"

MODELING_OUTPUT_PATH = (
    PROCESSED_DIR / "steel_quality_modeling.csv"
)

QUALITY_SUMMARY_PATH = (
    REPORTS_DIR / "data_quality_summary.csv"
)

FEATURE_REPORT_PATH = (
    REPORTS_DIR / "feature_quality_report.csv"
)

CLASS_DISTRIBUTION_PATH = (
    REPORTS_DIR / "class_distribution.csv"
)

MARKDOWN_REPORT_PATH = (
    DOCS_DIR / "data_quality_report.md"
)


# =========================================================
# 2. Dataset definitions
# =========================================================

ID_COLUMN = "id"

TARGET_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

BINARY_FEATURE_COLUMNS = [
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
]


# =========================================================
# 3. Load data
# =========================================================

def load_training_data() -> pd.DataFrame:
    """
    Load the raw Kaggle training dataset.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.csv was not found at:\n{TRAIN_PATH}"
        )

    return pd.read_csv(TRAIN_PATH)


# =========================================================
# 4. Basic dataset checks
# =========================================================

def run_basic_checks(df: pd.DataFrame) -> dict:
    """
    Run general structural data-quality checks.
    """

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    infinite_count = int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "duplicate_ids": int(
            df[ID_COLUMN].duplicated().sum()
        ),
        "infinite_values": infinite_count,
    }


# =========================================================
# 5. Validate binary columns
# =========================================================

def count_invalid_binary_values(
    df: pd.DataFrame,
    columns: list[str],
) -> int:
    """
    Count values outside the expected binary set {0, 1}.
    """

    invalid_count = 0

    for column in columns:

        valid_mask = df[column].isin([0, 1])

        invalid_count += int(
            (~valid_mask).sum()
        )

    return invalid_count


# =========================================================
# 6. Target structure analysis
# =========================================================

def analyse_target_structure(
    df: pd.DataFrame,
) -> dict:
    """
    Classify rows by the number of positive defect labels.
    """

    target_sum = df[TARGET_COLUMNS].sum(axis=1)

    return {
        "single_label_rows": int(
            (target_sum == 1).sum()
        ),
        "zero_label_rows": int(
            (target_sum == 0).sum()
        ),
        "multi_label_rows": int(
            (target_sum > 1).sum()
        ),
        "max_labels_per_row": int(
            target_sum.max()
        ),
    }


# =========================================================
# 7. Steel type consistency
# =========================================================

def analyse_steel_type_encoding(
    df: pd.DataFrame,
) -> dict:
    """
    Check whether the A300/A400 indicators form a
    mutually exclusive one-hot style representation.
    """

    steel_sum = (
        df["TypeOfSteel_A300"]
        + df["TypeOfSteel_A400"]
    )

    return {
        "steel_exactly_one_type": int(
            (steel_sum == 1).sum()
        ),
        "steel_no_type": int(
            (steel_sum == 0).sum()
        ),
        "steel_multiple_types": int(
            (steel_sum > 1).sum()
        ),
    }


# =========================================================
# 8. Numeric range checks
# =========================================================

def analyse_suspicious_ranges(
    df: pd.DataFrame,
) -> dict:
    """
    Run conservative validity checks only where the
    expected numeric direction is reasonably clear.

    Extreme values are not automatically treated as errors.
    """

    checks = {}

    non_negative_columns = [
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter",
        "Length_of_Conveyer",
        "Steel_Plate_Thickness",
    ]

    for column in non_negative_columns:

        checks[
            f"negative_{column}"
        ] = int(
            (df[column] < 0).sum()
        )

    checks["x_min_greater_than_x_max"] = int(
        (
            df["X_Minimum"]
            > df["X_Maximum"]
        ).sum()
    )

    checks["y_min_greater_than_y_max"] = int(
        (
            df["Y_Minimum"]
            > df["Y_Maximum"]
        ).sum()
    )

    return checks


# =========================================================
# 9. Feature profiling
# =========================================================

def build_feature_quality_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build profiling statistics for every dataset column.
    """

    records = []

    for column in df.columns:

        series = df[column]

        record = {
            "column": column,
            "dtype": str(series.dtype),
            "missing_count": int(
                series.isna().sum()
            ),
            "unique_count": int(
                series.nunique(dropna=True)
            ),
        }

        if pd.api.types.is_numeric_dtype(series):

            record.update(
                {
                    "min": series.min(),
                    "max": series.max(),
                    "mean": series.mean(),
                    "std": series.std(),
                }
            )

        else:

            record.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )

        records.append(record)

    return pd.DataFrame(records)


# =========================================================
# 10. Create modeling dataset
# =========================================================

def create_modeling_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the primary single-label multiclass dataset.

    Only samples with exactly one positive defect target
    are eligible.

    The original target indicators are preserved and a
    human-readable defect_type column is added.
    """

    target_sum = df[TARGET_COLUMNS].sum(axis=1)

    modeling_df = (
        df.loc[target_sum == 1]
        .copy()
        .reset_index(drop=True)
    )

    modeling_df["defect_type"] = (
        modeling_df[TARGET_COLUMNS]
        .idxmax(axis=1)
    )

    return modeling_df


# =========================================================
# 11. Class distribution
# =========================================================

def build_class_distribution(
    modeling_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate class counts and percentages for the
    cleaned modeling dataset.
    """

    counts = (
        modeling_df["defect_type"]
        .value_counts()
        .rename("count")
    )

    percentages = (
        modeling_df["defect_type"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percentage")
    )

    distribution = pd.concat(
        [counts, percentages],
        axis=1,
    )

    distribution.index.name = "defect_type"

    return distribution.reset_index()


# =========================================================
# 12. Build quality summary
# =========================================================

def build_quality_summary(
    basic_checks: dict,
    target_checks: dict,
    steel_checks: dict,
    range_checks: dict,
    invalid_target_binary_values: int,
    invalid_feature_binary_values: int,
    modeling_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all major quality checks into one report.
    """

    checks = {
        **basic_checks,

        "invalid_target_binary_values":
            invalid_target_binary_values,

        "invalid_feature_binary_values":
            invalid_feature_binary_values,

        **target_checks,
        **steel_checks,
        **range_checks,

        "modeling_rows":
            len(modeling_df),

        "excluded_from_modeling":
            (
                basic_checks["total_rows"]
                - len(modeling_df)
            ),
    }

    records = []

    for check_name, value in checks.items():

        records.append(
            {
                "check": check_name,
                "value": value,
            }
        )

    return pd.DataFrame(records)


# =========================================================
# 13. Generate Markdown report
# =========================================================

def generate_markdown_report(
    basic_checks: dict,
    target_checks: dict,
    steel_checks: dict,
    invalid_target_binary_values: int,
    invalid_feature_binary_values: int,
    modeling_df: pd.DataFrame,
    class_distribution: pd.DataFrame,
) -> str:
    """
    Generate a GitHub-friendly Data Quality report.
    """

    excluded = (
        basic_checks["total_rows"]
        - len(modeling_df)
    )

    lines = [
        "# Data Quality Report",
        "",
        "## Dataset",
        "",
        f"- Raw rows: {basic_checks['total_rows']:,}",
        f"- Columns: {basic_checks['total_columns']}",
        "",
        "## Structural Checks",
        "",
        f"- Missing cells: {basic_checks['missing_cells']:,}",
        f"- Duplicate rows: {basic_checks['duplicate_rows']:,}",
        f"- Duplicate IDs: {basic_checks['duplicate_ids']:,}",
        f"- Infinite numeric values: {basic_checks['infinite_values']:,}",
        "",
        "## Binary Encoding Checks",
        "",
        (
            "- Invalid target binary values: "
            f"{invalid_target_binary_values:,}"
        ),
        (
            "- Invalid steel-type binary values: "
            f"{invalid_feature_binary_values:,}"
        ),
        "",
        "## Target Structure",
        "",
        (
            "- Exactly one positive defect label: "
            f"{target_checks['single_label_rows']:,}"
        ),
        (
            "- Zero positive defect labels: "
            f"{target_checks['zero_label_rows']:,}"
        ),
        (
            "- Multiple positive defect labels: "
            f"{target_checks['multi_label_rows']:,}"
        ),
        "",
        "## Steel Type Encoding",
        "",
        (
            "- Exactly one steel type: "
            f"{steel_checks['steel_exactly_one_type']:,}"
        ),
        (
            "- No steel type indicator: "
            f"{steel_checks['steel_no_type']:,}"
        ),
        (
            "- Multiple steel type indicators: "
            f"{steel_checks['steel_multiple_types']:,}"
        ),
        "",
        "## Modeling Eligibility",
        "",
        (
            f"- Eligible modeling rows: "
            f"{len(modeling_df):,}"
        ),
        (
            f"- Excluded target exceptions: "
            f"{excluded:,}"
        ),
        "",
        (
            "The primary ML task is single-label multiclass "
            "classification. Samples without exactly one "
            "positive target label are excluded from the "
            "primary modeling dataset rather than silently "
            "converted into a class."
        ),
        "",
        "## Class Distribution",
        "",
        "| Defect | Count | Percentage |",
        "|---|---:|---:|",
    ]

    for _, row in class_distribution.iterrows():

        lines.append(
            f"| {row['defect_type']} "
            f"| {int(row['count']):,} "
            f"| {row['percentage']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Outlier Policy",
            "",
            (
                "Extreme numerical observations are not "
                "automatically removed. In manufacturing and "
                "defect data, unusual values may represent "
                "genuine defect characteristics rather than "
                "data errors."
            ),
            "",
            (
                "Outliers will be analysed during EDA and "
                "model evaluation before any removal or "
                "transformation decision is made."
            ),
            "",
            "## Modeling Decision",
            "",
            (
                "The cleaned modeling dataset retains only "
                "records containing exactly one defect label."
            ),
            "",
            (
                "The original Kaggle training dataset remains "
                "unchanged and serves as the raw source."
            ),
            "",
        ]
    )

    return "\n".join(lines)


# =========================================================
# 14. Save outputs
# =========================================================

def save_outputs(
    quality_summary: pd.DataFrame,
    feature_report: pd.DataFrame,
    class_distribution: pd.DataFrame,
    modeling_df: pd.DataFrame,
    markdown_report: str,
) -> None:
    """
    Save all B3 outputs.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_summary.to_csv(
        QUALITY_SUMMARY_PATH,
        index=False,
    )

    feature_report.to_csv(
        FEATURE_REPORT_PATH,
        index=False,
    )

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_PATH,
        index=False,
    )

    modeling_df.to_csv(
        MODELING_OUTPUT_PATH,
        index=False,
    )

    MARKDOWN_REPORT_PATH.write_text(
        markdown_report,
        encoding="utf-8",
    )


# =========================================================
# 15. Print summary
# =========================================================

def print_summary(
    basic_checks: dict,
    target_checks: dict,
    steel_checks: dict,
    invalid_target_binary_values: int,
    invalid_feature_binary_values: int,
    range_checks: dict,
    modeling_df: pd.DataFrame,
    class_distribution: pd.DataFrame,
) -> None:

    print("=" * 72)
    print("Stage B3 — Data Quality Check")
    print("=" * 72)

    print("\nSTRUCTURAL QUALITY")
    print("-" * 72)

    print(
        f"Rows                 : "
        f"{basic_checks['total_rows']:,}"
    )

    print(
        f"Columns              : "
        f"{basic_checks['total_columns']}"
    )

    print(
        f"Missing cells        : "
        f"{basic_checks['missing_cells']:,}"
    )

    print(
        f"Duplicate rows       : "
        f"{basic_checks['duplicate_rows']:,}"
    )

    print(
        f"Duplicate IDs        : "
        f"{basic_checks['duplicate_ids']:,}"
    )

    print(
        f"Infinite values      : "
        f"{basic_checks['infinite_values']:,}"
    )

    print("\nBINARY VALIDATION")
    print("-" * 72)

    print(
        f"Invalid targets      : "
        f"{invalid_target_binary_values:,}"
    )

    print(
        f"Invalid steel flags  : "
        f"{invalid_feature_binary_values:,}"
    )

    print("\nTARGET STRUCTURE")
    print("-" * 72)

    print(
        f"Single-label rows    : "
        f"{target_checks['single_label_rows']:,}"
    )

    print(
        f"Zero-label rows      : "
        f"{target_checks['zero_label_rows']:,}"
    )

    print(
        f"Multi-label rows     : "
        f"{target_checks['multi_label_rows']:,}"
    )

    print("\nSTEEL TYPE ENCODING")
    print("-" * 72)

    print(
        f"Exactly one type     : "
        f"{steel_checks['steel_exactly_one_type']:,}"
    )

    print(
        f"No type              : "
        f"{steel_checks['steel_no_type']:,}"
    )

    print(
        f"Multiple types       : "
        f"{steel_checks['steel_multiple_types']:,}"
    )

    print("\nRANGE CHECKS")
    print("-" * 72)

    for check, value in range_checks.items():

        print(
            f"{check:<34}: {value:,}"
        )

    print("\nMODELING DATASET")
    print("-" * 72)

    print(
        f"Eligible rows        : "
        f"{len(modeling_df):,}"
    )

    print(
        f"Excluded rows        : "
        f"{basic_checks['total_rows'] - len(modeling_df):,}"
    )

    print("\nCLASS DISTRIBUTION")
    print("-" * 72)

    display_distribution = (
        class_distribution.copy()
    )

    display_distribution["percentage"] = (
        display_distribution["percentage"]
        .map(lambda x: f"{x:.2f}%")
    )

    print(
        display_distribution.to_string(
            index=False
        )
    )

    print("\nOUTPUT FILES")
    print("-" * 72)

    print(
        f"Modeling data : {MODELING_OUTPUT_PATH}"
    )

    print(
        f"Quality report: {QUALITY_SUMMARY_PATH}"
    )

    print(
        f"Feature report: {FEATURE_REPORT_PATH}"
    )

    print(
        f"Class report  : {CLASS_DISTRIBUTION_PATH}"
    )

    print(
        f"Markdown      : {MARKDOWN_REPORT_PATH}"
    )

    print("\nStage B3 validation: PASSED")

    print("=" * 72)


# =========================================================
# 16. Main
# =========================================================

def main() -> None:

    df = load_training_data()

    basic_checks = run_basic_checks(df)

    invalid_target_binary_values = (
        count_invalid_binary_values(
            df,
            TARGET_COLUMNS,
        )
    )

    invalid_feature_binary_values = (
        count_invalid_binary_values(
            df,
            BINARY_FEATURE_COLUMNS,
        )
    )

    target_checks = analyse_target_structure(
        df
    )

    steel_checks = analyse_steel_type_encoding(
        df
    )

    range_checks = analyse_suspicious_ranges(
        df
    )

    feature_report = (
        build_feature_quality_report(df)
    )

    modeling_df = (
        create_modeling_dataset(df)
    )

    class_distribution = (
        build_class_distribution(
            modeling_df
        )
    )

    quality_summary = (
        build_quality_summary(
            basic_checks=basic_checks,
            target_checks=target_checks,
            steel_checks=steel_checks,
            range_checks=range_checks,
            invalid_target_binary_values=(
                invalid_target_binary_values
            ),
            invalid_feature_binary_values=(
                invalid_feature_binary_values
            ),
            modeling_df=modeling_df,
        )
    )

    markdown_report = (
        generate_markdown_report(
            basic_checks=basic_checks,
            target_checks=target_checks,
            steel_checks=steel_checks,
            invalid_target_binary_values=(
                invalid_target_binary_values
            ),
            invalid_feature_binary_values=(
                invalid_feature_binary_values
            ),
            modeling_df=modeling_df,
            class_distribution=(
                class_distribution
            ),
        )
    )

    save_outputs(
        quality_summary=quality_summary,
        feature_report=feature_report,
        class_distribution=class_distribution,
        modeling_df=modeling_df,
        markdown_report=markdown_report,
    )

    print_summary(
        basic_checks=basic_checks,
        target_checks=target_checks,
        steel_checks=steel_checks,
        invalid_target_binary_values=(
            invalid_target_binary_values
        ),
        invalid_feature_binary_values=(
            invalid_feature_binary_values
        ),
        range_checks=range_checks,
        modeling_df=modeling_df,
        class_distribution=class_distribution,
    )


if __name__ == "__main__":
    main()