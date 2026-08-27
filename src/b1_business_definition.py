from pathlib import Path

import pandas as pd


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.csv"


# =========================================================
# 2. Dataset definition
# =========================================================

TARGET_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

ID_COLUMN = "id"


# =========================================================
# 3. Load dataset
# =========================================================

def load_training_data() -> pd.DataFrame:
    """
    Load the Kaggle Steel Plate Defect Prediction training dataset.

    Returns
    -------
    pd.DataFrame
        Raw training dataframe.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.csv was not found at:\n{TRAIN_PATH}\n"
            "Please place the Kaggle training data inside the data/ folder."
        )

    return pd.read_csv(TRAIN_PATH)


# =========================================================
# 4. Validate expected schema
# =========================================================

def validate_schema(df: pd.DataFrame) -> None:
    """
    Confirm that the expected ID and target columns exist.
    """

    required_columns = [ID_COLUMN] + TARGET_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )


# =========================================================
# 5. Analyse target structure
# =========================================================

def analyse_target_structure(df: pd.DataFrame) -> dict:
    """
    Analyse how many defect labels are assigned to each sample.

    A valid sample for the primary multiclass task should have
    exactly one positive defect label.

    Returns
    -------
    dict
        Summary counts for single-label, zero-label,
        and multi-label samples.
    """

    target_sum = df[TARGET_COLUMNS].sum(axis=1)

    single_label_count = int((target_sum == 1).sum())
    zero_label_count = int((target_sum == 0).sum())
    multi_label_count = int((target_sum > 1).sum())

    return {
        "total_samples": len(df),
        "single_label_samples": single_label_count,
        "zero_label_samples": zero_label_count,
        "multi_label_samples": multi_label_count,
    }


# =========================================================
# 6. Analyse feature structure
# =========================================================

def analyse_feature_structure(df: pd.DataFrame) -> dict:
    """
    Identify predictor columns after excluding ID and targets.
    """

    feature_columns = [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS + [ID_COLUMN]
    ]

    return {
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }


# =========================================================
# 7. Defect distribution
# =========================================================

def calculate_defect_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Count positive labels for each defect category.
    """

    return (
        df[TARGET_COLUMNS]
        .sum()
        .astype(int)
        .sort_values(ascending=False)
    )


# =========================================================
# 8. Print business problem evidence
# =========================================================

def print_business_summary(
    target_summary: dict,
    feature_summary: dict,
    defect_distribution: pd.Series,
) -> None:
    """
    Print the dataset evidence supporting the Project B
    business problem definition.
    """

    print("=" * 70)
    print("Steel Quality Prediction, Explainability & AI Analytics Copilot")
    print("Stage B1 — Business Problem Definition")
    print("=" * 70)

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Total training samples : "
        f"{target_summary['total_samples']:,}"
    )

    print(
        f"Predictor features     : "
        f"{feature_summary['feature_count']}"
    )

    print(
        f"Defect categories      : "
        f"{len(TARGET_COLUMNS)}"
    )

    print("\nTARGET STRUCTURE")
    print("-" * 70)

    print(
        f"Exactly one label      : "
        f"{target_summary['single_label_samples']:,}"
    )

    print(
        f"No positive label      : "
        f"{target_summary['zero_label_samples']:,}"
    )

    print(
        f"Multiple labels        : "
        f"{target_summary['multi_label_samples']:,}"
    )

    print("\nDEFECT DISTRIBUTION")
    print("-" * 70)

    print(defect_distribution.to_string())

    print("\nPRIMARY ML TASK")
    print("-" * 70)

    print(
        "Single-label multiclass steel defect classification"
    )

    print("\nBUSINESS OBJECTIVE")
    print("-" * 70)

    print(
        "Predict the most likely steel defect category from "
        "structured quality characteristics to support "
        "manufacturing quality inspection triage."
    )

    print("\nSYSTEM RESPONSIBILITIES")
    print("-" * 70)

    print("SQL       : Query factual quality data and analytics")
    print("ML        : Predict the most likely defect category")
    print("SHAP      : Explain model drivers")
    print("LLM       : Select tools and summarize grounded results")
    print("Engineer  : Make the final quality decision")

    print("\nDATA QUALITY DECISION")
    print("-" * 70)

    print(
        "Samples without exactly one defect label will not be "
        "silently converted into multiclass labels."
    )

    print(
        "They will be investigated and documented during "
        "Stage B3 Data Quality."
    )

    print("\n" + "=" * 70)


# =========================================================
# 9. Main
# =========================================================

def main() -> None:
    df = load_training_data()

    validate_schema(df)

    target_summary = analyse_target_structure(df)

    feature_summary = analyse_feature_structure(df)

    defect_distribution = calculate_defect_distribution(df)

    print_business_summary(
        target_summary=target_summary,
        feature_summary=feature_summary,
        defect_distribution=defect_distribution,
    )


if __name__ == "__main__":
    main()