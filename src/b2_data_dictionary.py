from pathlib import Path

import pandas as pd


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
METADATA_DIR = PROJECT_ROOT / "metadata"

TRAIN_PATH = DATA_DIR / "train.csv"

CSV_OUTPUT_PATH = METADATA_DIR / "data_dictionary.csv"
MARKDOWN_OUTPUT_PATH = DOCS_DIR / "data_dictionary.md"


# =========================================================
# 2. Target columns
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


# =========================================================
# 3. Feature metadata
# =========================================================

FEATURE_METADATA = {
    "id": {
        "role": "Identifier",
        "feature_group": "Identifier",
        "description": (
            "Unique sample identifier provided by the Kaggle dataset."
        ),
        "model_usage": "Excluded from model training",
    },

    # -----------------------------------------------------
    # Geometry / position
    # -----------------------------------------------------

    "X_Minimum": {
        "role": "Feature",
        "feature_group": "Geometry / Position",
        "description": (
            "Minimum x-coordinate associated with the detected fault region."
        ),
        "model_usage": "Predictor",
    },

    "X_Maximum": {
        "role": "Feature",
        "feature_group": "Geometry / Position",
        "description": (
            "Maximum x-coordinate associated with the detected fault region."
        ),
        "model_usage": "Predictor",
    },

    "Y_Minimum": {
        "role": "Feature",
        "feature_group": "Geometry / Position",
        "description": (
            "Minimum y-coordinate associated with the detected fault region."
        ),
        "model_usage": "Predictor",
    },

    "Y_Maximum": {
        "role": "Feature",
        "feature_group": "Geometry / Position",
        "description": (
            "Maximum y-coordinate associated with the detected fault region."
        ),
        "model_usage": "Predictor",
    },

    "Pixels_Areas": {
        "role": "Feature",
        "feature_group": "Geometry / Position",
        "description": (
            "Area of the detected fault region measured in pixels."
        ),
        "model_usage": "Predictor",
    },

    "X_Perimeter": {
        "role": "Feature",
        "feature_group": "Geometry / Position",
        "description": (
            "Horizontal perimeter-related measurement of the fault region."
        ),
        "model_usage": "Predictor",
    },

    "Y_Perimeter": {
        "role": "Feature",
        "feature_group": "Geometry / Position",
        "description": (
            "Vertical perimeter-related measurement of the fault region."
        ),
        "model_usage": "Predictor",
    },

    # -----------------------------------------------------
    # Luminosity
    # -----------------------------------------------------

    "Sum_of_Luminosity": {
        "role": "Feature",
        "feature_group": "Luminosity",
        "description": (
            "Sum of pixel luminosity values within the fault region."
        ),
        "model_usage": "Predictor",
    },

    "Minimum_of_Luminosity": {
        "role": "Feature",
        "feature_group": "Luminosity",
        "description": (
            "Minimum observed luminosity value within the fault region."
        ),
        "model_usage": "Predictor",
    },

    "Maximum_of_Luminosity": {
        "role": "Feature",
        "feature_group": "Luminosity",
        "description": (
            "Maximum observed luminosity value within the fault region."
        ),
        "model_usage": "Predictor",
    },

    "Luminosity_Index": {
        "role": "Feature",
        "feature_group": "Luminosity",
        "description": (
            "Derived luminosity-related index describing the fault region."
        ),
        "model_usage": "Predictor",
    },

    # -----------------------------------------------------
    # Steel / production characteristics
    # -----------------------------------------------------

    "Length_of_Conveyer": {
        "role": "Feature",
        "feature_group": "Steel / Production",
        "description": (
            "Conveyor length-related variable provided in the dataset."
        ),
        "model_usage": "Predictor",
    },

    "TypeOfSteel_A300": {
        "role": "Feature",
        "feature_group": "Steel / Production",
        "description": (
            "Binary indicator representing steel type A300."
        ),
        "model_usage": "Predictor",
    },

    "TypeOfSteel_A400": {
        "role": "Feature",
        "feature_group": "Steel / Production",
        "description": (
            "Binary indicator representing steel type A400."
        ),
        "model_usage": "Predictor",
    },

    "Steel_Plate_Thickness": {
        "role": "Feature",
        "feature_group": "Steel / Production",
        "description": (
            "Steel plate thickness measurement provided by the dataset."
        ),
        "model_usage": "Predictor",
    },

    # -----------------------------------------------------
    # Shape / edge indices
    # -----------------------------------------------------

    "Edges_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived edge-related index describing fault geometry."
        ),
        "model_usage": "Predictor",
    },

    "Empty_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived index related to empty space within the fault region."
        ),
        "model_usage": "Predictor",
    },

    "Square_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived index describing how square-like the fault region is."
        ),
        "model_usage": "Predictor",
    },

    "Outside_X_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived horizontal outside-region index."
        ),
        "model_usage": "Predictor",
    },

    "Edges_X_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived index describing horizontal edge characteristics."
        ),
        "model_usage": "Predictor",
    },

    "Edges_Y_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived index describing vertical edge characteristics."
        ),
        "model_usage": "Predictor",
    },

    "Outside_Global_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived global outside-region index."
        ),
        "model_usage": "Predictor",
    },

    "Orientation_Index": {
        "role": "Feature",
        "feature_group": "Shape / Edge Index",
        "description": (
            "Derived index describing orientation characteristics "
            "of the fault region."
        ),
        "model_usage": "Predictor",
    },

    # -----------------------------------------------------
    # Log-transformed features
    # -----------------------------------------------------

    "LogOfAreas": {
        "role": "Feature",
        "feature_group": "Log Transformation",
        "description": (
            "Log-transformed representation of fault area."
        ),
        "model_usage": "Predictor",
    },

    "Log_X_Index": {
        "role": "Feature",
        "feature_group": "Log Transformation",
        "description": (
            "Log-transformed horizontal fault-related index."
        ),
        "model_usage": "Predictor",
    },

    "Log_Y_Index": {
        "role": "Feature",
        "feature_group": "Log Transformation",
        "description": (
            "Log-transformed vertical fault-related index."
        ),
        "model_usage": "Predictor",
    },

    # -----------------------------------------------------
    # Other transformation
    # -----------------------------------------------------

    "SigmoidOfAreas": {
        "role": "Feature",
        "feature_group": "Area Transformation",
        "description": (
            "Sigmoid-transformed representation of fault area."
        ),
        "model_usage": "Predictor",
    },
}


# =========================================================
# 4. Target metadata
# =========================================================

TARGET_DESCRIPTIONS = {
    "Pastry": "Binary indicator for the Pastry defect category.",
    "Z_Scratch": "Binary indicator for the Z_Scratch defect category.",
    "K_Scatch": "Binary indicator for the K_Scatch defect category.",
    "Stains": "Binary indicator for the Stains defect category.",
    "Dirtiness": "Binary indicator for the Dirtiness defect category.",
    "Bumps": "Binary indicator for the Bumps defect category.",
    "Other_Faults": (
        "Binary indicator for defects grouped under Other_Faults."
    ),
}


# =========================================================
# 5. Load dataset
# =========================================================

def load_training_data() -> pd.DataFrame:
    """
    Load the Kaggle training dataset.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.csv was not found at:\n{TRAIN_PATH}"
        )

    return pd.read_csv(TRAIN_PATH)


# =========================================================
# 6. Build data dictionary
# =========================================================

def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a structured data dictionary using actual dataframe
    dtypes and predefined semantic metadata.
    """

    records = []

    for column in df.columns:

        if column in TARGET_COLUMNS:

            metadata = {
                "role": "Target",
                "feature_group": "Defect Target",
                "description": TARGET_DESCRIPTIONS[column],
                "model_usage": (
                    "Used to construct multiclass target"
                ),
            }

        elif column in FEATURE_METADATA:

            metadata = FEATURE_METADATA[column]

        else:
            raise ValueError(
                f"No metadata definition found for column: {column}"
            )

        records.append(
            {
                "column_name": column,
                "data_type": str(df[column].dtype),
                "role": metadata["role"],
                "feature_group": metadata["feature_group"],
                "description": metadata["description"],
                "model_usage": metadata["model_usage"],
            }
        )

    return pd.DataFrame(records)


# =========================================================
# 7. Validate dictionary
# =========================================================

def validate_data_dictionary(
    df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
) -> None:
    """
    Verify that every dataset column is represented exactly once
    in the data dictionary.
    """

    dataset_columns = set(df.columns)

    dictionary_columns = set(
        dictionary_df["column_name"]
    )

    missing_from_dictionary = (
        dataset_columns - dictionary_columns
    )

    extra_in_dictionary = (
        dictionary_columns - dataset_columns
    )

    duplicated_dictionary_columns = (
        dictionary_df["column_name"]
        .duplicated()
        .sum()
    )

    if missing_from_dictionary:
        raise ValueError(
            "Columns missing from dictionary: "
            f"{sorted(missing_from_dictionary)}"
        )

    if extra_in_dictionary:
        raise ValueError(
            "Unexpected dictionary columns: "
            f"{sorted(extra_in_dictionary)}"
        )

    if duplicated_dictionary_columns > 0:
        raise ValueError(
            "Duplicate column definitions detected "
            "in the data dictionary."
        )


# =========================================================
# 8. Save CSV
# =========================================================

def save_data_dictionary_csv(
    dictionary_df: pd.DataFrame,
) -> None:
    """
    Save machine-readable metadata.
    """

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictionary_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
    )


# =========================================================
# 9. Generate Markdown documentation
# =========================================================

def generate_markdown(
    dictionary_df: pd.DataFrame,
) -> str:
    """
    Generate GitHub-friendly Markdown documentation.
    """

    lines = [
        "# Steel Quality Data Dictionary",
        "",
        "## Dataset Structure",
        "",
        "The dataset contains:",
        "",
        "- 1 sample identifier",
        "- 27 predictor features",
        "- 7 binary defect target columns",
        "",
        "The primary machine-learning task will later "
        "convert eligible records into a single-label "
        "multiclass classification dataset.",
        "",
        "## Important Interpretation Note",
        "",
        "Several variables are derived geometric or "
        "image-related indicators. Public documentation "
        "does not provide full manufacturing definitions "
        "for every derived index, so these variables are "
        "described conservatively and should not be "
        "interpreted as causal production parameters.",
        "",
        "## Columns",
        "",
        "| Column | Type | Role | Group | Description | Model Usage |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in dictionary_df.iterrows():

        line = (
            f"| {row['column_name']} "
            f"| {row['data_type']} "
            f"| {row['role']} "
            f"| {row['feature_group']} "
            f"| {row['description']} "
            f"| {row['model_usage']} |"
        )

        lines.append(line)

    lines.extend(
        [
            "",
            "## Target Categories",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        lines.append(
            f"- `{target}`"
        )

    lines.extend(
        [
            "",
            "## Modeling Note",
            "",
            "`id` is used only as a record identifier "
            "and will not be included as a predictive feature.",
            "",
            "Samples without exactly one positive target "
            "label will be handled as data-quality exceptions "
            "during Stage B3.",
            "",
        ]
    )

    return "\n".join(lines)


# =========================================================
# 10. Save Markdown
# =========================================================

def save_markdown(
    markdown_text: str,
) -> None:
    """
    Save human-readable documentation.
    """

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MARKDOWN_OUTPUT_PATH.write_text(
        markdown_text,
        encoding="utf-8",
    )


# =========================================================
# 11. Print summary
# =========================================================

def print_summary(
    dictionary_df: pd.DataFrame,
) -> None:

    print("=" * 70)
    print("Stage B2 — Data Dictionary")
    print("=" * 70)

    print(
        f"\nTotal documented columns : "
        f"{len(dictionary_df)}"
    )

    print("\nROLE DISTRIBUTION")
    print("-" * 70)

    print(
        dictionary_df["role"]
        .value_counts()
        .to_string()
    )

    print("\nFEATURE GROUPS")
    print("-" * 70)

    feature_rows = dictionary_df[
        dictionary_df["role"] == "Feature"
    ]

    print(
        feature_rows["feature_group"]
        .value_counts()
        .to_string()
    )

    print("\nOUTPUT FILES")
    print("-" * 70)

    print(
        f"CSV      : {CSV_OUTPUT_PATH}"
    )

    print(
        f"Markdown : {MARKDOWN_OUTPUT_PATH}"
    )

    print("\nData dictionary validation: PASSED")

    print("=" * 70)


# =========================================================
# 12. Main
# =========================================================

def main() -> None:

    df = load_training_data()

    dictionary_df = build_data_dictionary(df)

    validate_data_dictionary(
        df,
        dictionary_df,
    )

    save_data_dictionary_csv(
        dictionary_df
    )

    markdown_text = generate_markdown(
        dictionary_df
    )

    save_markdown(
        markdown_text
    )

    print_summary(
        dictionary_df
    )


if __name__ == "__main__":
    main()