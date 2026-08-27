from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "champion_model.joblib"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
    / "shap"
)

INDIVIDUAL_DIR = (
    REPORT_DIR
    / "individual_explanations"
)

DOCS_DIR = PROJECT_ROOT / "docs"

GLOBAL_IMPORTANCE_CSV_PATH = (
    REPORT_DIR
    / "global_shap_importance.csv"
)

GLOBAL_IMPORTANCE_IMAGE_PATH = (
    REPORT_DIR
    / "global_shap_importance.png"
)

PER_CLASS_IMPORTANCE_PATH = (
    REPORT_DIR
    / "per_class_shap_importance.csv"
)

PER_CLASS_TOP_FEATURES_PATH = (
    REPORT_DIR
    / "per_class_top_features.csv"
)

INDIVIDUAL_EXAMPLES_PATH = (
    REPORT_DIR
    / "individual_shap_examples.csv"
)

MARKDOWN_PATH = (
    DOCS_DIR
    / "shap_explainability.md"
)


# =========================================================
# 2. Dataset definitions
# =========================================================

TARGET_COLUMN = "defect_type"

ID_COLUMN = "id"

TARGET_BINARY_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]


# =========================================================
# 3. Load champion bundle
# =========================================================

def load_champion_bundle():
    """
    Load champion model selected during B12.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Champion model not found:\n{MODEL_PATH}"
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        bundle,
        dict,
    ):
        raise TypeError(
            "Champion model artifact must be a dictionary."
        )

    required_keys = [
        "model",
        "label_encoder",
        "feature_columns",
    ]

    missing = [
        key
        for key in required_keys
        if key not in bundle
    ]

    if missing:

        raise KeyError(
            f"Champion model bundle missing keys: {missing}"
        )

    return bundle


# =========================================================
# 4. Load test data
# =========================================================

def load_test_data():
    """
    Load held-out test split for post-hoc explanation.

    Model selection and final evaluation were already
    completed in B12.
    """

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    return pd.read_csv(
        TEST_PATH
    )


# =========================================================
# 5. Prepare model input
# =========================================================

def prepare_features(
    test_df,
    feature_columns,
):
    """
    Select the exact feature schema used by champion model.
    """

    missing = [
        feature
        for feature in feature_columns
        if feature not in test_df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing model features: {missing}"
        )

    return test_df[
        feature_columns
    ].copy()


# =========================================================
# 6. Build SHAP explainer
# =========================================================

def build_explainer(
    model,
):
    """
    Create tree-specific SHAP explainer for XGBoost.
    """

    explainer = shap.TreeExplainer(
        model
    )

    return explainer


# =========================================================
# 7. Calculate SHAP values
# =========================================================

def calculate_shap_values(
    explainer,
    X,
):
    """
    Compute SHAP values for multiclass XGBoost.
    """

    shap_values = explainer(
        X
    )

    values = shap_values.values

    if values.ndim != 3:

        raise ValueError(
            "Expected multiclass SHAP values with "
            "3 dimensions."
        )

    return shap_values


# =========================================================
# 8. Normalize SHAP orientation
# =========================================================

def normalize_shap_array(
    shap_values,
    number_of_features,
    number_of_classes,
):
    """
    Normalize SHAP array into:

    samples × features × classes

    SHAP versions may differ in multiclass output shape.
    """

    values = shap_values.values

    if (
        values.shape[1]
        == number_of_features
        and
        values.shape[2]
        == number_of_classes
    ):

        return values

    if (
        values.shape[1]
        == number_of_classes
        and
        values.shape[2]
        == number_of_features
    ):

        return np.transpose(
            values,
            (0, 2, 1),
        )

    raise ValueError(
        "Unable to determine SHAP multiclass array layout. "
        f"Received shape: {values.shape}"
    )


# =========================================================
# 9. Global SHAP importance
# =========================================================

def build_global_shap_importance(
    shap_array,
    feature_columns,
):
    """
    Global importance =
    mean absolute SHAP across samples and classes.
    """

    global_importance = (
        np.abs(
            shap_array
        )
        .mean(
            axis=(0, 2)
        )
    )

    df = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "mean_abs_shap":
                global_importance,
        }
    )

    df = (
        df
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "rank"
    ] = (
        df.index + 1
    )

    total = (
        df[
            "mean_abs_shap"
        ].sum()
    )

    if total > 0:

        df[
            "importance_share"
        ] = (
            df[
                "mean_abs_shap"
            ]
            / total
        )

    else:

        df[
            "importance_share"
        ] = 0.0

    return df


# =========================================================
# 10. Plot global SHAP importance
# =========================================================

def plot_global_shap_importance(
    global_df,
):
    """
    Save top-15 Global SHAP importance chart.
    """

    plot_df = (
        global_df
        .head(15)
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        plot_df[
            "feature"
        ],
        plot_df[
            "mean_abs_shap"
        ],
    )

    ax.set_xlabel(
        "Mean |SHAP Value|"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Champion Model — Global SHAP Importance"
    )

    fig.tight_layout()

    fig.savefig(
        GLOBAL_IMPORTANCE_IMAGE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# 11. Per-class SHAP importance
# =========================================================

def build_per_class_shap_importance(
    shap_array,
    feature_columns,
    class_names,
):
    """
    Calculate feature importance separately for each class.
    """

    records = []

    for class_index, class_name in enumerate(
        class_names
    ):

        class_values = (
            shap_array[
                :,
                :,
                class_index
            ]
        )

        class_importance = (
            np.abs(
                class_values
            )
            .mean(
                axis=0
            )
        )

        for feature, importance in zip(
            feature_columns,
            class_importance,
        ):

            records.append(
                {
                    "class":
                        class_name,

                    "feature":
                        feature,

                    "mean_abs_shap":
                        importance,
                }
            )

    df = pd.DataFrame(
        records
    )

    df[
        "rank"
    ] = (
        df
        .groupby(
            "class"
        )[
            "mean_abs_shap"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return (
        df
        .sort_values(
            [
                "class",
                "rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# 12. Extract per-class top features
# =========================================================

def build_per_class_top_features(
    per_class_df,
    top_n=5,
):
    """
    Keep top N SHAP features for each defect class.
    """

    return (
        per_class_df[
            per_class_df[
                "rank"
            ] <= top_n
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# =========================================================
# 13. Select individual explanation samples
# =========================================================

def select_individual_examples(
    test_df,
    model,
    X_test,
    label_encoder,
):
    """
    Select one correctly predicted example per defect class.

    Preference:
    highest-confidence correct prediction for each class.
    """

    encoded_predictions = (
        model.predict(
            X_test
        )
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    predicted_labels = (
        label_encoder.inverse_transform(
            encoded_predictions
        )
    )

    working_df = pd.DataFrame(
        {
            "row_position":
                np.arange(
                    len(test_df)
                ),

            "id":
                test_df[
                    ID_COLUMN
                ].values,

            "actual_defect":
                test_df[
                    TARGET_COLUMN
                ].values,

            "predicted_defect":
                predicted_labels,

            "confidence":
                probabilities.max(
                    axis=1
                ),
        }
    )

    correct_df = working_df[
        working_df[
            "actual_defect"
        ]
        ==
        working_df[
            "predicted_defect"
        ]
    ].copy()

    examples = []

    for class_name in (
        label_encoder.classes_
    ):

        candidates = (
            correct_df[
                correct_df[
                    "actual_defect"
                ]
                == class_name
            ]
            .sort_values(
                "confidence",
                ascending=False,
            )
        )

        if len(
            candidates
        ) == 0:

            continue

        examples.append(
            candidates.iloc[0]
        )

    return pd.DataFrame(
        examples
    )


# =========================================================
# 14. Individual SHAP explanation
# =========================================================

def build_individual_explanations(
    examples_df,
    shap_array,
    X_test,
    feature_columns,
    label_encoder,
):
    """
    Create top positive and negative SHAP contributions
    for selected individual predictions.
    """

    all_records = []

    class_name_to_index = {
        class_name: index
        for index, class_name
        in enumerate(
            label_encoder.classes_
        )
    }

    for _, example in (
        examples_df.iterrows()
    ):

        row_position = int(
            example[
                "row_position"
            ]
        )

        predicted_class = (
            example[
                "predicted_defect"
            ]
        )

        class_index = (
            class_name_to_index[
                predicted_class
            ]
        )

        row_shap = (
            shap_array[
                row_position,
                :,
                class_index
            ]
        )

        feature_values = (
            X_test.iloc[
                row_position
            ].values
        )

        detail_df = pd.DataFrame(
            {
                "feature":
                    feature_columns,

                "feature_value":
                    feature_values,

                "shap_value":
                    row_shap,
            }
        )

        detail_df[
            "abs_shap"
        ] = (
            detail_df[
                "shap_value"
            ].abs()
        )

        detail_df = (
            detail_df
            .sort_values(
                "abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        detail_df[
            "direction"
        ] = np.where(
            detail_df[
                "shap_value"
            ] > 0,
            "supports_prediction",
            "opposes_prediction",
        )

        top_10 = (
            detail_df
            .head(10)
            .copy()
        )

        top_10.insert(
            0,
            "id",
            example[
                "id"
            ],
        )

        top_10.insert(
            1,
            "actual_defect",
            example[
                "actual_defect"
            ],
        )

        top_10.insert(
            2,
            "predicted_defect",
            predicted_class,
        )

        top_10.insert(
            3,
            "confidence",
            example[
                "confidence"
            ],
        )

        filename = (
            f"id_{example['id']}"
            f"_{predicted_class}.csv"
        )

        top_10.to_csv(
            INDIVIDUAL_DIR
            / filename,
            index=False,
        )

        all_records.append(
            top_10
        )

    if not all_records:

        return pd.DataFrame()

    return pd.concat(
        all_records,
        ignore_index=True,
    )


# =========================================================
# 15. Build compact individual examples table
# =========================================================

def build_individual_summary(
    examples_df,
    individual_df,
):
    """
    Produce one compact summary row per selected sample.
    """

    records = []

    for _, example in (
        examples_df.iterrows()
    ):

        sample_df = (
            individual_df[
                individual_df[
                    "id"
                ]
                == example[
                    "id"
                ]
            ]
        )

        supports = (
            sample_df[
                sample_df[
                    "shap_value"
                ] > 0
            ]
            .head(3)
        )

        opposes = (
            sample_df[
                sample_df[
                    "shap_value"
                ] < 0
            ]
            .head(3)
        )

        records.append(
            {
                "id":
                    example[
                        "id"
                    ],

                "actual_defect":
                    example[
                        "actual_defect"
                    ],

                "predicted_defect":
                    example[
                        "predicted_defect"
                    ],

                "confidence":
                    example[
                        "confidence"
                    ],

                "top_supporting_features":
                    ", ".join(
                        supports[
                            "feature"
                        ].tolist()
                    ),

                "top_opposing_features":
                    ", ".join(
                        opposes[
                            "feature"
                        ].tolist()
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 16. Save outputs
# =========================================================

def save_outputs(
    global_df,
    per_class_df,
    per_class_top_df,
    individual_summary_df,
):
    """
    Save SHAP reports.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INDIVIDUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_df.to_csv(
        GLOBAL_IMPORTANCE_CSV_PATH,
        index=False,
    )

    per_class_df.to_csv(
        PER_CLASS_IMPORTANCE_PATH,
        index=False,
    )

    per_class_top_df.to_csv(
        PER_CLASS_TOP_FEATURES_PATH,
        index=False,
    )

    individual_summary_df.to_csv(
        INDIVIDUAL_EXAMPLES_PATH,
        index=False,
    )


# =========================================================
# 17. Markdown report
# =========================================================

def generate_markdown_report(
    global_df,
    per_class_top_df,
    individual_summary_df,
):
    """
    Create GitHub-friendly SHAP report.
    """

    lines = [
        "# SHAP Explainability",
        "",
        "## Purpose",
        "",
        (
            "SHAP is used to explain how the selected "
            "Tuned Weighted XGBoost model uses structured "
            "quality features when classifying steel defects."
        ),
        "",
        "## Global SHAP Importance",
        "",
        global_df[
            [
                "rank",
                "feature",
                "mean_abs_shap",
                "importance_share",
            ]
        ]
        .head(15)
        .to_markdown(
            index=False
        ),
        "",
        "## Top Features by Defect Class",
        "",
        per_class_top_df[
            [
                "class",
                "rank",
                "feature",
                "mean_abs_shap",
            ]
        ].to_markdown(
            index=False
        ),
        "",
        "## Individual Prediction Examples",
        "",
        individual_summary_df.to_markdown(
            index=False
        ),
        "",
        "## Interpretation Guardrail",
        "",
        (
            "Positive SHAP values indicate that a feature "
            "pushes the model output toward the explained "
            "class, while negative SHAP values push the "
            "model output away from that class."
        ),
        "",
        (
            "**SHAP explains model behavior, not physical "
            "manufacturing causality.**"
        ),
        "",
        (
            "A high SHAP contribution must therefore be "
            "interpreted as predictive evidence used by the "
            "model, not as confirmed root-cause evidence."
        ),
        "",
        "## Intended System Use",
        "",
        (
            "These explanations will later support the "
            "`explain_prediction()` backend tool so the "
            "AI Copilot can provide grounded model evidence "
            "to manufacturing engineers."
        ),
        "",
    ]

    return "\n".join(
        lines
    )


# =========================================================
# 18. Print summary
# =========================================================

def print_summary(
    global_df,
    per_class_top_df,
    individual_summary_df,
):
    """
    Print important B14 results.
    """

    print("=" * 72)

    print(
        "Stage B14 — SHAP Explainability"
    )

    print("=" * 72)

    print("\nGLOBAL TOP 15 SHAP FEATURES")
    print("-" * 72)

    print(
        global_df[
            [
                "rank",
                "feature",
                "mean_abs_shap",
                "importance_share",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print("\nTOP 5 FEATURES PER CLASS")
    print("-" * 72)

    print(
        per_class_top_df[
            [
                "class",
                "rank",
                "feature",
                "mean_abs_shap",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nINDIVIDUAL EXPLANATION EXAMPLES")
    print("-" * 72)

    print(
        individual_summary_df.to_string(
            index=False
        )
    )

    print("\nGUARDRAIL")
    print("-" * 72)

    print(
        "SHAP explains model behavior, "
        "NOT manufacturing causality."
    )

    print(
        "\nSHAP explainability: PASSED"
    )

    print("=" * 72)


# =========================================================
# 19. Main
# =========================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INDIVIDUAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = (
        load_champion_bundle()
    )

    model = (
        bundle[
            "model"
        ]
    )

    label_encoder = (
        bundle[
            "label_encoder"
        ]
    )

    feature_columns = list(
        bundle[
            "feature_columns"
        ]
    )

    test_df = (
        load_test_data()
    )

    X_test = (
        prepare_features(
            test_df,
            feature_columns,
        )
    )

    explainer = (
        build_explainer(
            model
        )
    )

    shap_values = (
        calculate_shap_values(
            explainer,
            X_test,
        )
    )

    shap_array = (
        normalize_shap_array(
            shap_values=(
                shap_values
            ),
            number_of_features=(
                len(
                    feature_columns
                )
            ),
            number_of_classes=(
                len(
                    label_encoder.classes_
                )
            ),
        )
    )

    global_df = (
        build_global_shap_importance(
            shap_array=(
                shap_array
            ),
            feature_columns=(
                feature_columns
            ),
        )
    )

    per_class_df = (
        build_per_class_shap_importance(
            shap_array=(
                shap_array
            ),
            feature_columns=(
                feature_columns
            ),
            class_names=(
                label_encoder.classes_
            ),
        )
    )

    per_class_top_df = (
        build_per_class_top_features(
            per_class_df,
            top_n=5,
        )
    )

    examples_df = (
        select_individual_examples(
            test_df=test_df,
            model=model,
            X_test=X_test,
            label_encoder=(
                label_encoder
            ),
        )
    )

    individual_df = (
        build_individual_explanations(
            examples_df=(
                examples_df
            ),
            shap_array=(
                shap_array
            ),
            X_test=X_test,
            feature_columns=(
                feature_columns
            ),
            label_encoder=(
                label_encoder
            ),
        )
    )

    individual_summary_df = (
        build_individual_summary(
            examples_df=(
                examples_df
            ),
            individual_df=(
                individual_df
            ),
        )
    )

    plot_global_shap_importance(
        global_df
    )

    save_outputs(
        global_df=(
            global_df
        ),
        per_class_df=(
            per_class_df
        ),
        per_class_top_df=(
            per_class_top_df
        ),
        individual_summary_df=(
            individual_summary_df
        ),
    )

    markdown = (
        generate_markdown_report(
            global_df=(
                global_df
            ),
            per_class_top_df=(
                per_class_top_df
            ),
            individual_summary_df=(
                individual_summary_df
            ),
        )
    )

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print_summary(
        global_df=(
            global_df
        ),
        per_class_top_df=(
            per_class_top_df
        ),
        individual_summary_df=(
            individual_summary_df
        ),
    )


if __name__ == "__main__":
    main()