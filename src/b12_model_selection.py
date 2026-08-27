from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "model_selection"
)

DOCS_DIR = PROJECT_ROOT / "docs"

CHAMPION_MODEL_PATH = (
    MODEL_DIR
    / "champion_model.joblib"
)

VALIDATION_COMPARISON_PATH = (
    REPORT_DIR
    / "validation_model_comparison.csv"
)

FINAL_TEST_METRICS_PATH = (
    REPORT_DIR
    / "final_test_metrics.csv"
)

FINAL_TEST_CLASSIFICATION_REPORT_PATH = (
    REPORT_DIR
    / "final_test_classification_report.csv"
)

FINAL_TEST_CONFUSION_MATRIX_CSV_PATH = (
    REPORT_DIR
    / "final_test_confusion_matrix.csv"
)

FINAL_TEST_CONFUSION_MATRIX_IMAGE_PATH = (
    REPORT_DIR
    / "final_test_confusion_matrix.png"
)

MODEL_SELECTION_SUMMARY_PATH = (
    REPORT_DIR
    / "model_selection_summary.csv"
)

MARKDOWN_PATH = (
    DOCS_DIR
    / "model_selection.md"
)


# =========================================================
# 2. Model / report paths
# =========================================================

MODEL_CONFIGS = {
    "logistic_regression": {
        "model_path":
            MODEL_DIR
            / "logistic_regression_baseline.joblib",

        "metrics_path":
            PROJECT_ROOT
            / "reports"
            / "models"
            / "logistic_regression"
            / "metrics.csv",
    },

    "random_forest": {
        "model_path":
            MODEL_DIR
            / "random_forest_baseline.joblib",

        "metrics_path":
            PROJECT_ROOT
            / "reports"
            / "models"
            / "random_forest"
            / "metrics.csv",
    },

    "xgboost_baseline": {
        "model_path":
            MODEL_DIR
            / "xgboost_baseline.joblib",

        "metrics_path":
            PROJECT_ROOT
            / "reports"
            / "models"
            / "xgboost"
            / "metrics.csv",
    },

    "xgboost_weighted": {
        "model_path":
            MODEL_DIR
            / "xgboost_weighted.joblib",

        "metrics_path":
            PROJECT_ROOT
            / "reports"
            / "models"
            / "xgboost_weighted"
            / "metrics.csv",
    },

    "xgboost_tuned": {
        "model_path":
            MODEL_DIR
            / "xgboost_tuned.joblib",

        "metrics_path":
            PROJECT_ROOT
            / "reports"
            / "models"
            / "xgboost_tuned"
            / "validation_metrics.csv",
    },
}


# =========================================================
# 3. Dataset definition
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
# 4. Load validation metrics
# =========================================================

def load_validation_metrics():
    """
    Load validation results produced by B7-B11.

    Test data is not used here.
    """

    rows = []

    for model_name, config in (
        MODEL_CONFIGS.items()
    ):

        path = config["metrics_path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Metrics file missing:\n{path}"
            )

        metrics_df = pd.read_csv(
            path
        )

        metrics = dict(
            zip(
                metrics_df["metric"],
                metrics_df["value"],
            )
        )

        rows.append(
            {
                "model": model_name,
                "accuracy":
                    metrics["accuracy"],
                "macro_precision":
                    metrics["macro_precision"],
                "macro_recall":
                    metrics["macro_recall"],
                "macro_f1":
                    metrics["macro_f1"],
                "weighted_f1":
                    metrics["weighted_f1"],
            }
        )

    comparison_df = pd.DataFrame(
        rows
    )

    return comparison_df


# =========================================================
# 5. Select champion from validation only
# =========================================================

def select_champion(
    comparison_df,
):
    """
    Select champion using validation Macro F1 as the
    primary metric and Macro Recall as tie-breaker.

    Test metrics are not involved in model selection.
    """

    ranked_df = (
        comparison_df
        .sort_values(
            by=[
                "macro_f1",
                "macro_recall",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    champion_name = (
        ranked_df.loc[
            0,
            "model"
        ]
    )

    return (
        champion_name,
        ranked_df,
    )


# =========================================================
# 6. Load champion model bundle
# =========================================================

def load_champion_bundle(
    champion_name,
):
    """
    Load selected model artifact.
    """

    model_path = (
        MODEL_CONFIGS[
            champion_name
        ]["model_path"]
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file missing:\n{model_path}"
        )

    bundle = joblib.load(
        model_path
    )

    return (
        bundle,
        model_path,
    )


# =========================================================
# 7. Load final test set
# =========================================================

def load_test_data():
    """
    Load held-out test split.

    This is the first evaluation stage where the
    test set is used.
    """

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test split missing:\n{TEST_PATH}"
        )

    return pd.read_csv(
        TEST_PATH
    )


# =========================================================
# 8. Extract bundle components
# =========================================================

def extract_bundle_components(
    champion_name,
    bundle,
):
    """
    Normalize different saved model formats.

    XGBoost models were saved as dictionaries.
    Logistic / RF may have been saved directly or
    as dictionaries depending on earlier stages.
    """

    if isinstance(bundle, dict):

        model = bundle.get(
            "model"
        )

        label_encoder = bundle.get(
            "label_encoder"
        )

        feature_columns = bundle.get(
            "feature_columns"
        )

    else:

        model = bundle
        label_encoder = None
        feature_columns = None

    if model is None:
        raise ValueError(
            f"Unable to extract model from "
            f"{champion_name} artifact."
        )

    return (
        model,
        label_encoder,
        feature_columns,
    )


# =========================================================
# 9. Infer feature columns if needed
# =========================================================

def get_feature_columns(
    test_df,
    saved_feature_columns,
):
    """
    Prefer feature schema saved with the model.

    Otherwise reconstruct the 27 predictor columns.
    """

    if saved_feature_columns:
        return list(
            saved_feature_columns
        )

    excluded = (
        [ID_COLUMN, TARGET_COLUMN]
        + TARGET_BINARY_COLUMNS
    )

    return [
        column
        for column in test_df.columns
        if column not in excluded
    ]


# =========================================================
# 10. Prepare target representation
# =========================================================

def prepare_test_target(
    test_df,
    label_encoder,
):
    """
    Encode test labels when the champion model uses
    numeric class IDs.

    For XGBoost, LabelEncoder is present.
    """

    y_text = (
        test_df[
            TARGET_COLUMN
        ].copy()
    )

    if label_encoder is not None:

        y_model = (
            label_encoder.transform(
                y_text
            )
        )

    else:

        y_model = y_text

    return (
        y_text,
        y_model,
    )


# =========================================================
# 11. Final test evaluation
# =========================================================

def evaluate_test(
    model,
    X_test,
    y_test_model,
    y_test_text,
    label_encoder,
):
    """
    Perform one-time final holdout evaluation.
    """

    predictions_model = (
        model.predict(
            X_test
        )
    )

    if label_encoder is not None:

        predictions_text = (
            label_encoder.inverse_transform(
                predictions_model
            )
        )

    else:

        predictions_text = (
            predictions_model
        )

    metrics = {
        "accuracy":
            accuracy_score(
                y_test_text,
                predictions_text,
            ),

        "macro_precision":
            precision_score(
                y_test_text,
                predictions_text,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_test_text,
                predictions_text,
                average="macro",
                zero_division=0,
            ),

        "macro_f1":
            f1_score(
                y_test_text,
                predictions_text,
                average="macro",
                zero_division=0,
            ),

        "weighted_f1":
            f1_score(
                y_test_text,
                predictions_text,
                average="weighted",
                zero_division=0,
            ),
    }

    return (
        predictions_text,
        metrics,
    )


# =========================================================
# 12. Classification report
# =========================================================

def build_classification_report(
    y_test,
    predictions,
):
    """
    Build per-class final test report.
    """

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(
            report
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class"
            }
        )
    )


# =========================================================
# 13. Confusion matrix
# =========================================================

def build_confusion_matrix(
    y_test,
    predictions,
):
    """
    Build final test confusion matrix.
    """

    class_names = sorted(
        y_test.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_names,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    return (
        matrix_df,
        class_names,
    )


# =========================================================
# 14. Plot confusion matrix
# =========================================================

def plot_confusion_matrix(
    matrix_df,
    class_names,
):
    """
    Save final test confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        matrix_df.values,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(class_names))
    )

    ax.set_yticks(
        range(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        class_names
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "Champion Model — "
        "Final Holdout Test Confusion Matrix"
    )

    for i in range(
        len(class_names)
    ):

        for j in range(
            len(class_names)
        ):

            ax.text(
                j,
                i,
                matrix_df.iloc[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        FINAL_TEST_CONFUSION_MATRIX_IMAGE_PATH,
        dpi=160,
    )

    plt.close(fig)


# =========================================================
# 15. Build model selection summary
# =========================================================

def build_selection_summary(
    champion_name,
    ranked_validation_df,
    test_metrics,
):
    """
    Create concise champion-model decision record.
    """

    champion_validation = (
        ranked_validation_df[
            ranked_validation_df[
                "model"
            ] == champion_name
        ]
        .iloc[0]
    )

    summary_df = pd.DataFrame(
        [
            {
                "champion_model":
                    champion_name,

                "selection_metric":
                    "validation_macro_f1",

                "validation_accuracy":
                    champion_validation[
                        "accuracy"
                    ],

                "validation_macro_recall":
                    champion_validation[
                        "macro_recall"
                    ],

                "validation_macro_f1":
                    champion_validation[
                        "macro_f1"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "test_macro_recall":
                    test_metrics[
                        "macro_recall"
                    ],

                "test_macro_f1":
                    test_metrics[
                        "macro_f1"
                    ],
            }
        ]
    )

    return summary_df


# =========================================================
# 16. Save outputs
# =========================================================

def save_outputs(
    champion_model_path,
    ranked_validation_df,
    test_metrics,
    classification_report_df,
    confusion_matrix_df,
    selection_summary_df,
):
    """
    Save all final model-selection artifacts.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranked_validation_df.to_csv(
        VALIDATION_COMPARISON_PATH,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value
            in test_metrics.items()
        ]
    ).to_csv(
        FINAL_TEST_METRICS_PATH,
        index=False,
    )

    classification_report_df.to_csv(
        FINAL_TEST_CLASSIFICATION_REPORT_PATH,
        index=False,
    )

    confusion_matrix_df.to_csv(
        FINAL_TEST_CONFUSION_MATRIX_CSV_PATH,
    )

    selection_summary_df.to_csv(
        MODEL_SELECTION_SUMMARY_PATH,
        index=False,
    )

    shutil.copy2(
        champion_model_path,
        CHAMPION_MODEL_PATH,
    )


# =========================================================
# 17. Markdown report
# =========================================================

def generate_markdown_report(
    champion_name,
    ranked_validation_df,
    test_metrics,
    classification_report_df,
):
    """
    Generate GitHub-friendly model-selection documentation.
    """

    defect_classes = [
        "Bumps",
        "Dirtiness",
        "K_Scatch",
        "Other_Faults",
        "Pastry",
        "Stains",
        "Z_Scratch",
    ]

    class_rows = (
        classification_report_df[
            classification_report_df[
                "class"
            ].isin(
                defect_classes
            )
        ]
    )

    lines = [
        "# Model Selection and Final Holdout Evaluation",
        "",
        "## Selection Policy",
        "",
        (
            "All model selection and hyperparameter tuning "
            "were completed using training and validation "
            "data only."
        ),
        "",
        (
            "The held-out test set was evaluated only after "
            "the champion model had been selected."
        ),
        "",
        "Primary selection metric: **Validation Macro F1**",
        "",
        "Macro Recall is used as a secondary consideration "
        "because minority defect detection is important.",
        "",
        "## Validation Model Ranking",
        "",
        ranked_validation_df.to_markdown(
            index=False
        ),
        "",
        "## Champion Model",
        "",
        f"**{champion_name}**",
        "",
        "## Final Holdout Test Performance",
        "",
        (
            f"- Accuracy: "
            f"{test_metrics['accuracy']:.4f}"
        ),
        (
            f"- Macro Precision: "
            f"{test_metrics['macro_precision']:.4f}"
        ),
        (
            f"- Macro Recall: "
            f"{test_metrics['macro_recall']:.4f}"
        ),
        (
            f"- Macro F1: "
            f"{test_metrics['macro_f1']:.4f}"
        ),
        (
            f"- Weighted F1: "
            f"{test_metrics['weighted_f1']:.4f}"
        ),
        "",
        "## Final Per-class Performance",
        "",
        class_rows[
            [
                "class",
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        ].to_markdown(
            index=False
        ),
        "",
        "## Deployment Interpretation",
        "",
        (
            "The selected model is treated as a quality "
            "triage and decision-support model rather than "
            "an autonomous product acceptance or rejection "
            "system."
        ),
        "",
        (
            "Production deployment would require validation "
            "on real manufacturing data, operating-threshold "
            "definition, domain-shift testing, monitoring, "
            "and engineer-in-the-loop review."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 18. Print summary
# =========================================================

def print_summary(
    champion_name,
    ranked_validation_df,
    test_metrics,
    classification_report_df,
):
    """
    Print B12 results.
    """

    print("=" * 72)
    print(
        "Stage B12 — Model Selection "
        "& Final Holdout Evaluation"
    )
    print("=" * 72)

    print("\nVALIDATION MODEL RANKING")
    print("-" * 72)

    print(
        ranked_validation_df.to_string(
            index=False
        )
    )

    print("\nCHAMPION MODEL")
    print("-" * 72)

    print(
        champion_name
    )

    print("\nFINAL HOLDOUT TEST METRICS")
    print("-" * 72)

    for metric, value in (
        test_metrics.items()
    ):

        print(
            f"{metric:<18}: "
            f"{value:.4f}"
        )

    print("\nFINAL PER-CLASS PERFORMANCE")
    print("-" * 72)

    defect_classes = [
        "Bumps",
        "Dirtiness",
        "K_Scatch",
        "Other_Faults",
        "Pastry",
        "Stains",
        "Z_Scratch",
    ]

    class_rows = (
        classification_report_df[
            classification_report_df[
                "class"
            ].isin(
                defect_classes
            )
        ]
    )

    print(
        class_rows[
            [
                "class",
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nMODEL ARTIFACT")
    print("-" * 72)

    print(
        f"Champion saved : "
        f"{CHAMPION_MODEL_PATH}"
    )

    print(
        "\nFinal holdout evaluation: PASSED"
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

    validation_comparison_df = (
        load_validation_metrics()
    )

    (
        champion_name,
        ranked_validation_df,
    ) = select_champion(
        validation_comparison_df
    )

    (
        bundle,
        champion_model_path,
    ) = load_champion_bundle(
        champion_name
    )

    (
        model,
        label_encoder,
        saved_feature_columns,
    ) = extract_bundle_components(
        champion_name,
        bundle,
    )

    test_df = (
        load_test_data()
    )

    feature_columns = (
        get_feature_columns(
            test_df,
            saved_feature_columns,
        )
    )

    X_test = (
        test_df[
            feature_columns
        ].copy()
    )

    (
        y_test_text,
        y_test_model,
    ) = prepare_test_target(
        test_df,
        label_encoder,
    )

    (
        predictions_text,
        test_metrics,
    ) = evaluate_test(
        model=model,
        X_test=X_test,
        y_test_model=y_test_model,
        y_test_text=y_test_text,
        label_encoder=label_encoder,
    )

    classification_report_df = (
        build_classification_report(
            y_test=y_test_text,
            predictions=(
                predictions_text
            ),
        )
    )

    (
        confusion_matrix_df,
        class_names,
    ) = build_confusion_matrix(
        y_test=y_test_text,
        predictions=(
            predictions_text
        ),
    )

    plot_confusion_matrix(
        confusion_matrix_df,
        class_names,
    )

    selection_summary_df = (
        build_selection_summary(
            champion_name=(
                champion_name
            ),
            ranked_validation_df=(
                ranked_validation_df
            ),
            test_metrics=(
                test_metrics
            ),
        )
    )

    save_outputs(
        champion_model_path=(
            champion_model_path
        ),
        ranked_validation_df=(
            ranked_validation_df
        ),
        test_metrics=(
            test_metrics
        ),
        classification_report_df=(
            classification_report_df
        ),
        confusion_matrix_df=(
            confusion_matrix_df
        ),
        selection_summary_df=(
            selection_summary_df
        ),
    )

    markdown = (
        generate_markdown_report(
            champion_name=(
                champion_name
            ),
            ranked_validation_df=(
                ranked_validation_df
            ),
            test_metrics=(
                test_metrics
            ),
            classification_report_df=(
                classification_report_df
            ),
        )
    )

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print_summary(
        champion_name=(
            champion_name
        ),
        ranked_validation_df=(
            ranked_validation_df
        ),
        test_metrics=(
            test_metrics
        ),
        classification_report_df=(
            classification_report_df
        ),
    )


if __name__ == "__main__":
    main()