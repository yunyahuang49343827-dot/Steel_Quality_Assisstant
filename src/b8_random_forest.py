from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "random_forest"
)

DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_PATH = (
    MODEL_DIR
    / "random_forest_baseline.joblib"
)

METRICS_PATH = (
    REPORT_DIR
    / "metrics.csv"
)

CLASSIFICATION_REPORT_PATH = (
    REPORT_DIR
    / "classification_report.csv"
)

CONFUSION_MATRIX_CSV_PATH = (
    REPORT_DIR
    / "confusion_matrix.csv"
)

CONFUSION_MATRIX_IMAGE_PATH = (
    REPORT_DIR
    / "confusion_matrix.png"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "validation_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = (
    REPORT_DIR
    / "feature_importance.csv"
)

MARKDOWN_PATH = (
    DOCS_DIR
    / "random_forest_baseline.md"
)


# =========================================================
# 2. Dataset definition
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

RANDOM_STATE = 42


# =========================================================
# 3. Load fixed B7 splits
# =========================================================

def load_dataset_splits():
    """
    Load the exact train / validation / test splits
    created during Stage B7.

    Test data is loaded only for validation of the split
    structure and is not used for model evaluation.
    """

    required_paths = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset split not found:\n{path}\n"
                "Please complete Stage B7 first."
            )

    train_df = pd.read_csv(TRAIN_PATH)

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    test_df = pd.read_csv(TEST_PATH)

    return (
        train_df,
        validation_df,
        test_df,
    )


# =========================================================
# 4. Feature columns
# =========================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return the 27 predictor features.

    Excludes:
    - ID
    - original binary target columns
    - multiclass defect target
    """

    excluded_columns = (
        [ID_COLUMN, TARGET_COLUMN]
        + TARGET_BINARY_COLUMNS
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return feature_columns


# =========================================================
# 5. Prepare X / y
# =========================================================

def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
):
    """
    Separate predictors and multiclass target.
    """

    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# =========================================================
# 6. Build Random Forest
# =========================================================

def build_model() -> RandomForestClassifier:
    """
    Build the unweighted Random Forest baseline.

    Class weighting is intentionally disabled here.
    Imbalance handling will be evaluated separately
    during Stage B10.
    """

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


# =========================================================
# 7. Train model
# =========================================================

def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Fit Random Forest using training data only.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# 8. Evaluate validation performance
# =========================================================

def evaluate_model(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
):
    """
    Evaluate model using validation data only.
    """

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )

    metrics = {
        "accuracy": accuracy_score(
            y_validation,
            predictions,
        ),

        "macro_precision": precision_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "macro_f1": f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return (
        predictions,
        probabilities,
        metrics,
    )


# =========================================================
# 9. Classification report
# =========================================================

def build_classification_report(
    y_validation,
    predictions,
) -> pd.DataFrame:
    """
    Build per-class validation metrics.
    """

    report = classification_report(
        y_validation,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class"
            }
        )
    )


# =========================================================
# 10. Confusion matrix
# =========================================================

def build_confusion_matrix(
    model,
    y_validation,
    predictions,
):
    """
    Build confusion matrix using model class order.
    """

    classes = list(
        model.classes_
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=classes,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    )

    return matrix_df, classes


# =========================================================
# 11. Plot confusion matrix
# =========================================================

def plot_confusion_matrix(
    matrix_df,
    classes,
) -> None:
    """
    Save validation confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        matrix_df.values,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(classes))
    )

    ax.set_yticks(
        range(len(classes))
    )

    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        classes,
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "Random Forest "
        "Validation Confusion Matrix"
    )

    for i in range(
        len(classes)
    ):
        for j in range(
            len(classes)
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
        CONFUSION_MATRIX_IMAGE_PATH,
        dpi=160,
    )

    plt.close(fig)


# =========================================================
# 12. Validation predictions
# =========================================================

def build_prediction_report(
    validation_df,
    model,
    predictions,
    probabilities,
) -> pd.DataFrame:
    """
    Save sample-level predictions and confidence.
    """

    classes = list(
        model.classes_
    )

    result = pd.DataFrame(
        {
            "id": validation_df[
                ID_COLUMN
            ].values,

            "actual_defect": validation_df[
                TARGET_COLUMN
            ].values,

            "predicted_defect":
                predictions,

            "prediction_confidence":
                probabilities.max(
                    axis=1
                ),
        }
    )

    for index, class_name in enumerate(
        classes
    ):

        result[
            f"probability_{class_name}"
        ] = probabilities[:, index]

    return result


# =========================================================
# 13. Feature importance
# =========================================================

def build_feature_importance(
    model,
    feature_columns,
) -> pd.DataFrame:
    """
    Extract Random Forest impurity-based feature importance.

    This is preliminary model inspection and must not be
    interpreted as causal feature importance.
    """

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "importance":
                model.feature_importances_,
        }
    )

    return (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =========================================================
# 14. Save outputs
# =========================================================

def save_outputs(
    model,
    metrics,
    classification_report_df,
    confusion_matrix_df,
    prediction_report,
    feature_importance_df,
) -> None:
    """
    Save trained model and validation reports.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value
            in metrics.items()
        ]
    ).to_csv(
        METRICS_PATH,
        index=False,
    )

    classification_report_df.to_csv(
        CLASSIFICATION_REPORT_PATH,
        index=False,
    )

    confusion_matrix_df.to_csv(
        CONFUSION_MATRIX_CSV_PATH,
    )

    prediction_report.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )


# =========================================================
# 15. Generate Markdown report
# =========================================================

def generate_markdown_report(
    metrics,
    classification_report_df,
    feature_importance_df,
) -> str:
    """
    Generate GitHub-friendly Random Forest documentation.
    """

    defect_classes = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
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

    highest_recall = (
        class_rows
        .sort_values(
            "recall",
            ascending=False,
        )
        .iloc[0]
    )

    lowest_recall = (
        class_rows
        .sort_values(
            "recall"
        )
        .iloc[0]
    )

    top_features = (
        feature_importance_df
        .head(10)
    )

    lines = [
        "# Random Forest Baseline",
        "",
        "## Purpose",
        "",
        (
            "Random Forest is evaluated as a nonlinear "
            "tree-based model using the same fixed "
            "training and validation split as the "
            "Logistic Regression baseline."
        ),
        "",
        "## Model Configuration",
        "",
        "- Trees: 400",
        "- max_depth: None",
        "- max_features: sqrt",
        "- class_weight: None",
        "",
        (
            "Class weighting is intentionally disabled "
            "during this stage so imbalance handling can "
            "be evaluated separately."
        ),
        "",
        "## Validation Performance",
        "",
        (
            f"- Accuracy: "
            f"{metrics['accuracy']:.4f}"
        ),
        (
            f"- Macro Precision: "
            f"{metrics['macro_precision']:.4f}"
        ),
        (
            f"- Macro Recall: "
            f"{metrics['macro_recall']:.4f}"
        ),
        (
            f"- Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        ),
        (
            f"- Weighted F1: "
            f"{metrics['weighted_f1']:.4f}"
        ),
        "",
        "## Per-class Performance",
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
        "## Recall Observations",
        "",
        (
            f"Highest recall: "
            f"`{highest_recall['class']}` "
            f"({highest_recall['recall']:.4f})"
        ),
        "",
        (
            f"Lowest recall: "
            f"`{lowest_recall['class']}` "
            f"({lowest_recall['recall']:.4f})"
        ),
        "",
        "## Preliminary Feature Importance",
        "",
        top_features[
            [
                "feature",
                "importance",
            ]
        ].to_markdown(
            index=False
        ),
        "",
        "## Interpretation",
        "",
        (
            "Random Forest feature importance describes "
            "how the trained model uses features to split "
            "the training data. It does not establish "
            "manufacturing causality."
        ),
        "",
        (
            "The test set remains reserved and is not "
            "evaluated during this stage."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 16. Validate split integrity
# =========================================================

def validate_splits(
    train_df,
    validation_df,
    test_df,
) -> None:
    """
    Confirm there is no sample overlap between splits.
    """

    train_ids = set(
        train_df[ID_COLUMN]
    )

    validation_ids = set(
        validation_df[ID_COLUMN]
    )

    test_ids = set(
        test_df[ID_COLUMN]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Train / validation overlap detected."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Train / test overlap detected."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation / test overlap detected."
        )


# =========================================================
# 17. Print summary
# =========================================================

def print_summary(
    feature_columns,
    train_df,
    validation_df,
    test_df,
    metrics,
    classification_report_df,
    feature_importance_df,
) -> None:

    print("=" * 72)
    print(
        "Stage B8 — Random Forest Baseline"
    )
    print("=" * 72)

    print("\nDATASET")
    print("-" * 72)

    print(
        f"Features      : "
        f"{len(feature_columns)}"
    )

    print(
        f"Train         : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation    : "
        f"{len(validation_df):,}"
    )

    print(
        f"Test reserved : "
        f"{len(test_df):,}"
    )

    print("\nVALIDATION METRICS")
    print("-" * 72)

    for metric, value in (
        metrics.items()
    ):

        print(
            f"{metric:<18}: "
            f"{value:.4f}"
        )

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 72)

    defect_classes = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
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

    print("\nTOP 10 FEATURES")
    print("-" * 72)

    print(
        feature_importance_df
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\nMODEL")
    print("-" * 72)

    print(
        f"Saved model : "
        f"{MODEL_PATH}"
    )

    print(
        "Test set    : NOT evaluated"
    )

    print(
        "\nRandom Forest validation: PASSED"
    )

    print("=" * 72)


# =========================================================
# 18. Main
# =========================================================

def main() -> None:

    (
        train_df,
        validation_df,
        test_df,
    ) = load_dataset_splits()

    validate_splits(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    feature_columns = (
        get_feature_columns(
            train_df
        )
    )

    (
        X_train,
        y_train,
    ) = prepare_xy(
        train_df,
        feature_columns,
    )

    (
        X_validation,
        y_validation,
    ) = prepare_xy(
        validation_df,
        feature_columns,
    )

    model = build_model()

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    (
        predictions,
        probabilities,
        metrics,
    ) = evaluate_model(
        model=model,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    classification_report_df = (
        build_classification_report(
            y_validation=y_validation,
            predictions=predictions,
        )
    )

    (
        confusion_matrix_df,
        classes,
    ) = build_confusion_matrix(
        model=model,
        y_validation=y_validation,
        predictions=predictions,
    )

    prediction_report = (
        build_prediction_report(
            validation_df=validation_df,
            model=model,
            predictions=predictions,
            probabilities=probabilities,
        )
    )

    feature_importance_df = (
        build_feature_importance(
            model=model,
            feature_columns=feature_columns,
        )
    )

    save_outputs(
        model=model,
        metrics=metrics,
        classification_report_df=(
            classification_report_df
        ),
        confusion_matrix_df=(
            confusion_matrix_df
        ),
        prediction_report=(
            prediction_report
        ),
        feature_importance_df=(
            feature_importance_df
        ),
    )

    plot_confusion_matrix(
        matrix_df=confusion_matrix_df,
        classes=classes,
    )

    markdown_report = (
        generate_markdown_report(
            metrics=metrics,
            classification_report_df=(
                classification_report_df
            ),
            feature_importance_df=(
                feature_importance_df
            ),
        )
    )

    MARKDOWN_PATH.write_text(
        markdown_report,
        encoding="utf-8",
    )

    print_summary(
        feature_columns=(
            feature_columns
        ),
        train_df=train_df,
        validation_df=(
            validation_df
        ),
        test_df=test_df,
        metrics=metrics,
        classification_report_df=(
            classification_report_df
        ),
        feature_importance_df=(
            feature_importance_df
        ),
    )


if __name__ == "__main__":
    main()  