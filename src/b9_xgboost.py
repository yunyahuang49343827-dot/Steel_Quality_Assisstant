from pathlib import Path

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
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


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

VALIDATION_PATH = (
    SPLIT_DIR / "validation.csv"
)

TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "xgboost"
)

DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_baseline.joblib"
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
    / "xgboost_baseline.md"
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
# 3. Load fixed dataset splits
# =========================================================

def load_dataset_splits():
    """
    Load the exact train / validation / test datasets
    created during Stage B7.

    The test set remains reserved and is not used
    for model evaluation during B9.
    """

    required_paths = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Dataset split not found:\n{path}\n"
                "Please complete Stage B7 first."
            )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# =========================================================
# 4. Validate split integrity
# =========================================================

def validate_splits(
    train_df,
    validation_df,
    test_df,
) -> None:
    """
    Ensure there is no record overlap between
    train, validation and test datasets.
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
# 5. Get feature columns
# =========================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return the 27 model predictor features.

    Excludes:
    - sample ID
    - original binary defect targets
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
# 6. Prepare feature matrices
# =========================================================

def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Select model predictor features.

    XGBoost is tree-based, so no feature scaling
    is applied.
    """

    return df[
        feature_columns
    ].copy()


# =========================================================
# 7. Encode target labels
# =========================================================

def prepare_targets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Fit LabelEncoder on training labels only.

    Convert human-readable defect labels into
    numerical class IDs required by XGBoost.
    """

    label_encoder = LabelEncoder()

    y_train = (
        label_encoder.fit_transform(
            train_df[TARGET_COLUMN]
        )
    )

    y_validation = (
        label_encoder.transform(
            validation_df[
                TARGET_COLUMN
            ]
        )
    )

    return (
        y_train,
        y_validation,
        label_encoder,
    )


# =========================================================
# 8. Build XGBoost baseline
# =========================================================

def build_model() -> XGBClassifier:
    """
    Build the unweighted XGBoost multiclass baseline.

    Hyperparameter tuning and imbalance handling are
    intentionally deferred to later stages.
    """

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=7,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


# =========================================================
# 9. Train model
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
):
    """
    Fit XGBoost using training data only.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# 10. Evaluate model
# =========================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    """
    Evaluate XGBoost on validation data only.

    Test data remains untouched.
    """

    encoded_predictions = (
        model.predict(
            X_validation
        )
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    metrics = {
        "accuracy": accuracy_score(
            y_validation,
            encoded_predictions,
        ),

        "macro_precision":
            precision_score(
                y_validation,
                encoded_predictions,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_validation,
                encoded_predictions,
                average="macro",
                zero_division=0,
            ),

        "macro_f1":
            f1_score(
                y_validation,
                encoded_predictions,
                average="macro",
                zero_division=0,
            ),

        "weighted_f1":
            f1_score(
                y_validation,
                encoded_predictions,
                average="weighted",
                zero_division=0,
            ),
    }

    return (
        encoded_predictions,
        probabilities,
        metrics,
    )


# =========================================================
# 11. Classification report
# =========================================================

def build_classification_report(
    y_validation,
    encoded_predictions,
    label_encoder,
) -> pd.DataFrame:
    """
    Build per-class validation performance report
    using human-readable defect names.
    """

    class_names = list(
        label_encoder.classes_
    )

    labels = list(
        range(len(class_names))
    )

    report = classification_report(
        y_validation,
        encoded_predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class"
            }
        )
    )

    return report_df


# =========================================================
# 12. Build confusion matrix
# =========================================================

def build_confusion_matrix(
    y_validation,
    encoded_predictions,
    label_encoder,
):
    """
    Build validation confusion matrix using
    human-readable class names.
    """

    class_names = list(
        label_encoder.classes_
    )

    labels = list(
        range(len(class_names))
    )

    matrix = confusion_matrix(
        y_validation,
        encoded_predictions,
        labels=labels,
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
# 13. Plot confusion matrix
# =========================================================

def plot_confusion_matrix(
    matrix_df,
    class_names,
) -> None:
    """
    Save XGBoost validation confusion matrix.
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
        class_names,
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "XGBoost Validation "
        "Confusion Matrix"
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
        CONFUSION_MATRIX_IMAGE_PATH,
        dpi=160,
    )

    plt.close(fig)


# =========================================================
# 14. Build validation predictions
# =========================================================

def build_prediction_report(
    validation_df,
    encoded_predictions,
    probabilities,
    label_encoder,
) -> pd.DataFrame:
    """
    Build sample-level validation prediction report.
    """

    predicted_labels = (
        label_encoder.inverse_transform(
            encoded_predictions
        )
    )

    result = pd.DataFrame(
        {
            "id":
                validation_df[
                    ID_COLUMN
                ].values,

            "actual_defect":
                validation_df[
                    TARGET_COLUMN
                ].values,

            "predicted_defect":
                predicted_labels,

            "prediction_confidence":
                probabilities.max(
                    axis=1
                ),
        }
    )

    for index, class_name in enumerate(
        label_encoder.classes_
    ):

        result[
            f"probability_{class_name}"
        ] = probabilities[
            :, index
        ]

    return result


# =========================================================
# 15. Feature importance
# =========================================================

def build_feature_importance(
    model,
    feature_columns,
) -> pd.DataFrame:
    """
    Extract preliminary XGBoost feature importance.

    This represents model usage and must not be
    interpreted as manufacturing causality.
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
# 16. Save outputs
# =========================================================

def save_outputs(
    model,
    label_encoder,
    feature_columns,
    metrics,
    classification_report_df,
    confusion_matrix_df,
    prediction_report,
    feature_importance_df,
) -> None:
    """
    Save model bundle and all validation reports.
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

    model_bundle = {
        "model": model,
        "label_encoder":
            label_encoder,
        "feature_columns":
            feature_columns,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value
            in metrics.items()
        ]
    )

    metrics_df.to_csv(
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
# 17. Generate Markdown report
# =========================================================

def generate_markdown_report(
    metrics,
    classification_report_df,
    feature_importance_df,
) -> str:
    """
    Generate concise GitHub-friendly XGBoost report.
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
        "# XGBoost Baseline",
        "",
        "## Purpose",
        "",
        (
            "XGBoost is evaluated as a gradient-boosted "
            "tree model using the same fixed train and "
            "validation datasets as the previous models."
        ),
        "",
        "## Model Configuration",
        "",
        "- n_estimators: 400",
        "- learning_rate: 0.05",
        "- max_depth: 6",
        "- subsample: 0.8",
        "- colsample_bytree: 0.8",
        "- class weighting: None",
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
            "Feature importance describes how the "
            "XGBoost model uses available predictors. "
            "It does not establish causal manufacturing "
            "relationships."
        ),
        "",
        (
            "The test set remains reserved and has not "
            "been used for model evaluation."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 18. Print summary
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
        "Stage B9 — XGBoost Baseline"
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
        "\nXGBoost validation: PASSED"
    )

    print("=" * 72)


# =========================================================
# 19. Main
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

    X_train = prepare_features(
        train_df,
        feature_columns,
    )

    X_validation = prepare_features(
        validation_df,
        feature_columns,
    )

    (
        y_train,
        y_validation,
        label_encoder,
    ) = prepare_targets(
        train_df=train_df,
        validation_df=validation_df,
    )

    model = build_model()

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    (
        encoded_predictions,
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
            encoded_predictions=(
                encoded_predictions
            ),
            label_encoder=(
                label_encoder
            ),
        )
    )

    (
        confusion_matrix_df,
        class_names,
    ) = build_confusion_matrix(
        y_validation=y_validation,
        encoded_predictions=(
            encoded_predictions
        ),
        label_encoder=(
            label_encoder
        ),
    )

    prediction_report = (
        build_prediction_report(
            validation_df=(
                validation_df
            ),
            encoded_predictions=(
                encoded_predictions
            ),
            probabilities=(
                probabilities
            ),
            label_encoder=(
                label_encoder
            ),
        )
    )

    feature_importance_df = (
        build_feature_importance(
            model=model,
            feature_columns=(
                feature_columns
            ),
        )
    )

    save_outputs(
        model=model,
        label_encoder=(
            label_encoder
        ),
        feature_columns=(
            feature_columns
        ),
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
        matrix_df=(
            confusion_matrix_df
        ),
        class_names=(
            class_names
        ),
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