from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.utils.class_weight import compute_class_weight

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
VALIDATION_PATH = SPLIT_DIR / "validation.csv"

MODEL_DIR = PROJECT_ROOT / "models"

BASELINE_MODEL_PATH = (
    MODEL_DIR
    / "xgboost_baseline.joblib"
)

WEIGHTED_MODEL_PATH = (
    MODEL_DIR
    / "xgboost_weighted.joblib"
)

BASELINE_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "xgboost"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "xgboost_weighted"
)

DOCS_DIR = PROJECT_ROOT / "docs"

METRICS_PATH = REPORT_DIR / "metrics.csv"

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

CLASS_WEIGHTS_PATH = (
    REPORT_DIR
    / "class_weights.csv"
)

COMPARISON_PATH = (
    REPORT_DIR
    / "baseline_vs_weighted.csv"
)

MARKDOWN_PATH = (
    DOCS_DIR
    / "imbalanced_classification.md"
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

RANDOM_STATE = 42


# =========================================================
# 3. Load data
# =========================================================

def load_data():
    """
    Load the fixed train and validation datasets.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training split not found:\n{TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found:\n{VALIDATION_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    return train_df, validation_df


# =========================================================
# 4. Feature columns
# =========================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return model predictor columns.
    """

    excluded = (
        [ID_COLUMN, TARGET_COLUMN]
        + TARGET_BINARY_COLUMNS
    )

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


# =========================================================
# 5. Encode labels
# =========================================================

def encode_targets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Fit label encoding using training labels only.
    """

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(
        train_df[TARGET_COLUMN]
    )

    y_validation = encoder.transform(
        validation_df[TARGET_COLUMN]
    )

    return (
        y_train,
        y_validation,
        encoder,
    )


# =========================================================
# 6. Compute class weights
# =========================================================

def build_class_weights(
    y_train,
    label_encoder,
):
    """
    Calculate balanced class weights based only on
    training-set class frequencies.
    """

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    weight_map = {
        class_id: weight
        for class_id, weight
        in zip(classes, weights)
    }

    class_weight_df = pd.DataFrame(
        {
            "class_id": classes,
            "defect_type":
                label_encoder.inverse_transform(
                    classes
                ),
            "class_weight": weights,
        }
    )

    sample_weights = np.array(
        [
            weight_map[class_id]
            for class_id in y_train
        ]
    )

    return (
        sample_weights,
        class_weight_df,
    )


# =========================================================
# 7. Build weighted XGBoost
# =========================================================

def build_model():
    """
    Use the same XGBoost configuration as B9.

    The only major experimental change is sample weighting.
    """

    return XGBClassifier(
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


# =========================================================
# 8. Train weighted model
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
    sample_weights,
):
    """
    Fit XGBoost with balanced sample weights.
    """

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    return model


# =========================================================
# 9. Evaluate model
# =========================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    """
    Evaluate weighted XGBoost on validation data.
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

        "macro_precision":
            precision_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "macro_f1":
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "weighted_f1":
            f1_score(
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
# 10. Classification report
# =========================================================

def build_classification_report(
    y_validation,
    predictions,
    label_encoder,
):
    """
    Build human-readable per-class performance report.
    """

    class_names = list(
        label_encoder.classes_
    )

    labels = list(
        range(len(class_names))
    )

    report = classification_report(
        y_validation,
        predictions,
        labels=labels,
        target_names=class_names,
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
# 11. Confusion matrix
# =========================================================

def build_confusion_matrix(
    y_validation,
    predictions,
    label_encoder,
):
    """
    Create confusion matrix.
    """

    class_names = list(
        label_encoder.classes_
    )

    labels = list(
        range(len(class_names))
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    return matrix_df, class_names


# =========================================================
# 12. Plot confusion matrix
# =========================================================

def plot_confusion_matrix(
    matrix_df,
    class_names,
):
    """
    Save validation confusion matrix chart.
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
        "Weighted XGBoost "
        "Validation Confusion Matrix"
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
# 13. Prediction report
# =========================================================

def build_prediction_report(
    validation_df,
    predictions,
    probabilities,
    label_encoder,
):
    """
    Save sample-level predictions.
    """

    predicted_labels = (
        label_encoder.inverse_transform(
            predictions
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
        ] = probabilities[:, index]

    return result


# =========================================================
# 14. Load baseline metrics
# =========================================================

def load_baseline_metrics():
    """
    Load B9 XGBoost validation metrics.
    """

    baseline_path = (
        BASELINE_REPORT_DIR
        / "metrics.csv"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"B9 metrics not found:\n{baseline_path}"
        )

    df = pd.read_csv(
        baseline_path
    )

    return dict(
        zip(
            df["metric"],
            df["value"],
        )
    )


# =========================================================
# 15. Build metric comparison
# =========================================================

def build_comparison(
    baseline_metrics,
    weighted_metrics,
):
    """
    Compare B9 baseline and weighted XGBoost.
    """

    records = []

    for metric in weighted_metrics:

        baseline = (
            baseline_metrics[
                metric
            ]
        )

        weighted = (
            weighted_metrics[
                metric
            ]
        )

        records.append(
            {
                "metric": metric,
                "baseline_xgboost":
                    baseline,
                "weighted_xgboost":
                    weighted,
                "difference":
                    weighted - baseline,
            }
        )

    return pd.DataFrame(records)


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
    class_weight_df,
    comparison_df,
):
    """
    Save weighted model and reports.
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

    bundle = {
        "model": model,
        "label_encoder":
            label_encoder,
        "feature_columns":
            feature_columns,
    }

    joblib.dump(
        bundle,
        WEIGHTED_MODEL_PATH,
    )

    pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value
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

    class_weight_df.to_csv(
        CLASS_WEIGHTS_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )


# =========================================================
# 17. Markdown report
# =========================================================

def generate_markdown_report(
    metrics,
    comparison_df,
    classification_report_df,
    class_weight_df,
):
    """
    Build concise imbalance experiment documentation.
    """

    defect_classes = list(
        class_weight_df[
            "defect_type"
        ]
    )

    class_rows = (
        classification_report_df[
            classification_report_df[
                "class"
            ].isin(defect_classes)
        ]
    )

    lines = [
        "# Imbalanced Classification Evaluation",
        "",
        "## Purpose",
        "",
        (
            "The baseline models showed weak recall for "
            "minority defect classes. This experiment "
            "tests balanced sample weighting using XGBoost."
        ),
        "",
        "## Why Sample Weighting",
        "",
        (
            "Minority-class training samples receive higher "
            "loss weights so classification errors on rare "
            "defect types have greater influence during "
            "model training."
        ),
        "",
        (
            "Synthetic oversampling such as SMOTE is not "
            "used in this stage because the dataset contains "
            "binary and derived geometric features, and the "
            "competition dataset itself is synthetic."
        ),
        "",
        "## Training Class Weights",
        "",
        class_weight_df.to_markdown(
            index=False
        ),
        "",
        "## Weighted Validation Performance",
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
        "## Baseline vs Weighted",
        "",
        comparison_df.to_markdown(
            index=False
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
        "## Decision Principle",
        "",
        (
            "The weighted model is not automatically better "
            "because minority recall increases. Model selection "
            "must consider the trade-off between missed defects "
            "and false-positive inspection workload."
        ),
        "",
        (
            "The test set remains reserved and is not used "
            "during this experiment."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 18. Print summary
# =========================================================

def print_summary(
    metrics,
    comparison_df,
    classification_report_df,
    class_weight_df,
):
    """
    Print B10 experiment results.
    """

    print("=" * 72)
    print(
        "Stage B10 — Imbalanced Classification Evaluation"
    )
    print("=" * 72)

    print("\nCLASS WEIGHTS")
    print("-" * 72)

    print(
        class_weight_df.to_string(
            index=False
        )
    )

    print("\nWEIGHTED VALIDATION METRICS")
    print("-" * 72)

    for metric, value in metrics.items():

        print(
            f"{metric:<18}: "
            f"{value:.4f}"
        )

    print("\nBASELINE VS WEIGHTED")
    print("-" * 72)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 72)

    class_names = list(
        class_weight_df[
            "defect_type"
        ]
    )

    class_rows = (
        classification_report_df[
            classification_report_df[
                "class"
            ].isin(class_names)
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

    print("\nMODEL")
    print("-" * 72)

    print(
        f"Saved model : "
        f"{WEIGHTED_MODEL_PATH}"
    )

    print(
        "Test set    : NOT evaluated"
    )

    print(
        "\nImbalance evaluation: PASSED"
    )

    print("=" * 72)


# =========================================================
# 19. Main
# =========================================================

def main():

    train_df, validation_df = (
        load_data()
    )

    feature_columns = (
        get_feature_columns(
            train_df
        )
    )

    X_train = train_df[
        feature_columns
    ].copy()

    X_validation = (
        validation_df[
            feature_columns
        ].copy()
    )

    (
        y_train,
        y_validation,
        label_encoder,
    ) = encode_targets(
        train_df,
        validation_df,
    )

    (
        sample_weights,
        class_weight_df,
    ) = build_class_weights(
        y_train,
        label_encoder,
    )

    model = build_model()

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        sample_weights=(
            sample_weights
        ),
    )

    (
        predictions,
        probabilities,
        metrics,
    ) = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    classification_report_df = (
        build_classification_report(
            y_validation,
            predictions,
            label_encoder,
        )
    )

    (
        confusion_matrix_df,
        class_names,
    ) = build_confusion_matrix(
        y_validation,
        predictions,
        label_encoder,
    )

    prediction_report = (
        build_prediction_report(
            validation_df,
            predictions,
            probabilities,
            label_encoder,
        )
    )

    baseline_metrics = (
        load_baseline_metrics()
    )

    comparison_df = (
        build_comparison(
            baseline_metrics,
            metrics,
        )
    )

    save_outputs(
        model=model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
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
        class_weight_df=(
            class_weight_df
        ),
        comparison_df=(
            comparison_df
        ),
    )

    plot_confusion_matrix(
        confusion_matrix_df,
        class_names,
    )

    markdown = generate_markdown_report(
        metrics=metrics,
        comparison_df=comparison_df,
        classification_report_df=(
            classification_report_df
        ),
        class_weight_df=(
            class_weight_df
        ),
    )

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print_summary(
        metrics=metrics,
        comparison_df=comparison_df,
        classification_report_df=(
            classification_report_df
        ),
        class_weight_df=(
            class_weight_df
        ),
    )


if __name__ == "__main__":
    main()