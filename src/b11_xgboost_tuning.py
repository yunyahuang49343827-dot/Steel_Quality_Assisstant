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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
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

VALIDATION_PATH = (
    SPLIT_DIR / "validation.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "xgboost_tuned"
)

DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_tuned.joblib"
)

BEST_PARAMS_PATH = (
    REPORT_DIR
    / "best_params.csv"
)

CV_RESULTS_PATH = (
    REPORT_DIR
    / "cv_results.csv"
)

METRICS_PATH = (
    REPORT_DIR
    / "validation_metrics.csv"
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

COMPARISON_PATH = (
    REPORT_DIR
    / "baseline_weighted_tuned_comparison.csv"
)

MARKDOWN_PATH = (
    DOCS_DIR
    / "xgboost_hyperparameter_tuning.md"
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
# 3. Load train / validation
# =========================================================

def load_data():
    """
    Load fixed train and validation splits.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training split not found:\n{TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found:\n{VALIDATION_PATH}"
        )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    return train_df, validation_df


# =========================================================
# 4. Feature columns
# =========================================================

def get_feature_columns(
    df: pd.DataFrame,
):
    """
    Return 27 predictor features.
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
# 5. Encode targets
# =========================================================

def encode_targets(
    train_df,
    validation_df,
):
    """
    Fit LabelEncoder on train only.
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
# 6. Build sample weights
# =========================================================

def build_sample_weights(
    y_train,
):
    """
    Compute balanced weights using train labels only.
    """

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    weight_map = dict(
        zip(
            classes,
            weights,
        )
    )

    sample_weights = np.array(
        [
            weight_map[label]
            for label in y_train
        ]
    )

    return sample_weights


# =========================================================
# 7. Base XGBoost estimator
# =========================================================

def build_base_model():
    """
    Base XGBoost estimator for tuning.
    """

    return XGBClassifier(
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =========================================================
# 8. Parameter distributions
# =========================================================

def get_parameter_distributions():
    """
    Controlled parameter search space.
    """

    return {
        "n_estimators": [
            250,
            350,
            450,
            550,
        ],

        "learning_rate": [
            0.03,
            0.05,
            0.08,
            0.10,
        ],

        "max_depth": [
            3,
            4,
            5,
            6,
            7,
        ],

        "min_child_weight": [
            1,
            3,
            5,
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "reg_alpha": [
            0.0,
            0.05,
            0.1,
            0.3,
        ],

        "reg_lambda": [
            0.5,
            1.0,
            2.0,
            5.0,
        ],
    }


# =========================================================
# 9. Randomized Search
# =========================================================

def run_randomized_search(
    model,
    X_train,
    y_train,
    sample_weights,
):
    """
    Tune XGBoost using Stratified 3-fold CV
    and Macro F1 as the optimization target.
    """

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=(
            get_parameter_distributions()
        ),
        n_iter=20,
        scoring="f1_macro",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    return search


# =========================================================
# 10. Evaluate tuned model
# =========================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    """
    Evaluate best model on untouched validation set.
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
# 11. Classification report
# =========================================================

def build_classification_report(
    y_validation,
    predictions,
    label_encoder,
):
    """
    Human-readable per-class metrics.
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
# 12. Confusion matrix
# =========================================================

def build_confusion_matrix(
    y_validation,
    predictions,
    label_encoder,
):
    """
    Build validation confusion matrix.
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
# 13. Plot confusion matrix
# =========================================================

def plot_confusion_matrix(
    matrix_df,
    class_names,
):
    """
    Save confusion matrix chart.
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
        "Tuned XGBoost "
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
# 14. Load previous model metrics
# =========================================================

def load_previous_metrics():
    """
    Load B9 baseline and B10 weighted metrics.
    """

    baseline_path = (
        PROJECT_ROOT
        / "reports"
        / "models"
        / "xgboost"
        / "metrics.csv"
    )

    weighted_path = (
        PROJECT_ROOT
        / "reports"
        / "models"
        / "xgboost_weighted"
        / "metrics.csv"
    )

    baseline_df = pd.read_csv(
        baseline_path
    )

    weighted_df = pd.read_csv(
        weighted_path
    )

    baseline = dict(
        zip(
            baseline_df["metric"],
            baseline_df["value"],
        )
    )

    weighted = dict(
        zip(
            weighted_df["metric"],
            weighted_df["value"],
        )
    )

    return baseline, weighted


# =========================================================
# 15. Build comparison
# =========================================================

def build_model_comparison(
    baseline,
    weighted,
    tuned,
):
    """
    Compare three XGBoost experiments.
    """

    records = []

    for metric in tuned:

        records.append(
            {
                "metric": metric,

                "baseline_xgboost":
                    baseline[metric],

                "weighted_xgboost":
                    weighted[metric],

                "tuned_weighted_xgboost":
                    tuned[metric],
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# 16. Save outputs
# =========================================================

def save_outputs(
    best_model,
    label_encoder,
    feature_columns,
    search,
    metrics,
    classification_report_df,
    confusion_matrix_df,
    comparison_df,
):
    """
    Persist tuned model and tuning reports.
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
        "model": best_model,
        "label_encoder":
            label_encoder,
        "feature_columns":
            feature_columns,
        "best_params":
            search.best_params_,
        "best_cv_macro_f1":
            search.best_score_,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    pd.DataFrame(
        [
            {
                "parameter": key,
                "value": value,
            }
            for key, value
            in search.best_params_.items()
        ]
    ).to_csv(
        BEST_PARAMS_PATH,
        index=False,
    )

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    cv_results = (
        cv_results
        .sort_values(
            "rank_test_score"
        )
    )

    cv_results.to_csv(
        CV_RESULTS_PATH,
        index=False,
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

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )


# =========================================================
# 17. Markdown report
# =========================================================

def generate_markdown_report(
    search,
    metrics,
    comparison_df,
    classification_report_df,
):
    """
    Create tuning documentation.
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

    best_params_df = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": value,
            }
            for key, value
            in search.best_params_.items()
        ]
    )

    lines = [
        "# XGBoost Hyperparameter Tuning",
        "",
        "## Purpose",
        "",
        (
            "RandomizedSearchCV is used to improve "
            "the weighted XGBoost model while keeping "
            "the search space computationally controlled."
        ),
        "",
        "## Optimization Metric",
        "",
        "**Macro F1**",
        "",
        (
            "Macro F1 is selected because the defect "
            "classes are imbalanced and performance on "
            "minority classes is important."
        ),
        "",
        "## Cross-validation",
        "",
        "- Stratified 3-fold CV",
        "- 20 randomized parameter combinations",
        "- Training split only",
        "",
        "## Best Cross-validation Score",
        "",
        f"{search.best_score_:.4f}",
        "",
        "## Best Parameters",
        "",
        best_params_df.to_markdown(
            index=False
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
        "## Model Comparison",
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
        "## Test Set Policy",
        "",
        (
            "The held-out test set remains untouched. "
            "Final test evaluation will be performed "
            "only after model selection."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 18. Print summary
# =========================================================

def print_summary(
    search,
    metrics,
    comparison_df,
    classification_report_df,
):
    """
    Print tuning results.
    """

    print("=" * 72)
    print(
        "Stage B11 — XGBoost Hyperparameter Tuning"
    )
    print("=" * 72)

    print("\nSEARCH")
    print("-" * 72)

    print(
        "Randomized combinations : 20"
    )

    print(
        "Cross-validation folds  : 3"
    )

    print(
        f"Best CV Macro F1        : "
        f"{search.best_score_:.4f}"
    )

    print("\nBEST PARAMETERS")
    print("-" * 72)

    for key, value in (
        search.best_params_.items()
    ):

        print(
            f"{key:<20}: {value}"
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

    print("\nMODEL COMPARISON")
    print("-" * 72)

    print(
        comparison_df.to_string(
            index=False
        )
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

    print("\nMODEL")
    print("-" * 72)

    print(
        f"Saved model : {MODEL_PATH}"
    )

    print(
        "Test set    : NOT evaluated"
    )

    print(
        "\nHyperparameter tuning: PASSED"
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

    sample_weights = (
        build_sample_weights(
            y_train
        )
    )

    base_model = (
        build_base_model()
    )

    search = (
        run_randomized_search(
            model=base_model,
            X_train=X_train,
            y_train=y_train,
            sample_weights=(
                sample_weights
            ),
        )
    )

    best_model = (
        search.best_estimator_
    )

    (
        predictions,
        probabilities,
        metrics,
    ) = evaluate_model(
        model=best_model,
        X_validation=X_validation,
        y_validation=y_validation,
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

    baseline, weighted = (
        load_previous_metrics()
    )

    comparison_df = (
        build_model_comparison(
            baseline=baseline,
            weighted=weighted,
            tuned=metrics,
        )
    )

    save_outputs(
        best_model=best_model,
        label_encoder=(
            label_encoder
        ),
        feature_columns=(
            feature_columns
        ),
        search=search,
        metrics=metrics,
        classification_report_df=(
            classification_report_df
        ),
        confusion_matrix_df=(
            confusion_matrix_df
        ),
        comparison_df=(
            comparison_df
        ),
    )

    plot_confusion_matrix(
        confusion_matrix_df,
        class_names,
    )

    markdown = (
        generate_markdown_report(
            search=search,
            metrics=metrics,
            comparison_df=(
                comparison_df
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
        search=search,
        metrics=metrics,
        comparison_df=(
            comparison_df
        ),
        classification_report_df=(
            classification_report_df
        ),
    )


if __name__ == "__main__":
    main()