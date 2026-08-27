from pathlib import Path


import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "steel_quality_modeling.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "logistic_regression"
)

DOCS_DIR = (
    PROJECT_ROOT
    / "docs"
)

MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression_baseline.joblib"
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

MARKDOWN_PATH = (
    DOCS_DIR
    / "logistic_regression_baseline.md"
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
# 3. Load modeling dataset
# =========================================================

def load_modeling_data() -> pd.DataFrame:
    """
    Load the B3 model-ready single-label dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# =========================================================
# 4. Feature definition
# =========================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Select the 27 predictor features.

    Excludes:
    - sample ID
    - original one-hot target columns
    - multiclass target
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
# 5. Create fixed stratified splits
# =========================================================

def create_dataset_splits(
    df: pd.DataFrame,
):
    """
    Create fixed 70/15/15 stratified train,
    validation, and test datasets.

    Test data is created here but will not be used
    for model comparison.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN],
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return (
        train_df,
        validation_df,
        test_df,
    )


# =========================================================
# 6. Save fixed splits
# =========================================================

def save_dataset_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Persist dataset splits so all later models use
    exactly the same records.
    """

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        SPLIT_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        SPLIT_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        SPLIT_DIR / "test.csv",
        index=False,
    )


# =========================================================
# 7. Prepare X / y
# =========================================================

def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
):
    """
    Separate predictors from multiclass target.
    """

    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# =========================================================
# 8. Build Logistic Regression pipeline
# =========================================================

def build_model(
    feature_columns: list[str],
) -> Pipeline:
    """
    Build StandardScaler + Logistic Regression pipeline.

    Scaling is fitted only on training data, which prevents
    validation/test information from leaking into training.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                RobustScaler(),
                feature_columns,
            )
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        C=1.0,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


# =========================================================
# 9. Train model
# =========================================================

def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fit the baseline model using training data only.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# 10. Evaluate validation performance
# =========================================================

def evaluate_model(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
):
    """
    Evaluate baseline performance on validation data.
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
# 11. Classification report
# =========================================================

def build_classification_report(
    y_validation: pd.Series,
    predictions,
) -> pd.DataFrame:
    """
    Build per-class precision, recall and F1 report.
    """

    report = classification_report(
        y_validation,
        predictions,
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
# 12. Confusion matrix
# =========================================================

def build_confusion_matrix(
    model: Pipeline,
    y_validation: pd.Series,
    predictions,
):
    """
    Build confusion matrix using model class ordering.
    """

    classes = list(
        model.named_steps[
            "classifier"
        ].classes_
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
# 13. Plot confusion matrix
# =========================================================

def plot_confusion_matrix(
    matrix_df: pd.DataFrame,
    classes: list[str],
) -> None:
    """
    Save validation confusion matrix visualization.
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
        "Logistic Regression "
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
# 14. Build validation prediction report
# =========================================================

def build_prediction_report(
    validation_df: pd.DataFrame,
    model: Pipeline,
    predictions,
    probabilities,
) -> pd.DataFrame:
    """
    Save sample-level validation predictions and confidence.
    """

    classes = list(
        model.named_steps[
            "classifier"
        ].classes_
    )

    result = pd.DataFrame(
        {
            "id": validation_df[
                ID_COLUMN
            ].values,

            "actual_defect": validation_df[
                TARGET_COLUMN
            ].values,

            "predicted_defect": predictions,

            "prediction_confidence":
                probabilities.max(axis=1),
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
# 15. Save model / reports
# =========================================================

def save_outputs(
    model: Pipeline,
    metrics: dict,
    classification_report_df: pd.DataFrame,
    confusion_matrix_df: pd.DataFrame,
    prediction_report: pd.DataFrame,
) -> None:
    """
    Persist baseline model and evaluation results.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
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


# =========================================================
# 16. Generate Markdown report
# =========================================================

def generate_markdown_report(
    metrics: dict,
    classification_report_df: pd.DataFrame,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> str:
    """
    Generate concise model documentation.
    """

    class_rows = classification_report_df[
        classification_report_df[
            "class"
        ].isin(
            [
                "Pastry",
                "Z_Scratch",
                "K_Scatch",
                "Stains",
                "Dirtiness",
                "Bumps",
                "Other_Faults",
            ]
        )
    ].copy()

    weakest_recall_row = (
        class_rows
        .sort_values("recall")
        .iloc[0]
    )

    strongest_recall_row = (
        class_rows
        .sort_values(
            "recall",
            ascending=False,
        )
        .iloc[0]
    )

    lines = [
        "# Logistic Regression Baseline",
        "",
        "## Purpose",
        "",
        (
            "This model establishes a simple linear "
            "classification baseline before evaluating "
            "tree-based models."
        ),
        "",
        "## Dataset Split",
        "",
        f"- Train: {train_count:,}",
        (
            f"- Validation: "
            f"{validation_count:,}"
        ),
        f"- Test: {test_count:,}",
        "",
        (
            "The split is stratified by defect class. "
            "The test set is reserved and is not used "
            "for model comparison."
        ),
        "",
        "## Pipeline",
        "",
        "StandardScaler",
        "",
        "→ Logistic Regression",
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
        "## Per-class Recall",
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
            f"`{strongest_recall_row['class']}` "
            f"({strongest_recall_row['recall']:.4f})"
        ),
        "",
        (
            f"Lowest recall: "
            f"`{weakest_recall_row['class']}` "
            f"({weakest_recall_row['recall']:.4f})"
        ),
        "",
        "## Interpretation",
        "",
        (
            "Because the defect classes are imbalanced, "
            "Macro F1 and per-class Recall are emphasized "
            "alongside Accuracy."
        ),
        "",
        (
            "This model does not use class weighting. "
            "It serves as the unweighted linear baseline "
            "for later model comparison."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 17. Validate splits
# =========================================================

def validate_splits(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Ensure all samples belong to exactly one split.
    """

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_split_rows != len(full_df):

        raise ValueError(
            "Split row counts do not match "
            "the full modeling dataset."
        )

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
# 18. Print results
# =========================================================

def print_summary(
    feature_count: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict,
    classification_report_df: pd.DataFrame,
) -> None:

    print("=" * 72)
    print(
        "Stage B7 — Logistic Regression Baseline"
    )
    print("=" * 72)

    print("\nDATASET SPLIT")
    print("-" * 72)

    print(
        f"Features      : {feature_count}"
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

    class_names = [
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
        f"Saved model : {MODEL_PATH}"
    )

    print(
        "Test set    : NOT evaluated"
    )

    print(
        "\nBaseline validation: PASSED"
    )

    print("=" * 72)


# =========================================================
# 19. Main
# =========================================================

def main() -> None:

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_modeling_data()

    feature_columns = (
        get_feature_columns(df)
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = create_dataset_splits(df)

    validate_splits(
        full_df=df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    save_dataset_splits(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    X_train, y_train = prepare_xy(
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

    model = build_model(
        feature_columns
    )

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
            train_count=len(train_df),
            validation_count=(
                len(validation_df)
            ),
            test_count=len(test_df),
        )
    )

    MARKDOWN_PATH.write_text(
        markdown_report,
        encoding="utf-8",
    )

    print_summary(
        feature_count=len(
            feature_columns
        ),
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        metrics=metrics,
        classification_report_df=(
            classification_report_df
        ),
    )


if __name__ == "__main__":
    main()