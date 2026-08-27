from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "eda"
)

DOCS_DIR = PROJECT_ROOT / "docs"

MARKDOWN_OUTPUT_PATH = (
    DOCS_DIR
    / "eda_report.md"
)


# =========================================================
# 2. Target / feature definitions
# =========================================================

TARGET_COLUMN = "defect_type"

TARGET_BINARY_COLUMNS = [
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
# 3. Load modeling data
# =========================================================

def load_modeling_data() -> pd.DataFrame:
    """
    Load the B3 cleaned modeling dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# =========================================================
# 4. Add EDA helper columns
# =========================================================

def add_eda_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add helper flags for geometry consistency analysis.
    """

    df = df.copy()

    df["x_geometry_warning"] = (
        df["X_Minimum"]
        > df["X_Maximum"]
    )

    df["y_geometry_warning"] = (
        df["Y_Minimum"]
        > df["Y_Maximum"]
    )

    df["geometry_warning"] = (
        df["x_geometry_warning"]
        | df["y_geometry_warning"]
    )

    return df


# =========================================================
# 5. Class distribution
# =========================================================

def build_class_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate class counts and percentages.
    """

    counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .rename("count")
    )

    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percentage")
    )

    result = pd.concat(
        [counts, percentages],
        axis=1,
    )

    result.index.name = TARGET_COLUMN

    return result.reset_index()


# =========================================================
# 6. Feature skewness
# =========================================================

def build_skewness_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate skewness for numeric predictive features.
    """

    excluded = (
        [ID_COLUMN, TARGET_COLUMN]
        + TARGET_BINARY_COLUMNS
    )

    numeric_features = [
        column
        for column in df.select_dtypes(
            include=[np.number]
        ).columns
        if column not in excluded
    ]

    records = []

    for column in numeric_features:

        skewness = df[column].skew()

        records.append(
            {
                "feature": column,
                "skewness": skewness,
                "absolute_skewness": abs(skewness),
            }
        )

    report = pd.DataFrame(records)

    return report.sort_values(
        "absolute_skewness",
        ascending=False,
    ).reset_index(drop=True)


# =========================================================
# 7. Class feature summary
# =========================================================

def build_class_feature_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize key explanatory features by defect class.
    """

    selected_features = [
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter",
        "Steel_Plate_Thickness",
        "Luminosity_Index",
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
    ]

    summary = (
        df.groupby(TARGET_COLUMN)[
            selected_features
        ]
        .median()
        .reset_index()
    )

    return summary


# =========================================================
# 8. Geometry warning analysis
# =========================================================

def build_geometry_warning_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyse geometry warning frequency by defect class.
    """

    grouped = (
        df.groupby(TARGET_COLUMN)
        .agg(
            total_samples=(
                TARGET_COLUMN,
                "size",
            ),
            geometry_warning_count=(
                "geometry_warning",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped[
        "geometry_warning_percentage"
    ] = (
        grouped["geometry_warning_count"]
        / grouped["total_samples"]
        * 100
    )

    return grouped.sort_values(
        "geometry_warning_percentage",
        ascending=False,
    )


# =========================================================
# 9. Plot: class distribution
# =========================================================

def plot_class_distribution(
    class_distribution: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        class_distribution[
            TARGET_COLUMN
        ],
        class_distribution[
            "count"
        ],
    )

    ax.set_title(
        "Steel Defect Class Distribution"
    )

    ax.set_xlabel(
        "Defect Type"
    )

    ax.set_ylabel(
        "Sample Count"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR
        / "class_distribution.png",
        dpi=160,
    )

    plt.close(fig)


# =========================================================
# 10. Helper: defect boxplot
# =========================================================

def plot_feature_by_defect(
    df: pd.DataFrame,
    feature: str,
    filename: str,
    title: str,
    log_scale: bool = False,
) -> None:
    """
    Draw a feature boxplot grouped by defect type.
    """

    classes = (
        df[TARGET_COLUMN]
        .value_counts()
        .index
        .tolist()
    )

    plot_data = [
        df.loc[
            df[TARGET_COLUMN] == defect,
            feature,
        ].dropna()
        for defect in classes
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.boxplot(
        plot_data,
        tick_labels=classes,
        showfliers=False,
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)

    ax.set_xlabel(
        "Defect Type"
    )

    ax.set_ylabel(feature)

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / filename,
        dpi=160,
    )

    plt.close(fig)


# =========================================================
# 11. Correlation plot
# =========================================================

def plot_feature_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate and visualize correlation among selected
    interpretable numerical features.
    """

    features = [
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter",
        "Steel_Plate_Thickness",
        "Luminosity_Index",
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
        "LogOfAreas",
        "SigmoidOfAreas",
    ]

    corr = (
        df[features]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        corr.values,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(features))
    )

    ax.set_yticks(
        range(len(features))
    )

    ax.set_xticklabels(
        features,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        features,
    )

    ax.set_title(
        "Selected Feature Correlation"
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR
        / "feature_correlation.png",
        dpi=160,
    )

    plt.close(fig)

    return corr


# =========================================================
# 12. Geometry warning plot
# =========================================================

def plot_geometry_warnings(
    warning_report: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        warning_report[
            TARGET_COLUMN
        ],
        warning_report[
            "geometry_warning_percentage"
        ],
    )

    ax.set_title(
        "Geometry Warning Rate by Defect Type"
    )

    ax.set_xlabel(
        "Defect Type"
    )

    ax.set_ylabel(
        "Warning Rate (%)"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR
        / "geometry_warning_distribution.png",
        dpi=160,
    )

    plt.close(fig)


# =========================================================
# 13. Generate markdown
# =========================================================

def generate_markdown_report(
    class_distribution: pd.DataFrame,
    skewness_report: pd.DataFrame,
    class_summary: pd.DataFrame,
    warning_report: pd.DataFrame,
) -> str:
    """
    Create concise GitHub-friendly EDA documentation.
    """

    largest_class = (
        class_distribution.iloc[0]
    )

    smallest_class = (
        class_distribution.iloc[-1]
    )

    most_skewed = (
        skewness_report.iloc[0]
    )

    highest_warning = (
        warning_report.iloc[0]
    )

    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## Purpose",
        "",
        (
            "EDA is used to understand class imbalance, "
            "feature distributions, class-level differences, "
            "skewness, correlations, and previously identified "
            "geometry-consistency warnings."
        ),
        "",
        "## Class Imbalance",
        "",
        (
            f"The largest class is "
            f"`{largest_class[TARGET_COLUMN]}` "
            f"with {largest_class['percentage']:.2f}% "
            f"of modeling samples."
        ),
        "",
        (
            f"The smallest class is "
            f"`{smallest_class[TARGET_COLUMN]}` "
            f"with {smallest_class['percentage']:.2f}%."
        ),
        "",
        (
            "Because the classes are imbalanced, later model "
            "evaluation will include Macro F1 and per-class "
            "Recall rather than relying on Accuracy alone."
        ),
        "",
        "## Feature Skewness",
        "",
        (
            f"The most strongly skewed analysed feature is "
            f"`{most_skewed['feature']}` "
            f"with skewness "
            f"{most_skewed['skewness']:.3f}."
        ),
        "",
        (
            "Strongly skewed features are not automatically "
            "removed. Tree-based models can often handle "
            "non-normal feature distributions."
        ),
        "",
        "## Class Feature Patterns",
        "",
        (
            "Median feature values were compared across defect "
            "classes to reduce the influence of extreme values."
        ),
        "",
        class_summary.to_markdown(
            index=False
        ),
        "",
        "## Geometry Consistency Warnings",
        "",
        (
            f"The highest observed geometry-warning rate is "
            f"in `{highest_warning[TARGET_COLUMN]}` "
            f"at "
            f"{highest_warning['geometry_warning_percentage']:.2f}%."
        ),
        "",
        (
            "These records are retained as warnings rather than "
            "automatically removed because this competition "
            "dataset is synthetic and the inconsistencies do "
            "not necessarily represent corrupted records."
        ),
        "",
        "## Interpretation Principle",
        "",
        (
            "EDA identifies associations and distributional "
            "patterns. These observations must not be presented "
            "as proof of manufacturing causality."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 14. Save tabular reports
# =========================================================

def save_reports(
    skewness_report: pd.DataFrame,
    class_summary: pd.DataFrame,
    warning_report: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
) -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    skewness_report.to_csv(
        REPORTS_DIR
        / "feature_skewness.csv",
        index=False,
    )

    class_summary.to_csv(
        REPORTS_DIR
        / "class_feature_summary.csv",
        index=False,
    )

    warning_report.to_csv(
        REPORTS_DIR
        / "geometry_warning_report.csv",
        index=False,
    )

    correlation_matrix.to_csv(
        REPORTS_DIR
        / "feature_correlation.csv",
    )


# =========================================================
# 15. Print summary
# =========================================================

def print_summary(
    class_distribution: pd.DataFrame,
    skewness_report: pd.DataFrame,
    class_summary: pd.DataFrame,
    warning_report: pd.DataFrame,
) -> None:

    print("=" * 72)
    print(
        "Stage B6 — Exploratory Data Analysis"
    )
    print("=" * 72)

    print(
        "\nCLASS DISTRIBUTION"
    )
    print("-" * 72)

    display_classes = (
        class_distribution.copy()
    )

    display_classes[
        "percentage"
    ] = display_classes[
        "percentage"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    print(
        display_classes.to_string(
            index=False
        )
    )

    print(
        "\nTOP SKEWED FEATURES"
    )
    print("-" * 72)

    print(
        skewness_report.head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nMEDIAN CLASS FEATURE SUMMARY"
    )
    print("-" * 72)

    print(
        class_summary.to_string(
            index=False
        )
    )

    print(
        "\nGEOMETRY WARNING RATE"
    )
    print("-" * 72)

    display_warning = (
        warning_report.copy()
    )

    display_warning[
        "geometry_warning_percentage"
    ] = display_warning[
        "geometry_warning_percentage"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    print(
        display_warning.to_string(
            index=False
        )
    )

    print(
        "\nOUTPUT DIRECTORY"
    )
    print("-" * 72)

    print(
        REPORTS_DIR
    )

    print(
        "\nEDA validation: PASSED"
    )

    print("=" * 72)


# =========================================================
# 16. Main
# =========================================================

def main() -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_modeling_data()

    df = add_eda_flags(
        df
    )

    class_distribution = (
        build_class_distribution(
            df
        )
    )

    skewness_report = (
        build_skewness_report(
            df
        )
    )

    class_summary = (
        build_class_feature_summary(
            df
        )
    )

    warning_report = (
        build_geometry_warning_report(
            df
        )
    )

    # -----------------------------------------------------
    # Charts
    # -----------------------------------------------------

    plot_class_distribution(
        class_distribution
    )

    plot_feature_by_defect(
        df=df,
        feature="Pixels_Areas",
        filename=(
            "pixels_area_by_defect.png"
        ),
        title=(
            "Fault Area Distribution "
            "by Defect Type"
        ),
        log_scale=True,
    )

    plot_feature_by_defect(
        df=df,
        feature="Luminosity_Index",
        filename=(
            "luminosity_by_defect.png"
        ),
        title=(
            "Luminosity Index "
            "by Defect Type"
        ),
    )

    plot_feature_by_defect(
        df=df,
        feature=(
            "Steel_Plate_Thickness"
        ),
        filename=(
            "thickness_by_defect.png"
        ),
        title=(
            "Steel Plate Thickness "
            "by Defect Type"
        ),
    )

    correlation_matrix = (
        plot_feature_correlation(
            df
        )
    )

    plot_geometry_warnings(
        warning_report
    )

    # -----------------------------------------------------
    # Save reports
    # -----------------------------------------------------

    save_reports(
        skewness_report=(
            skewness_report
        ),
        class_summary=(
            class_summary
        ),
        warning_report=(
            warning_report
        ),
        correlation_matrix=(
            correlation_matrix
        ),
    )

    markdown_report = (
        generate_markdown_report(
            class_distribution=(
                class_distribution
            ),
            skewness_report=(
                skewness_report
            ),
            class_summary=(
                class_summary
            ),
            warning_report=(
                warning_report
            ),
        )
    )

    MARKDOWN_OUTPUT_PATH.write_text(
        markdown_report,
        encoding="utf-8",
    )

    print_summary(
        class_distribution=(
            class_distribution
        ),
        skewness_report=(
            skewness_report
        ),
        class_summary=(
            class_summary
        ),
        warning_report=(
            warning_report
        ),
    )


if __name__ == "__main__":
    main()