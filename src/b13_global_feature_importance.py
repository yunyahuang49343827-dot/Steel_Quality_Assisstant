from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "champion_model.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
)

DOCS_DIR = (
    PROJECT_ROOT
    / "docs"
)

GLOBAL_IMPORTANCE_CSV_PATH = (
    REPORT_DIR
    / "global_feature_importance.csv"
)

GLOBAL_IMPORTANCE_IMAGE_PATH = (
    REPORT_DIR
    / "global_feature_importance.png"
)

GROUP_IMPORTANCE_CSV_PATH = (
    REPORT_DIR
    / "feature_group_importance.csv"
)

GROUP_IMPORTANCE_IMAGE_PATH = (
    REPORT_DIR
    / "feature_group_importance.png"
)

SUMMARY_CSV_PATH = (
    REPORT_DIR
    / "global_feature_importance_summary.csv"
)

MARKDOWN_PATH = (
    DOCS_DIR
    / "global_feature_importance.md"
)


# =========================================================
# 2. Feature groups
# =========================================================

FEATURE_GROUPS = {

    "Geometry / Position": [
        "X_Minimum",
        "X_Maximum",
        "Y_Minimum",
        "Y_Maximum",
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter",
    ],

    "Shape / Edge Index": [
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
        "Outside_X_Index",
        "Edges_X_Index",
        "Edges_Y_Index",
        "Outside_Global_Index",
        "Orientation_Index",
    ],

    "Luminosity": [
        "Sum_of_Luminosity",
        "Minimum_of_Luminosity",
        "Maximum_of_Luminosity",
        "Luminosity_Index",
    ],

    "Steel / Production": [
        "Length_of_Conveyer",
        "TypeOfSteel_A300",
        "TypeOfSteel_A400",
        "Steel_Plate_Thickness",
    ],

    "Log Transformation": [
        "LogOfAreas",
        "Log_X_Index",
        "Log_Y_Index",
    ],

    "Area Transformation": [
        "SigmoidOfAreas",
    ],
}


# =========================================================
# 3. Load champion model
# =========================================================

def load_champion_bundle():
    """
    Load the champion model selected in Stage B12.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Champion model not found:\n"
            f"{MODEL_PATH}\n"
            "Please complete Stage B12 first."
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        bundle,
        dict,
    ):
        raise TypeError(
            "Champion model artifact is expected "
            "to be a dictionary bundle."
        )

    required_keys = [
        "model",
        "feature_columns",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in bundle
    ]

    if missing_keys:

        raise KeyError(
            "Champion model bundle is missing: "
            f"{missing_keys}"
        )

    return bundle


# =========================================================
# 4. Validate feature schema
# =========================================================

def validate_feature_schema(
    feature_columns,
):
    """
    Ensure all 27 champion-model features are accounted
    for exactly once in the B2 feature-group taxonomy.
    """

    grouped_features = []

    for features in (
        FEATURE_GROUPS.values()
    ):
        grouped_features.extend(
            features
        )

    feature_set = set(
        feature_columns
    )

    grouped_set = set(
        grouped_features
    )

    missing_from_groups = (
        feature_set
        - grouped_set
    )

    extra_in_groups = (
        grouped_set
        - feature_set
    )

    duplicate_features = [
        feature
        for feature in grouped_features
        if grouped_features.count(
            feature
        ) > 1
    ]

    duplicate_features = sorted(
        set(
            duplicate_features
        )
    )

    if missing_from_groups:

        raise ValueError(
            "Features missing from group taxonomy: "
            f"{sorted(missing_from_groups)}"
        )

    if extra_in_groups:

        raise ValueError(
            "Grouped features not found in model: "
            f"{sorted(extra_in_groups)}"
        )

    if duplicate_features:

        raise ValueError(
            "Features assigned to multiple groups: "
            f"{duplicate_features}"
        )

    if len(
        feature_columns
    ) != 27:

        raise ValueError(
            "Expected 27 champion-model features, "
            f"found {len(feature_columns)}."
        )


# =========================================================
# 5. Feature-to-group mapping
# =========================================================

def build_feature_group_map():
    """
    Convert FEATURE_GROUPS into feature -> group mapping.
    """

    mapping = {}

    for group_name, features in (
        FEATURE_GROUPS.items()
    ):

        for feature in features:

            mapping[
                feature
            ] = group_name

    return mapping


# =========================================================
# 6. Global feature importance
# =========================================================

def build_global_feature_importance(
    model,
    feature_columns,
):
    """
    Extract native XGBoost feature importance from the
    champion model.

    This is model-level predictive importance and must
    not be interpreted as manufacturing causality.
    """

    if not hasattr(
        model,
        "feature_importances_",
    ):

        raise AttributeError(
            "Champion model does not expose "
            "feature_importances_."
        )

    importances = (
        model.feature_importances_
    )

    if len(
        importances
    ) != len(
        feature_columns
    ):

        raise ValueError(
            "Feature importance length does not match "
            "saved feature schema."
        )

    feature_group_map = (
        build_feature_group_map()
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "feature_group": [
                feature_group_map[
                    feature
                ]
                for feature
                in feature_columns
            ],

            "importance":
                importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_df[
        "rank"
    ] = (
        importance_df.index
        + 1
    )

    total_importance = (
        importance_df[
            "importance"
        ].sum()
    )

    if total_importance > 0:

        importance_df[
            "importance_share"
        ] = (
            importance_df[
                "importance"
            ]
            / total_importance
        )

    else:

        importance_df[
            "importance_share"
        ] = 0.0

    importance_df[
        "cumulative_importance_share"
    ] = (
        importance_df[
            "importance_share"
        ].cumsum()
    )

    return importance_df


# =========================================================
# 7. Feature-group importance
# =========================================================

def build_group_importance(
    importance_df,
):
    """
    Aggregate feature importance into the six B2
    feature groups.
    """

    group_df = (
        importance_df
        .groupby(
            "feature_group",
            as_index=False,
        )
        .agg(
            total_importance=(
                "importance",
                "sum",
            ),
            feature_count=(
                "feature",
                "count",
            ),
            mean_feature_importance=(
                "importance",
                "mean",
            ),
        )
    )

    total = (
        group_df[
            "total_importance"
        ].sum()
    )

    if total > 0:

        group_df[
            "importance_share"
        ] = (
            group_df[
                "total_importance"
            ]
            / total
        )

    else:

        group_df[
            "importance_share"
        ] = 0.0

    group_df = (
        group_df
        .sort_values(
            "total_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    group_df[
        "rank"
    ] = (
        group_df.index
        + 1
    )

    return group_df


# =========================================================
# 8. Plot global feature importance
# =========================================================

def plot_global_importance(
    importance_df,
):
    """
    Plot top 15 champion-model features.
    """

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        top_features[
            "feature"
        ],
        top_features[
            "importance"
        ],
    )

    ax.set_xlabel(
        "XGBoost Native Feature Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Champion Model — "
        "Top 15 Global Feature Importance"
    )

    fig.tight_layout()

    fig.savefig(
        GLOBAL_IMPORTANCE_IMAGE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# 9. Plot group importance
# =========================================================

def plot_group_importance(
    group_df,
):
    """
    Plot aggregated feature-group importance.
    """

    plot_df = (
        group_df
        .sort_values(
            "total_importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        plot_df[
            "feature_group"
        ],
        plot_df[
            "total_importance"
        ],
    )

    ax.set_xlabel(
        "Aggregated Feature Importance"
    )

    ax.set_ylabel(
        "Feature Group"
    )

    ax.set_title(
        "Champion Model — "
        "Feature Group Importance"
    )

    fig.tight_layout()

    fig.savefig(
        GROUP_IMPORTANCE_IMAGE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# 10. Build concise summary
# =========================================================

def build_summary(
    importance_df,
    group_df,
):
    """
    Produce concise report-level summary statistics.
    """

    top_feature = (
        importance_df.iloc[0]
    )

    top_group = (
        group_df.iloc[0]
    )

    features_for_50_percent = (
        importance_df[
            importance_df[
                "cumulative_importance_share"
            ] <= 0.50
        ]
        .shape[0]
    )

    if (
        features_for_50_percent
        < len(
            importance_df
        )
    ):
        features_for_50_percent += 1

    features_for_80_percent = (
        importance_df[
            importance_df[
                "cumulative_importance_share"
            ] <= 0.80
        ]
        .shape[0]
    )

    if (
        features_for_80_percent
        < len(
            importance_df
        )
    ):
        features_for_80_percent += 1

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "top_feature",

                "value":
                    top_feature[
                        "feature"
                    ],
            },

            {
                "metric":
                    "top_feature_importance",

                "value":
                    top_feature[
                        "importance"
                    ],
            },

            {
                "metric":
                    "top_feature_group",

                "value":
                    top_group[
                        "feature_group"
                    ],
            },

            {
                "metric":
                    "top_group_importance_share",

                "value":
                    top_group[
                        "importance_share"
                    ],
            },

            {
                "metric":
                    "features_for_50_percent_importance",

                "value":
                    features_for_50_percent,
            },

            {
                "metric":
                    "features_for_80_percent_importance",

                "value":
                    features_for_80_percent,
            },
        ]
    )

    return summary_df


# =========================================================
# 11. Save reports
# =========================================================

def save_outputs(
    importance_df,
    group_df,
    summary_df,
):
    """
    Save B13 analysis outputs.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        GLOBAL_IMPORTANCE_CSV_PATH,
        index=False,
    )

    group_df.to_csv(
        GROUP_IMPORTANCE_CSV_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )


# =========================================================
# 12. Markdown report
# =========================================================

def generate_markdown_report(
    importance_df,
    group_df,
):
    """
    Generate GitHub-friendly global-importance report.
    """

    top_15 = (
        importance_df
        .head(15)
    )

    top_feature = (
        importance_df.iloc[0]
    )

    top_group = (
        group_df.iloc[0]
    )

    lines = [
        "# Global Feature Importance",
        "",
        "## Purpose",
        "",
        (
            "This stage evaluates which structured "
            "quality features the selected champion "
            "XGBoost model relies on most strongly "
            "across all defect predictions."
        ),
        "",
        "## Champion Model",
        "",
        "**Tuned Weighted XGBoost**",
        "",
        "## Top 15 Features",
        "",
        top_15[
            [
                "rank",
                "feature",
                "feature_group",
                "importance",
                "importance_share",
            ]
        ].to_markdown(
            index=False
        ),
        "",
        "## Feature Group Importance",
        "",
        group_df[
            [
                "rank",
                "feature_group",
                "feature_count",
                "total_importance",
                "importance_share",
                "mean_feature_importance",
            ]
        ].to_markdown(
            index=False
        ),
        "",
        "## Main Observations",
        "",
        (
            f"- Highest-ranked individual feature: "
            f"`{top_feature['feature']}`."
        ),
        (
            f"- Highest-ranked feature group: "
            f"`{top_group['feature_group']}`."
        ),
        "",
        "## Interpretation Guardrail",
        "",
        (
            "XGBoost native feature importance measures "
            "how strongly the model uses available "
            "predictors during classification."
        ),
        "",
        (
            "**Feature importance does not establish "
            "manufacturing causality or root cause.**"
        ),
        "",
        (
            "Correlated or derived features can distribute "
            "importance across related variables, so this "
            "analysis should be treated as a global model "
            "overview rather than a causal explanation."
        ),
        "",
        "## Next Stage",
        "",
        (
            "Stage B14 applies SHAP to provide more detailed "
            "global, per-class, and individual prediction "
            "explanations."
        ),
        "",
    ]

    return "\n".join(
        lines
    )


# =========================================================
# 13. Print summary
# =========================================================

def print_summary(
    importance_df,
    group_df,
):
    """
    Print B13 results in terminal.
    """

    print("=" * 72)

    print(
        "Stage B13 — Global Feature Importance"
    )

    print("=" * 72)

    print("\nCHAMPION MODEL")
    print("-" * 72)

    print(
        "Tuned Weighted XGBoost"
    )

    print("\nTOP 15 FEATURES")
    print("-" * 72)

    print(
        importance_df[
            [
                "rank",
                "feature",
                "feature_group",
                "importance",
                "importance_share",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print("\nFEATURE GROUP IMPORTANCE")
    print("-" * 72)

    print(
        group_df[
            [
                "rank",
                "feature_group",
                "feature_count",
                "total_importance",
                "importance_share",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nINTERPRETATION")
    print("-" * 72)

    print(
        "Importance = predictive model usage, "
        "NOT manufacturing causality."
    )

    print(
        "\nGlobal feature importance: PASSED"
    )

    print("=" * 72)


# =========================================================
# 14. Main
# =========================================================

def main():

    bundle = (
        load_champion_bundle()
    )

    model = bundle[
        "model"
    ]

    feature_columns = list(
        bundle[
            "feature_columns"
        ]
    )

    validate_feature_schema(
        feature_columns
    )

    importance_df = (
        build_global_feature_importance(
            model=model,
            feature_columns=(
                feature_columns
            ),
        )
    )

    group_df = (
        build_group_importance(
            importance_df
        )
    )

    summary_df = (
        build_summary(
            importance_df,
            group_df,
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_global_importance(
        importance_df
    )

    plot_group_importance(
        group_df
    )

    save_outputs(
        importance_df=(
            importance_df
        ),
        group_df=(
            group_df
        ),
        summary_df=(
            summary_df
        ),
    )

    markdown = (
        generate_markdown_report(
            importance_df=(
                importance_df
            ),
            group_df=(
                group_df
            ),
        )
    )

    MARKDOWN_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print_summary(
        importance_df=(
            importance_df
        ),
        group_df=(
            group_df
        ),
    )


if __name__ == "__main__":
    main()