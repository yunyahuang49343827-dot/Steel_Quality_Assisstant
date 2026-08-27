from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.tools.quality_tools import (
    FEATURE_COLUMNS,
    VALID_DEFECT_TYPES,
    get_defect_distribution,
    get_defect_drivers,
    get_quality_overview,
)


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GLOBAL_SHAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
    / "shap"
    / "global_feature_importance.csv"
)

TEST_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)


# =========================================================
# 2. Locked model evaluation
# =========================================================

# These metrics are the one-time B12 holdout-test results.
# They are fixed evaluation evidence, not live production KPIs.

MODEL_SUMMARY = {
    "champion_model": "XGBoost",
    "selection_metric": "Validation Macro F1",
    "test_accuracy": 0.5851,
    "test_macro_precision": 0.5615,
    "test_macro_recall": 0.6471,
    "test_macro_f1": 0.5904,
    "test_weighted_f1": 0.5798,
}


PER_CLASS_TEST_METRICS = [
    {
        "defect_type": "Bumps",
        "precision": 0.5345,
        "recall": 0.5854,
        "f1": 0.5588,
        "support": 714,
    },
    {
        "defect_type": "Dirtiness",
        "precision": 0.3019,
        "recall": 0.4384,
        "f1": 0.3575,
        "support": 73,
    },
    {
        "defect_type": "K_Scatch",
        "precision": 0.8931,
        "recall": 0.9297,
        "f1": 0.9110,
        "support": 512,
    },
    {
        "defect_type": "Other_Faults",
        "precision": 0.5827,
        "recall": 0.3629,
        "f1": 0.4472,
        "support": 981,
    },
    {
        "defect_type": "Pastry",
        "precision": 0.3316,
        "recall": 0.5636,
        "f1": 0.4175,
        "support": 220,
    },
    {
        "defect_type": "Stains",
        "precision": 0.7653,
        "recall": 0.8824,
        "f1": 0.8197,
        "support": 85,
    },
    {
        "defect_type": "Z_Scratch",
        "precision": 0.5217,
        "recall": 0.7674,
        "f1": 0.6212,
        "support": 172,
    },
]


# =========================================================
# 3. Global SHAP fallback evidence
# =========================================================

# Used only if the B14 global SHAP CSV name differs or
# cannot be located. Values come from the completed B14 run.

GLOBAL_SHAP_FALLBACK = [
    {
        "rank": 1,
        "feature": "Steel_Plate_Thickness",
        "mean_abs_shap": 0.574158,
    },
    {
        "rank": 2,
        "feature": "Length_of_Conveyer",
        "mean_abs_shap": 0.330603,
    },
    {
        "rank": 3,
        "feature": "Orientation_Index",
        "mean_abs_shap": 0.197145,
    },
    {
        "rank": 4,
        "feature": "Pixels_Areas",
        "mean_abs_shap": 0.184514,
    },
    {
        "rank": 5,
        "feature": "Edges_Y_Index",
        "mean_abs_shap": 0.177971,
    },
    {
        "rank": 6,
        "feature": "Minimum_of_Luminosity",
        "mean_abs_shap": 0.177410,
    },
    {
        "rank": 7,
        "feature": "Luminosity_Index",
        "mean_abs_shap": 0.176544,
    },
    {
        "rank": 8,
        "feature": "Outside_X_Index",
        "mean_abs_shap": 0.167393,
    },
    {
        "rank": 9,
        "feature": "LogOfAreas",
        "mean_abs_shap": 0.151835,
    },
    {
        "rank": 10,
        "feature": "Log_X_Index",
        "mean_abs_shap": 0.147461,
    },
]


# =========================================================
# 4. Overview
# =========================================================

def build_quality_overview() -> Dict[str, Any]:
    """
    Combine live quality statistics with locked model
    evaluation metadata for Dashboard KPI cards.
    """

    overview = get_quality_overview()

    return {
        "modeling_samples":
            overview["total_samples"],

        "defect_classes":
            overview["defect_classes"],

        "champion_model":
            MODEL_SUMMARY["champion_model"],

        "test_macro_f1":
            MODEL_SUMMARY["test_macro_f1"],

        "test_macro_f1_percentage":
            round(
                MODEL_SUMMARY["test_macro_f1"] * 100,
                2,
            ),

        "feature_count":
            len(FEATURE_COLUMNS),
    }


# =========================================================
# 5. Defect distribution
# =========================================================

def build_defect_distribution() -> List[Dict[str, Any]]:
    """
    Return live PostgreSQL defect counts and percentages.
    """

    return get_defect_distribution()


# =========================================================
# 6. Model performance
# =========================================================

def build_model_performance() -> Dict[str, Any]:
    """
    Return locked B12 holdout-test evaluation.

    This is model evaluation evidence, not live
    manufacturing performance.
    """

    return {
        "summary":
            MODEL_SUMMARY,

        "per_class":
            PER_CLASS_TEST_METRICS,

        "interpretation_note": (
            "Metrics come from the one-time reserved "
            "holdout test set. Per-class recall is shown "
            "because class imbalance makes aggregate "
            "accuracy insufficient on its own."
        ),
    }


# =========================================================
# 7. Global SHAP
# =========================================================

def _try_load_global_shap() -> pd.DataFrame:
    """
    Try known B14 SHAP report paths before using the
    locked fallback values.
    """

    candidate_paths = [
        GLOBAL_SHAP_PATH,

        PROJECT_ROOT
        / "reports"
        / "explainability"
        / "shap"
        / "global_shap_importance.csv",

        PROJECT_ROOT
        / "reports"
        / "explainability"
        / "shap"
        / "global_top_features.csv",

        PROJECT_ROOT
        / "reports"
        / "explainability"
        / "shap_global_feature_importance.csv",
    ]

    for path in candidate_paths:

        if path.exists():

            df = pd.read_csv(path)

            if {
                "feature",
                "mean_abs_shap",
            }.issubset(
                df.columns
            ):

                return df

    return pd.DataFrame(
        GLOBAL_SHAP_FALLBACK
    )


def build_global_shap(
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Return global SHAP model drivers.
    """

    if top_n < 1 or top_n > 20:

        raise ValueError(
            "top_n must be between 1 and 20."
        )

    df = _try_load_global_shap()

    if "rank" not in df.columns:

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
            df.index
            + 1
        )

    df = (
        df
        .sort_values(
            "rank"
        )
        .head(
            top_n
        )
    )

    drivers = []

    for _, row in df.iterrows():

        drivers.append(
            {
                "rank":
                    int(
                        row["rank"]
                    ),

                "feature":
                    str(
                        row["feature"]
                    ),

                "mean_abs_shap":
                    float(
                        row["mean_abs_shap"]
                    ),
            }
        )

    return {
        "drivers":
            drivers,

        "interpretation_note": (
            "Global SHAP values describe how strongly "
            "features influence model predictions on "
            "average. They do not establish confirmed "
            "manufacturing root causes."
        ),
    }


# =========================================================
# 8. Defect intelligence
# =========================================================

def build_defect_intelligence(
    defect_type: str,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Combine class distribution, holdout performance,
    and per-class SHAP evidence.
    """

    if defect_type not in VALID_DEFECT_TYPES:

        raise ValueError(
            f"Unknown defect_type: {defect_type}"
        )

    distribution = (
        get_defect_distribution()
    )

    distribution_row = next(
        (
            item
            for item in distribution
            if item["defect_type"]
            == defect_type
        ),
        None,
    )

    performance_row = next(
        (
            item
            for item in PER_CLASS_TEST_METRICS
            if item["defect_type"]
            == defect_type
        ),
        None,
    )

    drivers = get_defect_drivers(
        defect_type=defect_type,
        top_n=top_n,
    )

    return {
        "defect_type":
            defect_type,

        "samples":
            (
                distribution_row[
                    "sample_count"
                ]
                if distribution_row
                else None
            ),

        "dataset_share":
            (
                distribution_row[
                    "percentage"
                ]
                if distribution_row
                else None
            ),

        "test_precision":
            (
                performance_row[
                    "precision"
                ]
                if performance_row
                else None
            ),

        "test_recall":
            (
                performance_row[
                    "recall"
                ]
                if performance_row
                else None
            ),

        "test_f1":
            (
                performance_row[
                    "f1"
                ]
                if performance_row
                else None
            ),

        "drivers":
            drivers,

        "interpretation_note": (
            "SHAP drivers explain predictive model "
            "behavior and are not confirmed physical "
            "root causes."
        ),
    }


# =========================================================
# 9. Demo sample
# =========================================================

def build_demo_sample() -> Dict[str, Any]:
    """
    Return one deterministic reserved test sample for
    the Prediction Lab.

    The label is intentionally not returned to the
    frontend so the sample behaves like inference input.
    """

    if not TEST_SPLIT_PATH.exists():

        raise FileNotFoundError(
            "Reserved test split not found."
        )

    df = pd.read_csv(
        TEST_SPLIT_PATH
    )

    if df.empty:

        raise RuntimeError(
            "Reserved test split is empty."
        )

    sample = df.iloc[0]

    features = {
        feature:
            (
                int(sample[feature])
                if feature.startswith(
                    "TypeOfSteel_"
                )
                else float(
                    sample[feature]
                )
            )
        for feature in FEATURE_COLUMNS
    }

    return {
        "sample_id":
            int(
                sample["id"]
            ),

        "features":
            features,

        "note": (
            "Reserved demonstration sample for "
            "Prediction Lab inference."
        ),
    }