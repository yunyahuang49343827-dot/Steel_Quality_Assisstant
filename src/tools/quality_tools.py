from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import psycopg
import shap

from pydantic import BaseModel, ConfigDict, Field

from src.config import DB_CONFIG


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

PER_CLASS_SHAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
    / "shap"
    / "per_class_top_features.csv"
)


# =========================================================
# 2. Database connection
# =========================================================

def get_database_connection():
    """
    Return PostgreSQL connection using the centralized
    application database configuration.

    Credentials are loaded through src.config and are
    never exposed through tool results.
    """

    if not DB_CONFIG["user"]:

        raise RuntimeError(
            "DB_USER is not configured."
        )

    return psycopg.connect(
        **DB_CONFIG
    )


# =========================================================
# 3. Load champion model
# =========================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Champion model not found: {MODEL_PATH}"
    )


MODEL_BUNDLE = joblib.load(
    MODEL_PATH
)


MODEL = MODEL_BUNDLE[
    "model"
]


LABEL_ENCODER = MODEL_BUNDLE[
    "label_encoder"
]


FEATURE_COLUMNS = list(
    MODEL_BUNDLE[
        "feature_columns"
    ]
)


VALID_DEFECT_TYPES = list(
    LABEL_ENCODER.classes_
)


SHAP_EXPLAINER = shap.TreeExplainer(
    MODEL
)


# =========================================================
# 4. Prediction input schema
# =========================================================

class PredictionFeatures(
    BaseModel
):
    """
    Exact 27-feature model input schema.

    extra='forbid' prevents unexpected fields.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    X_Minimum: float

    X_Maximum: float

    Y_Minimum: float

    Y_Maximum: float

    Pixels_Areas: float = Field(
        ge=0
    )

    X_Perimeter: float = Field(
        ge=0
    )

    Y_Perimeter: float = Field(
        ge=0
    )

    Sum_of_Luminosity: float

    Minimum_of_Luminosity: float

    Maximum_of_Luminosity: float

    Length_of_Conveyer: float = Field(
        ge=0
    )

    TypeOfSteel_A300: int = Field(
        ge=0,
        le=1,
    )

    TypeOfSteel_A400: int = Field(
        ge=0,
        le=1,
    )

    Steel_Plate_Thickness: float = Field(
        ge=0
    )

    Edges_Index: float

    Empty_Index: float

    Square_Index: float

    Outside_X_Index: float

    Edges_X_Index: float

    Edges_Y_Index: float

    Outside_Global_Index: float

    LogOfAreas: float

    Log_X_Index: float

    Log_Y_Index: float

    Orientation_Index: float

    Luminosity_Index: float

    SigmoidOfAreas: float


# =========================================================
# 5. Feature validation helper
# =========================================================

def features_to_dataframe(
    features: Dict[str, Any],
) -> pd.DataFrame:
    """
    Validate feature payload and preserve the exact
    training feature order.
    """

    validated = PredictionFeatures(
        **features
    )

    values = validated.model_dump()

    ordered = {
        feature:
            values[feature]
        for feature
        in FEATURE_COLUMNS
    }

    return pd.DataFrame(
        [ordered]
    )


# =========================================================
# 6. Internal prediction helper
# =========================================================

def _run_prediction(
    X: pd.DataFrame,
):
    """
    Run the champion classifier.
    """

    encoded_prediction = (
        MODEL.predict(
            X
        )
    )

    probabilities = (
        MODEL.predict_proba(
            X
        )[0]
    )

    predicted_defect = (
        LABEL_ENCODER
        .inverse_transform(
            encoded_prediction
        )[0]
    )

    probability_map = {
        defect_type:
            float(
                probabilities[index]
            )
        for index, defect_type
        in enumerate(
            VALID_DEFECT_TYPES
        )
    }

    return {
        "predicted_defect":
            predicted_defect,

        "confidence":
            float(
                probabilities.max()
            ),

        "probabilities":
            probability_map,

        "_class_id":
            int(
                encoded_prediction[0]
            ),
    }


# =========================================================
# 7. SHAP normalization helper
# =========================================================

def _normalize_shap(
    shap_values,
):
    """
    Normalize multiclass SHAP output to:

    samples × features × classes
    """

    values = shap_values.values

    feature_count = len(
        FEATURE_COLUMNS
    )

    class_count = len(
        VALID_DEFECT_TYPES
    )

    if (
        values.ndim == 3
        and values.shape[1] == feature_count
        and values.shape[2] == class_count
    ):

        return values

    if (
        values.ndim == 3
        and values.shape[1] == class_count
        and values.shape[2] == feature_count
    ):

        return np.transpose(
            values,
            (0, 2, 1),
        )

    raise RuntimeError(
        "Unexpected SHAP output format."
    )


# =========================================================
# 8. Tool — quality overview
# =========================================================

def get_quality_overview() -> Dict[str, Any]:
    """
    Return high-level modeling dataset statistics.

    The caller cannot supply SQL.
    """

    query = """
        SELECT
            COUNT(*) AS total_samples,
            COUNT(DISTINCT defect_type)
                AS defect_classes
        FROM modeling_steel_quality
    """

    with (
        get_database_connection()
        as connection
    ):

        with connection.cursor() as cursor:

            cursor.execute(
                query
            )

            row = cursor.fetchone()

    return {
        "total_samples":
            int(
                row[0]
            ),

        "defect_classes":
            int(
                row[1]
            ),
    }


# =========================================================
# 9. Tool — defect distribution
# =========================================================

def get_defect_distribution() -> List[Dict[str, Any]]:
    """
    Return defect class counts and percentages.

    The SQL statement is fixed on the server side.
    """

    query = """
        SELECT
            defect_type,
            COUNT(*) AS sample_count
        FROM modeling_steel_quality
        GROUP BY defect_type
        ORDER BY sample_count DESC
    """

    with (
        get_database_connection()
        as connection
    ):

        with connection.cursor() as cursor:

            cursor.execute(
                query
            )

            rows = cursor.fetchall()

    total = sum(
        int(
            row[1]
        )
        for row in rows
    )

    return [
        {
            "defect_type":
                row[0],

            "sample_count":
                int(
                    row[1]
                ),

            "percentage":
                round(
                    int(
                        row[1]
                    )
                    / total
                    * 100,
                    2,
                ),
        }
        for row in rows
    ]


# =========================================================
# 10. Tool — high-confidence predictions
# =========================================================

def get_high_confidence_predictions(
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return a bounded list of model predictions with
    the highest prediction confidence.

    Prediction confidence measures model certainty only.

    It does NOT represent:
    - manufacturing risk
    - defect severity
    - business impact
    - confirmed quality priority

    This tool uses the reserved demonstration dataset.
    """

    if not isinstance(
        limit,
        int,
    ):

        raise ValueError(
            "limit must be an integer."
        )

    if limit < 1 or limit > 100:

        raise ValueError(
            "limit must be between 1 and 100."
        )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            "Demo inference dataset not found."
        )

    df = pd.read_csv(
        TEST_PATH
    )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    predictions = MODEL.predict(
        X
    )

    probabilities = MODEL.predict_proba(
        X
    )

    predicted_labels = (
        LABEL_ENCODER
        .inverse_transform(
            predictions
        )
    )

    result_df = pd.DataFrame(
        {
            "id":
                df["id"].values,

            "predicted_defect":
                predicted_labels,

            "confidence":
                probabilities.max(
                    axis=1
                ),
        }
    )

    result_df = (
        result_df
        .sort_values(
            "confidence",
            ascending=False,
        )
        .head(
            limit
        )
    )

    return (
        result_df
        .to_dict(
            orient="records"
        )
    )


# =========================================================
# 11. Tool — predict defect
# =========================================================

def predict_defect(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict steel defect class from validated features.
    """

    X = features_to_dataframe(
        features
    )

    result = _run_prediction(
        X
    )

    result.pop(
        "_class_id"
    )

    return result


# =========================================================
# 12. Tool — explain prediction
# =========================================================

def explain_prediction(
    features: Dict[str, Any],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Predict and explain one sample using SHAP.

    SHAP describes predictive model behavior only.
    """

    if not isinstance(
        top_n,
        int,
    ):

        raise ValueError(
            "top_n must be an integer."
        )

    if top_n < 1 or top_n > 10:

        raise ValueError(
            "top_n must be between 1 and 10."
        )

    X = features_to_dataframe(
        features
    )

    prediction = _run_prediction(
        X
    )

    class_id = prediction[
        "_class_id"
    ]

    shap_values = (
        SHAP_EXPLAINER(
            X
        )
    )

    shap_array = _normalize_shap(
        shap_values
    )

    row_shap = (
        shap_array[
            0,
            :,
            class_id
        ]
    )

    detail_df = pd.DataFrame(
        {
            "feature":
                FEATURE_COLUMNS,

            "feature_value":
                X.iloc[
                    0
                ].values,

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
        .head(
            top_n
        )
    )

    drivers = []

    for _, row in (
        detail_df.iterrows()
    ):

        value = float(
            row[
                "shap_value"
            ]
        )

        drivers.append(
            {
                "feature":
                    row[
                        "feature"
                    ],

                "feature_value":
                    float(
                        row[
                            "feature_value"
                        ]
                    ),

                "shap_value":
                    value,

                "direction":
                    (
                        "supports_prediction"
                        if value > 0
                        else "opposes_prediction"
                    ),
            }
        )

    return {
        "predicted_defect":
            prediction[
                "predicted_defect"
            ],

        "confidence":
            prediction[
                "confidence"
            ],

        "top_drivers":
            drivers,

        "interpretation_note":
            (
                "SHAP explains predictive model behavior. "
                "It does not establish manufacturing "
                "causality or confirmed root cause."
            ),
    }


# =========================================================
# 13. Tool — defect drivers
# =========================================================

def get_defect_drivers(
    defect_type: str,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return per-class SHAP model drivers from B14.

    These are predictive drivers only, not confirmed
    manufacturing root causes.
    """

    if defect_type not in (
        VALID_DEFECT_TYPES
    ):

        raise ValueError(
            "Invalid defect_type. "
            f"Allowed values: {VALID_DEFECT_TYPES}"
        )

    if not isinstance(
        top_n,
        int,
    ):

        raise ValueError(
            "top_n must be an integer."
        )

    if top_n < 1 or top_n > 10:

        raise ValueError(
            "top_n must be between 1 and 10."
        )

    if not PER_CLASS_SHAP_PATH.exists():

        raise FileNotFoundError(
            "Per-class SHAP report not found."
        )

    df = pd.read_csv(
        PER_CLASS_SHAP_PATH
    )

    result = (
        df[
            df[
                "class"
            ]
            == defect_type
        ]
        .sort_values(
            "rank"
        )
        .head(
            top_n
        )
    )

    return (
        result[
            [
                "class",
                "rank",
                "feature",
                "mean_abs_shap",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )


# =========================================================
# 14. Allowlisted tool registry
# =========================================================

TOOL_REGISTRY = {
    "get_quality_overview":
        get_quality_overview,

    "get_defect_distribution":
        get_defect_distribution,

    "get_high_confidence_predictions":
        get_high_confidence_predictions,

    "predict_defect":
        predict_defect,

    "explain_prediction":
        explain_prediction,

    "get_defect_drivers":
        get_defect_drivers,
}


# =========================================================
# 15. Safe dispatcher
# =========================================================

def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
):
    """
    Execute only explicitly allowlisted backend tools.

    Arbitrary function execution, SQL execution,
    shell execution, and unknown tools are not allowed.
    """

    if tool_name not in (
        TOOL_REGISTRY
    ):

        raise ValueError(
            f"Tool not allowed: {tool_name}"
        )

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            "Tool arguments must be a dictionary."
        )

    tool = TOOL_REGISTRY[
        tool_name
    ]

    return tool(
        **arguments
    )