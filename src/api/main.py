

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from src.api.dashboard_service import (
    build_defect_distribution,
    build_defect_intelligence,
    build_demo_sample,
    build_global_shap,
    build_model_performance,
    build_quality_overview,
)

from src.config import (
    APP_ENV,
    APP_VERSION,
    CORS_ORIGINS,
    DEPLOYMENT_VARIANT,
)

from src.copilot.agent import (
    run_copilot,
)

from src.tools.quality_tools import (
    PredictionFeatures,
    explain_prediction,
    get_database_connection,
    predict_defect,
)


# =========================================================
# 1. FastAPI application
# =========================================================

app = FastAPI(
    title=(
        "Steel Quality Intelligence API"
    ),

    description=(
        "Backend API for steel quality analytics, "
        "XGBoost prediction, SHAP explainability, "
        "and grounded local AI Copilot."
    ),

    version="1.0.0",
)


# =========================================================
# 2. CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=(
        CORS_ORIGINS
    ),

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    allow_headers=[
        "*"
    ],
)


# =========================================================
# 3. Request models
# =========================================================

class PredictionRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    features: PredictionFeatures


class ExplanationRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    features: PredictionFeatures

    top_n: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class CopilotRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    question: str = Field(
        min_length=1,
        max_length=4000,
    )


# =========================================================
# 4. Health check
# =========================================================

@app.get(
    "/health",
    tags=[
        "System"
    ],
)
def health_check():

    database_status = (
        "healthy"
    )

    try:

        with (
            get_database_connection()
            as connection
        ):

            with (
                connection.cursor()
                as cursor
            ):

                cursor.execute(
                    "SELECT 1"
                )

                cursor.fetchone()

    except Exception:

        database_status = (
            "unavailable"
        )

    return {
        "status":
            (
                "healthy"
                if database_status
                == "healthy"
                else "degraded"
            ),

        "environment":
            APP_ENV,

        "model":
            "loaded",

        "database":
            database_status,
    }


# =========================================================
# 5. Deployment metadata
# =========================================================

@app.get(
    "/deployment",
    tags=[
        "System"
    ],
)
def deployment_metadata():
    """
    Return immutable deployment metadata.

    This endpoint is used to verify Stable / Canary
    routing without changing business behavior.
    """

    return {
        "variant":
            DEPLOYMENT_VARIANT,

        "version":
            APP_VERSION,

        "environment":
            APP_ENV,
    }


# =========================================================
# 5. Quality overview
# =========================================================

@app.get(
    "/quality/overview",
    tags=[
        "Quality Analytics"
    ],
)
def quality_overview():

    try:

        return (
            build_quality_overview()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "quality overview."
            ),
        ) from exc


# =========================================================
# 6. Defect distribution
# =========================================================

@app.get(
    "/quality/distribution",
    tags=[
        "Quality Analytics"
    ],
)
def quality_distribution():

    try:

        return {
            "distribution":
                build_defect_distribution()
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "defect distribution."
            ),
        ) from exc


# =========================================================
# 7. Model performance
# =========================================================

@app.get(
    "/model/performance",
    tags=[
        "Model"
    ],
)
def model_performance():

    return (
        build_model_performance()
    )


# =========================================================
# 8. Global SHAP drivers
# =========================================================

@app.get(
    "/explain/global",
    tags=[
        "Explainability"
    ],
)
def global_explanation(
    top_n: int = Query(
        default=10,
        ge=1,
        le=20,
    ),
):

    try:

        return (
            build_global_shap(
                top_n=top_n
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "global SHAP evidence."
            ),
        ) from exc


# =========================================================
# 9. Per-defect intelligence
# =========================================================

@app.get(
    "/explain/defect/{defect_type}",
    tags=[
        "Explainability"
    ],
)
def defect_explanation(
    defect_type: str,

    top_n: int = Query(
        default=5,
        ge=1,
        le=10,
    ),
):

    try:

        return (
            build_defect_intelligence(
                defect_type=defect_type,
                top_n=top_n,
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "defect intelligence."
            ),
        ) from exc


# =========================================================
# 10. Demo sample
# =========================================================

@app.get(
    "/demo/sample",
    tags=[
        "Prediction"
    ],
)
def demo_sample():

    try:

        return (
            build_demo_sample()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "demo sample."
            ),
        ) from exc


# =========================================================
# 11. Prediction
# =========================================================

@app.post(
    "/predict",
    tags=[
        "Prediction"
    ],
)
def prediction(
    request: PredictionRequest,
):

    try:

        return (
            predict_defect(
                request
                .features
                .model_dump()
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed."
            ),
        ) from exc


# =========================================================
# 12. Individual SHAP explanation
# =========================================================

@app.post(
    "/explain",
    tags=[
        "Explainability"
    ],
)
def individual_explanation(
    request: ExplanationRequest,
):

    try:

        return (
            explain_prediction(
                features=(
                    request
                    .features
                    .model_dump()
                ),

                top_n=(
                    request.top_n
                ),
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction explanation failed."
            ),
        ) from exc


# =========================================================
# 13. AI Quality Copilot
# =========================================================

@app.post(
    "/copilot/chat",
    tags=[
        "AI Copilot"
    ],
)
def copilot_chat(
    request: CopilotRequest,
):

    try:

        result = run_copilot(
            request.question
        )

        successful_tools = [
            item.get(
                "tool"
            )
            for item in (
                result.get(
                    "tool_trace",
                    []
                )
            )
            if item.get(
                "status"
            )
            == "success"
        ]

        return {
            "answer":
                result[
                    "answer"
                ],

            "model":
                result.get(
                    "model"
                ),

            "tools_used":
                successful_tools,

            "tool_trace":
                result.get(
                    "tool_trace",
                    []
                ),

            "policy_decision":
                result.get(
                    "policy_decision",
                    "allowed",
                ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "AI Copilot request failed."
            ),
        ) from exc