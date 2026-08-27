import json

import requests


# =========================================================
# 1. Configuration
# =========================================================

BASE_URL = (
    "http://127.0.0.1:8000"
)

TIMEOUT = 180


# =========================================================
# 2. GET helper
# =========================================================

def get_json(
    path: str,
):

    response = requests.get(
        f"{BASE_URL}{path}",
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# 3. POST helper
# =========================================================

def post_json(
    path: str,
    payload,
):

    response = requests.post(
        f"{BASE_URL}{path}",
        json=payload,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# 4. Main
# =========================================================

def main():

    print("=" * 72)
    print(
        "Stage B19.1 — Dashboard API Smoke Test"
    )
    print("=" * 72)

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    print("\n[1] /health")

    health = get_json(
        "/health"
    )

    print(
        json.dumps(
            health,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        health["status"]
        in {
            "healthy",
            "degraded",
        }
    )

    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------

    print("\n[2] /quality/overview")

    overview = get_json(
        "/quality/overview"
    )

    print(
        json.dumps(
            overview,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        overview[
            "modeling_samples"
        ]
        == 18380
    )

    assert (
        overview[
            "defect_classes"
        ]
        == 7
    )

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    print(
        "\n[3] /quality/distribution"
    )

    distribution = get_json(
        "/quality/distribution"
    )

    rows = (
        distribution[
            "distribution"
        ]
    )

    print(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        rows[0][
            "defect_type"
        ]
        == "Other_Faults"
    )

    # -----------------------------------------------------
    # Model performance
    # -----------------------------------------------------

    print(
        "\n[4] /model/performance"
    )

    performance = get_json(
        "/model/performance"
    )

    print(
        json.dumps(
            performance,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        performance[
            "summary"
        ][
            "champion_model"
        ]
        == "XGBoost"
    )

    assert (
        len(
            performance[
                "per_class"
            ]
        )
        == 7
    )

    # -----------------------------------------------------
    # Global explainability
    # -----------------------------------------------------

    print(
        "\n[5] /explain/global?top_n=6"
    )

    global_shap = get_json(
        "/explain/global?top_n=6"
    )

    print(
        json.dumps(
            global_shap,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        len(
            global_shap[
                "drivers"
            ]
        )
        == 6
    )

    # -----------------------------------------------------
    # Defect intelligence
    # -----------------------------------------------------

    print(
        "\n[6] /explain/defect/K_Scatch"
    )

    defect = get_json(
        "/explain/defect/K_Scatch"
    )

    print(
        json.dumps(
            defect,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        defect[
            "defect_type"
        ]
        == "K_Scatch"
    )

    assert (
        defect[
            "samples"
        ]
        == 3411
    )

    # -----------------------------------------------------
    # Demo sample
    # -----------------------------------------------------

    print(
        "\n[7] /demo/sample"
    )

    demo = get_json(
        "/demo/sample"
    )

    print(
        (
            f"sample_id = "
            f"{demo['sample_id']}"
        )
    )

    print(
        (
            f"feature_count = "
            f"{len(demo['features'])}"
        )
    )

    assert (
        len(
            demo[
                "features"
            ]
        )
        == 27
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    print(
        "\n[8] /predict"
    )

    prediction = post_json(
        "/predict",
        {
            "features":
                demo[
                    "features"
                ]
        },
    )

    print(
        json.dumps(
            prediction,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        "predicted_defect"
        in prediction
    )

    assert (
        "confidence"
        in prediction
    )

    # -----------------------------------------------------
    # Explain prediction
    # -----------------------------------------------------

    print(
        "\n[9] /explain"
    )

    explanation = post_json(
        "/explain",
        {
            "features":
                demo[
                    "features"
                ],

            "top_n":
                5,
        },
    )

    print(
        json.dumps(
            explanation,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        len(
            explanation[
                "top_drivers"
            ]
        )
        == 5
    )

    # -----------------------------------------------------
    # Copilot
    # -----------------------------------------------------

    print(
        "\n[10] /copilot/chat"
    )

    copilot = post_json(
        "/copilot/chat",
        {
            "question": (
                "哪一種鋼材缺陷最常見？"
                "請告訴我數量和比例。"
            )
        },
    )

    print(
        json.dumps(
            copilot,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert (
        "answer"
        in copilot
    )

    assert (
        "get_defect_distribution"
        in copilot[
            "tools_used"
        ]
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DASHBOARD API SMOKE TEST: PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()