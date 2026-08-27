from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

API_BASE_URL = "http://127.0.0.1:8000"

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


def load_sample():
    """
    Load one held-out sample and convert it into
    the 27-feature API request payload.
    """

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_PATH}"
        )

    test_df = pd.read_csv(TEST_PATH)

    sample = test_df.iloc[0]

    excluded = (
        [ID_COLUMN, TARGET_COLUMN]
        + TARGET_BINARY_COLUMNS
    )

    feature_columns = [
        column
        for column in test_df.columns
        if column not in excluded
    ]

    payload = {
        feature: (
            int(sample[feature])
            if feature
            in [
                "TypeOfSteel_A300",
                "TypeOfSteel_A400",
            ]
            else float(sample[feature])
        )
        for feature in feature_columns
    }

    expected = {
        "id": sample[ID_COLUMN],
        "actual_defect": sample[TARGET_COLUMN],
    }

    return payload, expected


def test_health():
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def test_quality_overview():
    response = requests.get(
        f"{API_BASE_URL}/quality/overview",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def test_prediction(payload):
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    result = response.json()

    required = {
        "predicted_defect",
        "confidence",
        "probabilities",
    }

    missing = (
        required
        - set(result.keys())
    )

    if missing:
        raise ValueError(
            f"/predict missing fields: {missing}"
        )

    probability_sum = sum(
        result["probabilities"].values()
    )

    if abs(
        probability_sum - 1.0
    ) > 1e-4:
        raise ValueError(
            "Prediction probabilities "
            "do not sum to approximately 1."
        )

    return result


def test_explanation(payload):
    response = requests.post(
        f"{API_BASE_URL}/explain",
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    result = response.json()

    required = {
        "predicted_defect",
        "confidence",
        "top_drivers",
        "interpretation_note",
    }

    missing = (
        required
        - set(result.keys())
    )

    if missing:
        raise ValueError(
            f"/explain missing fields: {missing}"
        )

    if len(
        result["top_drivers"]
    ) == 0:
        raise ValueError(
            "No SHAP drivers returned."
        )

    return result


def main():
    print("=" * 72)
    print("Stage B15 — FastAPI Smoke Test")
    print("=" * 72)

    payload, expected = (
        load_sample()
    )

    print("\nSAMPLE")
    print("-" * 72)

    print(
        f"ID            : "
        f"{expected['id']}"
    )

    print(
        f"Actual defect : "
        f"{expected['actual_defect']}"
    )

    print("\nHEALTH")
    print("-" * 72)

    health = test_health()

    print(health)

    print("\nQUALITY OVERVIEW")
    print("-" * 72)

    overview = (
        test_quality_overview()
    )

    print(
        f"Total samples  : "
        f"{overview['total_samples']:,}"
    )

    print(
        f"Defect classes : "
        f"{overview['defect_classes']}"
    )

    print("\nPREDICTION")
    print("-" * 72)

    prediction = (
        test_prediction(
            payload
        )
    )

    print(
        f"Predicted defect : "
        f"{prediction['predicted_defect']}"
    )

    print(
        f"Confidence       : "
        f"{prediction['confidence']:.4f}"
    )

    print("\nEXPLANATION")
    print("-" * 72)

    explanation = (
        test_explanation(
            payload
        )
    )

    for driver in (
        explanation["top_drivers"]
    ):
        print(
            f"{driver['feature']:<25}"
            f"{driver['shap_value']:>10.4f}   "
            f"{driver['direction']}"
        )

    print("\nGUARDRAIL")
    print("-" * 72)

    print(
        explanation[
            "interpretation_note"
        ]
    )

    print("\nAPI smoke test: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()