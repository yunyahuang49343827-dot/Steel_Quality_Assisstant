from pathlib import Path

import pandas as pd

from src.tools.quality_tools import (
    execute_tool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

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


def build_sample_features():

    df = pd.read_csv(
        TEST_PATH
    )

    sample = df.iloc[0]

    excluded = (
        [ID_COLUMN, TARGET_COLUMN]
        + TARGET_BINARY_COLUMNS
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    features = {
        feature: (
            int(
                sample[
                    feature
                ]
            )
            if feature in [
                "TypeOfSteel_A300",
                "TypeOfSteel_A400",
            ]
            else float(
                sample[
                    feature
                ]
            )
        )
        for feature in feature_columns
    }

    return (
        features,
        sample[
            ID_COLUMN
        ],
        sample[
            TARGET_COLUMN
        ],
    )


def main():

    print("=" * 72)
    print(
        "Stage B16 — Function Calling Tools Smoke Test"
    )
    print("=" * 72)

    print("\nQUALITY OVERVIEW")
    print("-" * 72)

    overview = execute_tool(
        "get_quality_overview",
        {},
    )

    print(
        overview
    )

    print("\nDEFECT DISTRIBUTION")
    print("-" * 72)

    distribution = execute_tool(
        "get_defect_distribution",
        {},
    )

    for row in distribution:

        print(
            f"{row['defect_type']:<15}"
            f"{row['sample_count']:>7,}   "
            f"{row['percentage']:>6.2f}%"
        )

    print("\nDEFECT DRIVERS — K_Scatch")
    print("-" * 72)

    drivers = execute_tool(
        "get_defect_drivers",
        {
            "defect_type":
                "K_Scatch",

            "top_n":
                5,
        },
    )

    for row in drivers:

        print(
            f"{row['rank']}  "
            f"{row['feature']:<25} "
            f"{row['mean_abs_shap']:.4f}"
        )

    (
        features,
        sample_id,
        actual_defect,
    ) = build_sample_features()

    print("\nPREDICT DEFECT")
    print("-" * 72)

    prediction = execute_tool(
        "predict_defect",
        {
            "features":
                features
        },
    )

    print(
        f"Sample ID        : {sample_id}"
    )

    print(
        f"Actual defect    : {actual_defect}"
    )

    print(
        f"Predicted defect : "
        f"{prediction['predicted_defect']}"
    )

    print(
        f"Confidence       : "
        f"{prediction['confidence']:.4f}"
    )

    print("\nEXPLAIN PREDICTION")
    print("-" * 72)

    explanation = execute_tool(
        "explain_prediction",
        {
            "features":
                features,

            "top_n":
                5,
        },
    )

    for row in (
        explanation[
            "top_drivers"
        ]
    ):

        print(
            f"{row['feature']:<25}"
            f"{row['shap_value']:>10.4f}   "
            f"{row['direction']}"
        )

    print("\nSECURITY — UNKNOWN TOOL")
    print("-" * 72)

    try:

        execute_tool(
            "run_arbitrary_sql",
            {
                "sql":
                    "SELECT * FROM users"
            },
        )

        raise RuntimeError(
            "Unknown tool was unexpectedly allowed."
        )

    except ValueError as exc:

        print(
            f"Blocked correctly: {exc}"
        )

    print("\nSECURITY — INVALID LIMIT")
    print("-" * 72)

    try:

        execute_tool(
            "get_high_confidence_predictions",
            {
                "limit":
                    100000
            },
        )

        raise RuntimeError(
            "Invalid limit was unexpectedly accepted."
        )

    except ValueError as exc:

        print(
            f"Blocked correctly: {exc}"
        )

    print(
        "\nFunction calling tools: PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()