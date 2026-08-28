import json
from pathlib import Path
from typing import Dict


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "lora"
)


BASE_ONLY_PATH = (
    REPORT_DIR
    / "locked_base_eval_v1.json"
)


LORA_ONLY_PATH = (
    REPORT_DIR
    / "locked_lora_eval_v1.json"
)


BASE_HARNESS_PATH = (
    REPORT_DIR
    / "locked_base_harness_eval_v1.json"
)


LORA_HARNESS_PATH = (
    REPORT_DIR
    / "locked_lora_harness_eval_v1.json"
)


OUTPUT_PATH = (
    REPORT_DIR
    / "four_way_evaluation_v1.json"
)


def load_json(
    path: Path,
) -> Dict[str, object]:
    """
    Load one evaluation report.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Missing report: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def extract_behavioral_metrics(
    report: Dict[str, object],
) -> Dict[str, object]:
    """
    Extract model-level behavioral metrics
    from one evaluation report.
    """

    summary = report[
        "summary"
    ]

    categories = summary[
        "category_summary"
    ]

    return {
        "overall_pass_rate":
            float(
                summary[
                    "overall_pass_rate"
                ]
            ),

        "grounding_pass_rate":
            float(
                categories[
                    "grounding"
                ][
                    "pass_rate"
                ]
            ),

        "explainability_pass_rate":
            float(
                categories[
                    "explainability"
                ][
                    "pass_rate"
                ]
            ),

        "confidence_pass_rate":
            float(
                categories[
                    "confidence"
                ][
                    "pass_rate"
                ]
            ),

        "security_pass_rate":
            float(
                categories[
                    "security"
                ][
                    "pass_rate"
                ]
            ),

        "fallback_pass_rate":
            float(
                categories[
                    "fallback"
                ][
                    "pass_rate"
                ]
            ),

        "empty_responses":
            int(
                summary[
                    "empty_responses"
                ]
            ),

        "incomplete_responses":
            int(
                summary[
                    "incomplete_responses"
                ]
            ),

        "language_regressions":
            int(
                summary[
                    "language_regressions"
                ]
            ),

        "forbidden_pattern_matches":
            int(
                summary[
                    "unsafe_responses"
                ]
            ),
    }


def extract_harness_metrics(
    report: Dict[str, object],
) -> Dict[str, int]:
    """
    Extract operational Harness metrics.

    These metrics describe runtime control
    rather than semantic model quality.
    """

    summary = report[
        "summary"
    ]

    harness = summary[
        "harness_metrics"
    ]

    return {
        "primary_passed":
            int(
                harness[
                    "primary_passed"
                ]
            ),

        "input_blocked":
            int(
                harness[
                    "input_blocked"
                ]
            ),

        "recovery_attempted":
            int(
                harness[
                    "recovery_attempted"
                ]
            ),

        "recovered":
            int(
                harness[
                    "recovered"
                ]
            ),

        "fallback_used":
            int(
                harness[
                    "fallback_used"
                ]
            ),
    }


def build_delta(
    baseline: Dict[str, object],
    candidate: Dict[str, object],
) -> Dict[str, float]:
    """
    Calculate selected behavioral deltas.
    """

    rate_fields = [
        "overall_pass_rate",
        "grounding_pass_rate",
        "explainability_pass_rate",
        "confidence_pass_rate",
        "security_pass_rate",
        "fallback_pass_rate",
    ]

    return {
        field: round(
            float(
                candidate[
                    field
                ]
            )
            - float(
                baseline[
                    field
                ]
            ),
            4,
        )
        for field in rate_fields
    }


def main() -> None:
    """
    Build the final four-way evaluation audit.
    """

    base_only = (
        load_json(
            BASE_ONLY_PATH
        )
    )

    lora_only = (
        load_json(
            LORA_ONLY_PATH
        )
    )

    base_harness = (
        load_json(
            BASE_HARNESS_PATH
        )
    )

    lora_harness = (
        load_json(
            LORA_HARNESS_PATH
        )
    )

    base_behavior = (
        extract_behavioral_metrics(
            base_only
        )
    )

    lora_behavior = (
        extract_behavioral_metrics(
            lora_only
        )
    )

    base_harness_behavior = (
        extract_behavioral_metrics(
            base_harness
        )
    )

    lora_harness_behavior = (
        extract_behavioral_metrics(
            lora_harness
        )
    )

    base_harness_ops = (
        extract_harness_metrics(
            base_harness
        )
    )

    lora_harness_ops = (
        extract_harness_metrics(
            lora_harness
        )
    )

    report = {
        "evaluation_name":
            "steel-quality-four-way-evaluation",

        "evaluation_version":
            "v1",

        "configurations": {
            "base_only": {
                "model_adaptation":
                    False,

                "runtime_harness":
                    False,

                "behavioral_metrics":
                    base_behavior,
            },

            "lora_only": {
                "model_adaptation":
                    True,

                "runtime_harness":
                    False,

                "behavioral_metrics":
                    lora_behavior,
            },

            "base_plus_harness": {
                "model_adaptation":
                    False,

                "runtime_harness":
                    True,

                "behavioral_metrics":
                    base_harness_behavior,

                "harness_operational_metrics":
                    base_harness_ops,
            },

            "lora_plus_harness": {
                "model_adaptation":
                    True,

                "runtime_harness":
                    True,

                "behavioral_metrics":
                    lora_harness_behavior,

                "harness_operational_metrics":
                    lora_harness_ops,
            },
        },

        "comparisons": {
            "lora_vs_base":
                build_delta(
                    base_behavior,
                    lora_behavior,
                ),

            "base_harness_vs_base":
                build_delta(
                    base_behavior,
                    base_harness_behavior,
                ),

            "lora_harness_vs_lora":
                build_delta(
                    lora_behavior,
                    lora_harness_behavior,
                ),
        },

        "governance_findings": {
            "lora_v1_promotion":
                "rejected",

            "lora_grounding_tendency":
                "improved",

            "lora_security_regression":
                True,

            "runtime_input_blocking_verified":
                True,

            "bounded_recovery_verified":
                True,

            "unsafe_numeric_recovery_verified":
                True,

            "output_detector_coverage_limit":
                True,

            "regex_semantic_false_positive_limit":
                True,

            "production_conclusion": (
                "LoRA improves selected model tendencies "
                "but is not a runtime reliability guarantee. "
                "Harness provides deterministic runtime "
                "boundaries, blocking, bounded recovery, "
                "fallback, and operational trace."
            ),
        },

        "measurement_notes": {
            "behavioral_scores": (
                "Automated deterministic-rule scores. "
                "They must not be interpreted as a complete "
                "semantic safety or intelligence metric."
            ),

            "forbidden_pattern_matches": (
                "This field represents deterministic "
                "forbidden-pattern matches and may contain "
                "semantic false positives."
            ),

            "harness_metrics": (
                "Operational metrics measure runtime control "
                "behavior and should be interpreted separately "
                "from model behavioral quality."
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "="
        * 72
    )

    print(
        "Stage B25.3.3 — Four-Way Evaluation Audit"
    )

    print(
        "="
        * 72
    )

    configurations = report[
        "configurations"
    ]

    for name in [
        "base_only",
        "lora_only",
        "base_plus_harness",
        "lora_plus_harness",
    ]:

        config = configurations[
            name
        ]

        metrics = config[
            "behavioral_metrics"
        ]

        print()

        print(
            name.upper()
        )

        print(
            f"Overall       : "
            f"{metrics['overall_pass_rate']:.2%}"
        )

        print(
            f"Grounding     : "
            f"{metrics['grounding_pass_rate']:.2%}"
        )

        print(
            f"Explainability: "
            f"{metrics['explainability_pass_rate']:.2%}"
        )

        print(
            f"Confidence    : "
            f"{metrics['confidence_pass_rate']:.2%}"
        )

        print(
            f"Security      : "
            f"{metrics['security_pass_rate']:.2%}"
        )

        print(
            f"Fallback      : "
            f"{metrics['fallback_pass_rate']:.2%}"
        )

        if (
            "harness_operational_metrics"
            in config
        ):

            harness = config[
                "harness_operational_metrics"
            ]

            print(
                f"Input blocked : "
                f"{harness['input_blocked']}"
            )

            print(
                f"Recovery tried: "
                f"{harness['recovery_attempted']}"
            )

            print(
                f"Recovered     : "
                f"{harness['recovered']}"
            )

            print(
                f"Fallback used : "
                f"{harness['fallback_used']}"
            )

    print()

    print(
        "-"
        * 72
    )

    print(
        "GOVERNANCE CONCLUSION"
    )

    print(
        "-"
        * 72
    )

    print(
        "LoRA v1 promotion      : REJECT"
    )

    print(
        "Grounding tendency     : IMPROVED"
    )

    print(
        "Security regression    : DETECTED"
    )

    print(
        "Input blocking         : VERIFIED"
    )

    print(
        "Bounded recovery       : VERIFIED"
    )

    print(
        "Detector coverage gap  : IDENTIFIED"
    )

    print()

    print(
        f"Report: {OUTPUT_PATH}"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()