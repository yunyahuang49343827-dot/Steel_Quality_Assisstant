import json
from pathlib import Path
from typing import Dict

from src.evaluation.promotion_gate import (
    EvaluationMetrics,
    evaluate_promotion,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


BASE_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lora"
    / "locked_base_eval_v1.json"
)


LORA_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lora"
    / "locked_lora_eval_v1.json"
)


COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lora"
    / "locked_base_vs_lora_v1.json"
)


def load_json(
    path: Path,
) -> Dict[str, object]:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def build_metrics(
    report: Dict[
        str,
        object
    ],
) -> EvaluationMetrics:

    summary = report[
        "summary"
    ]

    category_summary = summary[
        "category_summary"
    ]

    return EvaluationMetrics(
        overall_pass_rate=float(
            summary[
                "overall_pass_rate"
            ]
        ),

        grounding_pass_rate=float(
            category_summary[
                "grounding"
            ][
                "pass_rate"
            ]
        ),

        security_pass_rate=float(
            category_summary[
                "security"
            ][
                "pass_rate"
            ]
        ),

        empty_responses=int(
            summary[
                "empty_responses"
            ]
        ),

        unsafe_responses=int(
            summary[
                "unsafe_responses"
            ]
        ),

        language_regressions=int(
            summary[
                "language_regressions"
            ]
        ),
    )


def main() -> None:

    base_report = (
        load_json(
            BASE_REPORT_PATH
        )
    )

    lora_report = (
        load_json(
            LORA_REPORT_PATH
        )
    )

    base_metrics = (
        build_metrics(
            base_report
        )
    )

    lora_metrics = (
        build_metrics(
            lora_report
        )
    )

    decision = (
        evaluate_promotion(
            base_metrics,
            lora_metrics,
        )
    )

    comparison = {
        "evaluation_version":
            "locked_eval_v1",

        "base": {
            "overall_pass_rate":
                base_metrics.overall_pass_rate,

            "grounding_pass_rate":
                base_metrics.grounding_pass_rate,

            "security_pass_rate":
                base_metrics.security_pass_rate,

            "empty_responses":
                base_metrics.empty_responses,

            "unsafe_responses":
                base_metrics.unsafe_responses,

            "language_regressions":
                base_metrics.language_regressions,
        },

        "lora": {
            "overall_pass_rate":
                lora_metrics.overall_pass_rate,

            "grounding_pass_rate":
                lora_metrics.grounding_pass_rate,

            "security_pass_rate":
                lora_metrics.security_pass_rate,

            "empty_responses":
                lora_metrics.empty_responses,

            "unsafe_responses":
                lora_metrics.unsafe_responses,

            "language_regressions":
                lora_metrics.language_regressions,
        },

        "promotion_decision":
            decision.model_dump(
                mode="json"
            ),
    }

    COMPARISON_PATH.write_text(
        json.dumps(
            comparison,
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
        "Stage B25.2 — Locked Base vs LoRA"
    )

    print(
        "="
        * 72
    )

    print(
        "BASE"
    )

    print(
        f"Overall   : "
        f"{base_metrics.overall_pass_rate:.2%}"
    )

    print(
        f"Grounding : "
        f"{base_metrics.grounding_pass_rate:.2%}"
    )

    print(
        f"Security  : "
        f"{base_metrics.security_pass_rate:.2%}"
    )

    print()

    print(
        "LORA"
    )

    print(
        f"Overall   : "
        f"{lora_metrics.overall_pass_rate:.2%}"
    )

    print(
        f"Grounding : "
        f"{lora_metrics.grounding_pass_rate:.2%}"
    )

    print(
        f"Security  : "
        f"{lora_metrics.security_pass_rate:.2%}"
    )

    print()

    print(
        "DELTAS"
    )

    for (
        metric,
        delta,
    ) in decision.deltas.items():

        print(
            f"{metric:<12}: "
            f"{delta:+.2%}"
        )

    print()

    print(
        "PROMOTION DECISION"
    )

    print(
        "PROMOTE"
        if decision.promoted
        else "REJECT"
    )

    if decision.reasons:

        print()

        print(
            "Reasons:"
        )

        for reason in (
            decision.reasons
        ):

            print(
                f"- {reason}"
            )

    print()

    print(
        f"Report: {COMPARISON_PATH}"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()