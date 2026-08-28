import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from mlx_lm import load

from src.evaluation.harness_eval_adapter import (
    build_mlx_generator,
)
from src.evaluation.locked_eval import (
    OUTPUT_PATH,
)
from src.evaluation.locked_scoring import (
    evaluate_locked_answer,
)
from src.evaluation.runtime_harness import (
    run_runtime_harness,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


MODEL_NAME = (
    "mlx-community/"
    "Qwen3-4B-Instruct-2507-4bit"
)


LORA_ADAPTER_PATH = (
    PROJECT_ROOT
    / "adapters"
    / "qwen3_4b_sft_v1_selected"
)


REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "lora"
)


def load_cases() -> List[
    Dict[str, object]
]:
    """
    Load frozen B25 locked evaluation cases.
    """

    cases = []

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            cases.append(
                json.loads(
                    line
                )
            )

    return cases


def build_summary(
    results: List[
        Dict[str, object]
    ],
) -> Dict[str, object]:
    """
    Aggregate final-answer and harness metrics.
    """

    category_stats = defaultdict(
        lambda: {
            "passed": 0,
            "total": 0,
        }
    )

    total_passed = 0

    empty_responses = 0

    incomplete_responses = 0

    language_regressions = 0

    unsafe_responses = 0

    manual_review_cases = 0

    input_blocked = 0

    recovered = 0

    fallback_used = 0

    primary_passed = 0

    recovery_attempted = 0

    for result in results:

        category = str(
            result[
                "category"
            ]
        )

        scoring = result[
            "final_scoring"
        ]

        harness = result[
            "harness"
        ]

        category_stats[
            category
        ][
            "total"
        ] += 1

        if scoring[
            "behavioral_pass"
        ]:

            total_passed += 1

            category_stats[
                category
            ][
                "passed"
            ] += 1

        if scoring[
            "empty_response"
        ]:

            empty_responses += 1

        if not scoring[
            "complete_response"
        ]:

            incomplete_responses += 1

        if not scoring[
            "language_consistent"
        ]:

            language_regressions += 1

        if not scoring[
            "forbidden_pass"
        ]:

            unsafe_responses += 1

        if scoring[
            "manual_review_required"
        ]:

            manual_review_cases += 1

        status = str(
            harness[
                "status"
            ]
        )

        if status == "blocked":

            input_blocked += 1

        elif status == "recovered":

            recovered += 1

        elif status == "fallback":

            fallback_used += 1

        elif status == "passed":

            primary_passed += 1

        if harness[
            "recovery_attempted"
        ]:

            recovery_attempted += 1

    total = len(
        results
    )

    category_summary = {}

    for (
        category,
        values,
    ) in sorted(
        category_stats.items()
    ):

        category_passed = (
            values[
                "passed"
            ]
        )

        category_total = (
            values[
                "total"
            ]
        )

        category_summary[
            category
        ] = {
            "passed":
                category_passed,

            "total":
                category_total,

            "pass_rate":
                round(
                    category_passed
                    / category_total,
                    4,
                ),
        }

    return {
        "total_cases":
            total,

        "passed_cases":
            total_passed,

        "overall_pass_rate":
            round(
                total_passed
                / total,
                4,
            ),

        "empty_responses":
            empty_responses,

        "incomplete_responses":
            incomplete_responses,

        "language_regressions":
            language_regressions,

        "unsafe_responses":
            unsafe_responses,

        "manual_review_cases":
            manual_review_cases,

        "harness_metrics": {
            "primary_passed":
                primary_passed,

            "input_blocked":
                input_blocked,

            "recovery_attempted":
                recovery_attempted,

            "recovered":
                recovered,

            "fallback_used":
                fallback_used,
        },

        "category_summary":
            category_summary,
    }


def resolve_report_path(
    variant: str,
) -> Path:

    if variant == "base":

        return (
            REPORT_DIR
            / "locked_base_harness_eval_v1.json"
        )

    return (
        REPORT_DIR
        / "locked_lora_harness_eval_v1.json"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run locked evaluation through "
            "model-agnostic runtime harness."
        )
    )

    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "base",
            "lora",
        ],
    )

    args = parser.parse_args()

    variant = str(
        args.variant
    )

    print(
        "="
        * 72
    )

    print(
        "Stage B25.3.2 — Locked Harness Evaluation"
    )

    print(
        "="
        * 72
    )

    print(
        f"Variant    : {variant}"
    )

    print(
        f"Base model : {MODEL_NAME}"
    )

    if variant == "lora":

        print(
            f"Adapter    : "
            f"{LORA_ADAPTER_PATH}"
        )

    print(
        f"Locked eval: {OUTPUT_PATH}"
    )

    print()

    print(
        "Loading model..."
    )

    if variant == "base":

        model, tokenizer = load(
            MODEL_NAME
        )

    else:

        model, tokenizer = load(
            MODEL_NAME,
            adapter_path=str(
                LORA_ADAPTER_PATH
            ),
        )

    print(
        "Model loaded: PASSED"
    )

    generator = (
        build_mlx_generator(
            model,
            tokenizer,
        )
    )

    cases = (
        load_cases()
    )

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        prompt = str(
            case[
                "prompt"
            ]
        )

        print()

        print(
            "-"
            * 72
        )

        print(
            f"[{index}/{len(cases)}] "
            f"{case['eval_id']}"
        )

        print(
            f"Category : "
            f"{case['category']}"
        )

        print(
            f"Prompt   : {prompt}"
        )

        harness_result = (
            run_runtime_harness(
                prompt,
                generator,
            )
        )

        final_scoring = (
            evaluate_locked_answer(
                harness_result.final_answer,
                case,
            )
        )

        result = {
            "eval_id":
                case[
                    "eval_id"
                ],

            "category":
                case[
                    "category"
                ],

            "prompt":
                prompt,

            "harness":
                harness_result.to_dict(),

            "final_scoring":
                final_scoring,
        }

        results.append(
            result
        )

        print(
            f"Harness  : "
            f"{harness_result.status.value}"
        )

        print(
            f"Failure  : "
            f"{harness_result.failure_type.value}"
        )

        print(
            "Recovery : "
            + (
                "YES"
                if harness_result.recovery_attempted
                else "NO"
            )
        )

        print(
            "Behavior : "
            + (
                "PASS"
                if final_scoring[
                    "behavioral_pass"
                ]
                else "FAIL"
            )
        )

        print(
            "Complete : "
            + (
                "PASS"
                if final_scoring[
                    "complete_response"
                ]
                else "FAIL"
            )
        )

        print(
            "Language : "
            + (
                "PASS"
                if final_scoring[
                    "language_consistent"
                ]
                else "FAIL"
            )
        )

        print()

        print(
            "Raw Answer:"
        )

        print(
            harness_result.raw_answer
        )

        if (
            harness_result.recovery_answer
            is not None
        ):

            print()

            print(
                "Recovery Answer:"
            )

            print(
                harness_result.recovery_answer
            )

        print()

        print(
            "Final Answer:"
        )

        print(
            harness_result.final_answer
        )

    summary = (
        build_summary(
            results
        )
    )

    report = {
        "evaluation_version":
            "locked_eval_v1",

        "evaluation_mode":
            "runtime_harness",

        "variant":
            variant,

        "model":
            MODEL_NAME,

        "adapter":
            (
                str(
                    LORA_ADAPTER_PATH
                )
                if variant
                == "lora"
                else None
            ),

        "summary":
            summary,

        "results":
            results,
    }

    report_path = (
        resolve_report_path(
            variant
        )
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "="
        * 72
    )

    print(
        f"LOCKED {variant.upper()} + HARNESS SUMMARY"
    )

    print(
        "="
        * 72
    )

    print(
        f"Behavioral pass : "
        f"{summary['passed_cases']}/"
        f"{summary['total_cases']} "
        f"("
        f"{summary['overall_pass_rate']:.2%}"
        f")"
    )

    print(
        f"Empty responses : "
        f"{summary['empty_responses']}"
    )

    print(
        f"Incomplete      : "
        f"{summary['incomplete_responses']}"
    )

    print(
        f"Language issues : "
        f"{summary['language_regressions']}"
    )

    print(
        f"Unsafe outputs  : "
        f"{summary['unsafe_responses']}"
    )

    print()

    print(
        "HARNESS"
    )

    harness_metrics = (
        summary[
            "harness_metrics"
        ]
    )

    print(
        f"Primary passed  : "
        f"{harness_metrics['primary_passed']}"
    )

    print(
        f"Input blocked   : "
        f"{harness_metrics['input_blocked']}"
    )

    print(
        f"Recovery tried  : "
        f"{harness_metrics['recovery_attempted']}"
    )

    print(
        f"Recovered       : "
        f"{harness_metrics['recovered']}"
    )

    print(
        f"Fallback used   : "
        f"{harness_metrics['fallback_used']}"
    )

    print()

    for (
        category,
        category_result,
    ) in summary[
        "category_summary"
    ].items():

        print(
            f"{category:<18}"
            f"{category_result['passed']}/"
            f"{category_result['total']} "
            f"("
            f"{category_result['pass_rate']:.2%}"
            f")"
        )

    print()

    print(
        f"Report: {report_path}"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()