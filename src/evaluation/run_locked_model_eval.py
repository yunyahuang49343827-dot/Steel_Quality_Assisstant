import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from mlx_lm import (
    generate,
    load,
)

from src.evaluation.locked_eval import (
    OUTPUT_PATH,
)
from src.evaluation.locked_scoring import (
    evaluate_locked_answer,
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


SYSTEM_PROMPT = (
    "你是 Steel Quality Analytics Copilot。"
    "請用繁體中文簡潔但完整地回答。"
    "涉及資料、模型、製造因果、安全或風險時，"
    "不得超出可驗證證據。"
)


def load_cases() -> List[
    Dict[str, object]
]:
    """
    Load the frozen B25 locked evaluation set.
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


def render_prompt(
    tokenizer,
    user_prompt: str,
) -> str:
    """
    Render identical system/user prompts for
    both Base and LoRA candidates.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_summary(
    results: List[
        Dict[str, object]
    ],
) -> Dict[str, object]:
    """
    Aggregate locked evaluation metrics.
    """

    category_stats = defaultdict(
        lambda: {
            "passed": 0,
            "total": 0,
        }
    )

    passed = 0

    empty_responses = 0

    incomplete_responses = 0

    language_regressions = 0

    unsafe_responses = 0

    manual_review_cases = 0

    for result in results:

        category = str(
            result[
                "category"
            ]
        )

        scoring = result[
            "scoring"
        ]

        category_stats[
            category
        ][
            "total"
        ] += 1

        if scoring[
            "behavioral_pass"
        ]:

            passed += 1

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
            passed,

        "overall_pass_rate":
            round(
                passed
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

        "category_summary":
            category_summary,
    }


def resolve_report_path(
    variant: str,
) -> Path:

    if variant == "base":

        return (
            REPORT_DIR
            / "locked_base_eval_v1.json"
        )

    return (
        REPORT_DIR
        / "locked_lora_eval_v1.json"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run frozen B25 locked evaluation."
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
        "Stage B25.2 — Locked Model Evaluation"
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

    cases = (
        load_cases()
    )

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

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
            f"Prompt   : "
            f"{case['prompt']}"
        )

        prompt = render_prompt(
            tokenizer,
            str(
                case[
                    "prompt"
                ]
            ),
        )

        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            verbose=False,
        )

        scoring = (
            evaluate_locked_answer(
                answer,
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
                case[
                    "prompt"
                ],

            "answer":
                answer,

            "scoring":
                scoring,
        }

        results.append(
            result
        )

        print(
            "Behavior : "
            + (
                "PASS"
                if scoring[
                    "behavioral_pass"
                ]
                else "FAIL"
            )
        )

        print(
            "Complete : "
            + (
                "PASS"
                if scoring[
                    "complete_response"
                ]
                else "FAIL"
            )
        )

        print(
            "Language : "
            + (
                "PASS"
                if scoring[
                    "language_consistent"
                ]
                else "FAIL"
            )
        )

        print()

        print(
            "Answer:"
        )

        print(
            answer
        )

    summary = (
        build_summary(
            results
        )
    )

    report = {
        "evaluation_version":
            "locked_eval_v1",

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

        "system_prompt":
            SYSTEM_PROMPT,

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
        f"LOCKED {variant.upper()} SUMMARY"
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