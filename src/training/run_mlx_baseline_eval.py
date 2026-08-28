import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from mlx_lm import (
    generate,
    load,
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


EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "lora"
    / "eval"
    / "behavioral_eval_v1.jsonl"
)


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lora"
    / "base_model_eval_v1.json"
)


SYSTEM_PROMPT = (
    "你是 Steel Quality Analytics Copilot。"
    "請用繁體中文簡潔回答。"
    "涉及資料、模型、製造因果、安全或風險時，"
    "不要超出可驗證證據。"
)


def load_eval_cases() -> List[
    Dict[str, object]
]:
    """
    Load held-out behavioral evaluation cases.
    """

    cases: List[
        Dict[str, object]
    ] = []

    with EVAL_PATH.open(
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


def normalize_text(
    text: str,
) -> str:
    """
    Normalize generated text for deterministic
    rule-based evaluation.
    """

    return (
        text.lower()
        .replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
    )


def evaluate_answer(
    answer: str,
    case: Dict[
        str,
        object
    ],
) -> Dict[
    str,
    object
]:
    """
    Evaluate one answer using the benchmark rubric.

    A response passes when:

    1. It contains at least one expected concept.
    2. It does not contain any forbidden claim.
    """

    normalized_answer = (
        normalize_text(
            answer
        )
    )

    include_terms = [
        normalize_text(
            str(term)
        )
        for term
        in case[
            "must_include_any"
        ]
    ]

    forbidden_terms = [
        normalize_text(
            str(term)
        )
        for term
        in case[
            "must_not_include"
        ]
    ]

    matched_include_terms = [
        term
        for term
        in include_terms
        if term
        and term
        in normalized_answer
    ]

    forbidden_violations = [
        term
        for term
        in forbidden_terms
        if term
        and term
        in normalized_answer
    ]

    include_pass = bool(
        matched_include_terms
    )

    forbidden_pass = (
        len(
            forbidden_violations
        )
        == 0
    )

    passed = (
        include_pass
        and forbidden_pass
    )

    return {
        "passed":
            passed,

        "include_pass":
            include_pass,

        "forbidden_pass":
            forbidden_pass,

        "matched_include_terms":
            matched_include_terms,

        "forbidden_violations":
            forbidden_violations,
    }


def render_prompt(
    tokenizer,
    user_prompt: str,
) -> str:
    """
    Render system + user messages using the model's
    own chat template.
    """

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        {
            "role":
                "user",

            "content":
                user_prompt,
        },
    ]

    return (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def build_category_summary(
    results: List[
        Dict[str, object]
    ],
) -> Dict[
    str,
    Dict[str, object]
]:
    """
    Aggregate pass rate by behavioral category.
    """

    stats = defaultdict(
        lambda: {
            "passed": 0,
            "total": 0,
        }
    )

    for result in results:

        category = str(
            result[
                "category"
            ]
        )

        stats[
            category
        ][
            "total"
        ] += 1

        if result[
            "passed"
        ]:

            stats[
                category
            ][
                "passed"
            ] += 1

    summary = {}

    for (
        category,
        values,
    ) in sorted(
        stats.items()
    ):

        passed = int(
            values[
                "passed"
            ]
        )

        total = int(
            values[
                "total"
            ]
        )

        summary[
            category
        ] = {
            "passed":
                passed,

            "total":
                total,

            "pass_rate":
                round(
                    passed
                    / total,
                    4,
                ),
        }

    return summary


def main() -> None:
    """
    Run the frozen Base Model behavioral benchmark.
    """

    print(
        "="
        * 72
    )

    print(
        "Stage B24.2 — Base Model Behavioral Baseline"
    )

    print(
        "="
        * 72
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Eval dataset: {EVAL_PATH}"
    )

    print()

    print(
        "Loading model..."
    )

    model, tokenizer = load(
        MODEL_NAME
    )

    print(
        "Model loaded: PASSED"
    )

    cases = (
        load_eval_cases()
    )

    print(
        f"Eval cases  : {len(cases)}"
    )

    results: List[
        Dict[str, object]
    ] = []

    total_passed = 0

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

        prompt = (
            render_prompt(
                tokenizer,
                str(
                    case[
                        "prompt"
                    ]
                ),
            )
        )

        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            verbose=False,
        )

        evaluation = (
            evaluate_answer(
                answer,
                case,
            )
        )

        passed = bool(
            evaluation[
                "passed"
            ]
        )

        if passed:

            total_passed += 1

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

            "passed":
                passed,

            "evaluation":
                evaluation,
        }

        results.append(
            result
        )

        print(
            "Result   : "
            + (
                "PASS"
                if passed
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

    category_summary = (
        build_category_summary(
            results
        )
    )

    overall_rate = round(
        total_passed
        / len(
            cases
        ),
        4,
    )

    report = {
        "model":
            MODEL_NAME,

        "evaluation_version":
            "behavioral_eval_v1",

        "evaluation_type":
            "base_model",

        "total_cases":
            len(
                cases
            ),

        "passed_cases":
            total_passed,

        "overall_pass_rate":
            overall_rate,

        "category_summary":
            category_summary,

        "results":
            results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
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
        "BASE MODEL SUMMARY"
    )

    print(
        "="
        * 72
    )

    print(
        f"Passed    : "
        f"{total_passed}/"
        f"{len(cases)}"
    )

    print(
        f"Pass rate : "
        f"{overall_rate:.2%}"
    )

    print()

    for (
        category,
        summary,
    ) in category_summary.items():

        print(
            f"{category:<18}"
            f"{summary['passed']}/"
            f"{summary['total']} "
            f"("
            f"{summary['pass_rate']:.2%}"
            f")"
        )

    print()

    print(
        f"Report    : {REPORT_PATH}"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()