import json
from pathlib import Path
from typing import Dict, List

from mlx_lm import (
    generate,
    load,
)

from src.training.run_mlx_baseline_eval import (
    EVAL_PATH,
    MODEL_NAME,
    build_category_summary,
    evaluate_answer,
    load_eval_cases,
    render_prompt,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


ADAPTER_PATH = (
    PROJECT_ROOT
    / "adapters"
    / "qwen3_4b_sft_v1_selected"
)


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lora"
    / "lora_model_eval_v1.json"
)


CHECKPOINT = "step_150"


def main() -> None:
    """
    Run the frozen behavioral benchmark against
    the selected LoRA checkpoint.
    """

    print(
        "="
        * 72
    )

    print(
        "Stage B24.5 — LoRA Behavioral Evaluation"
    )

    print(
        "="
        * 72
    )

    print(
        f"Base model : {MODEL_NAME}"
    )

    print(
        f"Adapter    : {ADAPTER_PATH}"
    )

    print(
        f"Checkpoint : {CHECKPOINT}"
    )

    print(
        f"Eval data  : {EVAL_PATH}"
    )

    print()

    print(
        "Loading model + selected adapter..."
    )

    model, tokenizer = load(
        MODEL_NAME,
        adapter_path=str(
            ADAPTER_PATH
        ),
    )

    print(
        "Model + adapter loaded: PASSED"
    )

    cases = (
        load_eval_cases()
    )

    print(
        f"Eval cases: {len(cases)}"
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
        "base_model":
            MODEL_NAME,

        "adapter_path":
            str(
                ADAPTER_PATH
            ),

        "selected_checkpoint":
            CHECKPOINT,

        "evaluation_version":
            "behavioral_eval_v1",

        "evaluation_type":
            "lora_model",

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
        "LORA MODEL SUMMARY"
    )

    print(
        "="
        * 72
    )

    print(
        f"Selected checkpoint : "
        f"{CHECKPOINT}"
    )

    print(
        f"Passed              : "
        f"{total_passed}/"
        f"{len(cases)}"
    )

    print(
        f"Pass rate           : "
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
        f"Report: {REPORT_PATH}"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()