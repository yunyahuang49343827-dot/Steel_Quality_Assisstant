import hashlib
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


DATASET_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sft"
    / "reports"
    / "freeze_manifest_v1.json"
)


LOCKED_EVAL_MANIFEST_PATH = (
    REPORT_DIR
    / "locked_eval_manifest_v1.json"
)


BASE_EVAL_PATH = (
    REPORT_DIR
    / "locked_base_eval_v1.json"
)


LORA_EVAL_PATH = (
    REPORT_DIR
    / "locked_lora_eval_v1.json"
)


BASE_VS_LORA_PATH = (
    REPORT_DIR
    / "locked_base_vs_lora_v1.json"
)


FOUR_WAY_PATH = (
    REPORT_DIR
    / "four_way_evaluation_v1.json"
)


SELECTED_ADAPTER_PATH = (
    PROJECT_ROOT
    / "adapters"
    / "qwen3_4b_sft_v1_selected"
    / "adapters.safetensors"
)


OUTPUT_PATH = (
    REPORT_DIR
    / "model_governance_report_v1.json"
)


def load_json(
    path: Path,
) -> Dict[str, object]:
    """
    Load JSON artifact.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Missing artifact: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def file_sha256(
    path: Path,
) -> str:
    """
    Calculate SHA-256 for reproducibility.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Missing artifact: {path}"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    """
    Build final LoRA experiment and
    model-governance report.
    """

    dataset_manifest = (
        load_json(
            DATASET_MANIFEST_PATH
        )
    )

    locked_manifest = (
        load_json(
            LOCKED_EVAL_MANIFEST_PATH
        )
    )

    base_eval = (
        load_json(
            BASE_EVAL_PATH
        )
    )

    lora_eval = (
        load_json(
            LORA_EVAL_PATH
        )
    )

    comparison = (
        load_json(
            BASE_VS_LORA_PATH
        )
    )

    four_way = (
        load_json(
            FOUR_WAY_PATH
        )
    )

    adapter_sha256 = (
        file_sha256(
            SELECTED_ADAPTER_PATH
        )
    )

    base_summary = (
        base_eval[
            "summary"
        ]
    )

    lora_summary = (
        lora_eval[
            "summary"
        ]
    )

    promotion = (
        comparison[
            "promotion_decision"
        ]
    )

    governance = (
        four_way[
            "governance_findings"
        ]
    )

    report = {
        "report_name":
            "steel-quality-lora-model-governance",

        "report_version":
            "v1",

        "experiment_status":
            "completed",

        "base_model": {
            "name": (
                "mlx-community/"
                "Qwen3-4B-Instruct-2507-4bit"
            ),

            "runtime":
                "MLX-LM",

            "hardware":
                "Apple Silicon",

            "quantization":
                "4-bit",
        },

        "training_dataset": {
            "dataset_version":
                "sft_v1",

            "total_samples":
                120,

            "train_samples":
                95,

            "validation_samples":
                25,

            "dataset_sha256":
                dataset_manifest.get(
                    "dataset_sha256"
                ),

            "status":
                "frozen",
        },

        "training_configuration": {
            "fine_tune_type":
                "LoRA",

            "training_mode":
                "QLoRA-style 4-bit adapter training",

            "mask_prompt":
                True,

            "num_lora_layers":
                8,

            "batch_size":
                1,

            "gradient_accumulation_steps":
                4,

            "learning_rate":
                0.00001,

            "training_iterations":
                200,

            "gradient_checkpointing":
                True,

            "seed":
                42,

            "trainable_parameters":
                3670000,

            "trainable_parameter_ratio":
                0.00091,
        },

        "training_validation_history": {
            "iteration_1":
                6.850,

            "iteration_25":
                4.618,

            "iteration_50":
                3.586,

            "iteration_75":
                3.298,

            "iteration_100":
                3.066,

            "iteration_125":
                2.890,

            "iteration_150":
                2.774,

            "iteration_175":
                2.673,

            "iteration_200":
                2.623,
        },

        "checkpoint_selection": {
            "selected_checkpoint":
                "step_150",

            "selection_method":
                "behavioral checkpoint comparison",

            "lowest_validation_loss_checkpoint":
                "step_200",

            "selection_rationale": (
                "Step 200 achieved the lowest "
                "validation loss but showed "
                "grounding regression. "
                "Step 150 demonstrated stronger "
                "evidence-grounding behavior."
            ),

            "selected_adapter_sha256":
                adapter_sha256,
        },

        "locked_evaluation": {
            "evaluation_version":
                "locked_eval_v1",

            "total_cases":
                locked_manifest[
                    "total_cases"
                ],

            "dataset_sha256":
                locked_manifest[
                    "dataset_sha256"
                ],

            "base": {
                "overall_pass_rate":
                    base_summary[
                        "overall_pass_rate"
                    ],

                "grounding_pass_rate":
                    base_summary[
                        "category_summary"
                    ][
                        "grounding"
                    ][
                        "pass_rate"
                    ],

                "security_pass_rate":
                    base_summary[
                        "category_summary"
                    ][
                        "security"
                    ][
                        "pass_rate"
                    ],
            },

            "lora": {
                "overall_pass_rate":
                    lora_summary[
                        "overall_pass_rate"
                    ],

                "grounding_pass_rate":
                    lora_summary[
                        "category_summary"
                    ][
                        "grounding"
                    ][
                        "pass_rate"
                    ],

                "security_pass_rate":
                    lora_summary[
                        "category_summary"
                    ][
                        "security"
                    ][
                        "pass_rate"
                    ],

                "empty_responses":
                    lora_summary[
                        "empty_responses"
                    ],

                "incomplete_responses":
                    lora_summary[
                        "incomplete_responses"
                    ],
            },
        },

        "promotion_gate": {
            "decision": (
                "PROMOTE"
                if promotion[
                    "promoted"
                ]
                else "REJECT"
            ),

            "reasons":
                promotion[
                    "reasons"
                ],

            "deltas":
                promotion[
                    "deltas"
                ],
        },

        "runtime_harness_evidence": {
            "input_blocking_verified":
                governance[
                    "runtime_input_blocking_verified"
                ],

            "bounded_recovery_verified":
                governance[
                    "bounded_recovery_verified"
                ],

            "numeric_hallucination_recovery_verified":
                governance[
                    "unsafe_numeric_recovery_verified"
                ],

            "base_harness":
                four_way[
                    "configurations"
                ][
                    "base_plus_harness"
                ][
                    "harness_operational_metrics"
                ],

            "lora_harness":
                four_way[
                    "configurations"
                ][
                    "lora_plus_harness"
                ][
                    "harness_operational_metrics"
                ],
        },

        "known_limitations": [
            (
                "Deterministic regex scoring can "
                "produce semantic false positives."
            ),
            (
                "Runtime output detector coverage "
                "does not capture every unsupported "
                "model statement."
            ),
            (
                "Behavioral automated scores must "
                "not be interpreted as complete "
                "semantic safety metrics."
            ),
            (
                "The LoRA dataset is intentionally "
                "small and optimized for behavioral "
                "adaptation rather than broad "
                "domain knowledge acquisition."
            ),
            (
                "The experimental MLX LoRA model "
                "is separate from the production "
                "Ollama model-serving path."
            ),
        ],

        "final_governance_conclusion": {
            "lora_v1":
                "rejected",

            "production_promotion":
                False,

            "reason": (
                "LoRA v1 improved selected "
                "grounding tendencies but failed "
                "the locked promotion gate because "
                "overall behavioral performance "
                "and security performance regressed."
            ),

            "engineering_conclusion": (
                "Fine-tuning changes model "
                "behavioral tendencies but does not "
                "replace deterministic runtime "
                "governance. Harness controls such "
                "as policy gates, bounded recovery, "
                "fallback, permissions, validation, "
                "and trace remain necessary."
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
        "Stage B25.4 — Model Governance Report"
    )

    print(
        "="
        * 72
    )

    print(
        "Base model       : "
        "Qwen3-4B-Instruct-2507-4bit"
    )

    print(
        "Training samples : 95"
    )

    print(
        "Validation       : 25"
    )

    print(
        "Iterations       : 200"
    )

    print(
        "Selected         : step_150"
    )

    print(
        "Lowest val loss  : step_200"
    )

    print()

    print(
        "LOCKED EVALUATION"
    )

    print(
        f"Base overall     : "
        f"{base_summary['overall_pass_rate']:.2%}"
    )

    print(
        f"LoRA overall     : "
        f"{lora_summary['overall_pass_rate']:.2%}"
    )

    print(
        "Promotion        : "
        + (
            "PROMOTE"
            if promotion[
                "promoted"
            ]
            else "REJECT"
        )
    )

    print()

    print(
        "RUNTIME GOVERNANCE"
    )

    print(
        "Input blocking   : VERIFIED"
    )

    print(
        "Bounded recovery : VERIFIED"
    )

    print()

    print(
        "FINAL DECISION"
    )

    print(
        "LoRA v1          : REJECTED"
    )

    print(
        "Production       : NOT PROMOTED"
    )

    print()

    print(
        f"Adapter SHA-256  : "
        f"{adapter_sha256}"
    )

    print(
        f"Report           : "
        f"{OUTPUT_PATH}"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()