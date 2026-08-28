import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

from src.training.sft_schema import (
    SFTSample,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "sft"
    / "processed"
    / "train.jsonl"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "sft"
    / "processed"
    / "validation.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sft"
    / "reports"
    / "dataset_quality_report.json"
)

FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sft"
    / "reports"
    / "freeze_manifest_v1.json"
)


NEAR_DUPLICATE_THRESHOLD = 0.92

TRAIN_VALIDATION_LEAKAGE_THRESHOLD = 0.90


UNSAFE_TARGET_PATTERNS = [
    r"\bpassword\s*[:=]",
    r"\bapi[_\s-]?key\s*[:=]",
    r"\bsecret\s*[:=]",
    r"\bdb_password\s*[:=]",
    r"postgres(?:ql)?://[^\s]+",
    r"\bselect\s+.+\s+from\b",
    r"\bdrop\s+table\b",
    r"\bdelete\s+from\b",
    r"\binsert\s+into\b",
    r"\bupdate\s+.+\s+set\b",
]


def load_jsonl(
    path: Path,
) -> List[SFTSample]:
    """
    Load and validate an SFT JSONL file.
    """

    samples: List[
        SFTSample
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(
                    line
                )

                sample = (
                    SFTSample
                    .model_validate(
                        payload
                    )
                )

            except Exception as exc:

                raise ValueError(
                    f"Invalid SFT sample at "
                    f"{path}:{line_number}"
                ) from exc

            samples.append(
                sample
            )

    return samples


def get_user_text(
    sample: SFTSample,
) -> str:
    """
    Return normalized user message content.
    """

    for message in sample.messages:

        if message.role == "user":

            return " ".join(
                message.content
                .lower()
                .split()
            )

    return ""


def get_assistant_text(
    sample: SFTSample,
) -> str:
    """
    Return assistant target text.
    """

    for message in reversed(
        sample.messages
    ):

        if (
            message.role
            == "assistant"
        ):

            return (
                message.content
                .strip()
            )

    return ""


def normalized_pair_text(
    sample: SFTSample,
) -> str:
    """
    Normalize user + assistant content for hashing
    and similarity analysis.
    """

    user = get_user_text(
        sample
    )

    assistant = " ".join(
        get_assistant_text(
            sample
        )
        .lower()
        .split()
    )

    return (
        user
        + "\n"
        + assistant
    )


def stable_hash(
    text: str,
) -> str:
    """
    Return SHA-256 hash.
    """

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def dataset_fingerprint(
    samples: List[
        SFTSample
    ],
) -> str:
    """
    Create deterministic dataset fingerprint.

    Samples are sorted by sample_id before hashing.
    """

    normalized_rows = []

    for sample in sorted(
        samples,
        key=lambda item: (
            item.sample_id
        ),
    ):

        payload = (
            sample.model_dump(
                mode="json"
            )
        )

        normalized_rows.append(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

    dataset_text = "\n".join(
        normalized_rows
    )

    return stable_hash(
        dataset_text
    )


def find_exact_duplicates(
    samples: List[
        SFTSample
    ],
) -> List[
    Tuple[str, str]
]:
    """
    Find exact normalized user + assistant duplicates.
    """

    seen: Dict[
        str,
        str,
    ] = {}

    duplicates: List[
        Tuple[str, str]
    ] = []

    for sample in samples:

        content_hash = (
            stable_hash(
                normalized_pair_text(
                    sample
                )
            )
        )

        if content_hash in seen:

            duplicates.append(
                (
                    seen[
                        content_hash
                    ],
                    sample.sample_id,
                )
            )

        else:

            seen[
                content_hash
            ] = sample.sample_id

    return duplicates


def text_similarity(
    left: str,
    right: str,
) -> float:
    """
    Deterministic lexical similarity.

    This is intentionally lightweight and does not
    require embeddings.
    """

    return SequenceMatcher(
        None,
        left,
        right,
    ).ratio()


def find_near_duplicates(
    samples: List[
        SFTSample
    ],
    threshold: float = (
        NEAR_DUPLICATE_THRESHOLD
    ),
) -> List[
    Dict[str, object]
]:
    """
    Find high lexical-similarity pairs inside
    one dataset split.
    """

    results: List[
        Dict[str, object]
    ] = []

    texts = [
        (
            sample.sample_id,
            normalized_pair_text(
                sample
            ),
        )
        for sample in samples
    ]

    for index, (
        left_id,
        left_text,
    ) in enumerate(
        texts
    ):

        for (
            right_id,
            right_text,
        ) in texts[
            index + 1:
        ]:

            score = (
                text_similarity(
                    left_text,
                    right_text,
                )
            )

            if score >= threshold:

                results.append(
                    {
                        "sample_a":
                            left_id,

                        "sample_b":
                            right_id,

                        "similarity":
                            round(
                                score,
                                4,
                            ),
                    }
                )

    return results


def find_split_near_leakage(
    train_samples: List[
        SFTSample
    ],
    validation_samples: List[
        SFTSample
    ],
    threshold: float = (
        TRAIN_VALIDATION_LEAKAGE_THRESHOLD
    ),
) -> List[
    Dict[str, object]
]:
    """
    Detect train-validation lexical near-duplicate
    leakage using user prompts.

    User prompts are used instead of full user+assistant
    pairs because near-identical questions across splits
    can inflate evaluation even when answers differ.
    """

    results: List[
        Dict[str, object]
    ] = []

    train_texts = [
        (
            sample.sample_id,
            get_user_text(
                sample
            ),
        )
        for sample in train_samples
    ]

    validation_texts = [
        (
            sample.sample_id,
            get_user_text(
                sample
            ),
        )
        for sample in validation_samples
    ]

    for (
        train_id,
        train_text,
    ) in train_texts:

        for (
            validation_id,
            validation_text,
        ) in validation_texts:

            score = (
                text_similarity(
                    train_text,
                    validation_text,
                )
            )

            if score >= threshold:

                results.append(
                    {
                        "train_sample":
                            train_id,

                        "validation_sample":
                            validation_id,

                        "similarity":
                            round(
                                score,
                                4,
                            ),
                    }
                )

    return results


def scan_unsafe_targets(
    samples: List[
        SFTSample
    ],
) -> List[
    Dict[str, str]
]:
    """
    Scan assistant targets for dangerous patterns.

    This is a conservative static scan intended to
    detect accidental secret / SQL instruction leakage.
    """

    findings: List[
        Dict[str, str]
    ] = []

    for sample in samples:

        answer = (
            get_assistant_text(
                sample
            )
        )

        for pattern in (
            UNSAFE_TARGET_PATTERNS
        ):

            if re.search(
                pattern,
                answer,
                flags=(
                    re.IGNORECASE
                    | re.DOTALL
                ),
            ):

                findings.append(
                    {
                        "sample_id":
                            sample.sample_id,

                        "pattern":
                            pattern,
                    }
                )

    return findings


def category_distribution(
    samples: List[
        SFTSample
    ],
) -> Dict[str, int]:
    """
    Count samples by category.
    """

    counter = Counter(
        sample.category.value
        for sample
        in samples
    )

    return dict(
        sorted(
            counter.items()
        )
    )


def answer_length_statistics(
    samples: List[
        SFTSample
    ],
) -> Dict[
    str,
    float,
]:
    """
    Compute basic target-answer character statistics.
    """

    lengths = [
        len(
            get_assistant_text(
                sample
            )
        )
        for sample in samples
    ]

    if not lengths:

        return {
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0,
            "median_chars": 0,
        }

    return {
        "min_chars":
            min(
                lengths
            ),

        "max_chars":
            max(
                lengths
            ),

        "mean_chars":
            round(
                mean(
                    lengths
                ),
                2,
            ),

        "median_chars":
            round(
                median(
                    lengths
                ),
                2,
            ),
    }


def build_quality_report(
    train_samples: List[
        SFTSample
    ],
    validation_samples: List[
        SFTSample
    ],
) -> Dict[
    str,
    object,
]:
    """
    Run all B23.3 quality checks.
    """

    all_samples = (
        train_samples
        + validation_samples
    )

    exact_duplicates = (
        find_exact_duplicates(
            all_samples
        )
    )

    near_duplicates = (
        find_near_duplicates(
            all_samples
        )
    )

    split_leakage = (
        find_split_near_leakage(
            train_samples,
            validation_samples,
        )
    )

    unsafe_targets = (
        scan_unsafe_targets(
            all_samples
        )
    )

    train_distribution = (
        category_distribution(
            train_samples
        )
    )

    validation_distribution = (
        category_distribution(
            validation_samples
        )
    )

    all_distribution = (
        category_distribution(
            all_samples
        )
    )

    train_fingerprint = (
        dataset_fingerprint(
            train_samples
        )
    )

    validation_fingerprint = (
        dataset_fingerprint(
            validation_samples
        )
    )

    full_fingerprint = (
        dataset_fingerprint(
            all_samples
        )
    )

    passed = (
        len(
            exact_duplicates
        )
        == 0
        and
        len(
            split_leakage
        )
        == 0
        and
        len(
            unsafe_targets
        )
        == 0
    )

    return {
        "dataset_version":
            "v1",

        "quality_gate_passed":
            passed,

        "total_samples":
            len(
                all_samples
            ),

        "train_samples":
            len(
                train_samples
            ),

        "validation_samples":
            len(
                validation_samples
            ),

        "category_distribution":
            all_distribution,

        "train_category_distribution":
            train_distribution,

        "validation_category_distribution":
            validation_distribution,

        "exact_duplicate_count":
            len(
                exact_duplicates
            ),

        "exact_duplicates":
            exact_duplicates,

        "near_duplicate_threshold":
            NEAR_DUPLICATE_THRESHOLD,

        "near_duplicate_count":
            len(
                near_duplicates
            ),

        "near_duplicates":
            near_duplicates,

        "train_validation_leakage_threshold":
            TRAIN_VALIDATION_LEAKAGE_THRESHOLD,

        "train_validation_near_leakage_count":
            len(
                split_leakage
            ),

        "train_validation_near_leakage":
            split_leakage,

        "unsafe_target_count":
            len(
                unsafe_targets
            ),

        "unsafe_target_findings":
            unsafe_targets,

        "answer_length_statistics":
            answer_length_statistics(
                all_samples
            ),

        "fingerprints": {
            "train":
                train_fingerprint,

            "validation":
                validation_fingerprint,

            "full_dataset":
                full_fingerprint,
        },
    }


def build_freeze_manifest(
    report: Dict[
        str,
        object,
    ],
) -> Dict[
    str,
    object,
]:
    """
    Build immutable metadata for the frozen SFT v1 dataset.
    """

    if not report[
        "quality_gate_passed"
    ]:

        raise RuntimeError(
            "Dataset quality gate failed. "
            "Dataset cannot be frozen."
        )

    return {
        "dataset_name":
            "steel-quality-copilot-sft",

        "dataset_version":
            "v1",

        "status":
            "frozen",

        "total_samples":
            report[
                "total_samples"
            ],

        "train_samples":
            report[
                "train_samples"
            ],

        "validation_samples":
            report[
                "validation_samples"
            ],

        "full_dataset_sha256":
            report[
                "fingerprints"
            ][
                "full_dataset"
            ],

        "train_sha256":
            report[
                "fingerprints"
            ][
                "train"
            ],

        "validation_sha256":
            report[
                "fingerprints"
            ][
                "validation"
            ],

        "quality_gate_passed":
            True,

        "design_principle": (
            "Fine-tune behavioral policy; "
            "dynamic factual data remains tool-grounded."
        ),
    }


def main() -> None:
    """
    Run B23.3 Dataset Quality Gate and freeze SFT v1.
    """

    train_samples = (
        load_jsonl(
            TRAIN_PATH
        )
    )

    validation_samples = (
        load_jsonl(
            VALIDATION_PATH
        )
    )

    report = (
        build_quality_report(
            train_samples,
            validation_samples,
        )
    )

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

    print(
        "="
        * 72
    )

    print(
        "Stage B23.3 — SFT Dataset Quality Gate"
    )

    print(
        "="
        * 72
    )

    print(
        f"Total samples             : "
        f"{report['total_samples']}"
    )

    print(
        f"Train samples             : "
        f"{report['train_samples']}"
    )

    print(
        f"Validation samples        : "
        f"{report['validation_samples']}"
    )

    print()

    print(
        f"Exact duplicates          : "
        f"{report['exact_duplicate_count']}"
    )

    print(
        f"Near duplicates           : "
        f"{report['near_duplicate_count']}"
    )

    print(
        f"Train/Val near leakage    : "
        f"{report['train_validation_near_leakage_count']}"
    )

    print(
        f"Unsafe assistant targets  : "
        f"{report['unsafe_target_count']}"
    )

    print()

    print(
        "Answer length statistics"
    )

    print(
        "-"
        * 72
    )

    for (
        key,
        value,
    ) in report[
        "answer_length_statistics"
    ].items():

        print(
            f"{key:<24}: "
            f"{value}"
        )

    print()

    if not report[
        "quality_gate_passed"
    ]:

        print(
            "QUALITY GATE             : FAILED"
        )

        print(
            f"Report                   : "
            f"{REPORT_PATH}"
        )

        raise RuntimeError(
            "SFT dataset quality gate failed."
        )

    manifest = (
        build_freeze_manifest(
            report
        )
    )

    FREEZE_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "QUALITY GATE             : PASSED"
    )

    print(
        "Dataset status           : FROZEN v1"
    )

    print()

    print(
        f"Dataset SHA-256          : "
        f"{manifest['full_dataset_sha256']}"
    )

    print()

    print(
        f"Quality report           : "
        f"{REPORT_PATH}"
    )

    print(
        f"Freeze manifest          : "
        f"{FREEZE_MANIFEST_PATH}"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()