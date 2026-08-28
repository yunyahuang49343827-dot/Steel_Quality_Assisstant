import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "sft"
    / "processed"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "lora"
    / "qwen3_4b_v1"
)


TRAIN_SOURCE = (
    SOURCE_DIR
    / "train.jsonl"
)


VALID_SOURCE = (
    SOURCE_DIR
    / "validation.jsonl"
)


TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "train.jsonl"
)


VALID_OUTPUT = (
    OUTPUT_DIR
    / "valid.jsonl"
)


def load_jsonl(
    path: Path,
) -> List[
    Dict[str, object]
]:
    """
    Load JSONL records.
    """

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(
                    line
                )
            )

    return records


def to_mlx_chat_record(
    record: Dict[
        str,
        object,
    ],
) -> Dict[
    str,
    object,
]:
    """
    Convert one governed SFT record into
    MLX-LM chat dataset format.

    Training-only metadata is intentionally removed.
    """

    messages = record.get(
        "messages"
    )

    if not isinstance(
        messages,
        list,
    ):

        raise ValueError(
            "SFT record is missing messages."
        )

    return {
        "messages": messages
    }


def write_jsonl(
    path: Path,
    records: List[
        Dict[str, object]
    ],
) -> None:
    """
    Write one JSON object per line.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare_split(
    source_path: Path,
    output_path: Path,
) -> int:
    """
    Convert one SFT split to MLX chat format.
    """

    records = load_jsonl(
        source_path
    )

    mlx_records = [
        to_mlx_chat_record(
            record
        )
        for record in records
    ]

    write_jsonl(
        output_path,
        mlx_records,
    )

    return len(
        mlx_records
    )


def main() -> None:

    train_count = (
        prepare_split(
            TRAIN_SOURCE,
            TRAIN_OUTPUT,
        )
    )

    valid_count = (
        prepare_split(
            VALID_SOURCE,
            VALID_OUTPUT,
        )
    )

    print(
        "="
        * 72
    )

    print(
        "Stage B24.1 — Prepare MLX Training Dataset"
    )

    print(
        "="
        * 72
    )

    print(
        f"Train records : {train_count}"
    )

    print(
        f"Valid records : {valid_count}"
    )

    print()

    print(
        f"Train file    : {TRAIN_OUTPUT}"
    )

    print(
        f"Valid file    : {VALID_OUTPUT}"
    )

    print()

    print(
        "MLX chat format preparation: PASSED"
    )

    print(
        "="
        * 72
    )


if __name__ == "__main__":
    main()