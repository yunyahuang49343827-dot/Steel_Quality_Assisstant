from typing import Any, Dict, List

from pydantic import BaseModel


class EvidenceVerificationResult(
    BaseModel,
):
    """
    Result of deterministic tool evidence verification.
    """

    valid: bool

    usable_evidence_count: int

    reason: str


def verify_evidence_records(
    evidence_records: List[
        Dict[str, Any]
    ],
) -> EvidenceVerificationResult:
    """
    Verify that collected tool evidence is structurally
    usable before it is trusted by the final-answer layer.

    This does not determine whether the manufacturing
    conclusion is correct.

    It only verifies that backend evidence exists in a
    usable form and does not contain obvious execution
    errors.
    """

    if not evidence_records:

        return EvidenceVerificationResult(
            valid=True,
            usable_evidence_count=0,
            reason=(
                "No tool evidence was collected."
            ),
        )

    usable_count = 0

    for record in evidence_records:

        if not isinstance(
            record,
            dict,
        ):

            return EvidenceVerificationResult(
                valid=False,
                usable_evidence_count=(
                    usable_count
                ),
                reason=(
                    "Evidence record must be "
                    "a dictionary."
                ),
            )

        tool_name = record.get(
            "tool"
        )

        if not isinstance(
            tool_name,
            str,
        ) or not tool_name.strip():

            return EvidenceVerificationResult(
                valid=False,
                usable_evidence_count=(
                    usable_count
                ),
                reason=(
                    "Evidence record is missing "
                    "a valid tool name."
                ),
            )

        if "result" not in record:

            return EvidenceVerificationResult(
                valid=False,
                usable_evidence_count=(
                    usable_count
                ),
                reason=(
                    "Evidence record is missing "
                    "a tool result."
                ),
            )

        result = record[
            "result"
        ]

        if result is None:

            return EvidenceVerificationResult(
                valid=False,
                usable_evidence_count=(
                    usable_count
                ),
                reason=(
                    f"Tool {tool_name} returned "
                    "no usable result."
                ),
            )

        if (
            isinstance(
                result,
                dict,
            )
            and result.get(
                "error"
            )
        ):

            return EvidenceVerificationResult(
                valid=False,
                usable_evidence_count=(
                    usable_count
                ),
                reason=(
                    f"Tool {tool_name} returned "
                    "an error result."
                ),
            )

        usable_count += 1

    return EvidenceVerificationResult(
        valid=True,
        usable_evidence_count=(
            usable_count
        ),
        reason=(
            "Collected tool evidence passed "
            "structural verification."
        ),
    )