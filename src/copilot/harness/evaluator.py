import json
import re
from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field


CANONICAL_DEFECT_TYPES = {
    "Bumps",
    "Dirtiness",
    "K_Scatch",
    "Other_Faults",
    "Pastry",
    "Stains",
    "Z_Scratch",
}


ONTOLOGY_DRIFT_PATTERNS = [
    r"\bno[\s_-]?defect\b",
    r"\bnormal[\s_-]?class\b",
    r"無缺陷",
    r"正常類別",
    r"正常鋼材",
]


UNSUPPORTED_RISK_PATTERNS = [
    r"高信心.*高風險",
    r"高信心.*風險較高",
    r"信心度.*風險高",
    r"信心度.*嚴重",
    r"高信心.*嚴重",
    r"最需要關注",
    r"風險最低",
    r"風險最高",
    r"較容易清理",
    r"容易清理",
]


RISK_DISCLAIMER_PATTERNS = [
    r"高信心.*不代表.*(?:高)?風險",
    r"高信心.*並不代表.*(?:高)?風險",
    r"信心度.*不代表.*風險",
    r"信心度.*並不代表.*風險",
    r"confidence.*does not.*risk",
    r"confidence.*not.*risk",
    r"不代表.*缺陷嚴重性",
    r"不代表.*嚴重程度",
    r"不直接反映.*製造風險",
    r"不直接代表.*製造風險",
]


UNSUPPORTED_CAUSAL_PATTERNS = [
    r"SHAP.*證明.*(?:製造)?(?:根本)?原因",
    r"SHAP.*代表.*(?:已)?確認.*(?:製造)?根本原因",
    r"SHAP.*就是.*(?:製造)?根本原因",
    r"SHAP.*確認.*(?:這|該).*(?:製造)?根本原因",
    r"特徵.*導致.*缺陷",
    r"特徵.*造成.*缺陷",
    r"可以確認.*製造原因",
    r"已確認.*(?:這|該).*(?:製造)?根本原因",
]


CAUSAL_DISCLAIMER_PATTERNS = [
    r"不代表.*(?:已)?確認.*(?:製造)?根本原因",
    r"不代表.*製造原因",
    r"不能.*確認.*根本原因",
    r"無法.*確認.*根本原因",
    r"並非.*根本原因",
    r"不等於.*根本原因",
]


class EvaluationIssue(
    BaseModel,
):
    """
    One deterministic output-evaluation finding.
    """

    code: str

    message: str


class OutputEvaluationResult(
    BaseModel,
):
    """
    Final-answer evaluation result.
    """

    passed: bool

    issues: List[
        EvaluationIssue
    ] = Field(
        default_factory=list
    )


def _matches_any(
    text: str,
    patterns: List[str],
) -> bool:
    """
    Case-insensitive deterministic pattern matching.
    """

    return any(
        re.search(
            pattern,
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )
        is not None
        for pattern in patterns
    )


def _normalize_numeric_text(
    text: str,
) -> str:
    """
    Normalize comma-separated integers so that
    18,380 and 18380 can be compared.
    """

    return re.sub(
        r"(?<=\d),(?=\d)",
        "",
        text,
    )


def _extract_large_integers(
    text: str,
) -> Set[str]:
    """
    Extract integers >= 100.

    Small numbers are intentionally ignored because
    answers often contain rankings, top-N values,
    class counts, percentages, and presentation values
    that do not always require direct evidence matching.
    """

    normalized = (
        _normalize_numeric_text(
            text
        )
    )

    candidates = re.findall(
        r"(?<![\d.])\d{3,}(?![\d.])",
        normalized,
    )

    return set(
        candidates
    )


def _serialize_evidence(
    evidence_records: List[
        Dict[str, Any]
    ],
) -> str:
    """
    Convert evidence records into text for
    deterministic numeric grounding checks.
    """

    return json.dumps(
        evidence_records,
        ensure_ascii=False,
        default=str,
    )


def evaluate_output(
    answer: str,
    evidence_records: List[
        Dict[str, Any]
    ],
) -> OutputEvaluationResult:
    """
    Deterministically evaluate the final LLM answer.

    Current checks:
    - empty answer
    - defect ontology drift
    - unsupported risk / severity assumptions
    - unsupported manufacturing causality
    - large numeric claims not found in tool evidence

    Disclaimer-aware behavior:
    - "confidence does not mean risk" is allowed
    - "SHAP does not confirm root cause" is allowed
    """

    issues: List[
        EvaluationIssue
    ] = []

    # =====================================================
    # Empty answer
    # =====================================================

    if not isinstance(
        answer,
        str,
    ) or not answer.strip():

        issues.append(
            EvaluationIssue(
                code="empty_answer",
                message=(
                    "Final answer is empty."
                ),
            )
        )

        return OutputEvaluationResult(
            passed=False,
            issues=issues,
        )

    # =====================================================
    # Ontology drift
    # =====================================================

    if _matches_any(
        answer,
        ONTOLOGY_DRIFT_PATTERNS,
    ):

        issues.append(
            EvaluationIssue(
                code="ontology_drift",
                message=(
                    "Answer introduced a defect class "
                    "outside the supported ontology."
                ),
            )
        )

    # =====================================================
    # Unsupported risk / severity assumptions
    # =====================================================

    has_risk_disclaimer = (
        _matches_any(
            answer,
            RISK_DISCLAIMER_PATTERNS,
        )
    )

    has_unsupported_risk = (
        _matches_any(
            answer,
            UNSUPPORTED_RISK_PATTERNS,
        )
    )

    if (
        has_unsupported_risk
        and not has_risk_disclaimer
    ):

        issues.append(
            EvaluationIssue(
                code=(
                    "unsupported_risk_claim"
                ),
                message=(
                    "Answer introduced unsupported "
                    "manufacturing risk, severity, "
                    "or operational assumptions."
                ),
            )
        )

    # =====================================================
    # Unsupported causality
    # =====================================================

    has_causal_disclaimer = (
        _matches_any(
            answer,
            CAUSAL_DISCLAIMER_PATTERNS,
        )
    )

    has_unsupported_causality = (
        _matches_any(
            answer,
            UNSUPPORTED_CAUSAL_PATTERNS,
        )
    )

    if (
        has_unsupported_causality
        and not has_causal_disclaimer
    ):

        issues.append(
            EvaluationIssue(
                code=(
                    "unsupported_causality"
                ),
                message=(
                    "Answer converted predictive "
                    "evidence into unsupported "
                    "manufacturing causality."
                ),
            )
        )

    # =====================================================
    # Numeric grounding
    # =====================================================

    if evidence_records:

        answer_numbers = (
            _extract_large_integers(
                answer
            )
        )

        evidence_text = (
            _serialize_evidence(
                evidence_records
            )
        )

        evidence_numbers = (
            _extract_large_integers(
                evidence_text
            )
        )

        unsupported_numbers = (
            answer_numbers
            - evidence_numbers
        )

        if unsupported_numbers:

            issues.append(
                EvaluationIssue(
                    code=(
                        "unsupported_numeric_claim"
                    ),
                    message=(
                        "Answer contains large numeric "
                        "claims not found in available "
                        "tool evidence: "
                        + ", ".join(
                            sorted(
                                unsupported_numbers
                            )
                        )
                    ),
                )
            )

    return OutputEvaluationResult(
        passed=(
            len(
                issues
            )
            == 0
        ),
        issues=issues,
    )
