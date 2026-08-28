import pytest

from src.copilot.harness.evaluator import (
    evaluate_output,
)


pytestmark = pytest.mark.unit


EVIDENCE = [
    {
        "tool":
            "get_quality_overview",
        "arguments": {},
        "result": {
            "total_samples":
                18380,
            "defect_classes":
                7,
        },
    }
]


def test_grounded_answer_passes():

    result = evaluate_output(
        answer=(
            "目前共有 18,380 筆資料，"
            "包含 7 種缺陷類別。"
        ),
        evidence_records=EVIDENCE,
    )

    assert result.passed is True


def test_wrong_large_number_is_blocked():

    result = evaluate_output(
        answer=(
            "目前共有 18,420 筆資料。"
        ),
        evidence_records=EVIDENCE,
    )

    assert result.passed is False

    codes = {
        issue.code
        for issue
        in result.issues
    }

    assert (
        "unsupported_numeric_claim"
        in codes
    )


def test_no_defect_class_is_blocked():

    result = evaluate_output(
        answer=(
            "資料另外包含一個無缺陷類別。"
        ),
        evidence_records=EVIDENCE,
    )

    assert result.passed is False

    codes = {
        issue.code
        for issue
        in result.issues
    }

    assert (
        "ontology_drift"
        in codes
    )


def test_confidence_risk_claim_is_blocked():

    result = evaluate_output(
        answer=(
            "模型高信心代表這個樣本風險較高。"
        ),
        evidence_records=[],
    )

    assert result.passed is False

    codes = {
        issue.code
        for issue
        in result.issues
    }

    assert (
        "unsupported_risk_claim"
        in codes
    )


def test_shap_root_cause_claim_is_blocked():

    result = evaluate_output(
        answer=(
            "SHAP 已確認這是製造根本原因。"
        ),
        evidence_records=[],
    )

    assert result.passed is False

    codes = {
        issue.code
        for issue
        in result.issues
    }

    assert (
        "unsupported_causality"
        in codes
    )


def test_shap_disclaimer_is_allowed():

    result = evaluate_output(
        answer=(
            "SHAP 反映模型預測行為，"
            "不代表已確認的製造根本原因。"
        ),
        evidence_records=[],
    )

    assert result.passed is True


def test_operational_assumption_is_blocked():

    result = evaluate_output(
        answer=(
            "Dirtiness 風險最低，而且比較容易清理。"
        ),
        evidence_records=[],
    )

    assert result.passed is False

def test_confidence_risk_disclaimer_is_allowed():

    result = evaluate_output(
        answer=(
            "模型高信心不代表高製造風險，"
            "也不代表缺陷嚴重程度。"
        ),
        evidence_records=[],
    )

    assert result.passed is True


def test_confidence_high_risk_claim_still_blocked():

    result = evaluate_output(
        answer=(
            "模型高信心代表這個樣本具有高風險。"
        ),
        evidence_records=[],
    )

    assert result.passed is False

    codes = {
        issue.code
        for issue
        in result.issues
    }

    assert (
        "unsupported_risk_claim"
        in codes
    )