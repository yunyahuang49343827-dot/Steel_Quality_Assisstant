import pytest

from src.copilot.harness.evidence import (
    verify_evidence_records,
)


pytestmark = pytest.mark.unit


def test_empty_evidence_is_valid():

    result = (
        verify_evidence_records(
            []
        )
    )

    assert result.valid is True
    assert (
        result.usable_evidence_count
        == 0
    )


def test_valid_evidence_record():

    result = (
        verify_evidence_records(
            [
                {
                    "tool":
                        "get_quality_overview",
                    "arguments": {},
                    "result": {
                        "total_samples":
                            18380,
                    },
                }
            ]
        )
    )

    assert result.valid is True
    assert (
        result.usable_evidence_count
        == 1
    )


def test_missing_result_is_invalid():

    result = (
        verify_evidence_records(
            [
                {
                    "tool":
                        "get_quality_overview",
                    "arguments": {},
                }
            ]
        )
    )

    assert result.valid is False


def test_error_result_is_invalid():

    result = (
        verify_evidence_records(
            [
                {
                    "tool":
                        "get_quality_overview",
                    "arguments": {},
                    "result": {
                        "error":
                            "database failed"
                    },
                }
            ]
        )
    )

    assert result.valid is False