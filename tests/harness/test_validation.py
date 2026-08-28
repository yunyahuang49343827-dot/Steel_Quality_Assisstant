import pytest

from src.copilot.harness.validation import (
    validate_tool_arguments,
)


pytestmark = pytest.mark.unit


def test_empty_arguments_allowed_for_overview():

    result = (
        validate_tool_arguments(
            tool_name=(
                "get_quality_overview"
            ),
            arguments={},
        )
    )

    assert result.valid is True


def test_unknown_argument_is_blocked():

    result = (
        validate_tool_arguments(
            tool_name=(
                "get_quality_overview"
            ),
            arguments={
                "sql":
                    "SELECT * FROM users"
            },
        )
    )

    assert result.valid is False


def test_limit_accepts_valid_integer():

    result = (
        validate_tool_arguments(
            tool_name=(
                "get_high_confidence_predictions"
            ),
            arguments={
                "limit": 3,
            },
        )
    )

    assert result.valid is True


def test_limit_rejects_string():

    result = (
        validate_tool_arguments(
            tool_name=(
                "get_high_confidence_predictions"
            ),
            arguments={
                "limit": "3",
            },
        )
    )

    assert result.valid is False


def test_limit_rejects_excessive_value():

    result = (
        validate_tool_arguments(
            tool_name=(
                "get_high_confidence_predictions"
            ),
            arguments={
                "limit": 999,
            },
        )
    )

    assert result.valid is False


def test_sample_id_accepts_positive_integer():

    result = (
        validate_tool_arguments(
            tool_name=(
                "predict_defect"
            ),
            arguments={
                "sample_id": 100,
            },
        )
    )

    assert result.valid is True


def test_sample_id_rejects_negative_integer():

    result = (
        validate_tool_arguments(
            tool_name=(
                "predict_defect"
            ),
            arguments={
                "sample_id": -1,
            },
        )
    )

    assert result.valid is False


def test_defect_driver_arguments_are_valid():

    result = (
        validate_tool_arguments(
            tool_name=(
                "get_defect_drivers"
            ),
            arguments={
                "defect_type":
                    "K_Scatch",
                "top_n":
                    5,
            },
        )
    )

    assert result.valid is True


def test_empty_defect_type_is_blocked():

    result = (
        validate_tool_arguments(
            tool_name=(
                "get_defect_drivers"
            ),
            arguments={
                "defect_type":
                    "",
                "top_n":
                    5,
            },
        )
    )

    assert result.valid is False


def test_unknown_tool_has_no_argument_policy():

    result = (
        validate_tool_arguments(
            tool_name=(
                "run_arbitrary_sql"
            ),
            arguments={},
        )
    )

    assert result.valid is False