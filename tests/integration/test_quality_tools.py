import pytest

from src.tools.quality_tools import (
    execute_tool,
    get_defect_distribution,
    get_quality_overview,
)


pytestmark = pytest.mark.integration


def test_quality_overview_reads_real_database():

    result = get_quality_overview()

    assert result == {
        "total_samples": 10,
        "defect_classes": 7,
    }


def test_defect_distribution_reads_real_database():

    result = (
        get_defect_distribution()
    )

    distribution = {
        item["defect_type"]: item
        for item in result
    }

    assert (
        distribution[
            "Other_Faults"
        ][
            "sample_count"
        ]
        == 3
    )

    assert (
        distribution[
            "Other_Faults"
        ][
            "percentage"
        ]
        == 30.0
    )

    assert (
        distribution[
            "Bumps"
        ][
            "sample_count"
        ]
        == 2
    )

    assert (
        distribution[
            "Bumps"
        ][
            "percentage"
        ]
        == 20.0
    )

    assert len(
        distribution
    ) == 7


def test_allowlisted_overview_tool_uses_database():

    result = execute_tool(
        tool_name=(
            "get_quality_overview"
        ),
        arguments={},
    )

    assert (
        result[
            "total_samples"
        ]
        == 10
    )

    assert (
        result[
            "defect_classes"
        ]
        == 7
    )