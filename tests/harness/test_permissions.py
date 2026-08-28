import pytest

from src.copilot.harness.permissions import (
    ALLOWED_TOOLS,
    check_tool_permission,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "tool_name",
    sorted(
        ALLOWED_TOOLS
    ),
)
def test_allowlisted_tools_are_permitted(
    tool_name,
):

    result = (
        check_tool_permission(
            tool_name
        )
    )

    assert result.allowed is True

    assert (
        result.tool_name
        == tool_name
    )


def test_arbitrary_sql_tool_is_blocked():

    result = (
        check_tool_permission(
            "run_arbitrary_sql"
        )
    )

    assert result.allowed is False


def test_unknown_tool_is_blocked():

    result = (
        check_tool_permission(
            "delete_database"
        )
    )

    assert result.allowed is False


def test_empty_tool_name_is_blocked():

    result = (
        check_tool_permission(
            ""
        )
    )

    assert result.allowed is False