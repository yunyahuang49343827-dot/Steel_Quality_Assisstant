import pytest

from src.copilot.harness.failures import (
    FailureType,
    build_safe_fallback,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "failure_type",
    list(
        FailureType
    ),
)
def test_all_failure_types_have_fallback(
    failure_type,
):

    message = (
        build_safe_fallback(
            failure_type
        )
    )

    assert isinstance(
        message,
        str,
    )

    assert message.strip()