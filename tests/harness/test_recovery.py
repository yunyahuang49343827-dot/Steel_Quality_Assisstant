import pytest

from src.copilot.harness.evaluator import (
    EvaluationIssue,
)
from src.copilot.harness.recovery import (
    MAX_RECOVERY_ATTEMPTS,
    recover_answer,
)


pytestmark = pytest.mark.unit


def test_recovery_attempt_limit():

    issues = [
        EvaluationIssue(
            code="test_issue",
            message="test",
        )
    ]

    with pytest.raises(
        RuntimeError
    ):

        recover_answer(
            messages=[],
            issues=issues,
            attempt_number=(
                MAX_RECOVERY_ATTEMPTS
                + 1
            ),
        )