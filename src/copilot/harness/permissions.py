from typing import Set

from src.copilot.harness.schemas import (
    ToolPermissionResult,
)


ALLOWED_TOOLS: Set[str] = {
    "get_quality_overview",
    "get_defect_distribution",
    "get_high_confidence_predictions",
    "predict_defect",
    "explain_prediction",
    "get_defect_drivers",
}


def check_tool_permission(
    tool_name: str,
) -> ToolPermissionResult:
    """
    Verify that an LLM-requested tool is explicitly
    permitted by the Harness.

    The Harness permission boundary is intentionally
    separate from the backend TOOL_REGISTRY.

    Defense in depth:
    1. Harness permission layer
    2. Tool dispatcher allowlist
    """

    if not isinstance(
        tool_name,
        str,
    ):

        return ToolPermissionResult(
            allowed=False,
            tool_name=str(
                tool_name
            ),
            reason=(
                "Tool name must be a string."
            ),
        )

    normalized = (
        tool_name.strip()
    )

    if not normalized:

        return ToolPermissionResult(
            allowed=False,
            tool_name=normalized,
            reason=(
                "Tool name cannot be empty."
            ),
        )

    if normalized not in ALLOWED_TOOLS:

        return ToolPermissionResult(
            allowed=False,
            tool_name=normalized,
            reason=(
                "Tool is not included in the "
                "Harness allowlist."
            ),
        )

    return ToolPermissionResult(
        allowed=True,
        tool_name=normalized,
        reason=(
            "Tool is explicitly allowed."
        ),
    )