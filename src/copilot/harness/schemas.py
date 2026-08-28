from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TraceStatus(
    str,
    Enum,
):
    """
    Runtime Harness event status.
    """

    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


class TraceStage(
    str,
    Enum,
):
    """
    Standard Harness execution stages.
    """

    POLICY_CHECK = "policy_check"

    TOOL_PERMISSION = (
        "tool_permission"
    )

    ARGUMENT_VALIDATION = (
        "argument_validation"
    )

    TOOL_EXECUTION = (
        "tool_execution"
    )

    EVIDENCE_VERIFICATION = (
        "evidence_verification"
    )

    OUTPUT_EVALUATION = (
        "output_evaluation"
    )

    RECOVERY = "recovery"

    FINAL_RESPONSE = (
        "final_response"
    )


class TraceEvent(
    BaseModel,
):
    """
    One externally safe runtime execution event.

    This is operational trace data,
    NOT model chain-of-thought.
    """

    stage: TraceStage

    status: TraceStatus

    message: str

    metadata: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    timestamp: datetime = Field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            )
        )
    )


class HarnessTrace(
    BaseModel,
):
    """
    Complete runtime Harness execution trace.
    """

    events: List[
        TraceEvent
    ] = Field(
        default_factory=list
    )

    final_status: Optional[
        str
    ] = None


class ToolPermissionResult(
    BaseModel,
):
    """
    Result of a Harness tool permission check.
    """

    allowed: bool

    tool_name: str

    reason: str

class ToolExecutionResult(
    BaseModel,
):
    """
    Structured result returned after Harness-controlled
    tool execution.
    """

    tool_name: str

    arguments: Dict[
        str,
        Any,
    ]

    result: Optional[
        Any
    ] = None

    success: bool

    error: Optional[
        str
    ] = None