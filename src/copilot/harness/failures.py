from enum import Enum
from typing import Any, Dict, Optional

from pydantic import (
    BaseModel,
    Field,
)


class FailureType(
    str,
    Enum,
):
    """
    Standard Harness failure categories.
    """

    POLICY_BLOCKED = (
        "policy_blocked"
    )

    TOOL_PERMISSION_BLOCKED = (
        "tool_permission_blocked"
    )

    ARGUMENT_VALIDATION_FAILED = (
        "argument_validation_failed"
    )

    TOOL_EXECUTION_FAILED = (
        "tool_execution_failed"
    )

    EVIDENCE_FAILED = (
        "evidence_failed"
    )

    OUTPUT_EVALUATION_FAILED = (
        "output_evaluation_failed"
    )

    RECOVERY_FAILED = (
        "recovery_failed"
    )

    MAX_TOOL_ROUNDS_REACHED = (
        "max_tool_rounds_reached"
    )


class HarnessFailure(
    BaseModel,
):
    """
    Standardized Harness failure record.
    """

    failure_type: FailureType

    message: str

    tool_name: Optional[
        str
    ] = None

    metadata: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


def build_safe_fallback(
    failure_type: FailureType,
) -> str:
    """
    Return user-safe fallback text for a known
    Harness failure category.
    """

    fallback_map = {
        FailureType.POLICY_BLOCKED: (
            "此請求被安全政策阻擋。"
        ),

        FailureType.TOOL_PERMISSION_BLOCKED: (
            "要求的工具未被系統允許，"
            "因此無法執行此操作。"
        ),

        FailureType.ARGUMENT_VALIDATION_FAILED: (
            "工具輸入參數未通過驗證，"
            "請調整問題後再試一次。"
        ),

        FailureType.TOOL_EXECUTION_FAILED: (
            "後端分析工具暫時無法完成查詢，"
            "請稍後再試或縮小查詢範圍。"
        ),

        FailureType.EVIDENCE_FAILED: (
            "目前取得的工具證據未通過完整性驗證，"
            "因此不提供可能誤導的分析結論。"
        ),

        FailureType.OUTPUT_EVALUATION_FAILED: (
            "模型回答未通過安全與證據檢查，"
            "因此暫不輸出該分析結論。"
        ),

        FailureType.RECOVERY_FAILED: (
            "模型回答未通過檢查，"
            "且自動修正未成功，"
            "請改以更明確的問題重新查詢。"
        ),

        FailureType.MAX_TOOL_ROUNDS_REACHED: (
            "目前無法在允許的工具呼叫次數內"
            "完成查詢，請縮小問題範圍後再試一次。"
        ),
    }

    return fallback_map[
        failure_type
    ]