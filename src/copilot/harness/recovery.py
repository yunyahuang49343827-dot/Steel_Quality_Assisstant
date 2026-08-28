from typing import Any, Dict, List

from src.copilot.harness.evaluator import (
    EvaluationIssue,
)
from src.copilot.ollama_client import (
    chat_with_ollama,
)


MAX_RECOVERY_ATTEMPTS = 1


def build_recovery_instruction(
    issues: List[
        EvaluationIssue
    ],
) -> str:
    """
    Build a concise recovery instruction from
    deterministic evaluator findings.
    """

    issue_lines = [
        (
            f"- {issue.code}: "
            f"{issue.message}"
        )
        for issue in issues
    ]

    issue_text = "\n".join(
        issue_lines
    )

    return (
        "上一版回答沒有通過 Harness Output Evaluator。\n\n"
        "偵測到的問題：\n"
        f"{issue_text}\n\n"
        "請重新產生回答，並遵守以下規則：\n"
        "1. 只能使用前面 tool result 中已有的證據。\n"
        "2. 不得新增不存在的 defect class。\n"
        "3. 不得把 SHAP 或模型特徵解釋成製造根本原因。\n"
        "4. 不得把 confidence 解釋成風險或嚴重程度。\n"
        "5. 不得新增 tool evidence 中不存在的數字。\n"
        "6. 如果證據不足，直接說證據不足。\n"
        "7. 使用繁體中文簡潔回答。\n"
    )


def recover_answer(
    messages: List[
        Dict[str, Any]
    ],
    issues: List[
        EvaluationIssue
    ],
    attempt_number: int = 1,
) -> str:
    """
    Perform one bounded recovery rewrite.

    Recovery:
    - cannot call tools
    - cannot exceed MAX_RECOVERY_ATTEMPTS
    - must only rewrite from existing evidence
    """

    if (
        attempt_number
        > MAX_RECOVERY_ATTEMPTS
    ):

        raise RuntimeError(
            "Maximum Harness recovery "
            "attempts exceeded."
        )

    recovery_messages = (
        messages
        + [
            {
                "role": "system",
                "content": (
                    build_recovery_instruction(
                        issues
                    )
                ),
            }
        ]
    )

    response = chat_with_ollama(
        messages=recovery_messages,
        tools=None,
    )

    return (
        response.message
        .content
        .strip()
    )