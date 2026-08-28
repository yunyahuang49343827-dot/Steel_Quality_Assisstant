import re
from typing import Any, Dict, List, Optional


SECRET_PATTERNS = [
    r"\bpassword\b",
    r"\bpasswd\b",
    r"\bdb_password\b",
    r"\bdb_user\b",
    r"\bapi[_\s-]?key\b",
    r"\bsecret\b",
    r"\bcredential",
    r"\.env",
    r"connection string",
    r"連線字串",
    r"密碼",
    r"憑證",
    r"環境變數",
]


SQL_EXECUTION_PATTERNS = [
    r"\bselect\b.+\bfrom\b",
    r"\binsert\s+into\b",
    r"\bupdate\b.+\bset\b",
    r"\bdelete\s+from\b",
    r"\bdrop\s+table\b",
    r"\balter\s+table\b",
    r"\btruncate\b",
    r"\brun_arbitrary_sql\b",
    r"執行.*sql",
    r"直接.*sql",
]


PROMPT_INJECTION_PATTERNS = [
    r"ignore .*previous",
    r"ignore .*rules",
    r"ignore .*instructions",
    r"disregard .*rules",
    r"override .*rules",
    r"忽略.*規則",
    r"忽略.*指令",
    r"忽略.*限制",
    r"繞過.*規則",
]


def _matches_any(
    text: str,
    patterns: List[str],
) -> bool:
    """
    Perform case-insensitive deterministic pattern matching.
    """

    return any(
        re.search(
            pattern,
            text,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )
        is not None
        for pattern in patterns
    )


def apply_policy_gate(
    user_question: str,
) -> Optional[Dict[str, Any]]:
    """
    Block clearly unsafe requests before LLM routing.

    Returns:
    - None when the request may continue.
    - A complete blocked response when policy denies it.
    """

    if _matches_any(
        user_question,
        SECRET_PATTERNS,
    ):

        return {
            "answer": (
                "我不能提供、推測或還原資料庫密碼、"
                "帳號憑證、API Key、`.env` 內容、"
                "連線字串或其他內部秘密。"
            ),
            "model": "policy_gate",
            "tool_trace": [],
            "tool_rounds": 0,
            "policy_decision":
                "blocked_secret_request",
        }

    if _matches_any(
        user_question,
        SQL_EXECUTION_PATTERNS,
    ):

        return {
            "answer": (
                "我不能直接產生或執行任意 SQL。"
                "資料庫查詢只能透過目前允許的"
                "後端分析工具進行。"
            ),
            "model": "policy_gate",
            "tool_trace": [],
            "tool_rounds": 0,
            "policy_decision":
                "blocked_arbitrary_sql",
        }

    if (
        _matches_any(
            user_question,
            PROMPT_INJECTION_PATTERNS,
        )
        and (
            "sql"
            in user_question.lower()
            or
            "password"
            in user_question.lower()
            or
            "密碼"
            in user_question
            or
            "run_arbitrary_sql"
            in user_question.lower()
        )
    ):

        return {
            "answer": (
                "我不能忽略或繞過既有的安全限制。"
                "只能透過已允許的分析工具存取資料，"
                "也不能提供任何內部憑證或秘密。"
            ),
            "model": "policy_gate",
            "tool_trace": [],
            "tool_rounds": 0,
            "policy_decision":
                "blocked_prompt_injection",
        }

    return None