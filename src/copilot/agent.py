import json
from typing import Any, Dict, List

from src.copilot.harness.evaluator import (
    evaluate_output,
)
from src.copilot.harness.evidence import (
    verify_evidence_records,
)
from src.copilot.harness.instructions import (
    SYSTEM_PROMPT,
)
from src.copilot.harness.permissions import (
    check_tool_permission,
)
from src.copilot.harness.policies import (
    apply_policy_gate,
)
from src.copilot.harness.recovery import (
    recover_answer,
)
from src.copilot.harness.schemas import (
    TraceStage,
    TraceStatus,
)
from src.copilot.harness.trace import (
    ExecutionTracer,
)
from src.copilot.harness.validation import (
    validate_tool_arguments,
)
from src.copilot.ollama_client import (
    OLLAMA_MODEL,
    chat_with_ollama,
)
from src.tools.quality_tools import (
    execute_tool,
)
from src.tools.tool_schemas import (
    TOOL_SCHEMAS,
)


# =========================================================
# 1. Normalize Ollama message
# =========================================================


def message_to_dict(
    message,
) -> Dict[str, Any]:
    """
    Convert Ollama message object into a plain dictionary.
    """

    result = {
        "role": message.role,
        "content": message.content or "",
    }

    if getattr(
        message,
        "tool_calls",
        None,
    ):

        result["tool_calls"] = []

        for call in message.tool_calls:

            result["tool_calls"].append(
                {
                    "function": {
                        "name": (
                            call.function.name
                        ),
                        "arguments": (
                            call.function.arguments
                        ),
                    }
                }
            )

    return result


# =========================================================
# 2. Execute requested tool through Harness
# =========================================================


def execute_requested_tool(
    tool_call,
    tracer: ExecutionTracer,
):
    """
    Execute one model-requested tool through the
    Harness control boundary.
    """

    tool_name = (
        tool_call
        .function
        .name
    )

    arguments = (
        tool_call
        .function
        .arguments
        or {}
    )

    # =====================================================
    # Tool permission
    # =====================================================

    permission = (
        check_tool_permission(
            tool_name
        )
    )

    if not permission.allowed:

        tracer.record(
            stage=(
                TraceStage.TOOL_PERMISSION
            ),
            status=(
                TraceStatus.BLOCKED
            ),
            message=(
                "Tool permission check blocked."
            ),
            metadata={
                "tool": tool_name,
                "reason": (
                    permission.reason
                ),
            },
        )

        raise PermissionError(
            permission.reason
        )

    tracer.record(
        stage=(
            TraceStage.TOOL_PERMISSION
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Tool permission check passed."
        ),
        metadata={
            "tool": tool_name,
        },
    )

    # =====================================================
    # Argument validation
    # =====================================================

    validation = (
        validate_tool_arguments(
            tool_name=tool_name,
            arguments=arguments,
        )
    )

    if not validation.valid:

        tracer.record(
            stage=(
                TraceStage.ARGUMENT_VALIDATION
            ),
            status=(
                TraceStatus.BLOCKED
            ),
            message=(
                "Tool argument validation blocked."
            ),
            metadata={
                "tool": tool_name,
                "reason": (
                    validation.reason
                ),
            },
        )

        raise ValueError(
            validation.reason
        )

    tracer.record(
        stage=(
            TraceStage.ARGUMENT_VALIDATION
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Tool argument validation passed."
        ),
        metadata={
            "tool": tool_name,
        },
    )

    # =====================================================
    # Backend tool execution
    # =====================================================

    try:

        result = execute_tool(
            tool_name=tool_name,
            arguments=arguments,
        )

    except Exception as exc:

        tracer.record(
            stage=(
                TraceStage.TOOL_EXECUTION
            ),
            status=(
                TraceStatus.FAILED
            ),
            message=(
                "Backend tool execution failed."
            ),
            metadata={
                "tool": tool_name,
                "error": str(
                    exc
                ),
            },
        )

        raise

    tracer.record(
        stage=(
            TraceStage.TOOL_EXECUTION
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Backend tool execution succeeded."
        ),
        metadata={
            "tool": tool_name,
        },
    )

    return (
        tool_name,
        arguments,
        result,
    )


# =========================================================
# 3. Final-answer fallback generation
# =========================================================


def generate_final_answer_from_tools(
    messages,
):
    """
    Ask Ollama one additional time without tools after
    tool results are already available.
    """

    fallback_messages = (
        messages
        + [
            {
                "role": "system",
                "content": (
                    "請根據上方已取得的 tool result "
                    "直接回答使用者問題。"
                    "不得新增沒有證據的數字或結論。"
                ),
            }
        ]
    )

    response = chat_with_ollama(
        messages=fallback_messages,
        tools=None,
    )

    return (
        response.message
        .content
        .strip()
    )


# =========================================================
# 4. Build response
# =========================================================


def build_response(
    *,
    answer: str,
    model: str,
    tool_trace: List[
        Dict[str, Any]
    ],
    tool_rounds: int,
    policy_decision: str,
    tracer: ExecutionTracer,
    harness_status: str,
) -> Dict[str, Any]:

    tracer.record(
        stage=(
            TraceStage.FINAL_RESPONSE
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Final response prepared."
        ),
    )

    harness_trace = (
        tracer.finalize(
            harness_status
        )
    )

    return {
        "answer": answer,
        "model": model,
        "tool_trace": tool_trace,
        "tool_rounds": tool_rounds,
        "policy_decision": (
            policy_decision
        ),
        "harness_trace": (
            harness_trace.model_dump(
                mode="json"
            )
        ),
    }


# =========================================================
# 5. Evidence verification + output evaluation
# =========================================================


def evaluate_and_recover_answer(
    *,
    answer: str,
    messages: List[
        Dict[str, Any]
    ],
    evidence_records: List[
        Dict[str, Any]
    ],
    tracer: ExecutionTracer,
) -> tuple[
    str,
    str,
]:
    """
    Validate evidence, evaluate the draft answer,
    and perform at most one bounded recovery rewrite.

    Returns:
        final_answer,
        harness_status
    """

    # =====================================================
    # Evidence verification
    # =====================================================

    evidence_result = (
        verify_evidence_records(
            evidence_records
        )
    )

    if evidence_records:

        if evidence_result.valid:

            tracer.record(
                stage=(
                    TraceStage.EVIDENCE_VERIFICATION
                ),
                status=(
                    TraceStatus.PASSED
                ),
                message=(
                    "Tool evidence verification passed."
                ),
                metadata={
                    "usable_evidence_count": (
                        evidence_result
                        .usable_evidence_count
                    ),
                },
            )

        else:

            tracer.record(
                stage=(
                    TraceStage.EVIDENCE_VERIFICATION
                ),
                status=(
                    TraceStatus.FAILED
                ),
                message=(
                    "Tool evidence verification failed."
                ),
                metadata={
                    "reason": (
                        evidence_result.reason
                    ),
                },
            )

            return (
                (
                    "目前取得的工具證據未通過"
                    "完整性驗證，因此暫不提供"
                    "可能誤導的分析結論。"
                ),
                "evidence_failed",
            )

    else:

        tracer.record(
            stage=(
                TraceStage.EVIDENCE_VERIFICATION
            ),
            status=(
                TraceStatus.SKIPPED
            ),
            message=(
                "No tool evidence required "
                "for verification."
            ),
        )

    # =====================================================
    # Initial output evaluation
    # =====================================================

    evaluation = (
        evaluate_output(
            answer=answer,
            evidence_records=(
                evidence_records
            ),
        )
    )

    if evaluation.passed:

        tracer.record(
            stage=(
                TraceStage.OUTPUT_EVALUATION
            ),
            status=(
                TraceStatus.PASSED
            ),
            message=(
                "Final answer passed "
                "output evaluation."
            ),
        )

        return (
            answer,
            "completed",
        )

    tracer.record(
        stage=(
            TraceStage.OUTPUT_EVALUATION
        ),
        status=(
            TraceStatus.FAILED
        ),
        message=(
            "Initial answer failed "
            "output evaluation."
        ),
        metadata={
            "issues": [
                issue.code
                for issue
                in evaluation.issues
            ],
        },
    )

    # =====================================================
    # One bounded recovery attempt
    # =====================================================

    try:

        recovered_answer = (
            recover_answer(
                messages=messages,
                issues=(
                    evaluation.issues
                ),
            )
        )

    except Exception as exc:

        tracer.record(
            stage=(
                TraceStage.RECOVERY
            ),
            status=(
                TraceStatus.FAILED
            ),
            message=(
                "Recovery generation failed."
            ),
            metadata={
                "error": str(
                    exc
                ),
            },
        )

        return (
            (
                "模型回答未通過安全與證據檢查，"
                "且自動修正失敗。"
                "請改以更明確的分析問題重新查詢。"
            ),
            "safe_fallback",
        )

    second_evaluation = (
        evaluate_output(
            answer=recovered_answer,
            evidence_records=(
                evidence_records
            ),
        )
    )

    if second_evaluation.passed:

        tracer.record(
            stage=(
                TraceStage.RECOVERY
            ),
            status=(
                TraceStatus.PASSED
            ),
            message=(
                "Answer recovered successfully."
            ),
        )

        tracer.record(
            stage=(
                TraceStage.OUTPUT_EVALUATION
            ),
            status=(
                TraceStatus.PASSED
            ),
            message=(
                "Recovered answer passed "
                "output evaluation."
            ),
            metadata={
                "recovered": True,
            },
        )

        return (
            recovered_answer,
            "recovered",
        )

    tracer.record(
        stage=(
            TraceStage.RECOVERY
        ),
        status=(
            TraceStatus.FAILED
        ),
        message=(
            "Recovered answer still failed "
            "output evaluation."
        ),
        metadata={
            "issues": [
                issue.code
                for issue
                in second_evaluation.issues
            ],
        },
    )

    return (
        (
            "模型回答未通過安全與證據檢查，"
            "因此 Harness 已停止輸出該分析結論。"
            "目前可確認的是後端工具已取得資料，"
            "但最終自然語言回答需要人工覆核。"
        ),
        "safe_fallback",
    )


# =========================================================
# 6. Run Copilot
# =========================================================


def run_copilot(
    user_question: str,
    max_tool_rounds: int = 4,
) -> Dict[str, Any]:
    """
    Run local Qwen tool-calling agent through the
    Harness runtime control layer.
    """

    if not isinstance(
        user_question,
        str,
    ):

        raise ValueError(
            "user_question must be a string."
        )

    user_question = (
        user_question.strip()
    )

    if not user_question:

        raise ValueError(
            "user_question cannot be empty."
        )

    tracer = ExecutionTracer()

    # =====================================================
    # Policy gate
    # =====================================================

    policy_response = (
        apply_policy_gate(
            user_question
        )
    )

    if policy_response is not None:

        tracer.record(
            stage=(
                TraceStage.POLICY_CHECK
            ),
            status=(
                TraceStatus.BLOCKED
            ),
            message=(
                "Request blocked by deterministic "
                "policy gate."
            ),
            metadata={
                "policy_decision": (
                    policy_response[
                        "policy_decision"
                    ]
                ),
            },
        )

        policy_response[
            "harness_trace"
        ] = (
            tracer.finalize(
                "blocked"
            )
            .model_dump(
                mode="json"
            )
        )

        return policy_response

    tracer.record(
        stage=(
            TraceStage.POLICY_CHECK
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Request passed deterministic "
            "policy gate."
        ),
    )

    # =====================================================
    # LLM conversation
    # =====================================================

    messages: List[
        Dict[str, Any]
    ] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    tool_trace: List[
        Dict[str, Any]
    ] = []

    evidence_records: List[
        Dict[str, Any]
    ] = []

    for round_number in range(
        1,
        max_tool_rounds + 1,
    ):

        response = (
            chat_with_ollama(
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        )

        assistant_message = (
            response.message
        )

        messages.append(
            message_to_dict(
                assistant_message
            )
        )

        tool_calls = (
            assistant_message
            .tool_calls
            or []
        )

        # =================================================
        # No tool call
        # =================================================

        if not tool_calls:

            answer = (
                assistant_message
                .content
                .strip()
            )

            if (
                not answer
                and tool_trace
            ):

                answer = (
                    generate_final_answer_from_tools(
                        messages
                    )
                )

            if not answer:

                answer = (
                    "目前沒有足夠的可用證據回答此問題。"
                )

            (
                final_answer,
                harness_status,
            ) = (
                evaluate_and_recover_answer(
                    answer=answer,
                    messages=messages,
                    evidence_records=(
                        evidence_records
                    ),
                    tracer=tracer,
                )
            )

            return build_response(
                answer=final_answer,
                model=OLLAMA_MODEL,
                tool_trace=tool_trace,
                tool_rounds=(
                    round_number - 1
                ),
                policy_decision=(
                    "allowed"
                ),
                tracer=tracer,
                harness_status=(
                    harness_status
                ),
            )

        # =================================================
        # Execute tool calls
        # =================================================

        for tool_call in tool_calls:

            try:

                (
                    tool_name,
                    arguments,
                    tool_result,
                ) = (
                    execute_requested_tool(
                        tool_call=tool_call,
                        tracer=tracer,
                    )
                )

                tool_trace.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "status": "success",
                    }
                )

                evidence_records.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": tool_result,
                    }
                )

                tool_content = json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    default=str,
                )

            except Exception as exc:

                tool_name = (
                    getattr(
                        getattr(
                            tool_call,
                            "function",
                            None,
                        ),
                        "name",
                        "unknown",
                    )
                )

                tool_trace.append(
                    {
                        "tool": tool_name,
                        "status": (
                            "blocked_or_failed"
                        ),
                        "error": str(
                            exc
                        ),
                    }
                )

                tool_content = json.dumps(
                    {
                        "error": (
                            "Tool execution "
                            "was blocked or failed."
                        )
                    },
                    ensure_ascii=False,
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": tool_content,
                }
            )

    # =====================================================
    # Max tool rounds reached
    # =====================================================

    if tool_trace:

        answer = (
            generate_final_answer_from_tools(
                messages
            )
        )

        if answer:

            (
                final_answer,
                harness_status,
            ) = (
                evaluate_and_recover_answer(
                    answer=answer,
                    messages=messages,
                    evidence_records=(
                        evidence_records
                    ),
                    tracer=tracer,
                )
            )

            return build_response(
                answer=final_answer,
                model=OLLAMA_MODEL,
                tool_trace=tool_trace,
                tool_rounds=(
                    max_tool_rounds
                ),
                policy_decision=(
                    "allowed"
                ),
                tracer=tracer,
                harness_status=(
                    harness_status
                ),
            )

    return build_response(
        answer=(
            "目前無法在允許的工具呼叫次數內完成查詢，"
            "請將問題縮小範圍後再試一次。"
        ),
        model=OLLAMA_MODEL,
        tool_trace=tool_trace,
        tool_rounds=(
            max_tool_rounds
        ),
        policy_decision=(
            "allowed"
        ),
        tracer=tracer,
        harness_status=(
            "safe_fallback"
        ),
    )