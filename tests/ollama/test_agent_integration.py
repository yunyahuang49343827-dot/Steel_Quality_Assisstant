import pytest

from src.copilot.agent import (
    run_copilot,
)


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.integration,
]


def get_successful_tools(
    result,
):
    """
    Return successfully executed tool names.
    """

    return [
        item.get(
            "tool"
        )
        for item in result[
            "tool_trace"
        ]
        if item.get(
            "status"
        )
        == "success"
    ]


def assert_no_forbidden_tool(
    result,
):
    """
    Arbitrary SQL must never appear in tool execution.
    """

    tools = [
        item.get(
            "tool"
        )
        for item in result[
            "tool_trace"
        ]
    ]

    assert (
        "run_arbitrary_sql"
        not in tools
    )


def test_dataset_overview_uses_correct_tool():

    result = run_copilot(
        (
            "目前這份鋼材品質資料總共有多少筆資料，"
            "以及幾種缺陷類別？"
        )
    )

    tools = get_successful_tools(
        result
    )

    assert (
        result[
            "policy_decision"
        ]
        == "allowed"
    )

    assert (
        "get_quality_overview"
        in tools
    )

    assert result[
        "answer"
    ].strip()

    assert (
        "18,380"
        in result["answer"]
        or
        "18380"
        in result["answer"]
    )

    assert (
        "7"
        in result[
            "answer"
        ]
    )

    assert_no_forbidden_tool(
        result
    )


def test_defect_distribution_uses_correct_tool():

    result = run_copilot(
        (
            "哪一種鋼材缺陷最常見？"
            "請告訴我數量和比例。"
        )
    )

    tools = get_successful_tools(
        result
    )

    assert (
        result[
            "policy_decision"
        ]
        == "allowed"
    )

    assert (
        "get_defect_distribution"
        in tools
    )

    answer = result[
        "answer"
    ]

    assert (
        "Other_Faults"
        in answer
    )

    assert (
        "6,540"
        in answer
        or
        "6540"
        in answer
    )

    assert (
        "35.58"
        in answer
    )

    assert_no_forbidden_tool(
        result
    )


def test_k_scatch_drivers_use_explainability_tool():

    result = run_copilot(
        (
            "請列出 K_Scatch 最重要的 "
            "5 個模型判斷特徵。"
        )
    )

    tools = get_successful_tools(
        result
    )

    assert (
        result[
            "policy_decision"
        ]
        == "allowed"
    )

    assert (
        "get_defect_drivers"
        in tools
    )

    answer = result[
        "answer"
    ]

    assert (
        "K_Scatch"
        in answer
    )

    assert (
        "Steel_Plate_Thickness"
        in answer
    )

    assert_no_forbidden_tool(
        result
    )


def test_high_confidence_predictions_use_correct_tool():

    result = run_copilot(
        (
            "請列出 3 筆目前模型信心度最高的"
            "缺陷預測樣本。"
        )
    )

    tools = get_successful_tools(
        result
    )

    assert (
        result[
            "policy_decision"
        ]
        == "allowed"
    )

    assert (
        "get_high_confidence_predictions"
        in tools
    )

    assert result[
        "answer"
    ].strip()

    assert_no_forbidden_tool(
        result
    )