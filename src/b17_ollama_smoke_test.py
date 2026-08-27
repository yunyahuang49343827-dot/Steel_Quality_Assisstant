from src.copilot.agent import (
    run_copilot,
)


TEST_QUESTIONS = [
    (
        "目前這份鋼材品質資料總共有多少筆資料，"
        "以及幾種缺陷類別？"
    ),

    (
        "哪一種鋼材缺陷最常見？"
        "請告訴我數量和比例。"
    ),

    (
        "請列出 K_Scatch 最重要的 "
        "5 個模型判斷特徵。"
    ),

    (
        "請列出 3 筆目前模型信心度最高的"
        "缺陷預測樣本。"
    ),
]


def main():

    print("=" * 72)
    print(
        "Stage B17 — Qwen Function Calling Agent"
    )
    print("=" * 72)

    for index, question in enumerate(
        TEST_QUESTIONS,
        start=1,
    ):

        print(
            f"\nQUESTION {index}"
        )

        print("-" * 72)

        print(
            question
        )

        result = run_copilot(
            question
        )

        print("\nANSWER")

        print(
            result[
                "answer"
            ]
        )

        print("\nTOOL TRACE")

        if (
            result[
                "tool_trace"
            ]
        ):

            for trace in (
                result[
                    "tool_trace"
                ]
            ):

                print(
                    trace
                )

        else:

            print(
                "No tool used"
            )

        print(
            f"\nTool rounds : "
            f"{result['tool_rounds']}"
        )

    print(
        "\nQwen function calling agent: PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()