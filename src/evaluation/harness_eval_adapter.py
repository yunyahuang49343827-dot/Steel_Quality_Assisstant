from typing import Callable

from mlx_lm import generate


DEFAULT_SYSTEM_PROMPT = (
    "你是 Steel Quality Analytics Copilot。"
    "請用繁體中文簡潔但完整地回答。"
    "涉及資料、模型、製造因果、安全或風險時，"
    "不得超出可驗證證據。"
)


def build_mlx_generator(
    model,
    tokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 256,
) -> Callable[
    [
        str,
        str,
    ],
    str,
]:
    """
    Adapt an MLX model into the generic generator
    interface expected by runtime_harness.

    The harness itself remains model-agnostic.
    """

    def generator(
        user_prompt: str,
        extra_instruction: str,
    ) -> str:

        effective_system_prompt = (
            system_prompt
        )

        if extra_instruction:

            effective_system_prompt = (
                effective_system_prompt
                + "\n\n"
                + extra_instruction
            )

        messages = [
            {
                "role": "system",
                "content":
                    effective_system_prompt,
            },
            {
                "role": "user",
                "content":
                    user_prompt,
            },
        ]

        prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

    return generator