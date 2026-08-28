from enum import Enum
from typing import List

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class SFTCategory(
    str,
    Enum,
):
    """
    Behavioral categories used by the SFT dataset.
    """

    GROUNDING = "grounding"

    EXPLAINABILITY = "explainability"

    CONFIDENCE = "confidence"

    SECURITY = "security"

    FALLBACK = "fallback"


class SFTMessage(
    BaseModel,
):
    """
    One chat message in a supervised fine-tuning sample.
    """

    role: str

    content: str

    @field_validator(
        "role"
    )
    @classmethod
    def validate_role(
        cls,
        value: str,
    ) -> str:

        allowed_roles = {
            "system",
            "user",
            "assistant",
        }

        if value not in allowed_roles:

            raise ValueError(
                "Unsupported SFT message role."
            )

        return value

    @field_validator(
        "content"
    )
    @classmethod
    def validate_content(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:

            raise ValueError(
                "SFT message content cannot be empty."
            )

        return value


class SFTSample(
    BaseModel,
):
    """
    One supervised fine-tuning sample.

    Metadata is retained during dataset construction
    and evaluation.

    The messages field can later be directly converted
    into the chat format required by the target model.
    """

    sample_id: str

    category: SFTCategory

    messages: List[
        SFTMessage
    ]

    source: str = "curated"

    tags: List[
        str
    ] = Field(
        default_factory=list
    )

    @field_validator(
        "sample_id"
    )
    @classmethod
    def validate_sample_id(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:

            raise ValueError(
                "sample_id cannot be empty."
            )

        return value

    @field_validator(
        "messages"
    )
    @classmethod
    def validate_messages(
        cls,
        value: List[
            SFTMessage
        ],
    ) -> List[
        SFTMessage
    ]:

        if len(
            value
        ) < 2:

            raise ValueError(
                "SFT sample must contain "
                "at least user and assistant messages."
            )

        roles = [
            message.role
            for message
            in value
        ]

        if "user" not in roles:

            raise ValueError(
                "SFT sample must contain "
                "a user message."
            )

        if (
            roles[-1]
            != "assistant"
        ):

            raise ValueError(
                "SFT sample must end with "
                "an assistant response."
            )

        return value