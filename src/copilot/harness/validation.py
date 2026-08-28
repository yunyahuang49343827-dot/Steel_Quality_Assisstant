from typing import Any, Dict

from pydantic import BaseModel


class ArgumentValidationResult(
    BaseModel,
):
    """
    Result of Harness tool argument validation.
    """

    valid: bool

    tool_name: str

    arguments: Dict[
        str,
        Any,
    ]

    reason: str


TOOL_ARGUMENT_RULES = {
    "get_quality_overview": {
        "allowed_keys": set(),
    },

    "get_defect_distribution": {
        "allowed_keys": set(),
    },

    "get_high_confidence_predictions": {
        "allowed_keys": {
            "limit",
        },
    },

    "predict_defect": {
        "allowed_keys": {
            "sample_id",
        },
    },

    "explain_prediction": {
        "allowed_keys": {
            "sample_id",
            "top_n",
        },
    },

    "get_defect_drivers": {
        "allowed_keys": {
            "defect_type",
            "top_n",
        },
    },
}


def validate_tool_arguments(
    tool_name: str,
    arguments: Any,
) -> ArgumentValidationResult:
    """
    Deterministically validate arguments requested
    by the LLM before backend tool execution.

    Validation goals:
    - arguments must be a dictionary
    - tool must have a known argument policy
    - unknown argument names are blocked
    - basic value constraints are enforced
    """

    if not isinstance(
        arguments,
        dict,
    ):

        return ArgumentValidationResult(
            valid=False,
            tool_name=tool_name,
            arguments={},
            reason=(
                "Tool arguments must be a dictionary."
            ),
        )

    rule = TOOL_ARGUMENT_RULES.get(
        tool_name
    )

    if rule is None:

        return ArgumentValidationResult(
            valid=False,
            tool_name=tool_name,
            arguments=arguments,
            reason=(
                "No Harness argument policy exists "
                "for this tool."
            ),
        )

    allowed_keys = rule[
        "allowed_keys"
    ]

    received_keys = set(
        arguments.keys()
    )

    unknown_keys = (
        received_keys
        - allowed_keys
    )

    if unknown_keys:

        return ArgumentValidationResult(
            valid=False,
            tool_name=tool_name,
            arguments=arguments,
            reason=(
                "Unexpected tool arguments: "
                + ", ".join(
                    sorted(
                        unknown_keys
                    )
                )
            ),
        )

    # -----------------------------------------------------
    # Generic limit / top_n validation
    # -----------------------------------------------------

    for key in (
        "limit",
        "top_n",
    ):

        if key not in arguments:
            continue

        value = arguments[
            key
        ]

        if (
            not isinstance(
                value,
                int,
            )
            or isinstance(
                value,
                bool,
            )
        ):

            return ArgumentValidationResult(
                valid=False,
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    f"{key} must be an integer."
                ),
            )

        if value < 1:

            return ArgumentValidationResult(
                valid=False,
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    f"{key} must be greater than 0."
                ),
            )

        if value > 20:

            return ArgumentValidationResult(
                valid=False,
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    f"{key} cannot exceed 20."
                ),
            )

    # -----------------------------------------------------
    # sample_id validation
    # -----------------------------------------------------

    if "sample_id" in arguments:

        sample_id = arguments[
            "sample_id"
        ]

        if (
            not isinstance(
                sample_id,
                int,
            )
            or isinstance(
                sample_id,
                bool,
            )
        ):

            return ArgumentValidationResult(
                valid=False,
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    "sample_id must be an integer."
                ),
            )

        if sample_id < 1:

            return ArgumentValidationResult(
                valid=False,
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    "sample_id must be greater than 0."
                ),
            )

    # -----------------------------------------------------
    # defect_type validation
    # -----------------------------------------------------

    if "defect_type" in arguments:

        defect_type = arguments[
            "defect_type"
        ]

        if not isinstance(
            defect_type,
            str,
        ):

            return ArgumentValidationResult(
                valid=False,
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    "defect_type must be a string."
                ),
            )

        defect_type = (
            defect_type.strip()
        )

        if not defect_type:

            return ArgumentValidationResult(
                valid=False,
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    "defect_type cannot be empty."
                ),
            )

    return ArgumentValidationResult(
        valid=True,
        tool_name=tool_name,
        arguments=arguments,
        reason=(
            "Tool arguments passed Harness validation."
        ),
    )