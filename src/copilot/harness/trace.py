from typing import Any, Dict, Optional

from src.copilot.harness.schemas import (
    HarnessTrace,
    TraceEvent,
    TraceStage,
    TraceStatus,
)


class ExecutionTracer:
    """
    Collect externally safe Harness runtime events.

    The tracer records control-flow outcomes only.
    It must never store model chain-of-thought.
    """

    def __init__(
        self,
    ) -> None:

        self._trace = (
            HarnessTrace()
        )


    def _is_duplicate(
        self,
        event: TraceEvent,
    ) -> bool:
        """
        Prevent accidental duplicate consecutive events.

        Events are considered duplicates when stage,
        status, message, and metadata are identical
        to the immediately previous event.
        """

        if not self._trace.events:

            return False

        previous = (
            self._trace.events[
                -1
            ]
        )

        return (
            previous.stage
            == event.stage
            and
            previous.status
            == event.status
            and
            previous.message
            == event.message
            and
            previous.metadata
            == event.metadata
        )


    def record(
        self,
        stage: TraceStage,
        status: TraceStatus,
        message: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:

        event = TraceEvent(
            stage=stage,
            status=status,
            message=message,
            metadata=(
                metadata
                or {}
            ),
        )

        if self._is_duplicate(
            event
        ):

            return

        self._trace.events.append(
            event
        )


    def finalize(
        self,
        final_status: str,
    ) -> HarnessTrace:

        self._trace.final_status = (
            final_status
        )

        return self._trace


    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize trace for API responses or audit logs.
        """

        return (
            self._trace
            .model_dump(
                mode="json"
            )
        )