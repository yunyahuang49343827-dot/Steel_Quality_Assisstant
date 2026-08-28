import pytest

from src.copilot.harness.schemas import (
    TraceStage,
    TraceStatus,
)

from src.copilot.harness.trace import (
    ExecutionTracer,
)


pytestmark = pytest.mark.unit


def test_execution_trace_records_event():

    tracer = ExecutionTracer()

    tracer.record(
        stage=(
            TraceStage.POLICY_CHECK
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Policy check passed."
        ),
    )

    trace = tracer.finalize(
        "grounded"
    )

    assert (
        len(
            trace.events
        )
        == 1
    )

    assert (
        trace.events[
            0
        ].stage
        == TraceStage.POLICY_CHECK
    )

    assert (
        trace.events[
            0
        ].status
        == TraceStatus.PASSED
    )

    assert (
        trace.final_status
        == "grounded"
    )


def test_trace_metadata_is_preserved():

    tracer = ExecutionTracer()

    tracer.record(
        stage=(
            TraceStage.TOOL_PERMISSION
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Tool permission passed."
        ),
        metadata={
            "tool":
                "get_quality_overview"
        },
    )

    trace = tracer.finalize(
        "grounded"
    )

    assert (
        trace.events[
            0
        ].metadata[
            "tool"
        ]
        == "get_quality_overview"
    )


def test_trace_can_be_serialized():

    tracer = ExecutionTracer()

    tracer.record(
        stage=(
            TraceStage.FINAL_RESPONSE
        ),
        status=(
            TraceStatus.PASSED
        ),
        message=(
            "Final response generated."
        ),
    )

    tracer.finalize(
        "grounded"
    )

    payload = (
        tracer.to_dict()
    )

    assert (
        payload[
            "final_status"
        ]
        == "grounded"
    )

    assert (
        payload[
            "events"
        ][0][
            "stage"
        ]
        == "final_response"
    )

def test_duplicate_trace_event_is_ignored():

    tracer = ExecutionTracer()

    for _ in range(
        2
    ):

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
                "tool":
                    "get_quality_overview"
            },
        )

    trace = tracer.finalize(
        "completed"
    )

    assert (
        len(
            trace.events
        )
        == 1
    )