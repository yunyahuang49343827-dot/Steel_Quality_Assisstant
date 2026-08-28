import pytest

from src.evaluation.promotion_gate import (
    EvaluationMetrics,
    evaluate_promotion,
)


pytestmark = pytest.mark.unit


def build_base_metrics():

    return EvaluationMetrics(
        overall_pass_rate=0.80,
        grounding_pass_rate=0.50,
        security_pass_rate=1.00,
        empty_responses=0,
        unsafe_responses=0,
        language_regressions=0,
    )


def test_candidate_promotes_when_all_gates_pass():

    base = (
        build_base_metrics()
    )

    candidate = EvaluationMetrics(
        overall_pass_rate=0.90,
        grounding_pass_rate=0.75,
        security_pass_rate=1.00,
        empty_responses=0,
        unsafe_responses=0,
        language_regressions=0,
    )

    decision = (
        evaluate_promotion(
            base,
            candidate,
        )
    )

    assert decision.promoted


def test_overall_regression_blocks_promotion():

    base = (
        build_base_metrics()
    )

    candidate = EvaluationMetrics(
        overall_pass_rate=0.70,
        grounding_pass_rate=0.75,
        security_pass_rate=1.00,
    )

    decision = (
        evaluate_promotion(
            base,
            candidate,
        )
    )

    assert not decision.promoted


def test_grounding_must_improve():

    base = (
        build_base_metrics()
    )

    candidate = EvaluationMetrics(
        overall_pass_rate=0.90,
        grounding_pass_rate=0.50,
        security_pass_rate=1.00,
    )

    decision = (
        evaluate_promotion(
            base,
            candidate,
        )
    )

    assert not decision.promoted


def test_security_regression_blocks_promotion():

    base = (
        build_base_metrics()
    )

    candidate = EvaluationMetrics(
        overall_pass_rate=0.95,
        grounding_pass_rate=0.75,
        security_pass_rate=0.75,
    )

    decision = (
        evaluate_promotion(
            base,
            candidate,
        )
    )

    assert not decision.promoted


def test_empty_response_blocks_promotion():

    base = (
        build_base_metrics()
    )

    candidate = EvaluationMetrics(
        overall_pass_rate=0.90,
        grounding_pass_rate=0.75,
        security_pass_rate=1.00,
        empty_responses=1,
    )

    decision = (
        evaluate_promotion(
            base,
            candidate,
        )
    )

    assert not decision.promoted


def test_unsafe_response_blocks_promotion():

    base = (
        build_base_metrics()
    )

    candidate = EvaluationMetrics(
        overall_pass_rate=0.90,
        grounding_pass_rate=0.75,
        security_pass_rate=1.00,
        unsafe_responses=1,
    )

    decision = (
        evaluate_promotion(
            base,
            candidate,
        )
    )

    assert not decision.promoted