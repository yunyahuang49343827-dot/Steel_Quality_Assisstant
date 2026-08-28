import pytest

from src.evaluation.locked_eval import (
    build_locked_eval_cases,
)


pytestmark = pytest.mark.unit


def test_locked_eval_has_20_cases():

    cases = (
        build_locked_eval_cases()
    )

    assert len(
        cases
    ) == 20


def test_locked_eval_has_five_categories():

    cases = (
        build_locked_eval_cases()
    )

    categories = {
        case[
            "category"
        ]
        for case in cases
    }

    assert categories == {
        "grounding",
        "explainability",
        "confidence",
        "security",
        "fallback",
    }


def test_each_category_has_four_cases():

    cases = (
        build_locked_eval_cases()
    )

    categories = {
        "grounding",
        "explainability",
        "confidence",
        "security",
        "fallback",
    }

    for category in categories:

        count = sum(
            1
            for case
            in cases
            if case[
                "category"
            ]
            == category
        )

        assert count == 4


def test_locked_eval_ids_are_unique():

    cases = (
        build_locked_eval_cases()
    )

    ids = [
        case[
            "eval_id"
        ]
        for case in cases
    ]

    assert len(
        ids
    ) == len(
        set(
            ids
        )
    )


def test_locked_cases_have_required_metadata():

    cases = (
        build_locked_eval_cases()
    )

    for case in cases:

        assert case[
            "prompt"
        ]

        assert case[
            "required_concepts"
        ]

        assert (
            "forbidden_behaviors"
            in case
        )