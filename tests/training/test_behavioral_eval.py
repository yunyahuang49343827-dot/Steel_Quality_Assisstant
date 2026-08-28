import pytest

from src.training.build_behavioral_eval import (
    build_eval_cases,
)


pytestmark = pytest.mark.unit


def test_behavioral_eval_has_15_cases():

    cases = build_eval_cases()

    assert len(
        cases
    ) == 15


def test_behavioral_eval_has_five_categories():

    cases = build_eval_cases()

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


def test_each_category_has_three_cases():

    cases = build_eval_cases()

    for category in {
        "grounding",
        "explainability",
        "confidence",
        "security",
        "fallback",
    }:

        count = sum(
            1
            for case
            in cases
            if case[
                "category"
            ]
            == category
        )

        assert count == 3


def test_eval_ids_are_unique():

    cases = build_eval_cases()

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


def test_all_cases_have_rules():

    cases = build_eval_cases()

    for case in cases:

        assert case[
            "must_include_any"
        ]

        assert (
            "must_not_include"
            in case
        )