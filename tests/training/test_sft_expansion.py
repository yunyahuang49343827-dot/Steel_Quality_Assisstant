import pytest

from src.training.build_sft_dataset import (
    build_curated_samples,
)
from src.training.sft_expansion import (
    build_expansion_samples,
)
from src.training.sft_schema import (
    SFTCategory,
)


pytestmark = pytest.mark.unit


def test_expansion_has_expected_size():

    samples = (
        build_expansion_samples()
    )

    assert len(
        samples
    ) == 80


def test_expansion_has_all_categories():

    samples = (
        build_expansion_samples()
    )

    categories = {
        sample.category
        for sample
        in samples
    }

    assert categories == set(
        SFTCategory
    )


def test_each_category_has_16_expansion_samples():

    samples = (
        build_expansion_samples()
    )

    for category in SFTCategory:

        count = sum(
            1
            for sample
            in samples
            if sample.category
            == category
        )

        assert count == 16


def test_combined_dataset_has_120_samples():

    samples = (
        build_curated_samples()
        + build_expansion_samples()
    )

    assert len(
        samples
    ) == 120


def test_combined_sample_ids_are_unique():

    samples = (
        build_curated_samples()
        + build_expansion_samples()
    )

    ids = [
        sample.sample_id
        for sample
        in samples
    ]

    assert len(
        ids
    ) == len(
        set(
            ids
        )
    )