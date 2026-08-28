import pytest

from src.training.build_sft_dataset import (
    build_curated_samples,
    ensure_no_split_leakage,
    sample_hash,
    stratified_split,
)
from src.training.sft_schema import (
    SFTCategory,
)


pytestmark = pytest.mark.unit


def test_curated_dataset_has_expected_size():

    samples = (
        build_curated_samples()
    )

    assert (
        len(
            samples
        )
        == 40
    )


def test_all_categories_exist():

    samples = (
        build_curated_samples()
    )

    categories = {
        sample.category
        for sample
        in samples
    }

    assert categories == set(
        SFTCategory
    )


def test_each_category_has_eight_samples():

    samples = (
        build_curated_samples()
    )

    for category in (
        SFTCategory
    ):

        count = sum(
            1
            for sample
            in samples
            if (
                sample.category
                == category
            )
        )

        assert count == 8


def test_sample_ids_are_unique():

    samples = (
        build_curated_samples()
    )

    ids = [
        sample.sample_id
        for sample
        in samples
    ]

    assert (
        len(
            ids
        )
        == len(
            set(
                ids
            )
        )
    )


def test_content_hashes_are_unique():

    samples = (
        build_curated_samples()
    )

    hashes = [
        sample_hash(
            sample
        )
        for sample
        in samples
    ]

    assert (
        len(
            hashes
        )
        == len(
            set(
                hashes
            )
        )
    )


def test_split_preserves_all_samples():

    samples = (
        build_curated_samples()
    )

    (
        train_samples,
        validation_samples,
    ) = stratified_split(
        samples
    )

    assert (
        len(
            train_samples
        )
        + len(
            validation_samples
        )
        == len(
            samples
        )
    )


def test_validation_contains_all_categories():

    samples = (
        build_curated_samples()
    )

    (
        _,
        validation_samples,
    ) = stratified_split(
        samples
    )

    categories = {
        sample.category
        for sample
        in validation_samples
    }

    assert categories == set(
        SFTCategory
    )


def test_no_train_validation_leakage():

    samples = (
        build_curated_samples()
    )

    (
        train_samples,
        validation_samples,
    ) = stratified_split(
        samples
    )

    ensure_no_split_leakage(
        train_samples,
        validation_samples,
    )