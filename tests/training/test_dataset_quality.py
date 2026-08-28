
import pytest

from src.training.build_sft_dataset import (
    build_curated_samples,
)
from src.training.dataset_quality import (
    answer_length_statistics,
    build_freeze_manifest,
    build_quality_report,
    dataset_fingerprint,
    find_exact_duplicates,
    find_split_near_leakage,
    scan_unsafe_targets,
)
from src.training.sft_expansion import (
    build_expansion_samples,
)


pytestmark = pytest.mark.unit


def get_combined_samples():

    return (
        build_curated_samples()
        + build_expansion_samples()
    )


def test_exact_duplicate_check_passes():

    samples = (
        get_combined_samples()
    )

    duplicates = (
        find_exact_duplicates(
            samples
        )
    )

    assert duplicates == []


def test_dataset_fingerprint_is_deterministic():

    samples = (
        get_combined_samples()
    )

    first = (
        dataset_fingerprint(
            samples
        )
    )

    second = (
        dataset_fingerprint(
            samples
        )
    )

    assert first == second

    assert len(
        first
    ) == 64


def test_unsafe_target_scan_passes():

    samples = (
        get_combined_samples()
    )

    findings = (
        scan_unsafe_targets(
            samples
        )
    )

    assert findings == []


def test_answer_length_statistics_are_valid():

    samples = (
        get_combined_samples()
    )

    stats = (
        answer_length_statistics(
            samples
        )
    )

    assert (
        stats[
            "min_chars"
        ]
        > 0
    )

    assert (
        stats[
            "max_chars"
        ]
        >= stats[
            "min_chars"
        ]
    )

    assert (
        stats[
            "mean_chars"
        ]
        > 0
    )


def test_split_near_leakage_detects_identical_prompt():

    samples = (
        get_combined_samples()
    )

    train_sample = (
        samples[
            0
        ]
    )

    validation_sample = (
        train_sample.model_copy(
            deep=True
        )
    )

    validation_sample.sample_id = (
        "synthetic_validation_copy"
    )

    leakage = (
        find_split_near_leakage(
            [
                train_sample
            ],
            [
                validation_sample
            ],
        )
    )

    assert len(
        leakage
    ) == 1


def test_quality_report_has_120_samples():

    samples = (
        get_combined_samples()
    )

    train_samples = (
        samples[
            :95
        ]
    )

    validation_samples = (
        samples[
            95:
        ]
    )

    report = (
        build_quality_report(
            train_samples,
            validation_samples,
        )
    )

    assert (
        report[
            "total_samples"
        ]
        == 120
    )


def test_freeze_manifest_requires_passed_gate():

    report = {
        "quality_gate_passed":
            False
    }

    with pytest.raises(
        RuntimeError
    ):

        build_freeze_manifest(
            report
        )


def test_freeze_manifest_contains_hash():

    samples = (
        get_combined_samples()
    )

    train_samples = (
        samples[
            :95
        ]
    )

    validation_samples = (
        samples[
            95:
        ]
    )

    report = (
        build_quality_report(
            train_samples,
            validation_samples,
        )
    )

    if not report[
        "quality_gate_passed"
    ]:

        pytest.skip(
            "Synthetic split contains "
            "near leakage."
        )

    manifest = (
        build_freeze_manifest(
            report
        )
    )

    assert (
        manifest[
            "status"
        ]
        == "frozen"
    )

    assert len(
        manifest[
            "full_dataset_sha256"
        ]
    ) == 64