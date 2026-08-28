import pytest

from fastapi.testclient import (
    TestClient,
)

from src.api.main import app


pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures(
        "integration_database"
    ),
]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures(
        "integration_database"
    ),
]

client = TestClient(
    app
)


def test_health_with_real_postgresql():

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["status"]
        == "healthy"
    )

    assert (
        payload["database"]
        == "healthy"
    )

    assert (
        payload["model"]
        == "loaded"
    )


def test_quality_overview_with_real_postgresql():

    response = client.get(
        "/quality/overview"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload[
            "modeling_samples"
        ]
        == 10
    )

    assert (
        payload[
            "defect_classes"
        ]
        == 7
    )

    assert (
        payload[
            "champion_model"
        ]
        == "XGBoost"
    )

    assert (
        payload[
            "test_macro_f1"
        ]
        == 0.5904
    )

    assert (
        payload[
            "feature_count"
        ]
        == 27
    )