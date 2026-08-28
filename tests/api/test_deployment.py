
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import (
    APP_ENV,
    APP_VERSION,
    DEPLOYMENT_VARIANT,
)


client = TestClient(
    app
)


def test_deployment_endpoint_returns_metadata():

    response = client.get(
        "/deployment"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert payload == {
        "variant":
            DEPLOYMENT_VARIANT,

        "version":
            APP_VERSION,

        "environment":
            APP_ENV,
    }


def test_deployment_metadata_is_non_empty():

    response = client.get(
        "/deployment"
    )

    payload = (
        response.json()
    )

    assert payload[
        "variant"
    ]

    assert payload[
        "version"
    ]

    assert payload[
        "environment"
    ]