import pytest

from fastapi.testclient import (
    TestClient,
)

from src.api.main import app


client = TestClient(
    app
)


@pytest.mark.api
def test_openapi_is_available():

    response = client.get(
        "/openapi.json"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        "openapi"
        in payload
    )

    assert (
        "paths"
        in payload
    )


@pytest.mark.api
def test_health_route_exists():

    response = client.get(
        "/openapi.json"
    )

    payload = response.json()

    assert (
        "/health"
        in payload["paths"]
    )


@pytest.mark.api
def test_health_supports_get():

    response = client.get(
        "/openapi.json"
    )

    payload = response.json()

    health_schema = (
        payload["paths"][
            "/health"
        ]
    )

    assert (
        "get"
        in health_schema
    )