import pytest

from fastapi.testclient import (
    TestClient,
)

from src.api.main import app


client = TestClient(
    app
)


@pytest.mark.api
def test_quality_overview_route_exists():

    response = client.get(
        "/openapi.json"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        "/quality/overview"
        in payload["paths"]
    )


@pytest.mark.api
def test_quality_overview_supports_get():

    response = client.get(
        "/openapi.json"
    )

    payload = response.json()

    route_schema = (
        payload["paths"][
            "/quality/overview"
        ]
    )

    assert (
        "get"
        in route_schema
    )


@pytest.mark.api
def test_global_shap_route_exists():

    response = client.get(
        "/openapi.json"
    )

    payload = response.json()

    assert (
        "/explain/global"
        in payload["paths"]
    )