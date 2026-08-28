import os

import httpx
import pytest


GATEWAY_BASE_URL = os.getenv(
    "GATEWAY_BASE_URL",
    "http://localhost:8008",
)

FRONTEND_BASE_URL = os.getenv(
    "FRONTEND_BASE_URL",
    "http://localhost:5173",
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.docker,
]


# =========================================================
# Helpers
# =========================================================

def get(
    url: str,
) -> httpx.Response:
    """
    Send one HTTP request with a bounded timeout.
    """

    return httpx.get(
        url,
        timeout=10.0,
    )


# =========================================================
# 1. Kong health routing
# =========================================================

def test_gateway_routes_health_to_fastapi():

    response = get(
        f"{GATEWAY_BASE_URL}/api/health"
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
        payload["environment"]
        == "docker"
    )

    assert (
        payload["database"]
        == "healthy"
    )


# =========================================================
# 2. Kong proxy evidence
# =========================================================

def test_gateway_response_contains_kong_headers():

    response = get(
        f"{GATEWAY_BASE_URL}/api/deployment"
    )

    assert (
        response.status_code
        == 200
    )

    via = response.headers.get(
        "Via",
        "",
    )

    assert "kong" in via.lower()

    assert (
        "X-Kong-Request-Id"
        in response.headers
    )

    assert (
        "X-Kong-Proxy-Latency"
        in response.headers
    )

    assert (
        "X-Kong-Upstream-Latency"
        in response.headers
    )


# =========================================================
# 3. Rate-limit policy evidence
# =========================================================

def test_gateway_exposes_rate_limit_headers():

    response = get(
        f"{GATEWAY_BASE_URL}/api/deployment"
    )

    assert (
        response.status_code
        == 200
    )

    limit = response.headers.get(
        "X-RateLimit-Limit-Minute"
    )

    remaining = response.headers.get(
        "X-RateLimit-Remaining-Minute"
    )

    assert limit is not None
    assert remaining is not None

    assert int(limit) > 0

    assert (
        0
        <= int(remaining)
        <= int(limit)
    )


# =========================================================
# 4. Frontend → Nginx → Kong → Backend
# =========================================================

def test_frontend_api_path_still_passes_through_kong():

    response = get(
        f"{FRONTEND_BASE_URL}/api/deployment"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert payload[
        "variant"
    ] in {
        "stable",
        "canary",
    }

    assert payload[
        "version"
    ] in {
        "v1",
        "v2",
    }

    via = response.headers.get(
        "Via",
        "",
    )

    assert "kong" in via.lower()


# =========================================================
# 5. Deployment metadata consistency
# =========================================================

def test_deployment_variant_matches_version():

    response = get(
        f"{GATEWAY_BASE_URL}/api/deployment"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    valid_deployments = {
        ("stable", "v1"),
        ("canary", "v2"),
    }

    deployment = (
        payload["variant"],
        payload["version"],
    )

    assert (
        deployment
        in valid_deployments
    )


# =========================================================
# 6. Weighted Canary routing
# =========================================================

def test_weighted_canary_routes_to_both_variants():

    request_count = 100

    stable_count = 0
    canary_count = 0
    unexpected_count = 0

    for _ in range(
        request_count
    ):

        response = get(
            f"{GATEWAY_BASE_URL}/api/deployment"
        )

        assert (
            response.status_code
            == 200
        )

        payload = response.json()

        variant = payload.get(
            "variant"
        )

        if variant == "stable":

            stable_count += 1

        elif variant == "canary":

            canary_count += 1

        else:

            unexpected_count += 1

    assert unexpected_count == 0

    assert stable_count > 0
    assert canary_count > 0

    assert (
        stable_count
        > canary_count
    )

    canary_ratio = (
        canary_count
        / request_count
    )

    # -----------------------------------------------------
    # Configured target is 10%.
    #
    # We intentionally allow a wider range because
    # weighted routing should not be tested as an exact
    # random/statistical equality assertion.
    # -----------------------------------------------------

    assert (
        0.03
        <= canary_ratio
        <= 0.25
    )


# =========================================================
# 7. Unknown Gateway route
# =========================================================

def test_unknown_gateway_route_is_rejected():

    response = get(
        f"{GATEWAY_BASE_URL}/"
        "not-a-real-api"
    )

    assert (
        response.status_code
        == 404
    )