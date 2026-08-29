#!/usr/bin/env bash

set -Eeuo pipefail


# =========================================================
# B26.7 — Continuous Delivery
#
# CI-gated containerized staging delivery validation.
#
# This script creates an ephemeral Docker Compose staging
# environment, validates the deployed application, and
# removes the environment afterwards.
#
# This is NOT a persistent production deployment.
# =========================================================


# =========================================================
# 1. Configuration
# =========================================================

PROJECT_NAME="steel-quality-staging-${GITHUB_RUN_ID:-local}"

BASE_COMPOSE_FILE="docker-compose.yml"

GATEWAY_COMPOSE_FILE="docker-compose.gateway.yml"


# ---------------------------------------------------------
# Staging-only configuration
# ---------------------------------------------------------

export POSTGRES_DB="${POSTGRES_DB:-steel_quality_staging}"

export POSTGRES_USER="${POSTGRES_USER:-staging_user}"

export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-staging_password}"

export APP_ENV="staging"

export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:9b}"

export FRONTEND_PORT="${FRONTEND_PORT:-15173}"

export KONG_PROXY_PORT="${KONG_PROXY_PORT:-18008}"

export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}}"


# ---------------------------------------------------------
# Public staging URLs
# ---------------------------------------------------------

GATEWAY_BASE_URL="http://localhost:${KONG_PROXY_PORT}"

FRONTEND_BASE_URL="http://localhost:${FRONTEND_PORT}"

GATEWAY_HEALTH_URL="${GATEWAY_BASE_URL}/api/health"

FRONTEND_HEALTH_URL="${FRONTEND_BASE_URL}/api/health"

DEPLOYMENT_URL="${GATEWAY_BASE_URL}/api/deployment"


# ---------------------------------------------------------
# Retry policy
# ---------------------------------------------------------

MAX_ATTEMPTS=60

SLEEP_SECONDS=2

MAX_VARIANT_ATTEMPTS=40


# =========================================================
# 2. Docker Compose helper
# =========================================================

compose() {

    docker compose \
        -p "${PROJECT_NAME}" \
        -f "${BASE_COMPOSE_FILE}" \
        -f "${GATEWAY_COMPOSE_FILE}" \
        "$@"
}


# =========================================================
# 3. Cleanup / diagnostics
# =========================================================

cleanup() {

    exit_code=$?


    echo
    echo "============================================================"
    echo "Staging Delivery Cleanup"
    echo "============================================================"


    if [ "${exit_code}" -ne 0 ]; then

        echo
        echo "Staging delivery validation FAILED."
        echo "Collecting diagnostics..."
        echo


        echo "---------------- Container status ----------------"

        compose ps || true


        echo
        echo "---------------- Kong logs ----------------"

        compose logs \
            kong \
            --tail=200 \
            || true


        echo
        echo "--------------- Stable backend logs --------------"

        compose logs \
            backend \
            --tail=200 \
            || true


        echo
        echo "--------------- Canary backend logs --------------"

        compose logs \
            backend-canary \
            --tail=200 \
            || true


        echo
        echo "---------------- Frontend logs -------------------"

        compose logs \
            frontend \
            --tail=200 \
            || true


        echo
        echo "--------------- PostgreSQL logs ------------------"

        compose logs \
            postgres \
            --tail=200 \
            || true
    fi


    echo
    echo "Removing ephemeral staging environment..."


    compose down \
        --volumes \
        --remove-orphans \
        || true


    echo

    if [ "${exit_code}" -eq 0 ]; then

        echo "Staging environment removed successfully."

    else

        echo "Staging environment removed after failure."
    fi


    exit "${exit_code}"
}


trap cleanup EXIT


# =========================================================
# 4. HTTP readiness helper
# =========================================================

wait_for_url() {

    name="$1"

    url="$2"


    echo
    echo "Waiting for ${name}:"
    echo "${url}"


    for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do

        status_code=$(

            curl \
                --silent \
                --output /dev/null \
                --write-out "%{http_code}" \
                "${url}" \
                || true
        )


        if [ "${status_code}" = "200" ]; then

            echo
            echo "${name} ready after attempt ${attempt}."

            return 0
        fi


        echo \
            "${name} not ready " \
            "(attempt ${attempt}/${MAX_ATTEMPTS}, HTTP ${status_code})."


        sleep "${SLEEP_SECONDS}"
    done


    echo
    echo "${name} failed to become ready."

    return 1
}


# =========================================================
# 5. Health payload validation
# =========================================================

validate_health_payload() {

    name="$1"

    url="$2"


    echo
    echo "Validating ${name} health payload..."


    payload=$(

        curl \
            --fail \
            --silent \
            --show-error \
            "${url}"
    )


    echo "${payload}"


    HEALTH_PAYLOAD="${payload}" \
    python - <<'PY'
import json
import os

payload = json.loads(
    os.environ["HEALTH_PAYLOAD"]
)

status = payload.get("status")
database = payload.get("database")

if status != "healthy":
    raise SystemExit(
        f"Expected status=healthy, got {status!r}"
    )

if database != "healthy":
    raise SystemExit(
        f"Expected database=healthy, got {database!r}"
    )

print(
    "Health payload verified: "
    "status=healthy, database=healthy"
)
PY
}


# =========================================================
# 6. Deployment metadata validation
# =========================================================

validate_deployment_metadata() {

    echo
    echo "Validating deployment metadata..."


    payload=$(

        curl \
            --fail \
            --silent \
            --show-error \
            "${DEPLOYMENT_URL}"
    )


    echo "${payload}"


    DEPLOYMENT_PAYLOAD="${payload}" \
    python - <<'PY'
import json
import os

payload = json.loads(
    os.environ["DEPLOYMENT_PAYLOAD"]
)

required_fields = {
    "variant",
    "version",
    "environment",
}

missing = (
    required_fields
    - payload.keys()
)

if missing:
    raise SystemExit(
        "Deployment metadata missing fields: "
        + ", ".join(
            sorted(missing)
        )
    )

variant = payload["variant"]

version = payload["version"]

environment = payload["environment"]


if variant not in {
    "stable",
    "canary",
}:
    raise SystemExit(
        f"Unexpected variant: {variant!r}"
    )


expected_versions = {
    "stable": "v1",
    "canary": "v2",
}

expected_version = (
    expected_versions[variant]
)

if version != expected_version:
    raise SystemExit(
        "Unexpected version for "
        f"{variant}: "
        f"expected {expected_version!r}, "
        f"got {version!r}"
    )


if not environment:
    raise SystemExit(
        "Deployment environment must not be empty."
    )


print(
    "Deployment metadata verified: "
    f"variant={variant}, "
    f"version={version}, "
    f"environment={environment}"
)
PY
}


# =========================================================
# 7. Kong response validation
# =========================================================

validate_kong_header() {

    echo
    echo "Validating Kong proxy evidence..."


    headers=$(

        curl \
            --silent \
            --show-error \
            --dump-header - \
            --output /dev/null \
            "${GATEWAY_HEALTH_URL}"
    )


    echo "${headers}"


    if ! printf '%s\n' "${headers}" \
        | grep \
            --ignore-case \
            --extended-regexp \
            '^via:.*kong' \
            > /dev/null; then

        echo
        echo "Expected Kong Via header was not found."

        return 1
    fi


    echo "Kong Via header verified."
}


# =========================================================
# 8. Stable / Canary availability validation
# =========================================================

validate_variants() {

    echo
    echo "Validating Stable / Canary availability..."


    stable_count=0

    canary_count=0

    unknown_count=0


    for attempt in $(seq 1 "${MAX_VARIANT_ATTEMPTS}"); do

        payload=$(

            curl \
                --fail \
                --silent \
                --show-error \
                "${DEPLOYMENT_URL}"
        )


        variant=$(

            DEPLOYMENT_PAYLOAD="${payload}" \
            python - <<'PY'
import json
import os

payload = json.loads(
    os.environ["DEPLOYMENT_PAYLOAD"]
)

print(
    payload.get(
        "variant",
        "unknown",
    )
)
PY
        )


        case "${variant}" in

            stable)

                stable_count=$((stable_count + 1))

                ;;

            canary)

                canary_count=$((canary_count + 1))

                ;;

            *)

                unknown_count=$((unknown_count + 1))

                ;;
        esac


        if \
            [ "${stable_count}" -gt 0 ] \
            && \
            [ "${canary_count}" -gt 0 ]; then

            break
        fi
    done


    echo
    echo "Stable responses : ${stable_count}"
    echo "Canary responses : ${canary_count}"
    echo "Unknown responses: ${unknown_count}"


    if [ "${stable_count}" -eq 0 ]; then

        echo
        echo "Stable backend did not serve traffic."

        return 1
    fi


    if [ "${canary_count}" -eq 0 ]; then

        echo
        echo "Canary backend did not serve traffic."

        return 1
    fi


    if [ "${unknown_count}" -ne 0 ]; then

        echo
        echo "Unknown deployment variant detected."

        return 1
    fi


    echo
    echo "Stable and Canary availability verified."
}


# =========================================================
# 9. Compose validation
# =========================================================

echo "============================================================"
echo "B26.7 — Continuous Delivery"
echo "Ephemeral Staging Delivery Validation"
echo "============================================================"

echo
echo "Project name:"
echo "${PROJECT_NAME}"

echo
echo "Frontend staging URL:"
echo "${FRONTEND_BASE_URL}"

echo
echo "Kong staging URL:"
echo "${GATEWAY_BASE_URL}"


echo
echo "Validating Docker Compose configuration..."


compose config \
    > /dev/null


echo "Docker Compose configuration valid."


# =========================================================
# 10. Build and deploy staging stack
# =========================================================

echo
echo "Building and starting ephemeral staging environment..."


compose up \
    --build \
    --detach


# =========================================================
# 11. Container status
# =========================================================

echo
echo "Current staging container status:"


compose ps


# =========================================================
# 12. Wait for deployed public paths
# =========================================================

wait_for_url \
    "Kong Gateway" \
    "${GATEWAY_HEALTH_URL}"


wait_for_url \
    "Frontend → Nginx → Kong → FastAPI" \
    "${FRONTEND_HEALTH_URL}"


# =========================================================
# 13. Post-deployment smoke tests
# =========================================================

echo
echo "============================================================"
echo "Post-deployment Smoke Tests"
echo "============================================================"


# ---------------------------------------------------------
# A. Kong → FastAPI health
# ---------------------------------------------------------

validate_health_payload \
    "Kong Gateway" \
    "${GATEWAY_HEALTH_URL}"


# ---------------------------------------------------------
# B. Frontend → Nginx → Kong → FastAPI health
# ---------------------------------------------------------

validate_health_payload \
    "Frontend Gateway Path" \
    "${FRONTEND_HEALTH_URL}"


# ---------------------------------------------------------
# C. Deployment metadata
# ---------------------------------------------------------

validate_deployment_metadata


# ---------------------------------------------------------
# D. Kong response evidence
# ---------------------------------------------------------

validate_kong_header


# ---------------------------------------------------------
# E. Stable / Canary availability
# ---------------------------------------------------------

validate_variants


# =========================================================
# 14. Final delivery result
# =========================================================

echo
echo "============================================================"
echo "Staging Delivery Verification PASSED"
echo "============================================================"

echo
echo "Validated:"
echo "- Kong → FastAPI health"
echo "- Frontend → Nginx → Kong → FastAPI health"
echo "- Deployment metadata"
echo "- Kong proxy evidence"
echo "- Stable backend availability"
echo "- Canary backend availability"

echo
echo "This was an ephemeral staging deployment."
echo "No persistent production environment was modified."