#!/usr/bin/env bash

set -Eeuo pipefail


# =========================================================
# Configuration
# =========================================================

PROJECT_NAME="steel-quality-gateway-ci"

BASE_COMPOSE_FILE="docker-compose.yml"

GATEWAY_COMPOSE_FILE="docker-compose.gateway.yml"

GATEWAY_URL="http://localhost:8008/api/health"

FRONTEND_URL="http://localhost:5173/api/health"

MAX_ATTEMPTS=60

SLEEP_SECONDS=2


# =========================================================
# Docker Compose helper
# =========================================================

compose() {

    docker compose \
        -p "${PROJECT_NAME}" \
        -f "${BASE_COMPOSE_FILE}" \
        -f "${GATEWAY_COMPOSE_FILE}" \
        "$@"
}


# =========================================================
# Cleanup / diagnostics
# =========================================================

cleanup() {

    exit_code=$?

    echo
    echo "============================================================"
    echo "Gateway CI Cleanup"
    echo "============================================================"


    if [ "${exit_code}" -ne 0 ]; then

        echo
        echo "Gateway validation failed."
        echo "Collecting Docker diagnostics..."
        echo

        compose ps || true

        echo
        echo "---------------- Kong logs ----------------"
        compose logs kong --tail=200 || true

        echo
        echo "--------------- Backend logs --------------"
        compose logs backend --tail=200 || true

        echo
        echo "------------ Canary backend logs ----------"
        compose logs backend-canary --tail=200 || true

        echo
        echo "-------------- Frontend logs --------------"
        compose logs frontend --tail=200 || true

        echo
        echo "-------------- PostgreSQL logs ------------"
        compose logs postgres --tail=200 || true

    fi


    echo
    echo "Stopping isolated Gateway CI stack..."

    compose down \
        --volumes \
        --remove-orphans \
        || true


    exit "${exit_code}"
}


trap cleanup EXIT


# =========================================================
# Wait helper
# =========================================================

wait_for_url() {

    name="$1"

    url="$2"


    echo
    echo "Waiting for ${name}: ${url}"


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

            echo "${name} ready after attempt ${attempt}."

            return 0

        fi


        echo \
            "${name} not ready " \
            "(attempt ${attempt}/${MAX_ATTEMPTS}, status=${status_code})."


        sleep "${SLEEP_SECONDS}"

    done


    echo "${name} failed to become ready."

    return 1
}


# =========================================================
# 1. Compose validation
# =========================================================

echo "============================================================"
echo "B26.6 — Gateway Docker CI Validation"
echo "============================================================"

echo
echo "Validating Docker Compose configuration..."

compose config > /dev/null

echo "Compose configuration valid."


# =========================================================
# 2. Start isolated stack
# =========================================================

echo
echo "Building and starting Gateway stack..."

compose up \
    --build \
    --detach


# =========================================================
# 3. Container status
# =========================================================

echo
echo "Current container status:"

compose ps


# =========================================================
# 4. Wait for public paths
# =========================================================

wait_for_url \
    "Kong Gateway" \
    "${GATEWAY_URL}"


wait_for_url \
    "Frontend Gateway Path" \
    "${FRONTEND_URL}"


# =========================================================
# 5. Gateway tests
# =========================================================

echo
echo "Running Gateway integration tests..."

pytest \
    tests/integration/test_gateway.py \
    -m "integration and docker" \
    -v


# =========================================================
# 6. Final status
# =========================================================

echo
echo "============================================================"
echo "Gateway Docker CI Validation PASSED"
echo "============================================================"