#!/usr/bin/env bash

set -euo pipefail


URL="${1:-http://localhost:8008/api/deployment}"

REQUESTS="${2:-7}"


echo "============================================================"
echo "Kong Rate Limit Test"
echo "============================================================"
echo "URL      : ${URL}"
echo "Requests : ${REQUESTS}"
echo


for i in $(seq 1 "${REQUESTS}"); do

    status_code=$(
        curl \
            --silent \
            --output /dev/null \
            --write-out "%{http_code}" \
            "${URL}"
    )

    echo "Request ${i}: ${status_code}"

done