#!/usr/bin/env bash

set -euo pipefail


URL="${1:-http://localhost:8008/api/deployment}"
REQUESTS="${2:-100}"


stable_count=0
canary_count=0
unknown_count=0


echo "============================================================"
echo "Kong Canary Distribution Test"
echo "============================================================"
echo "URL      : ${URL}"
echo "Requests : ${REQUESTS}"
echo


for i in $(seq 1 "${REQUESTS}"); do

    response=$(
        curl \
            --silent \
            "${URL}"
    )


    if echo "${response}" \
        | grep -q '"variant":"stable"'; then

        stable_count=$((stable_count + 1))

    elif echo "${response}" \
        | grep -q '"variant":"canary"'; then

        canary_count=$((canary_count + 1))

    else

        unknown_count=$((unknown_count + 1))

        echo
        echo "Unexpected response on request ${i}:"
        echo "${response}"

    fi

done


stable_percentage=$(
    awk \
        "BEGIN {
            printf \"%.2f\",
            (${stable_count} / ${REQUESTS}) * 100
        }"
)


canary_percentage=$(
    awk \
        "BEGIN {
            printf \"%.2f\",
            (${canary_count} / ${REQUESTS}) * 100
        }"
)


echo
echo "Stable"
echo "  Count      : ${stable_count}"
echo "  Percentage : ${stable_percentage}%"
echo

echo "Canary"
echo "  Count      : ${canary_count}"
echo "  Percentage : ${canary_percentage}%"
echo

echo "Unknown"
echo "  Count      : ${unknown_count}"
echo


if [ "${stable_count}" -eq 0 ]; then

    echo "FAIL: Stable received no traffic."

    exit 1

fi


if [ "${canary_count}" -eq 0 ]; then

    echo "FAIL: Canary received no traffic."

    exit 1

fi


if [ "${unknown_count}" -ne 0 ]; then

    echo "FAIL: Unexpected responses detected."

    exit 1

fi


echo "PASS: Stable and Canary both received traffic."