import os

import psycopg
import pytest


pytestmark = pytest.mark.integration


def get_test_connection():

    return psycopg.connect(
        host=os.getenv(
            "DB_HOST",
            "127.0.0.1",
        ),
        port=int(
            os.getenv(
                "DB_PORT",
                "55432",
            )
        ),
        dbname=os.getenv(
            "DB_NAME",
            "steel_quality_ci",
        ),
        user=os.getenv(
            "DB_USER",
            "ci_user",
        ),
        password=os.getenv(
            "DB_PASSWORD",
            "ci_password",
        ),
    )


def test_postgresql_connection():

    with get_test_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT 1"
            )

            result = cursor.fetchone()

    assert result == (1,)


def test_fixture_contains_ten_rows():

    with get_test_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM modeling_steel_quality
                """
            )

            result = cursor.fetchone()

    assert result[0] == 10


def test_fixture_contains_seven_defect_classes():

    with get_test_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    COUNT(
                        DISTINCT defect_type
                    )
                FROM modeling_steel_quality
                """
            )

            result = cursor.fetchone()

    assert result[0] == 7