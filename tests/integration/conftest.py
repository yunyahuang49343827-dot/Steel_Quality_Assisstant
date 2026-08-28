import os
import time

import psycopg
import pytest


TEST_DB_CONFIG = {
    "host": os.getenv(
        "DB_HOST",
        "127.0.0.1",
    ),
    "port": int(
        os.getenv(
            "DB_PORT",
            "55432",
        )
    ),
    "dbname": os.getenv(
        "DB_NAME",
        "steel_quality_ci",
    ),
    "user": os.getenv(
        "DB_USER",
        "ci_user",
    ),
    "password": os.getenv(
        "DB_PASSWORD",
        "ci_password",
    ),
}


TEST_DEFECT_TYPES = [
    "Other_Faults",
    "Other_Faults",
    "Other_Faults",
    "Bumps",
    "Bumps",
    "K_Scatch",
    "Pastry",
    "Z_Scratch",
    "Stains",
    "Dirtiness",
]


def wait_for_postgres(
    attempts: int = 20,
    delay_seconds: float = 1.0,
) -> None:
    """
    Wait until the PostgreSQL integration-test
    instance accepts connections.
    """

    last_error = None

    for _ in range(
        attempts
    ):

        try:

            with psycopg.connect(
                **TEST_DB_CONFIG
            ) as connection:

                with connection.cursor() as cursor:

                    cursor.execute(
                        "SELECT 1"
                    )

                    cursor.fetchone()

            return

        except psycopg.Error as exc:

            last_error = exc

            time.sleep(
                delay_seconds
            )

    raise RuntimeError(
        "PostgreSQL integration-test database "
        "did not become ready."
    ) from last_error


@pytest.fixture(
    scope="session",
)
def integration_database():
    """
    Create deterministic integration-test data.

    The temporary database contains only the columns
    required by the current analytics queries.
    """

    wait_for_postgres()

    with psycopg.connect(
        **TEST_DB_CONFIG
    ) as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                DROP TABLE IF EXISTS
                    modeling_steel_quality
                """
            )

            cursor.execute(
                """
                CREATE TABLE
                    modeling_steel_quality
                (
                    id BIGSERIAL PRIMARY KEY,
                    defect_type TEXT NOT NULL
                )
                """
            )

            cursor.executemany(
                """
                INSERT INTO
                    modeling_steel_quality
                    (
                        defect_type
                    )
                VALUES
                    (%s)
                """,
                [
                    (
                        defect_type,
                    )
                    for defect_type
                    in TEST_DEFECT_TYPES
                ],
            )

        connection.commit()

    yield

    with psycopg.connect(
        **TEST_DB_CONFIG
    ) as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                DROP TABLE IF EXISTS
                    modeling_steel_quality
                """
            )

        connection.commit()