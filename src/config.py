import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    ENV_PATH,
    override=False,
)


# =========================================================
# 2. Application configuration
# =========================================================

APP_ENV = os.getenv(
    "APP_ENV",
    "development",
)


DEPLOYMENT_VARIANT = os.getenv(
    "DEPLOYMENT_VARIANT",
    "local",
)


APP_VERSION = os.getenv(
    "APP_VERSION",
    "dev",
)


API_HOST = os.getenv(
    "API_HOST",
    "0.0.0.0",
)



API_PORT = int(
    os.getenv(
        "API_PORT",
        "8000",
    )
)


# =========================================================
# 3. PostgreSQL configuration
# =========================================================

DB_HOST = os.getenv(
    "DB_HOST",
    "localhost",
)

DB_PORT = int(
    os.getenv(
        "DB_PORT",
        "5432",
    )
)

DB_NAME = os.getenv(
    "DB_NAME",
    "steel_quality",
)

DB_USER = os.getenv(
    "DB_USER",
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD",
    "",
)


DB_CONFIG = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
}


# =========================================================
# 4. Ollama configuration
# =========================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3.5:9b",
)


# =========================================================
# 5. CORS configuration
# =========================================================

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)


def parse_cors_origins(
    raw_value: str,
) -> List[str]:
    """
    Convert comma-separated CORS origins into
    a clean list.
    """

    return [
        origin.strip()
        for origin in raw_value.split(",")
        if origin.strip()
    ]


CORS_ORIGINS = parse_cors_origins(
    os.getenv(
        "CORS_ORIGINS",
        DEFAULT_CORS_ORIGINS,
    )
)


# =========================================================
# 6. Runtime validation
# =========================================================

def validate_runtime_configuration() -> None:
    """
    Validate configuration required by the
    application runtime.

    Password is intentionally optional because
    local PostgreSQL installations may use
    trusted/local authentication.
    """

    if not DB_USER:

        raise ValueError(
            "DB_USER is missing. "
            "Configure it through environment variables."
        )

    if not OLLAMA_BASE_URL:

        raise ValueError(
            "OLLAMA_BASE_URL is missing."
        )

    if not OLLAMA_MODEL:

        raise ValueError(
            "OLLAMA_MODEL is missing."
        )