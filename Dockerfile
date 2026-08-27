# =========================================================
# Steel Quality Intelligence
# FastAPI Backend Container
# =========================================================


# ---------------------------------------------------------
# 1. Python runtime
# ---------------------------------------------------------

FROM python:3.12-slim


# ---------------------------------------------------------
# 2. Runtime environment
# ---------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=docker \
    API_HOST=0.0.0.0 \
    API_PORT=8000


# ---------------------------------------------------------
# 3. Application working directory
# ---------------------------------------------------------

WORKDIR /app


# ---------------------------------------------------------
# 4. System dependencies
#
# libgomp1:
# Required by XGBoost / numerical runtime.
#
# build-essential:
# Provides compiler toolchain when a Python package
# requires native compilation during installation.
# ---------------------------------------------------------

RUN apt-get update \
    && apt-get install -y \
        --no-install-recommends \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------
# 5. Python dependencies
#
# Copy dependency definition separately so Docker can
# cache the dependency layer when application code changes.
# ---------------------------------------------------------

COPY requirements.txt .


RUN python -m pip install \
        --no-cache-dir \
        --upgrade pip \
    && python -m pip install \
        --no-cache-dir \
        -r requirements.txt


# ---------------------------------------------------------
# 6. Application source
# ---------------------------------------------------------

COPY src ./src


# ---------------------------------------------------------
# 7. ML model artifacts
# ---------------------------------------------------------

COPY models ./models


# ---------------------------------------------------------
# 8. Explainability / analytical artifacts
# ---------------------------------------------------------

COPY reports ./reports


# ---------------------------------------------------------
# 9. Metadata
# ---------------------------------------------------------

COPY metadata ./metadata


# ---------------------------------------------------------
# 10. Processed runtime data
# ---------------------------------------------------------

COPY data/processed ./data/processed


# ---------------------------------------------------------
# 11. Internal application port
#
# Docker Compose controls whether this port is exposed
# externally or only available inside the Docker network.
# ---------------------------------------------------------

EXPOSE 8000


# ---------------------------------------------------------
# 12. FastAPI startup
#
# IMPORTANT:
# Keep exec-form JSON CMD on ONE LINE.
#
# Multi-line JSON CMD caused Dockerfile parsing errors
# because subsequent lines were interpreted as Dockerfile
# instructions.
# ---------------------------------------------------------

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]