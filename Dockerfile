# 1. Build Stage: Use uv to install dependencies
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
# Copy mode is required for Docker layer caching to work reliably
ENV UV_LINK_MODE=copy

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

# Install dependencies specifically (this layer is cached until pyproject.toml changes)
# We install into a virtual env at /app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 2. Runtime Stage: Copy only the environment
FROM python:3.13-slim
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update && \
    apt-get install -y git wget build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for devcontainer
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

# Copy the pre-built virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Add the virtual environment to the PATH
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code AND configuration, preserving folder structure
# This ensures /app/src/main.py exists and /app/configs/config.yaml exists
COPY src/ src/
COPY configs/ configs/

# Run the app pointing to the file inside src/
ENTRYPOINT ["python", "src/main.py"]
