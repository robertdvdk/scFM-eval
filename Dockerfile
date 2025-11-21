# 1. Build Stage: Use uv to install dependencies
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
# Copy mode is required for Docker layer caching to work reliably
ENV UV_LINK_MODE=copy

# Install dependencies specifically (this layer is cached until pyproject.toml changes)
# We install into a virtual env at /app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. Runtime Stage: Copy only the environment
FROM python:3.13-slim
WORKDIR /app

# Copy the pre-built virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code AND configuration, preserving folder structure
# This ensures /app/src/main.py exists and /app/configs/config.yaml exists
COPY src/ src/
COPY configs/ configs/

# Run the app pointing to the file inside src/
ENTRYPOINT ["python", "src/main.py"]
