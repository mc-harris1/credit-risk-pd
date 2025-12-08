FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      build-essential \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy project metadata separately to maximize Docker build caching
COPY pyproject.toml ./
# If you have a uv.lock, uncomment this line:
# COPY uv.lock ./

# Install only production dependencies
RUN uv sync --no-dev --frozen || uv sync --no-dev

# Copy source code after dependencies for better caching
COPY src ./src
COPY streamlit_app ./streamlit_app

# Copy model directory structure (metadata will be populated at runtime/training)
COPY models ./models

# Copy only processed/interim data needed at runtime; do NOT copy raw CSVs
COPY data/interim ./data/interim
COPY data/processed ./data/processed

COPY README.md .

EXPOSE 8000

# Default container command
CMD ["uv", "run", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
