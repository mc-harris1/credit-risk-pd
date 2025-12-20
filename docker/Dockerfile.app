FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1

# System deps
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

# Copy metadata first for caching
COPY pyproject.toml ./
COPY README.md ./
# If you have a uv.lock, uncomment:
COPY uv.lock ./

# Install only production deps
RUN uv sync --no-dev --frozen || uv sync --no-dev

# Copy Streamlit + shared code it imports
COPY .streamlit ./.streamlit
COPY streamlit_app ./streamlit_app
COPY src ./src
COPY models ./models

EXPOSE 8501

# Streamlit must bind 0.0.0.0 and use App Runner PORT if set
CMD ["sh","-c","uv run streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
