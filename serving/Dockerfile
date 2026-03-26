FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install only serving dependencies (no dev extras)
RUN uv sync --frozen --no-dev --extra serving

# Copy application code
COPY src/ src/
COPY serving/ serving/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
